"""
Attention with TTT/DeltaNet-style Memory

Key features:
- Position-invariant memory keys (theta_K projection)
- Position-invariant memory queries (theta_Q projection)  
- Values are NEXT token embeddings (not current hidden states)
- Learned per-token writing strength beta (DeltaNet-style)
"""
import math
import random

import torch
import torch.nn.functional as F
from torch import nn


def get_freqs(n: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    """Compute RoPE frequencies."""
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(nn.Module):
    """
    Attention layer with associative memory using delta rule updates.
    
    Memory stores associations: (current_token -> next_token_embedding)
    so queries with current_token retrieve info about next token.
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        nh = config.n_head
        N = config.N
        D = config.n_embd

        # RoPE frequencies
        self.freqs = nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )
        
        # Memory gate (learned blend between standard attention and memory)
        self.memory_gate = nn.Parameter(torch.full((1, nh, 1, 1), -2.0))
        self.memory_ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        
        # Memory matrix buffer
        self.register_buffer('memory_M', None, persistent=False)
        self.register_buffer('memory_norm_tracker', torch.tensor(0.0), persistent=False)

        # TTT/DeltaNet-style memory projections
        # theta_K: projects input to position-invariant memory key
        # theta_Q: projects input to position-invariant memory query
        # beta_proj: learned writing strength per head
        self.theta_K = nn.Linear(D, N, bias=False)
        self.theta_Q = nn.Linear(D, N, bias=False)
        self.beta_proj = nn.Linear(D, nh, bias=False)

        # Initialize
        nn.init.normal_(self.theta_K.weight, std=0.02)
        nn.init.normal_(self.theta_Q.weight, std=0.02)
        nn.init.zeros_(self.beta_proj.weight)  # beta starts at 0.5 (sigmoid(0))

    @staticmethod
    def phases_cos_sin(phases: torch.Tensor):
        """Convert phases to cos/sin for RoPE."""
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding."""
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = Attention.phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def reset_memory(self):
        """Clear the memory matrix."""
        self.memory_M = None
        self.memory_norm_tracker = torch.tensor(0.0, device=self.memory_norm_tracker.device)

    def _delta_update(self, x_raw: torch.Tensor, x_next: torch.Tensor, 
                      B: int, nh: int, N: int, D: int):
        """
        TTT/DeltaNet-style delta update.

        Args:
            x_raw: Current token embeddings [B, T, D] - RAW, no position encoding
            x_next: Next token embeddings [B, T, D] - what we want to retrieve
            B, nh, N, D: dimensions
        """
        if self.memory_M is None:
            self.memory_M = torch.zeros(B, nh, N, D, device=x_raw.device, dtype=x_raw.dtype)

        T = x_raw.size(1)

        # Compute position-invariant memory keys from current tokens
        K_mem = self.theta_K(x_raw)  # [B, T, N]
        K_mem = K_mem.unsqueeze(1).expand(-1, nh, -1, -1)  # [B, nh, T, N]
        K_mem = F.normalize(K_mem, p=2, dim=-1)

        # Value is the NEXT token's embedding (what we want to retrieve later)
        V_mem = x_next.unsqueeze(1).expand(-1, nh, -1, -1)  # [B, nh, T, D]

        # Compute per-token writing strength beta
        beta = torch.sigmoid(self.beta_proj(x_raw))  # [B, T, nh]
        beta = beta.permute(0, 2, 1).unsqueeze(-1)  # [B, nh, T, 1]

        # Detach for memory update (train theta_K, theta_Q, beta_proj via retrieval loss)
        M = self.memory_M.detach().clone()
        K_det = K_mem.detach()
        V_det = V_mem.detach()
        beta_det = beta.detach()

        # Delta rule update for all positions
        for t in range(T):
            k_t = K_det[:, :, t:t+1, :]  # [B, nh, 1, N]
            v_t = V_det[:, :, t:t+1, :]  # [B, nh, 1, D]
            b_t = beta_det[:, :, t:t+1, :]  # [B, nh, 1, 1]

            # Retrieve old value for this key
            v_old = k_t @ M  # [B, nh, 1, D]

            # Interpolate: new value is mix of target and retrieved
            v_new = b_t * v_t + (1 - b_t) * v_old

            # Update: outer product
            v_diff = v_new - v_old  # [B, nh, 1, D]
            update = k_t.transpose(-2, -1) @ v_diff  # [B, nh, N, D]
            M = M + update

        self.memory_M = M
        self.memory_norm_tracker = M.norm().clone().detach()

    def _memory_retrieve(self, x_raw: torch.Tensor, 
                         B: int, nh: int, N: int, D: int) -> torch.Tensor | None:
        """
        Retrieve from memory using position-invariant query.

        Args:
            x_raw: Current token embeddings [B, T, D] - RAW, no position encoding

        Returns:
            y_memory: Retrieved values [B, nh, T, D] or None
        """
        if self.memory_M is None:
            return None

        # Compute position-invariant query
        Q_mem = self.theta_Q(x_raw)  # [B, T, N]
        Q_mem = Q_mem.unsqueeze(1).expand(-1, nh, -1, -1)  # [B, nh, T, N]
        Q_mem = F.normalize(Q_mem, p=2, dim=-1)

        # Retrieve
        y_memory = Q_mem @ self.memory_M  # [B, nh, T, D]
        return y_memory

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                x_raw: torch.Tensor = None, x_next: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional memory.

        Args:
            Q, K, V: Standard attention inputs
            x_raw: Raw token embeddings (no position) for memory K/Q [B, T, D]
            x_next: Next token embeddings (shifted) for memory values [B, T, D]
        """
        assert K is Q
        B, nh, T, N = Q.size()
        D = V.size(-1)

        # Standard causal attention with RoPE
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR

        scores = (QR @ KR.mT).tril(diagonal=-1)
        y_standard = scores @ V

        # Determine if we should use memory
        use_mem = False
        if self.training and self.config.memory_train_prob > 0:
            use_mem = (random.random() < self.config.memory_train_prob)
        elif not self.training and self.config.use_memory:
            use_mem = True

        if use_mem and x_raw is not None:
            # Retrieve from memory
            y_memory = self._memory_retrieve(x_raw, B, nh, N, D)
            if y_memory is not None:
                y_memory = self.memory_ln(y_memory)
                y_memory = y_memory * self.config.memory_retrieval_scale
            else:
                y_memory = torch.zeros_like(y_standard)

            # Update memory (if not frozen and we have next tokens)
            if not self.config.memory_freeze and x_next is not None:
                self._delta_update(x_raw, x_next, B, nh, N, D)

            # Blend standard attention with memory
            gate = torch.sigmoid(self.memory_gate)
            return (1 - gate) * y_standard + gate * y_memory

        return y_standard

    def gate_values(self) -> list:
        """Get current gate values (after sigmoid)."""
        gates = torch.sigmoid(self.memory_gate).squeeze().tolist()
        if not isinstance(gates, list):
            gates = [gates]
        return gates

    def get_diagnostics(self) -> dict:
        """Get diagnostic info about memory state."""
        gates = self.gate_values()
        if not isinstance(gates, list):
            gates = [gates]
        if self.memory_M is None:
            return {"status": "empty", "norm": 0, "gates": gates}
        M = self.memory_M
        return {
            "status": "active",
            "norm": M.norm().item(),
            "mean": M.mean().item(),
            "std": M.std().item(),
            "max_abs": M.abs().max().item(),
            "shape": list(M.shape),
            "gates": gates,
        }
