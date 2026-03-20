"""
BDH Model - Baby Dragon Hatchling Architecture

Combines:
- Sparse encoding/decoding
- Attention with delta-rule memory
- Output-level Latent RAG cache
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import BDHConfig
from .attention import Attention


class BDH(nn.Module):
    """
    BDH: Baby Dragon Hatchling Architecture model with associative memory.
    
    Features:
    - Per-layer sparse encoders/decoders
    - TTT/DeltaNet-style attention memory
    - Zero-training output-level Latent RAG cache
    """
    
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh, D, N = config.n_head, config.n_embd, config.N

        # Per-layer components
        self.encoders = nn.ParameterList()
        self.encoders_v = nn.ParameterList()
        self.decoders = nn.ParameterList()
        self.attns = nn.ModuleList()
        self.drops = nn.ModuleList()
        self.lns = nn.ModuleList()

        for li in range(config.n_layer):
            self.encoders.append(
                nn.Parameter(torch.empty(nh, D, N).normal_(std=0.02))
            )
            self.encoders_v.append(
                nn.Parameter(torch.empty(nh, D, N).normal_(std=0.02))
            )
            self.decoders.append(
                nn.Parameter(torch.empty(nh * N, D).normal_(std=0.02 / math.sqrt(2 * config.n_layer)))
            )
            self.attns.append(Attention(config, li))
            self.drops.append(nn.Dropout(config.dropout))
            self.lns.append(nn.LayerNorm(D, elementwise_affine=False, bias=False))

        # Final layer norm and embeddings
        self.ln_f = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.pos_embed = nn.Embedding(config.block_size, D)
        
        self.apply(self._init_weights)

        # Zero-Training Output-Level Latent RAG Cache
        self.output_memory_keys = []
        self.output_memory_values = []
        self.rag_threshold = 0.85
        self.rag_gate = 0.95

        n_params = sum(p.numel() for p in self.parameters())
        print(f"BDH: {n_params/1e6:.1f}M params")
        print(f"  {config.n_layer}L D={D} {nh}H N={N}/head")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ══════════════════════════════════════════════════════════════
    # RAG Helper Methods
    # ══════════════════════════════════════════════════════════════
    
    def _get_rag_key(self, token_ids: torch.Tensor | list, n_context: int = 5) -> torch.Tensor:
        """Create parameter-free positional fingerprint for RAG."""
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, device=self.embed.weight.device)
        ctx = token_ids[-n_context:]
        embeds = self.embed.weight[ctx]
        # Exponential weighting: [1, 2, 4, 8, 16]
        weights = torch.tensor([2.0 ** i for i in range(len(ctx))], device=embeds.device)
        weights = weights / weights.sum()
        key = (embeds * weights.unsqueeze(1)).sum(dim=0)
        return F.normalize(key, p=2, dim=0)

    def memorize(self, tokens: list):
        """Store a sequence (list of ints) into the Latent RAG cache."""
        for i in range(4, len(tokens) - 1):
            ctx = tokens[:i+1]
            next_tok = tokens[i+1]
            key = self._get_rag_key(ctx)
            val = self.embed.weight[next_tok].detach().clone()
            self.output_memory_keys.append(key)
            self.output_memory_values.append(val)

    # ══════════════════════════════════════════════════════════════
    # Forward Pass
    # ══════════════════════════════════════════════════════════════
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass.
        
        Args:
            idx: Input token indices [B, T]
            targets: Target token indices [B, T] for loss computation
            
        Returns:
            logits: Output logits [B, T, vocab_size]
            loss: Cross-entropy loss if targets provided, else None
        """
        C = self.config
        B, T = idx.size()
        D, nh, N = C.n_embd, C.n_head, C.N

        # Token embeddings
        x_raw = self.embed(idx)  # [B, T, D]

        # Create shifted next-token embeddings for memory
        if T > 1:
            x_next = torch.cat([
                x_raw[:, 1:, :],
                torch.zeros(B, 1, D, device=idx.device, dtype=x_raw.dtype)
            ], dim=1)
        else:
            x_next = torch.zeros_like(x_raw)

        # Add position embeddings
        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = x_raw + self.pos_embed(pos)
        x = x.unsqueeze(1)  # [B, 1, T, D]

        # Process through layers
        for i in range(C.n_layer):
            x_sparse = F.relu(x @ self.encoders[i])

            yKV = self.lns[i](self.attns[i](
                Q=x_sparse,
                K=x_sparse,
                V=x,
                x_raw=x_raw,
                x_next=x_next
            ))

            y_sparse = F.relu(yKV @ self.encoders_v[i])
            xy = self.drops[i](x_sparse * y_sparse)
            yMLP = xy.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoders[i]
            x = self.lns[i](x + self.lns[i](yMLP))

        x = x.view(B, T, D)

        # Latent RAG Interceptor (inference only)
        if not self.training and len(self.output_memory_keys) > 0 and targets is None:
            token_list = idx[0].tolist()
            query_key = self._get_rag_key(token_list)
            keys_tensor = torch.stack(self.output_memory_keys)  # [M, D]

            # Fast Cosine Similarity search
            sims = F.cosine_similarity(query_key.unsqueeze(0), keys_tensor, dim=1)
            best_idx = sims.argmax().item()
            best_sim = sims[best_idx].item()

            if best_sim > self.rag_threshold:
                retrieved_val = self.output_memory_values[best_idx]
                # Overwrite the residual stream right before LayerNorm
                x[0, -1, :] = (1 - self.rag_gate) * x[0, -1, :] + (self.rag_gate * retrieved_val)

        # Output projection
        logits = self.ln_f(x) @ self.embed.weight.T

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, C.vocab_size), targets.view(-1))
        
        return logits, loss

    # ══════════════════════════════════════════════════════════════
    # Memory Management
    # ══════════════════════════════════════════════════════════════
    
    def reset_all_memory(self):
        """Flush both internal attention matrices and Latent RAG cache."""
        for a in self.attns:
            a.reset_memory()
        self.output_memory_keys.clear()
        self.output_memory_values.clear()

    def save_all_synapses(self, path: str):
        """Save memory matrices to disk."""
        states = {}
        for i, a in enumerate(self.attns):
            states[f'L{i}'] = {
                'M': a.memory_M.cpu() if a.memory_M is not None else None,
                'norm': a.memory_norm_tracker.cpu(),
            }
        torch.save(states, path)
        print(f"  Synapses saved → {path}")

    def load_all_synapses(self, path: str, device: str = None):
        """Load memory matrices from disk."""
        states = torch.load(path, map_location=device, weights_only=True)
        for i, a in enumerate(self.attns):
            s = states[f'L{i}']
            a.memory_M = s['M'].to(device) if s['M'] is not None else None
            a.memory_norm_tracker = s.get('norm', torch.tensor(0.0)).to(device)
        print(f"  Synapses loaded ← {path}")

    def get_all_diagnostics(self) -> dict:
        """Get diagnostics for all attention layers."""
        return {f'L{i}': a.get_diagnostics() for i, a in enumerate(self.attns)}

    def print_gates(self):
        """Print gate values for all layers."""
        for i, a in enumerate(self.attns):
            gv = a.gate_values()
            if not isinstance(gv, list):
                gv = [gv]
            gs = ', '.join(f'{g:.3f}' for g in gv)
            print(f"    L{i} gates: [{gs}]")

    # ══════════════════════════════════════════════════════════════
    # Generation
    # ══════════════════════════════════════════════════════════════
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: int = None,
                 repetition_penalty: float = 1.2) -> torch.Tensor:
        """
        Generate with memory frozen (read-only) and repetition penalty.
        
        Args:
            idx: Starting token indices [B, T]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for no filtering)
            repetition_penalty: Penalty for repeated tokens
            
        Returns:
            idx: Generated sequence [B, T + max_new_tokens]
        """
        old_freeze = self.config.memory_freeze
        self.config.memory_freeze = True

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.config.memory_freeze = old_freeze
        return idx
