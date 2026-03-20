"""
Phase 2: Memory Gate Training

Trains the memory gates, theta_K/Q projections, and beta
using contrastive loss to encourage orthogonal keys.
"""
import os
import time
import math
import types
import random
from functools import partial

import torch
import torch.nn.functional as F

from .utils import (
    create_optimizer,
    save_checkpoint,
    TrainConfig,
)


# ══════════════════════════════════════════════════════════════
# Phase 2 Attention Patches
# ══════════════════════════════════════════════════════════════

def _phase2_delta_update(self, x_raw, x_next, B, nh, N, D, chunk_size=64):
    """
    Chunked delta update for Phase 2 training.
    
    Processes tokens in chunks for better speed/memory tradeoff.
    """
    if self.memory_M is None:
        self.memory_M = torch.zeros(B, nh, N, D, device=x_raw.device, dtype=x_raw.dtype)

    T = x_raw.size(1)

    K_mem = self.theta_K(x_raw)
    K_mem = K_mem.unsqueeze(1).expand(-1, nh, -1, -1)
    K_mem = F.normalize(K_mem, p=2, dim=-1)

    V_mem = x_next.unsqueeze(1).expand(-1, nh, -1, -1)

    beta = torch.sigmoid(self.beta_proj(x_raw))
    beta = beta.permute(0, 2, 1).unsqueeze(-1)

    M = self.memory_M.detach()
    K_det = K_mem.detach()
    V_det = V_mem.detach()
    beta_det = beta.detach()

    # Process in chunks
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)

        k_chunk = K_det[:, :, start:end, :]
        v_chunk = V_det[:, :, start:end, :]
        b_chunk = beta_det[:, :, start:end, :]

        # Vectorized within chunk
        v_old = k_chunk @ M
        v_new = b_chunk * v_chunk + (1 - b_chunk) * v_old
        v_diff = v_new - v_old
        update = k_chunk.transpose(-2, -1) @ v_diff

        M = M + self.config.memory_lr * update

    self.memory_M = M
    self.memory_norm_tracker = M.norm().clone().detach()


def _phase2_forward(self, Q, K, V, x_raw=None, x_next=None):
    """
    Phase 2 forward - allows gradients through memory retrieval for theta_Q.
    """
    assert K is Q
    B, nh, T, N = Q.size()
    D = V.size(-1)

    r_phases = (
        torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
        .view(1, 1, -1, 1)
    ) * self.freqs
    QR = self.rope(r_phases, Q)
    KR = QR
    scores = (QR @ KR.mT).tril(diagonal=-1)
    y_standard = scores @ V

    use_mem = False
    if self.training and self.config.memory_train_prob > 0:
        use_mem = (random.random() < self.config.memory_train_prob)
    elif not self.training and self.config.use_memory:
        use_mem = True

    if use_mem and x_raw is not None:
        if self.memory_M is not None and self.memory_M.shape[0] == B:
            # Compute position-invariant query (WITH gradients for theta_Q)
            Q_mem = self.theta_Q(x_raw)
            Q_mem = Q_mem.unsqueeze(1).expand(-1, nh, -1, -1)
            Q_mem = F.normalize(Q_mem, p=2, dim=-1)

            # Retrieve (detach M to avoid backprop through it)
            y_memory = self.memory_ln(Q_mem @ self.memory_M.detach())
            y_memory = y_memory * self.config.memory_retrieval_scale
        else:
            y_memory = torch.zeros_like(y_standard)

        if not self.config.memory_freeze and x_next is not None:
            self._delta_update(x_raw, x_next, B, nh, N, D)

        gate = torch.sigmoid(self.memory_gate)
        return (1 - gate) * y_standard + gate * y_memory

    return y_standard


def apply_phase2_patches(model):
    """Apply Phase 2 method patches to model attention layers."""
    for attn in model.attns:
        attn._delta_update = types.MethodType(_phase2_delta_update, attn)
        attn.forward = types.MethodType(_phase2_forward, attn)
    print("  Attention patched for Phase 2")


def remove_phase2_patches(model):
    """Remove Phase 2 patches, restore original methods."""
    from bdh.attention import Attention
    for attn in model.attns:
        attn._delta_update = types.MethodType(Attention._delta_update, attn)
        attn.forward = types.MethodType(Attention.forward, attn)
    print("  Attention patches removed")


# ══════════════════════════════════════════════════════════════
# Contrastive Loss
# ══════════════════════════════════════════════════════════════

def compute_contrastive_loss(model, x_raw: torch.Tensor) -> torch.Tensor:
    """
    Compute contrastive loss to push memory keys apart.
    
    Encourages orthogonal keys for different tokens.
    
    Args:
        model: BDH model
        x_raw: Raw token embeddings [B, T, D]
        
    Returns:
        loss: Scalar loss encouraging orthogonal keys
    """
    B, T, D = x_raw.shape

    total_loss = 0
    for attn in model.attns:
        # Compute keys for all tokens
        K = attn.theta_K(x_raw)  # [B, T, N]
        K = F.normalize(K, p=2, dim=-1)

        # Compute pairwise similarities within each batch
        sim = K @ K.transpose(-1, -2)  # [B, T, T]

        # Mask out diagonal (self-similarity)
        mask = torch.eye(T, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, 0)

        # Loss: penalize high similarity between different tokens
        loss = (sim ** 2).mean()
        total_loss = total_loss + loss

    return total_loss / len(model.attns)


# ══════════════════════════════════════════════════════════════
# Phase 2 Training
# ══════════════════════════════════════════════════════════════

def phase2_step(model, X: torch.Tensor, Y: torch.Tensor, 
                context_len: int, vocab_size: int) -> torch.Tensor:
    """
    Single Phase 2 training step.
    
    1. Process context to fill memory
    2. Generate on target portion
    3. Compute loss only on target
    
    Args:
        model: BDH model
        X: Input tokens [B, T]
        Y: Target tokens [B, T]
        context_len: Length of context portion
        vocab_size: Vocabulary size
        
    Returns:
        loss: Cross-entropy loss on target portion
    """
    x_ctx = X[:, :context_len]
    x_tgt = X[:, context_len:]
    y_tgt = Y[:, context_len:]
    
    model.config.use_memory = True
    
    # Fill memory with context
    with torch.no_grad():
        model(x_ctx)
    
    # Generate on target
    logits, _ = model(x_tgt)
    
    return F.cross_entropy(logits.view(-1, vocab_size), y_tgt.reshape(-1))


@torch.no_grad()
def eval_phase2(model, get_batch_fn, step_fn, eval_iters: int,
                device_type: str) -> float:
    """Evaluate Phase 2 loss."""
    model.eval()
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        model.reset_all_memory()
        X, Y = get_batch_fn('val')
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            losses[k] = step_fn(X, Y).item()
    
    model.train()
    return losses.mean().item()


def train_phase2(
    model,
    get_batch_fn,
    config: TrainConfig,
    device: str,
    device_type: str,
    contrastive_weight: float = 0.1,
):
    """
    Train Phase 2: Memory gates and projections.
    
    Args:
        model: BDH model (should have Phase 1 weights loaded)
        get_batch_fn: Function(split) -> (x, y) batch
        config: Training configuration
        device: Device string
        device_type: 'cuda' or 'cpu'
        contrastive_weight: Weight for contrastive loss
        
    Returns:
        Best validation loss achieved
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: Memory Gate Training")
    print(f"{'='*60}")
    
    # Apply Phase 2 patches
    apply_phase2_patches(model)
    
    # Configure memory
    model.config.memory_train_prob = 1.0
    model.config.memory_lr = 0.1
    model.config.memory_decay = 1.0
    
    block_size = model.config.block_size
    context_len = block_size // 2
    vocab_size = model.config.vocab_size
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        lr=config.p2_lr_gate,
        weight_decay=0.01,
        memory_lr=config.p2_lr_memory,
        body_lr=config.p2_lr_body,
        phase=2
    )
    
    # Step function
    def step_fn(X, Y):
        return phase2_step(model, X, Y, context_len, vocab_size)
    
    # Training loop
    best_val = float('inf')
    patience_ctr = 0
    t0 = time.time()
    model.train()
    
    print(f"\nTraining {config.p2_iters} iterations...")
    
    for it in range(config.p2_iters):
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        
        for micro in range(config.p2_grad_accum):
            model.reset_all_memory()
            X, Y = get_batch_fn('train')
            
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                # Main loss
                loss = step_fn(X, Y) / config.p2_grad_accum
                
                # Contrastive loss
                if contrastive_weight > 0:
                    x_raw = model.embed(X)
                    c_loss = compute_contrastive_loss(model, x_raw)
                    loss = loss + contrastive_weight * c_loss / config.p2_grad_accum
            
            loss_accum += loss.item()
            loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Evaluation
        if it % config.eval_interval == 0 or it == config.p2_iters - 1:
            val_loss = eval_phase2(model, get_batch_fn, step_fn, 15, device_type)
            dt = time.time() - t0
            
            print(f"  step {it:5d} | val {val_loss:.4f} | {dt:.0f}s")
            model.print_gates()
            
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, config.checkpoint_dir, "best_p2")
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= config.p2_patience:
                    print(f"    → Early stopping")
                    break
        
        # Logging
        if it > 0 and it % config.log_interval == 0:
            dt = time.time() - t0
            print(f"    iter {it:5d} | loss {loss_accum:.4f} | {dt:.0f}s")
    
    total_time = time.time() - t0
    print(f"\nPhase 2 done in {total_time/60:.1f} min | Best val: {best_val:.4f}")
    
    return best_val
