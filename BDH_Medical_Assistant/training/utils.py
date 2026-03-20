"""
Training utilities

Contains:
- Learning rate scheduler
- Loss estimation
- Device setup
- Training configuration
"""
import os
import math
import dataclasses
import random

import numpy as np
import torch


@dataclasses.dataclass
class TrainConfig:
    """Training configuration."""
    # Phase 1 settings
    max_iters: int = 5000
    warmup_iters: int = 300
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    batch_size: int = 64
    grad_accum: int = 2
    
    # Phase 2 settings
    p2_iters: int = 3500
    p2_warmup: int = 200
    p2_lr_gate: float = 5e-4
    p2_lr_memory: float = 1e-3
    p2_lr_body: float = 1e-5
    p2_batch_size: int = 40
    p2_grad_accum: int = 2
    
    # Common settings
    eval_interval: int = 500
    eval_iters: int = 25
    log_interval: int = 100
    patience: int = 5
    p2_patience: int = 7
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    
    # Random seed
    seed: int = 1337


def setup_device(seed: int = 1337) -> tuple[str, str]:
    """
    Setup device and random seeds.
    
    Returns:
        (device, device_type): Device strings
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {gb:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    return device, device_type


def get_lr(it: int, warmup_iters: int, max_iters: int, 
           lr: float, min_lr: float) -> float:
    """
    Cosine learning rate schedule with warmup.
    
    Args:
        it: Current iteration
        warmup_iters: Number of warmup iterations
        max_iters: Total iterations
        lr: Peak learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate for current iteration
    """
    if it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model, get_batch_fn, eval_iters: int, 
                  device_type: str) -> dict[str, float]:
    """
    Estimate loss on train and val splits.
    
    Args:
        model: The model to evaluate
        get_batch_fn: Function that takes split name and returns (x, y)
        eval_iters: Number of evaluation iterations
        device_type: 'cuda' or 'cpu'
        
    Returns:
        Dictionary with 'train' and 'val' losses
    """
    model.eval()
    model.reset_all_memory()
    
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_fn(split)
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    
    model.train()
    return out


def create_optimizer(model, lr: float, weight_decay: float,
                     memory_lr: float = None, body_lr: float = None,
                     phase: int = 1) -> torch.optim.Optimizer:
    """
    Create optimizer with parameter groups.
    
    Args:
        model: The model
        lr: Main learning rate
        weight_decay: Weight decay
        memory_lr: Learning rate for memory params (Phase 2)
        body_lr: Learning rate for body params (Phase 2)
        phase: Training phase (1 or 2)
        
    Returns:
        AdamW optimizer
    """
    if phase == 1:
        # Phase 1: Standard parameter groups
        decay_params = [
            p for n, p in model.named_parameters()
            if 'embed' not in n and 'memory_gate' not in n and p.dim() >= 2
        ]
        no_decay_params = [
            p for n, p in model.named_parameters()
            if 'embed' in n or 'memory_gate' in n or p.dim() < 2
        ]
        
        return torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=lr, betas=(0.9, 0.95))
    
    else:
        # Phase 2: Separate groups for gates, memory projections, body
        gate_params = []
        memory_params = []
        ln_params = []
        body_params = []
        
        for n, p in model.named_parameters():
            if 'memory_gate' in n:
                gate_params.append(p)
            elif any(x in n for x in ['theta_K', 'theta_Q', 'beta_proj']):
                memory_params.append(p)
            elif 'memory_ln' in n:
                ln_params.append(p)
            else:
                body_params.append(p)
        
        print(f"  Gates: {sum(p.numel() for p in gate_params)} params")
        print(f"  Memory projections: {sum(p.numel() for p in memory_params)} params")
        print(f"  Memory LN: {sum(p.numel() for p in ln_params)} params")
        print(f"  Body: {sum(p.numel() for p in body_params)/1e6:.1f}M params")
        
        return torch.optim.AdamW([
            {'params': gate_params, 'lr': lr, 'weight_decay': 0.0},
            {'params': memory_params, 'lr': memory_lr or lr, 'weight_decay': 0.0},
            {'params': ln_params, 'lr': lr, 'weight_decay': 0.0},
            {'params': body_params, 'lr': body_lr or lr * 0.01, 'weight_decay': 0.01},
        ], betas=(0.9, 0.99))


def save_checkpoint(model, path: str, name: str = "checkpoint"):
    """Save model checkpoint."""
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{name}.pt")
    torch.save(model.state_dict(), filepath)
    print(f"    → Saved: {filepath}")


def load_checkpoint(model, path: str, name: str = "checkpoint", 
                    device: str = 'cuda'):
    """Load model checkpoint."""
    filepath = os.path.join(path, f"{name}.pt")
    model.load_state_dict(
        torch.load(filepath, map_location=device, weights_only=True)
    )
    print(f"    ← Loaded: {filepath}")
