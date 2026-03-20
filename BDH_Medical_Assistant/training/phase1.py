"""
Phase 1: Base Language Model Training

Trains the BDH model on mixed FineWeb + PubMed data
without memory active (standard language modeling).
"""
import os
import time
import math
from functools import partial

import torch

from .utils import (
    get_lr,
    estimate_loss,
    create_optimizer,
    save_checkpoint,
    TrainConfig,
)


def train_phase1(
    model,
    get_batch_fn,
    config: TrainConfig,
    device: str,
    device_type: str,
    use_compile: bool = True,
):
    """
    Train Phase 1: Base language model on mixed data.
    
    Args:
        model: BDH model
        get_batch_fn: Function(split) -> (x, y) batch
        config: Training configuration
        device: Device string
        device_type: 'cuda' or 'cpu'
        use_compile: Whether to use torch.compile
        
    Returns:
        Best validation loss achieved
    """
    block_size = model.config.block_size
    tokens_per_iter = config.batch_size * config.grad_accum * block_size
    total_tokens = config.max_iters * tokens_per_iter

    print(f"\n{'='*60}")
    print(f"PHASE 1: Base Model Training")
    print(f"{'='*60}")
    print(f"  Tokens/iter: {tokens_per_iter:,}")
    print(f"  Total budget: {total_tokens/1e6:.0f}M tokens")
    
    # Create optimizer
    optimizer = create_optimizer(
        model, 
        lr=config.lr, 
        weight_decay=config.weight_decay,
        phase=1
    )
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Optional compilation
    if use_compile and device == 'cuda' and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model_compiled = torch.compile(model, mode='reduce-overhead')
    else:
        model_compiled = model
    
    # Training loop
    best_val = float('inf')
    patience_ctr = 0
    t0 = time.time()
    model.train()
    
    # Wrap get_batch for estimate_loss
    eval_batch_fn = partial(get_batch_fn)
    
    print(f"\nTraining {config.max_iters} iterations...")
    
    for it in range(config.max_iters):
        # Update learning rate
        lr = get_lr(it, config.warmup_iters, config.max_iters, 
                    config.lr, config.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        # Evaluation
        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            losses = estimate_loss(model, eval_batch_fn, config.eval_iters, device_type)
            ppl = math.exp(min(losses['val'], 20))
            dt = time.time() - t0
            
            print(f"  step {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | ppl {ppl:.1f} | "
                  f"lr {lr:.2e} | {dt:.0f}s")
            
            if losses['val'] < best_val:
                best_val = losses['val']
                save_checkpoint(model, config.checkpoint_dir, "best_p1")
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= config.patience:
                    print(f"    → Early stopping")
                    break
        
        # Training step
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        
        for micro in range(config.grad_accum):
            model.reset_all_memory()
            X, Y = get_batch_fn('train')
            
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model_compiled(X, Y)
                loss = loss / config.grad_accum
            
            loss_accum += loss.item()
            loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if it > 0 and it % config.log_interval == 0:
            dt = time.time() - t0
            tps = (it + 1) * tokens_per_iter / dt
            print(f"    iter {it:5d} | loss {loss_accum:.4f} | "
                  f"{tps/1e3:.0f}K tok/s | {dt:.0f}s")
    
    total_time = time.time() - t0
    print(f"\nPhase 1 done in {total_time/60:.1f} min | Best val: {best_val:.4f}")
    
    return best_val
