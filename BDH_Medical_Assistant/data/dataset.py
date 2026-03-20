"""
Dataset utilities for BDH training

Handles:
- FineWeb + PubMed mixed data loading
- Train/val splitting
- Batch generation for Phase 1 and Phase 2
"""
import os
import dataclasses
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .tokenizer import get_tokenizer, encode_ordinary, EOT_TOKEN


@dataclasses.dataclass
class DataConfig:
    """Configuration for data loading."""
    data_dir: str = "data"
    fineweb_tokens: int = 200_000_000  # 200M tokens from FineWeb
    pubmed_tokens: int = 100_000_000   # 100M tokens from PubMed
    val_fraction: float = 0.005
    block_size: int = 512
    batch_size: int = 64
    phase2_batch_size: int = 40
    phase2_context_ratio: float = 0.5  # Half context, half target


def tokenize_mixed_data(config: DataConfig = None) -> tuple[str, str]:
    """
    Download and tokenize FineWeb + PubMed data.
    
    Args:
        config: Data configuration
        
    Returns:
        (train_path, val_path): Paths to memory-mapped token files
    """
    if config is None:
        config = DataConfig()
    
    os.makedirs(config.data_dir, exist_ok=True)
    train_path = os.path.join(config.data_dir, "train.bin")
    val_path = os.path.join(config.data_dir, "val.bin")

    # Skip if already exists
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Data already exists at {config.data_dir}/")
        return train_path, val_path

    # Import datasets here to avoid import overhead
    from datasets import load_dataset

    all_tokens = []
    enc = get_tokenizer()

    # ── Part 1: FineWeb-Edu (general English) ──
    print(f"\nDownloading FineWeb-Edu ({config.fineweb_tokens/1e6:.0f}M tokens)...")
    
    ds_fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    count = 0
    pbar = tqdm(total=config.fineweb_tokens, unit='tok', unit_scale=True, desc="FineWeb")
    for ex in ds_fw:
        toks = encode_ordinary(ex["text"])
        toks.append(EOT_TOKEN)
        all_tokens.extend(toks)
        count += len(toks)
        pbar.update(len(toks))
        if count >= config.fineweb_tokens:
            break
    pbar.close()
    print(f"  FineWeb: {count:,} tokens")

    # ── Part 2: PubMed Abstracts (medical English) ──
    print(f"\nDownloading PubMed abstracts ({config.pubmed_tokens/1e6:.0f}M tokens)...")

    pubmed_loaded = False
    pubmed_sources = [
        ("scientific_papers", {"name": "pubmed", "split": "train", "streaming": True, "trust_remote_code": True}, "article"),
        ("ccdv/pubmed-summarization", {"split": "train", "streaming": True}, "article"),
        ("pubmed_qa", {"name": "pqa_artificial", "split": "train", "streaming": True, "trust_remote_code": True}, "context"),
    ]

    for ds_name, ds_kwargs, text_field in pubmed_sources:
        try:
            print(f"  Trying {ds_name}...")
            ds_pm = load_dataset(ds_name, **ds_kwargs)

            count_pm = 0
            pbar = tqdm(total=config.pubmed_tokens, unit='tok', unit_scale=True, desc="PubMed")
            for ex in ds_pm:
                text = ex.get(text_field, "")
                if isinstance(text, dict):
                    text = str(text)
                if not text or len(text) < 50:
                    continue
                toks = encode_ordinary(text)
                toks.append(EOT_TOKEN)
                all_tokens.extend(toks)
                count_pm += len(toks)
                pbar.update(len(toks))
                if count_pm >= config.pubmed_tokens:
                    break
            pbar.close()
            print(f"  PubMed ({ds_name}): {count_pm:,} tokens")
            pubmed_loaded = True
            break
        except Exception as e:
            print(f"  Failed to load {ds_name}: {e}")
            continue

    if not pubmed_loaded:
        print("  WARNING: Could not load PubMed data. Using more FineWeb as fallback.")
        count_extra = 0
        pbar = tqdm(total=config.pubmed_tokens, unit='tok', unit_scale=True, desc="FineWeb-extra")
        for ex in ds_fw:
            toks = encode_ordinary(ex["text"])
            toks.append(EOT_TOKEN)
            all_tokens.extend(toks)
            count_extra += len(toks)
            pbar.update(len(toks))
            if count_extra >= config.pubmed_tokens:
                break
        pbar.close()

    # ── Shuffle and split ──
    print(f"\nTotal tokens: {len(all_tokens):,}")
    
    arr = np.array(all_tokens, dtype=np.uint16)
    split_idx = int(len(arr) * (1 - config.val_fraction))

    for path, data in [(train_path, arr[:split_idx]), (val_path, arr[split_idx:])]:
        mm = np.memmap(path, dtype=np.uint16, mode='w+', shape=data.shape)
        mm[:] = data
        mm.flush()

    print(f"Saved: {split_idx:,} train + {len(arr)-split_idx:,} val tokens")
    return train_path, val_path


def get_batch(split: str, train_path: str, val_path: str,
              batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a batch for Phase 1 training.
    
    Args:
        split: 'train' or 'val'
        train_path: Path to training data
        val_path: Path to validation data
        batch_size: Batch size
        block_size: Sequence length
        device: Target device
        
    Returns:
        (x, y): Input and target tensors [B, T]
    """
    path = train_path if split == 'train' else val_path
    data = np.memmap(path, dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    
    if device == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y


def get_batch_phase2(split: str, train_path: str, val_path: str,
                     batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a batch for Phase 2 training.
    
    Same as get_batch but with Phase 2 batch size.
    """
    return get_batch(split, train_path, val_path, batch_size, block_size, device)


def generate_infinite_batch(batch_size: int, block_size: int = 64, 
                           device: str = 'cuda') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate infinite synthetic data for memory forcing experiments.
    
    Creates random token pairs and tests if the model can recall them.
    
    Args:
        batch_size: Number of sequences
        block_size: Sequence length
        device: Target device
        
    Returns:
        (x, y): Input and target tensors with -100 for ignored positions
    """
    import random
    
    enc = get_tokenizer()
    batch_x, batch_y = [], []
    
    for _ in range(batch_size):
        # Generate 4-6 random pairs for this sequence
        vocab_pool = random.sample(range(5000, 20000), 12)
        pairs = [(vocab_pool[i], vocab_pool[i+1]) for i in range(0, 10, 2)]

        context_tokens = []
        for a, b in pairs:
            context_tokens.extend([a, 318, b, 13])  # "X is Y."

        query_a, query_b = random.choice(pairs)
        query_tokens = [query_a, 318]
        full_tokens = context_tokens + query_tokens

        if len(full_tokens) > block_size - 1:
            full_tokens = full_tokens[-(block_size - 1):]

        target_tokens = [-100] * len(full_tokens)
        target_tokens[-1] = query_b

        while len(full_tokens) < block_size:
            full_tokens = [EOT_TOKEN] + full_tokens
            target_tokens = [-100] + target_tokens

        batch_x.append(full_tokens)
        batch_y.append(target_tokens)

    return (
        torch.tensor(batch_x, dtype=torch.long, device=device),
        torch.tensor(batch_y, dtype=torch.long, device=device)
    )
