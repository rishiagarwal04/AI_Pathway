#!/usr/bin/env python3
"""
Main training script for BDH Medical Assistant

Usage:
    python scripts/train.py                    # Full training (Phase 1 + 2)
    python scripts/train.py --phase 1          # Phase 1 only
    python scripts/train.py --phase 2          # Phase 2 only (requires Phase 1 checkpoint)
    python scripts/train.py --skip-data        # Skip data download (use existing)
"""
import argparse
import os
import sys
from functools import partial

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bdh import BDH, BDHConfig
from data import tokenize_mixed_data, get_batch, DataConfig
from training import (
    train_phase1,
    train_phase2,
    setup_device,
    TrainConfig,
    load_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description="Train BDH Medical Assistant")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Train only specific phase (default: both)")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data download, use existing files")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory for data files")
    parser.add_argument("--max-iters", type=int, default=5000,
                        help="Max iterations for Phase 1")
    parser.add_argument("--p2-iters", type=int, default=3500,
                        help="Max iterations for Phase 2")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for Phase 1")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    args = parser.parse_args()

    # Setup device
    device, device_type = setup_device()

    # Configuration
    data_config = DataConfig(data_dir=args.data_dir)
    train_config = TrainConfig(
        checkpoint_dir=args.checkpoint_dir,
        max_iters=args.max_iters,
        p2_iters=args.p2_iters,
        batch_size=args.batch_size,
    )
    model_config = BDHConfig()

    # Prepare data
    if not args.skip_data:
        print("\n" + "="*60)
        print("Preparing Data")
        print("="*60)
        train_path, val_path = tokenize_mixed_data(data_config)
    else:
        train_path = os.path.join(args.data_dir, "train.bin")
        val_path = os.path.join(args.data_dir, "val.bin")
        if not os.path.exists(train_path):
            print(f"Error: {train_path} not found. Run without --skip-data first.")
            sys.exit(1)
        print(f"Using existing data: {train_path}")

    # Create batch function
    def get_batch_fn(split):
        return get_batch(
            split, train_path, val_path,
            train_config.batch_size, model_config.block_size, device
        )

    def get_batch_p2_fn(split):
        return get_batch(
            split, train_path, val_path,
            train_config.p2_batch_size, model_config.block_size, device
        )

    # Initialize model
    model = BDH(model_config).to(device)

    # Phase 1
    if args.phase is None or args.phase == 1:
        train_phase1(
            model=model,
            get_batch_fn=get_batch_fn,
            config=train_config,
            device=device,
            device_type=device_type,
            use_compile=not args.no_compile,
        )

    # Phase 2
    if args.phase is None or args.phase == 2:
        # Load Phase 1 checkpoint if only running Phase 2
        if args.phase == 2:
            load_checkpoint(model, train_config.checkpoint_dir, "best_p1", device)

        train_phase2(
            model=model,
            get_batch_fn=get_batch_p2_fn,
            config=train_config,
            device=device,
            device_type=device_type,
        )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")
    print("  - best_p1.pt (Phase 1)")
    print("  - best_p2.pt (Phase 2)")


if __name__ == "__main__":
    main()
