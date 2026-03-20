#!/usr/bin/env python3
"""
Interactive demo for BDH Medical Assistant

Usage:
    python scripts/demo.py                      # Interactive mode
    python scripts/demo.py --learn "fact"       # Learn a fact
    python scripts/demo.py --ask "question"     # Ask a question
"""
import argparse
import os
import sys

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bdh import BDH, BDHConfig, InferenceLearner
from data import get_tokenizer
from training import setup_device, load_checkpoint


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║           BDH Medical Assistant - Interactive Demo        ║
╠══════════════════════════════════════════════════════════╣
║  Commands:                                                ║
║    learn <fact>    - Teach the model a fact               ║
║    ask <question>  - Ask the model a question             ║
║    list            - Show learned facts                   ║
║    reset           - Clear all memories                   ║
║    gates           - Show memory gate values              ║
║    help            - Show this message                    ║
║    quit            - Exit                                 ║
╚══════════════════════════════════════════════════════════╝
    """)


def interactive_mode(learner):
    """Run interactive REPL."""
    print_banner()
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "quit" or cmd == "exit":
            print("Goodbye!")
            break

        elif cmd == "help":
            print_banner()

        elif cmd == "learn":
            if not arg:
                print("Usage: learn <fact>")
                print("Example: learn Patient John Smith has Type 2 Diabetes.")
                continue
            
            print(f"Learning: '{arg}'")
            learner.learn(arg, repetitions=50)
            
            # Also memorize in RAG cache
            tokens = learner.enc.encode(arg)
            learner.model.memorize(tokens)
            
            print(f"✓ Learned! ({len(learner.model.output_memory_keys)} keys in cache)")

        elif cmd == "ask":
            if not arg:
                print("Usage: ask <question>")
                print("Example: ask What condition does John Smith have?")
                continue
            
            response = learner.ask(arg, max_tokens=20, temperature=0.7)
            print(f"Response: {response}")

        elif cmd == "list":
            if not learner.facts:
                print("No facts learned yet.")
            else:
                print(f"Learned facts ({len(learner.facts)}):")
                for i, fact in enumerate(learner.facts, 1):
                    print(f"  {i}. {fact}")

        elif cmd == "reset":
            learner.reset()
            print("✓ Memory cleared.")

        elif cmd == "gates":
            learner.model.print_gates()

        elif cmd == "diag" or cmd == "diagnostics":
            diag = learner.model.get_all_diagnostics()
            for layer, info in diag.items():
                print(f"{layer}: {info['status']}", end="")
                if info['status'] == 'active':
                    print(f" (norm={info['norm']:.2f})", end="")
                print()

        else:
            # Treat as a question
            response = learner.ask(user_input, max_tokens=20, temperature=0.7)
            print(f"Response: {response}")


def main():
    parser = argparse.ArgumentParser(description="BDH Medical Assistant Demo")
    parser.add_argument("--checkpoint", type=str, default="best_p2",
                        help="Checkpoint name to load")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints")
    parser.add_argument("--learn", type=str, default=None,
                        help="Learn a fact (non-interactive)")
    parser.add_argument("--ask", type=str, default=None,
                        help="Ask a question (non-interactive)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Maximum tokens to generate")
    args = parser.parse_args()

    # Setup
    device, _ = setup_device()
    enc = get_tokenizer()

    # Load model
    print("\nLoading model...")
    config = BDHConfig()
    model = BDH(config).to(device)

    try:
        load_checkpoint(model, args.checkpoint_dir, args.checkpoint, device)
    except FileNotFoundError:
        print(f"\nWarning: Checkpoint '{args.checkpoint}' not found.")
        print("Using untrained model. Run training first for better results:")
        print("  python scripts/train.py")
        print()

    model.eval()

    # Create learner
    learner = InferenceLearner(model, enc, device)
    learner.enable(memory_lr=0.1, memory_decay=1.0, retrieval_scale=1.0)

    # Non-interactive mode
    if args.learn:
        print(f"Learning: '{args.learn}'")
        learner.learn(args.learn, repetitions=50)
        tokens = enc.encode(args.learn)
        model.memorize(tokens)
        print("✓ Learned!")
        
        if args.ask:
            response = learner.ask(args.ask, max_tokens=args.max_tokens, 
                                   temperature=args.temperature)
            print(f"\nQ: {args.ask}")
            print(f"A: {response}")
        return

    if args.ask:
        response = learner.ask(args.ask, max_tokens=args.max_tokens,
                               temperature=args.temperature)
        print(f"Q: {args.ask}")
        print(f"A: {response}")
        return

    # Interactive mode
    interactive_mode(learner)


if __name__ == "__main__":
    main()
