#!/usr/bin/env python3
"""
Evaluation script for BDH Medical Assistant

Runs recall tests to evaluate memory performance.

Usage:
    python scripts/evaluate.py                           # Run all tests
    python scripts/evaluate.py --checkpoint best_p2      # Use specific checkpoint
    python scripts/evaluate.py --test medical            # Run only medical tests
    python scripts/evaluate.py --test general            # Run only general tests
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bdh import BDH, BDHConfig
from data import get_tokenizer
from training import setup_device, load_checkpoint


def test_medical_recall(model, enc, device, verbose=True):
    """
    Test medical fact recall using the Latent RAG cache.
    
    Returns:
        (passed, total): Number of tests passed and total tests
    """
    if verbose:
        print("\n" + "="*60)
        print("MEDICAL RECALL TEST")
        print("="*60)

    # Facts to memorize
    facts = [
        "Patient Profile - Name: John Martinez. Condition: Stomach Bacteria.",
        "Patient Profile - Name: Sarah Chen. Condition: Headache Migraine.",
    ]

    # Reset and memorize
    model.reset_all_memory()
    model.eval()

    for fact in facts:
        tokens = enc.encode(fact)
        model.memorize(tokens)

    if verbose:
        print(f"Stored {len(model.output_memory_keys)} keys in RAG cache.\n")

    # Queries and expected answers
    tests = [
        ("Patient Profile - Name: John Martinez. Condition:", "Stomach"),
        ("Patient Profile - Name: Sarah Chen. Condition:", "Headache"),
    ]

    passed = 0
    for query, expected in tests:
        q_tokens = enc.encode(query)
        q_idx = torch.tensor([q_tokens], device=device)

        with torch.no_grad():
            out = model.generate(q_idx, max_new_tokens=4, temperature=0.5)

        response = enc.decode(out[0].tolist())
        success = expected.lower() in response.lower()
        
        if success:
            passed += 1

        if verbose:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  Q: '{query}'")
            print(f"  A: '{response}'")
            print(f"  Expected: '{expected}' {status}\n")

    return passed, len(tests)


def test_general_generation(model, enc, device, verbose=True):
    """
    Test that general language generation still works.
    
    Returns:
        (passed, total): Number of tests with coherent output
    """
    if verbose:
        print("\n" + "="*60)
        print("GENERAL GENERATION TEST")
        print("="*60)
        print(f"[RAG Cache: {len(model.output_memory_keys)} keys loaded]\n")

    prompts = [
        "The quick brown fox jumps over the lazy",
        "In the morning, I always like to drink a fresh cup of",
        "The primary ingredients for baking a cake are flour, sugar, and",
        "Once upon a time, in a faraway land, there lived a",
    ]

    passed = 0
    for prompt in prompts:
        q_tokens = enc.encode(prompt)
        q_idx = torch.tensor([q_tokens], device=device)

        with torch.no_grad():
            out = model.generate(q_idx, max_new_tokens=10, temperature=0.8, top_k=40)

        response = enc.decode(out[0].tolist())
        
        # Basic coherence check: response should be longer and not repeat prompt excessively
        generated = response[len(prompt):]
        coherent = len(generated.strip()) > 3 and generated.count(prompt[-10:]) < 2
        
        if coherent:
            passed += 1

        if verbose:
            status = "✓" if coherent else "?"
            print(f"  {status} Prompt: '{prompt}'")
            print(f"    Output: '{response}'\n")

    return passed, len(prompts)


def test_memory_diagnostics(model, verbose=True):
    """
    Print memory diagnostics for each layer.
    """
    if verbose:
        print("\n" + "="*60)
        print("MEMORY DIAGNOSTICS")
        print("="*60)

        diag = model.get_all_diagnostics()
        for layer, info in diag.items():
            print(f"\n  {layer}:")
            print(f"    Status: {info['status']}")
            if info['status'] == 'active':
                print(f"    Norm: {info['norm']:.4f}")
                print(f"    Mean: {info['mean']:.6f}")
                print(f"    Std: {info['std']:.4f}")
            gates = ', '.join(f'{g:.3f}' for g in info['gates'])
            print(f"    Gates: [{gates}]")


def test_retrieval_similarity(model, enc, device, verbose=True):
    """
    Test the cosine similarity of retrieval.
    """
    if verbose:
        print("\n" + "="*60)
        print("RETRIEVAL SIMILARITY TEST")
        print("="*60)

    model.reset_all_memory()
    
    # Memorize a fact
    fact = "Patient Profile - Name: John Martinez. Condition: Stomach Bacteria."
    tokens = enc.encode(fact)
    model.memorize(tokens)

    # Query
    query = "Patient Profile - Name: John Martinez. Condition:"
    query_tokens = enc.encode(query)

    # Get retrieval similarity
    query_key = model._get_rag_key(query_tokens)
    keys_tensor = torch.stack(model.output_memory_keys)
    
    sims = F.cosine_similarity(query_key.unsqueeze(0), keys_tensor, dim=1)
    best_idx = sims.argmax().item()
    best_sim = sims[best_idx].item()

    if verbose:
        print(f"\n  Query: '{query}'")
        print(f"  Best similarity: {best_sim:.4f}")
        print(f"  Threshold: {model.rag_threshold}")
        print(f"  Above threshold: {'Yes' if best_sim > model.rag_threshold else 'No'}")

    return best_sim


def main():
    parser = argparse.ArgumentParser(description="Evaluate BDH Medical Assistant")
    parser.add_argument("--checkpoint", type=str, default="best_p2",
                        help="Checkpoint name to load (default: best_p2)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints")
    parser.add_argument("--test", type=str, choices=["medical", "general", "all"],
                        default="all", help="Which tests to run")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    args = parser.parse_args()

    # Setup
    device, _ = setup_device()
    enc = get_tokenizer()
    
    # Load model
    config = BDHConfig()
    model = BDH(config).to(device)
    
    try:
        load_checkpoint(model, args.checkpoint_dir, args.checkpoint, device)
    except FileNotFoundError:
        print(f"Error: Checkpoint '{args.checkpoint}' not found in {args.checkpoint_dir}/")
        print("Run training first: python scripts/train.py")
        sys.exit(1)

    model.eval()
    verbose = not args.quiet

    # Run tests
    results = {}

    if args.test in ["medical", "all"]:
        passed, total = test_medical_recall(model, enc, device, verbose)
        results["medical"] = (passed, total)

    if args.test in ["general", "all"]:
        passed, total = test_general_generation(model, enc, device, verbose)
        results["general"] = (passed, total)

    if args.test == "all":
        test_memory_diagnostics(model, verbose)
        sim = test_retrieval_similarity(model, enc, device, verbose)
        results["retrieval_sim"] = sim

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        if isinstance(result, tuple):
            passed, total = result
            pct = 100 * passed / total if total > 0 else 0
            print(f"  {test_name}: {passed}/{total} ({pct:.0f}%)")
        else:
            print(f"  {test_name}: {result:.4f}")


if __name__ == "__main__":
    main()
