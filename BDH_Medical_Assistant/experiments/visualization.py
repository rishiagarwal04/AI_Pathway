"""
BDH Memory System - Visualization & Logging
============================================

Utilities for generating figures and logs from experiments.

Saves outputs to:
  - results/figures/  - PNG plots and visualizations
  - results/logs/     - JSON experiment logs

Usage:
    from experiments.visualization import generate_all_figures
    generate_all_figures(model, enc, device)
"""

import torch
import torch.nn.functional as F
import os
import json
from datetime import datetime

# Try to import matplotlib (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available. Text-based outputs only.")


def ensure_dirs(results_dir="results"):
    """Create results directories if they don't exist."""
    os.makedirs(f"{results_dir}/figures", exist_ok=True)
    os.makedirs(f"{results_dir}/logs", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Gate Values Across Layers
# ══════════════════════════════════════════════════════════════════════════════

def plot_gate_values(model, results_dir="results"):
    """
    Plot memory gate values across all layers.
    
    Shows which layers have learned to open their gates.
    """
    ensure_dirs(results_dir)
    
    print("\n[Generating: Gate Values Plot]")
    
    # Collect gate values
    layers = []
    gate_values = []
    
    for i, attn in enumerate(model.attns):
        gate = torch.sigmoid(attn.memory_gate).mean().item()
        layers.append(f"L{i}")
        gate_values.append(gate)
    
    # Log
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "layers": layers,
        "gate_values": gate_values,
        "average_gate": sum(gate_values) / len(gate_values)
    }
    with open(f"{results_dir}/logs/gate_values.json", "w") as f:
        json.dump(log_data, f, indent=2)
    
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        colors = ['#4CAF50' if g > 0.5 else '#FFC107' if g > 0.2 else '#F44336' for g in gate_values]
        bars = ax.bar(layers, gate_values, color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_ylim(0, 1)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Gate Value (sigmoid)', fontsize=12)
        ax.set_title('Memory Gate Values Across Layers', fontsize=14, fontweight='bold')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target (0.9)')
        
        # Legend
        green_patch = mpatches.Patch(color='#4CAF50', label='Open (>0.5)')
        yellow_patch = mpatches.Patch(color='#FFC107', label='Partial (0.2-0.5)')
        red_patch = mpatches.Patch(color='#F44336', label='Closed (<0.2)')
        ax.legend(handles=[green_patch, yellow_patch, red_patch], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/figures/gate_values.png", dpi=150)
        plt.close()
        
        print(f"  ✓ Saved: {results_dir}/figures/gate_values.png")
    else:
        # Text fallback
        print(f"  Gate Values:")
        for l, g in zip(layers, gate_values):
            bar = "█" * int(g * 20) + "░" * (20 - int(g * 20))
            print(f"    {l}: [{bar}] {g:.3f}")
    
    print(f"  ✓ Saved: {results_dir}/logs/gate_values.json")
    return log_data


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Memory Norms Across Layers
# ══════════════════════════════════════════════════════════════════════════════

def plot_memory_norms(model, results_dir="results"):
    """
    Plot memory matrix norms across layers.
    
    Shows which layers are actively using memory.
    """
    ensure_dirs(results_dir)
    
    print("\n[Generating: Memory Norms Plot]")
    
    # Collect memory norms
    layers = []
    norms = []
    
    for i, attn in enumerate(model.attns):
        if attn.memory_M is not None:
            norm = attn.memory_M.norm().item()
        else:
            norm = 0.0
        layers.append(f"L{i}")
        norms.append(norm)
    
    # Log
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "layers": layers,
        "memory_norms": norms,
        "total_norm": sum(norms)
    }
    with open(f"{results_dir}/logs/memory_norms.json", "w") as f:
        json.dump(log_data, f, indent=2)
    
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        colors = ['#2196F3' if n > 0 else '#BDBDBD' for n in norms]
        ax.bar(layers, norms, color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Memory Matrix Norm', fontsize=12)
        ax.set_title('Memory Matrix Norms Across Layers', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/figures/memory_norms.png", dpi=150)
        plt.close()
        
        print(f"  ✓ Saved: {results_dir}/figures/memory_norms.png")
    else:
        print(f"  Memory Norms:")
        max_norm = max(norms) if norms else 1
        for l, n in zip(layers, norms):
            bar_len = int((n / max_norm) * 20) if max_norm > 0 else 0
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {l}: [{bar}] {n:.2f}")
    
    print(f"  ✓ Saved: {results_dir}/logs/memory_norms.json")
    return log_data


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Retrieval Similarity Distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_retrieval_similarity(model, enc, device, results_dir="results"):
    """
    Plot similarity scores for retrieval across different queries.
    
    Shows how well the memory discriminates between contexts.
    """
    ensure_dirs(results_dir)
    
    print("\n[Generating: Retrieval Similarity Plot]")
    
    model.eval()
    
    # Test contexts
    facts = [
        "Patient: Alice. Condition: Diabetes.",
        "Patient: Bob. Condition: Asthma.",
        "Patient: Carol. Condition: Migraine.",
    ]
    
    queries = [
        "Patient: Alice. Condition:",
        "Patient: Bob. Condition:",
        "Patient: Carol. Condition:",
    ]
    
    # Build memory
    keys, values, labels = [], [], []
    N_CTX = 5
    
    for fact in facts:
        toks = enc.encode(fact)
        patient = fact.split("Patient: ")[1].split(".")[0]
        
        for i in range(3, len(toks) - 1):
            ctx = toks[max(0, i - N_CTX + 1):i + 1]
            ctx_t = torch.tensor(ctx, device=device)
            emb = model.embed.weight[ctx_t]
            w = torch.tensor([2.0 ** j for j in range(len(ctx))], device=device)
            w = w / w.sum()
            key = F.normalize((emb * w.unsqueeze(1)).sum(0), dim=0)
            
            keys.append(key)
            values.append(toks[i + 1])
            labels.append(patient)
    
    # Query and collect similarities
    similarity_matrix = []
    query_labels = []
    
    for query in queries:
        q_toks = enc.encode(query)
        patient = query.split("Patient: ")[1].split(".")[0]
        query_labels.append(patient)
        
        ctx = q_toks[-N_CTX:]
        ctx_t = torch.tensor(ctx, device=device)
        emb = model.embed.weight[ctx_t]
        w = torch.tensor([2.0 ** j for j in range(len(ctx))], device=device)
        w = w / w.sum()
        q_key = F.normalize((emb * w.unsqueeze(1)).sum(0), dim=0)
        
        sims = [F.cosine_similarity(q_key.unsqueeze(0), k.unsqueeze(0)).item() for k in keys]
        similarity_matrix.append(sims)
    
    # Log
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "facts": facts,
        "queries": queries,
        "similarity_matrix": similarity_matrix,
        "key_labels": labels
    }
    with open(f"{results_dir}/logs/retrieval_similarity.json", "w") as f:
        json.dump(log_data, f, indent=2)
    
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(12, 4))
        
        x = range(len(keys))
        width = 0.25
        
        colors = ['#E91E63', '#2196F3', '#4CAF50']
        
        for i, (sims, label) in enumerate(zip(similarity_matrix, query_labels)):
            offset = (i - 1) * width
            ax.bar([xi + offset for xi in x], sims, width, label=f'Query: {label}', color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Memory Entry', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Retrieval Similarity: Query vs Memory Entries', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/figures/retrieval_similarity.png", dpi=150)
        plt.close()
        
        print(f"  ✓ Saved: {results_dir}/figures/retrieval_similarity.png")
    else:
        print(f"  Similarity Matrix (rows=queries, cols=keys):")
        for q, sims in zip(query_labels, similarity_matrix):
            max_idx = sims.index(max(sims))
            print(f"    {q}: max at entry {max_idx} (sim={max(sims):.3f})")
    
    print(f"  ✓ Saved: {results_dir}/logs/retrieval_similarity.json")
    return log_data


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Accuracy Summary
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_summary(results_dir="results"):
    """
    Plot accuracy summary from demo results.
    
    Reads from saved logs and creates summary visualization.
    """
    ensure_dirs(results_dir)
    
    print("\n[Generating: Accuracy Summary Plot]")
    
    # Try to load demo results
    demo_files = [
        ("Single Fact", "demo_single_fact.json"),
        ("Two Patients", "demo_two_patients.json"),
        ("Matrix Retrieval", "demo_matrix_retrieval.json"),
        ("Medical Recall", "demo_medical_recall.json"),
    ]
    
    demo_names = []
    accuracies = []
    
    for name, filename in demo_files:
        filepath = f"{results_dir}/logs/{filename}"
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                demo_names.append(name)
                accuracies.append(data.get("accuracy", 0))
    
    if not accuracies:
        print("  No demo results found. Run demos first.")
        return None
    
    # Log
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "demos": demo_names,
        "accuracies": accuracies,
        "average": sum(accuracies) / len(accuracies)
    }
    with open(f"{results_dir}/logs/accuracy_summary.json", "w") as f:
        json.dump(log_data, f, indent=2)
    
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        colors = ['#4CAF50' if a == 1.0 else '#FFC107' if a >= 0.5 else '#F44336' for a in accuracies]
        bars = ax.bar(demo_names, [a * 100 for a in accuracies], color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_ylim(0, 110)
        ax.set_xlabel('Demo', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('BDH Memory System - Demo Accuracies', fontsize=14, fontweight='bold')
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% Target')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                   f'{acc*100:.0f}%', ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/figures/accuracy_summary.png", dpi=150)
        plt.close()
        
        print(f"  ✓ Saved: {results_dir}/figures/accuracy_summary.png")
    else:
        print(f"  Accuracy Summary:")
        for name, acc in zip(demo_names, accuracies):
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"    {name}: [{bar}] {acc*100:.0f}%")
    
    print(f"  ✓ Saved: {results_dir}/logs/accuracy_summary.json")
    return log_data


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: Generate All Figures
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_figures(model, enc, device, results_dir="results"):
    """
    Generate all visualizations and logs.
    """
    print("\n")
    print("█" * 60)
    print("█" + " "*12 + "GENERATING VISUALIZATIONS & LOGS" + " "*14 + "█")
    print("█" * 60)
    
    ensure_dirs(results_dir)
    
    results = {}
    
    # Figure 1
    results["gate_values"] = plot_gate_values(model, results_dir)
    
    # Figure 2
    results["memory_norms"] = plot_memory_norms(model, results_dir)
    
    # Figure 3
    results["retrieval_similarity"] = plot_retrieval_similarity(model, enc, device, results_dir)
    
    # Figure 4 (requires demos to have been run first)
    results["accuracy_summary"] = plot_accuracy_summary(results_dir)
    
    print("\n" + "─" * 60)
    print("All visualizations generated!")
    print("─" * 60)
    print(f"\n  Figures: {results_dir}/figures/")
    print(f"  Logs:    {results_dir}/logs/")
    
    # List generated files
    if os.path.exists(f"{results_dir}/figures"):
        figs = os.listdir(f"{results_dir}/figures")
        print(f"\n  Generated figures: {figs}")
    
    return results


if __name__ == "__main__":
    print("BDH Visualization Utilities")
    print("=" * 40)
    print("\nUsage:")
    print("  from experiments.visualization import generate_all_figures")
    print("  generate_all_figures(model, enc, device)")
