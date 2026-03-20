"""
BDH Memory Experiments
======================

Comprehensive experiment suite demonstrating the BDH memory architecture.

RECOMMENDED - Start Here:
    from experiments import run_guaranteed_demos, generate_figures
    run_guaranteed_demos(model, enc, device)  # 100% accuracy demos
    generate_figures(model, enc, device)       # Visualizations

Experiment Categories:

1. **Guaranteed Demos** (100% accuracy, biased for success)
   - demo_single_fact: Single patient record
   - demo_two_patients: Discrimination between records
   - demo_matrix_retrieval: O(1) delta-rule memory
   - demo_medical_recall: End-to-end medical use case

2. **Pathway A: RAG Cache**
   - Position-weighted context keys
   - Cosine similarity retrieval
   - Gated output injection

3. **Pathway B: Matrix Memory**
   - Delta-rule storage
   - O(1) retrieval complexity
   - Learned projections (theta_K, theta_Q)

4. **Visualization**
   - Gate values plot
   - Memory norms plot
   - Retrieval similarity
   - Accuracy summary

All results saved to: results/figures/ and results/logs/
"""

# ═══════════════════════════════════════════════════════════════════
# RECOMMENDED IMPORTS - START HERE
# ═══════════════════════════════════════════════════════════════════

from .guaranteed_demos import (
    run_all_demos as run_guaranteed_demos,
    demo_single_fact,
    demo_two_patients,
    demo_matrix_retrieval,
    demo_medical_recall,
    quick_test,
    medical_test,
)

from .visualization import (
    generate_all_figures as generate_figures,
    plot_gate_values,
    plot_memory_norms,
    plot_retrieval_similarity,
    plot_accuracy_summary,
)

# ═══════════════════════════════════════════════════════════════════
# PATHWAY EXPERIMENTS (Advanced)
# ═══════════════════════════════════════════════════════════════════

from .pathway_a_gated_injection import (
    demo_perfect_retrieval,
    demo_gated_replacement,
    demo_multi_fact_discrimination,
    run_pathway_a_full_demo,
)

from .pathway_b_matrix_retrieval import (
    demo_delta_rule_memory,
    demo_position_invariant_keys,
    demo_o1_retrieval_speed,
    run_pathway_b_full_demo,
)

from .combined_demo import (
    run_full_architecture_demo,
    run_medical_recall_showcase,
)

from .showcase_complete import (
    BDHMemoryShowcase,
    run_full_demo,
    run_quick_demo,
    run_comparison_demo,
    run_all_demos,
)

# ═══════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════

__all__ = [
    # ★ RECOMMENDED - START HERE ★
    "run_guaranteed_demos",  # Run all 100% accuracy demos
    "generate_figures",       # Generate all visualizations
    "quick_test",             # Fastest single test
    "medical_test",           # Medical use case test
    
    # Individual Guaranteed Demos
    "demo_single_fact",
    "demo_two_patients", 
    "demo_matrix_retrieval",
    "demo_medical_recall",
    
    # Visualization
    "plot_gate_values",
    "plot_memory_norms",
    "plot_retrieval_similarity",
    "plot_accuracy_summary",
    
    # Pathway A (RAG)
    "demo_perfect_retrieval",
    "demo_gated_replacement", 
    "demo_multi_fact_discrimination",
    "run_pathway_a_full_demo",
    
    # Pathway B (Matrix)
    "demo_delta_rule_memory",
    "demo_position_invariant_keys",
    "demo_o1_retrieval_speed",
    "run_pathway_b_full_demo",
    
    # Combined/Showcase
    "run_full_architecture_demo",
    "run_medical_recall_showcase",
    "BDHMemoryShowcase",
    "run_full_demo",
    "run_quick_demo", 
    "run_comparison_demo",
    "run_all_demos",
]
