<div align="center">

# BDH NEXUS: The Comprehensive BDH Suite

### A Multi-Disciplinary Exploration of the Baby Dragon Hatchling (BDH) Neural Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Svelte](https://img.shields.io/badge/Svelte-4-orange)](https://svelte.dev/)
[![Status](https://img.shields.io/badge/Status-Research_Preview-important)](https://pathway.com/bdh)

</div>

---

## Table of Contents

1. [What We Built](#what-we-built)
2. [The Core Technology: Baby Dragon Hatchling (BDH)](#the-core-technology-baby-dragon-hatchling-bdh)
3. [Project 1: BDH Explainer](#path-c1-bdh-explainer-interactive-visualization)
4. [Project 2: BDH Medical Assistant](#path-c2-bdh-medical-assistant-inference-time-learning)
5. [Project 3: BDH Game of Life Analysis](#path-b-bdh-game-of-life-mechanistic-interpretability)
6. [Key Research Insights](#key-research-insights)
7. [Installation, Usage Guide, and Hosted Demo](#installation-usage-guide-and-hosted-demo)
8. [Limitations and Future Scope](#limitations-and-future-scope)
9. [Team & Contributions](#team--contributions)
10. [References](#references)

---

## What We Built

This repository is a unified research suite containing three distinct but interconnected projects that rigorously test, visualize, and apply the **Baby Dragon Hatchling (BDH)** architecture. Unlike standard Transformers, which rely on dense matrix multiplications and $O(N^2)$ attention, BDH utilizes **linear attention as fast weights**, **sparse spatial hubs**, and **Hebbian learning rules**.

This suite answers three fundamental questions:

1. **Can we visualize the "brain" of a BDH model?** (Yes, via *BDH Explainer*)
2. **Can BDH learn facts instantly without retraining?** (Yes, via *BDH Medical Assistant*)
3. **Is BDH truly more interpretable than Transformers?** (Yes, via *BDH Game of Life*)

---

## Repository Structure

```
bdh-nexus/
├── README.md
├── LICENSE
├── BDH_Explainer/
├── BDH_Medical_Assistant/
└── game_of_life/
```

---

## The Core Technology: Baby Dragon Hatchling (BDH)

The BDH architecture represents a paradigm shift from standard deep learning:

- **Fast Weights vs. KV Cache:** Instead of storing a growing history of Key-Value pairs (like Llama/GPT), BDH compresses history into a fixed-size matrix $\sigma$ using an outer-product update rule: $\sigma_{t} = \lambda \sigma_{t-1} + \eta (k_t^T v_t)$.
- **Sparsity:** BDH uses a "Hub" system where only a small fraction of neurons fire for any given input, making it biologically plausible and computationally efficient.
- **Monosemanticity:** Our research suggests that BDH neurons naturally converge to single, interpretable concepts without the "superposition" issues that plague Transformers.

---

## Path C1: BDH Explainer (Interactive Visualization)

**Objective:** Demystify BDH by visualizing its internal tensor flow in real time and explaining the underlying theory.

### Key Features

- **Full-Stack Architecture:** Built with a **Svelte 4 + D3.js** frontend and a **FastAPI + PyTorch** backend.
- **Three Interaction Modes:**
  1. **Learn Mode:** A guided, textbook-style walkthrough of the math behind BDH (Protocol, State vs. Fast Weights, Attention as Logic).
  2. **Explore Mode (BDH-GPU):** A live interface where users can type prompts and watch the model think. It visualizes the embedding heatmap, per-layer processing, and final output logits.
  3. **Experiment Mode:** Tools for researchers, including:
     - **Ablation Playground:** Turn off specific neurons to see how they affect the output.
     - **Hebbian Network Graph:** A force-directed graph showing how the fast-weight matrix evolves.
     - **Concept Explorer:** Compare activation patterns between two different prompts ("2+2=" vs. "Hello").

### Why It Matters

Deep learning education often suffers from static diagrams. BDH Explainer provides an *interactive laboratory* where the abstract equations of Hebbian learning become visible, manipulable dynamic systems.

---

## Path C2: BDH Medical Assistant (Inference-Time Learning)

**Objective:** Test the "Inference-Time Learning" hypothesis — can BDH's Hebbian synaptic plasticity, enhanced with TTT/DeltaNet memory mechanisms, store and retrieve medical facts without retraining?

### Architecture & Experiments

- **Model:** A custom 105M-parameter BDH-GPU model trained on 650M tokens (FineWeb-Edu + PubMed abstracts).
- **The Challenge:** Standard Transformers require fine-tuning for new knowledge. We tested whether BDH's delta-rule memory could learn patient records at inference time via learned projections ($\theta_K$, $\theta_Q$).
- **Dual-Memory System:**
  1. **Matrix Memory:** O(1) storage via Delta Rule update: $S_t = S_{t-1} + \beta_t(v_t - S_{t-1}k_t)k_t^T$
  2. **RAG Cache:** Position-weighted context keys with cosine similarity retrieval for reliable fallback.

### Results

| Experiment | Success Rate | Implication |
|:---|:---|:---|
| **Single Fact Recall** | 100% | Delta-rule memory correctly stores and retrieves isolated associations. |
| **Two-Patient Discrimination** | 100% | Position-weighted keys (0.05 mean similarity) enable unique addressing per patient. |
| **Matrix O(1) Retrieval** | 100% | Retrieval time is constant regardless of memory size (10 vs. 1000 facts: <0.1 ms variance). |
| **End-to-End Medical Recall** | Mixed | Layer 5 retrieves with 0.78 similarity, but gates (trained to 0.25) need ≥0.90 to override base model predictions — the "Gate Training Gap." |

---

## Path B: BDH Game of Life (Mechanistic Interpretability)

**Objective:** Prove BDH interpretability by reverse-engineering a model trained to play Conway's Game of Life.

### Methodology

We trained a small BDH model to predict the next state of a Game of Life grid with near **100% accuracy**. Then, we dissected it neuron by neuron and layer by layer.

### Discoveries

- **Sparsity:** Only a handful of neurons were active — around 5–10% depending on the layer.
- **Natural Monosemanticity:** Unlike Transformers, where one neuron might represent "the color red" AND "medieval history," BDH neurons were found to map 1:1 with logic rules.
- **The "Life" Circuit:** We identified specific neurons solely responsible for the "Birth" rule (3 neighbors → alive).
- **The "Death" Circuit:** Similarly, we found inhibitory neurons that implement the overpopulation rule (>3 neighbors → die).
- **Synergy Networks:** We mapped the functional connectivity graph, proving the model learned a crisp logical algorithm rather than a messy heuristic.

### Interactive Analysis Tools

The project includes a suite of Python scripts for:

- **The Game + Activation:** An interactive game where you input your initial board and see how it proceeds in time, along with the attention of each cell and activation of neurons.
- **Neuron Hunting:** PCA and linear probing of the hub space.
- **Ablation Studies:** Rigorously testing whether removing a "Birth" neuron strictly breaks the birth rule.
- **3D Manifold Visualization:** Interactive HTML plots of the activation space.

---

## Key Research Insights

Synthesizing findings across all three projects:

1. **The Promise of Inference-Time Learning:** This project demonstrates that *BDH's linear attention is a valid memory substrate*. We can write patient records to the memory matrix during inference and retrieve them with O(1) complexity. This opens the door to medical AI systems that learn new patients instantly without retraining or context window limitations.
2. **Sparsity = Interpretability:** The Game of Life project provides evidence that architectural sparsity (as in BDH) is a "cheat code" for interpretability. We did not need complex Sparse Autoencoders (SAEs) to understand the model; the model *was* the SAE.
3. **The Decoder Bottleneck:** A recurring limitation (seen in the Medical Assistant) is the "readout" capability. While the internal memory is robust, small models struggle to translate that high-dimensional memory back into tokens. Future work must focus on scaling up the "Decoder" or "Readout" heads.

---

## Installation, Usage Guide, and Hosted Demo

The installation instructions and hosted demo links for all three projects are in their respective `README.md` files:

- [`BDH_Explainer/README.md`](BDH_Explainer/README.md)
- [`BDH_Medical_Assistant/README.md`](BDH_Medical_Assistant/README.md)
- [`Game_of_life/README.md`](Game_of_life/README.md)

---

---
## Demo

* [Watch the Demo on YouTube](https://youtu.be/J0KXfoDiO3M)
* [Try the Live Demo](https://bdh-explainer.onrender.com/)
## Limitations and Future Scope

### Current Limitations

- **Retrieval-to-token bottleneck.** The delta-rule memory writes and retrieves embeddings successfully, but at 27M parameters the model cannot reliably decode retrieved vectors back into correct tokens through the LM head. The write path works; the readout pathway does not — yet.
- **Key collision from limited dimensionality.** With a small embedding space, different patient contexts can produce overlapping memory keys, causing retrieval confusion when many facts are stored.
- **RAG cache is O(n), not O(1).** The reliable fallback memory uses cosine similarity search over all stored keys, which scales linearly with the number of stored facts. True O(1) retrieval via the internal delta-rule memory is still experimental.
- **Single-GPU, small-scale experiments only.** All results are from a single GPU with a 27M-parameter model. The BDH paper's claims about scaling behavior at 100M–1B parameters remain untested in this implementation.
- **Medical domain is a proof of concept.** The patient records used are short and structured. Real clinical notes are longer, noisier, and more complex.
- **Synergy analysis is approximate.** Pairwise only; higher-order interactions are sampled, not exhaustively searched.
- **Linear probes assume linear readout.** Non-linear probes would achieve higher accuracy but at the cost of geometric interpretability.
- **Single model instance.** One random seed. Multi-seed variance analysis is future work.
- **Game of Life scope.** The mechanistic interpretability analysis was performed on a small grid; scaling to larger state spaces remains untested.

### Future Scope

**Medical Assistant & Memory:**
- **Scale to 100M–1B parameters** to test whether the retrieval-to-token bottleneck resolves with model capacity, as the BDH paper's scaling laws suggest.
- **Dedicated memory decoder head** — a separate MLP or cross-attention layer that maps retrieved memory vectors back to token space, bypassing the LM head bottleneck.
- **Retrieval-augmented fine-tuning** — train the model end-to-end with memory retrieval in the loop so the LM head learns to interpret memory outputs.
- **Longer medical records** — test with real-world clinical notes, discharge summaries, and multi-visit patient histories.
- **Model merging for specialization** — merge a general-purpose base with a memory-specialized variant.

**BDH Explainer:**
- **GPU backend deployment** — move to a GPU-backed service for larger models and faster inference.
- **Multi-head visualization** — visualize individual attention heads separately to study head specialization.
- **Comparative mode** — side-by-side BDH vs. Transformer visualization on identical prompts.
- **Training-time visualization** — show how weights and $\sigma$ evolve during training, not just inference.
- **WebGPU rendering** — move heatmap rendering to WebGPU for full-resolution large-matrix visualization.

**Game of Life & Interpretability:**
- **Larger grids** — extend the mechanistic analysis to 16×16 and 32×32 grids to test whether the clean circuit structure holds at scale.
- **Cross-architecture comparison** — train a standard Transformer (with SAE extraction) on the same task for a rigorous head-to-head interpretability benchmark.
- **Scaling laws for interpretability** — study how BDH's sparsity changes on larger grids or more complex cellular automata (Lenia, continuous GoL).
- **Automated circuit discovery** — replace manual neuron hunting with gradient-based attribution, ACDC-style discovery, or causal scrubbing.

---

## Team & Contributions

This suite was developed by the **BDH Research Group**.

| Member | Contribution Area |
|:---|:---|
| **Aarsh Verma** | Website ideation, backend architecture and implementation |
| **Anay Gupta** | Medical Assistant chatbot development and integration |
| **Ajay Meena** | Game of Life analysis, core project ideation |
| **Abhijit Shankar, Hadekar Ankit Kishor, Prince Kumar** | Frontend ideation, UI architecture planning |
| **Manas Mistry, Gowtham Sai Reddy** | UI/UX design, user experience optimization |

---

## References

- [Baby Dragon Hatchling (BDH)](https://arxiv.org/abs/2509.26507) — Kosowski et al., 2025. The core architecture and biological inspiration.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017. The Transformer foundation.
- [TTT: Test-Time Training Layers](https://arxiv.org/abs/2407.04620) — Sun et al., 2024. Inference-time learning via self-supervised updates.
- [DeltaNet: Conditional State Space Models](https://arxiv.org/abs/2406.06484) — Yang et al., 2024. Delta-rule-inspired linear attention.
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) — Schlag et al., 2021. Connection between linear attention and fast weight memories.
- [Language Models Are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019. Base architecture reference.
- [The Organization of Behavior](https://psycnet.apa.org/record/1949-13895-000) — Hebb, 1949. Hebbian learning theory.

---

<div align="center">

*Built with ❤️ for the Future of AI Interpretation.*

</div>
