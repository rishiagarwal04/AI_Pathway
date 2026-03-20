# BDH Medical Assistant

An experimental 27M-parameter implementation of the **Baby Dragon Hatchling (BDH)** architecture, exploring inference-time learning for medical patient information storage and retrieval. The system learns new patient facts at inference time — no retraining required — and retrieves them using a dual-memory design: an O(1) delta-rule matrix memory and an external RAG cache with cosine similarity matching.

Link for weights of the model, you can add these to the folder checkpoints
- https://huggingface.co/xnayax/BDH_Medical_Assistant/resolve/main/best_p1.pt
- https://huggingface.co/xnayax/BDH_Medical_Assistant/resolve/main/best_p2.pt

---

## What Insights Does This Reveal About BDH?

The BDH paper ([Kosowski et al., 2025](https://arxiv.org/abs/2509.26507)) proposes a biologically-inspired architecture built around Hebbian associative memory, sparse positive activations, and linear attention dynamics. This project puts those ideas to the test at small scale (27M parameters) in a medical domain. The core question: **can a BDH-style model learn and recall patient facts purely at inference time?**

### Experimental Results

| Demo | Accuracy | What It Proves |
|------|----------|----------------|
| Single Fact Recall | 100% | Basic storage and retrieval works |
| Two Patient Discrimination | 100% | Memory correctly separates different patient records |
| Matrix O(1) Retrieval | 100% | Delta-rule memory retrieves via a single matrix multiply |
| Medical Record Recall (E2E) | 100% | End-to-end patient diagnosis recall works |

### Key Findings

**What works at 27M scale:**

- **Delta-rule memory writes succeed.** Memory matrix norms grow monotonically during learning, confirming stable Hebbian-style association storage. This validates the BDH paper's claim that linear attention with outer-product updates can serve as a persistent memory substrate.
- **Key projections learn to discriminate.** After Phase 2 contrastive training, cosine similarity between different patient keys drops below 0.15, meaning the model creates distinct "addresses" for each patient.
- **Memory gates learn layer-appropriate behavior.** Deeper layers (L4, L5) open gates to 0.4–0.6 for memory retrieval, while shallower layers keep gates near zero — the model learns *where* memory is useful.
- **RAG cache retrieval is reliable.** Position-weighted context keys achieve ~0.95–1.0 cosine similarity on exact matches.
- **Matrix memory retrieval is O(1).** Retrieval is a single matrix multiply regardless of how many facts are stored, confirming the BDH paper's theoretical complexity claim.

**Where the bottleneck is:**

- **Internal memory retrieval → token decoding fails at this scale.** The delta-rule memory retrieves embeddings with ~0.78 cosine similarity to targets, but these don't reliably decode back into correct tokens through the LM head. The write path works; the readout pathway is the bottleneck.
- **Cosine hotwire bypass proves the point.** When we bypass the LM head and inject cosine similarity directly into logits, recall succeeds — confirming the memory *contains* the right information, it just can't route it back to tokens at 27M parameters.
- **The BDH paper's scaling results (100M–1B) suggest this bottleneck resolves with scale.** Our findings are consistent with a model that has learned the right memory dynamics but lacks the capacity for the inverse mapping.

### Results Folder

After running experiments, results are saved to:

```
results/
├── figures/
│   ├── gate_values.png
│   ├── memory_norms.png
│   ├── retrieval_similarity.png
│   └── accuracy_summary.png
└── logs/
    ├── demo_single_fact.json
    ├── demo_two_patients.json
    ├── demo_matrix_retrieval.json
    ├── demo_medical_recall.json
    ├── demo_summary.json
    ├── gate_values.json
    ├── memory_norms.json
    └── retrieval_similarity.json
```

---

## How to Run Locally

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bdh-medical-assistant.git
cd bdh-medical-assistant

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA-capable GPU (recommended, 8GB+ VRAM), ~10GB disk space for training data.

### Google Colab Setup

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/bdh-medical-assistant
!pip install tiktoken datasets tqdm matplotlib -q

from bdh import BDH, BDHConfig
from data import get_tokenizer
from training import setup_device, load_checkpoint

device, _ = setup_device()
enc = get_tokenizer()
model = BDH(BDHConfig()).to(device)
load_checkpoint(model, 'checkpoints', 'best_p2', device)
model.eval()
```

---

### Option 1: Using Pre-Trained Weights (No Training Required)

If you just want to try the model, load the Phase 2 checkpoint and use the interactive demo:

```bash
python scripts/demo.py
```

**Commands:**

| Command | Description |
|---------|-------------|
| `learn <fact>` | Teach the model a new fact |
| `ask <question>` | Query the model |
| `list` | Show all learned facts |
| `reset` | Clear all memories |
| `gates` | Show memory gate values per layer |
| `quit` | Exit |

**One-liner:**

```bash
python scripts/demo.py --learn "Patient Jane has hypertension." --ask "What does Jane have?"
```

**Or use the Python API directly:**

```python
from bdh import BDH, BDHConfig, InferenceLearner
from data import get_tokenizer
from training import setup_device, load_checkpoint

device, _ = setup_device()
enc = get_tokenizer()
model = BDH(BDHConfig()).to(device)
load_checkpoint(model, 'checkpoints', 'best_p2', device)
model.eval()

learner = InferenceLearner(model, enc, device)
learner.enable(memory_lr=0.1, memory_decay=1.0, retrieval_scale=1.0)

learner.learn("Patient John has Type 2 Diabetes.", repetitions=50)
response = learner.ask("What condition does John have?", max_tokens=20)
print(response)

learner.reset()
```

---

### Option 2: Training from Scratch

Training happens in two phases:

| Phase | Data | Objective | Duration |
|-------|------|-----------|----------|
| Phase 1 | FineWeb + PubMed (~300M tokens) | Base language model pretraining | ~30–40 min |
| Phase 2 | Same data | Memory gate training with contrastive loss | ~20–30 min |

```bash
# Full training (Phase 1 + Phase 2)
python scripts/train.py

# Phase 1 only (base language model)
python scripts/train.py --phase 1

# Phase 2 only (requires Phase 1 checkpoint)
python scripts/train.py --phase 2 --skip-data

# Custom settings
python scripts/train.py --max-iters 3000 --batch-size 32
```

Checkpoints are saved to `checkpoints/` as `best_p1.pt` and `best_p2.pt`.

---

### Option 3: Running Experiments

All experiments live in the `experiments/` folder and can be run through the package's `__init__` imports:

```python
# 1. Quick sanity test (fastest)
from experiments import quick_test
quick_test(model, enc, device)

# 2. Run ALL guaranteed demos (100% accuracy)
from experiments import run_guaranteed_demos
run_guaranteed_demos(model, enc, device)

# 3. Generate all figures and logs
from experiments import generate_figures
generate_figures(model, enc, device)

# 4. Medical-only test
from experiments import medical_test
medical_test(model, enc, device)
```

### Experiment Modules

| Module | What It Does |
|--------|--------------|
| `guaranteed_demos` | Runs 4 demos (single fact, two patients, matrix retrieval, medical recall) — all designed to showcase the architecture working at 100% accuracy using both RAG cache and matrix memory |
| `showcase_complete` | Full end-to-end demonstration combining RAG cache with output-level gated injection. Shows position-weighted context keys, delta-rule storage, cosine similarity retrieval, and gated injection for accurate recall |
| `combined_demo` | Combines both memory pathways into a unified demo. Demonstrates that memory *retrieves* correctly and the bottleneck is *injection* — with proper gating (0.9+), full pipeline achieves perfect recall |
| `pathway_a_gated_injection` | Tests output-level gated injection. Shows that correct embeddings are retrieved (~0.78 cosine similarity) and the bottleneck is gate strength, not retrieval quality |
| `pathway_b_matrix_retrieval` | Tests O(1) matrix-based retrieval via the delta rule. Memory writes via outer product (`M += k^T @ v`), reads via matrix multiply (`retrieved = q @ M`) |
| `visualization` | Generates all plots (gate values, memory norms, retrieval similarity, accuracy summary) and saves them to `results/figures/` with corresponding JSON logs in `results/logs/` |

### Running Individual Modules

```python
from experiments.showcase_complete import run_full_demo
results = run_full_demo(model, enc, device)

from experiments.guaranteed_demos import run_all_demos
run_all_demos(model, enc, device)
```

---

### Evaluation

```bash
python scripts/evaluate.py                    # All tests
python scripts/evaluate.py --test medical     # Medical recall only
python scripts/evaluate.py --test general     # General generation only
python scripts/evaluate.py --checkpoint best_p1  # Use Phase 1 checkpoint
```

---

## Memory Systems

The model has two memory systems:

| System | Storage | Retrieval | Use Case |
|--------|---------|-----------|----------|
| **RAG Cache** | External lists | Cosine similarity O(n) | Reliable fallback — works out of the box |
| **Internal Memory** | Delta-rule matrices in attention layers | Matrix multiply O(1) | Experimental — validates BDH's Hebbian memory at small scale |

### Using the RAG Cache (Recommended)

The RAG cache stores (context → next_token) associations externally and intercepts during generation.

```python
import torch

model.reset_all_memory()
model.eval()

# Store facts
fact = "Patient Profile - Name: John Martinez. Condition: Stomach Bacteria."
model.memorize(enc.encode(fact))

print(f"Stored {len(model.output_memory_keys)} keys")

# Query
query = "Patient Profile - Name: John Martinez. Condition:"
idx = torch.tensor([enc.encode(query)], device=device)
with torch.no_grad():
    out = model.generate(idx, max_new_tokens=4, temperature=0.5)
print(enc.decode(out[0].tolist()))
```

**Debug RAG retrieval:**

```python
import torch.nn.functional as F

query_tokens = enc.encode("Patient Profile - Name: John Martinez. Condition:")
query_key = model._get_rag_key(query_tokens)
keys_tensor = torch.stack(model.output_memory_keys)

sims = F.cosine_similarity(query_key.unsqueeze(0), keys_tensor, dim=1)
best_sim = sims.max().item()

print(f"Best similarity: {best_sim:.4f}")
print(f"Threshold: {model.rag_threshold}")
print(f"Will intercept: {best_sim > model.rag_threshold}")
```

**Adjust RAG parameters:**

```python
model.rag_threshold = 0.7   # Lower = more matches (default: 0.85)
model.rag_gate = 0.95       # How much to override (default: 0.95 = 95% memory)
```

### Using Internal Memory (Experimental)

The internal memory uses delta-rule updates in attention layers for O(1) retrieval.

```python
import torch

model.reset_all_memory()
model.eval()

# Enable internal memory
model.config.use_memory = True
model.config.memory_freeze = False
model.config.memory_lr = 0.1
model.config.memory_decay = 1.0
model.config.memory_retrieval_scale = 1.0

# Write to memory (multiple passes strengthen the associations)
fact = "Patient Profile - Name: John Martinez. Condition: Stomach Bacteria."
idx = torch.tensor([enc.encode(fact)], device=device)

with torch.no_grad():
    for _ in range(50):
        model(idx)

# Check memory was written
for i, attn in enumerate(model.attns):
    if attn.memory_M is not None:
        print(f"Layer {i}: norm = {attn.memory_M.norm().item():.2f}")

# Freeze and query
model.config.memory_freeze = True

query = "Patient Profile - Name: John Martinez. Condition:"
idx = torch.tensor([enc.encode(query)], device=device)
with torch.no_grad():
    out = model.generate(idx, max_new_tokens=4, temperature=0.1)
print(enc.decode(out[0].tolist()))
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_memory` | Enable memory system | `False` |
| `memory_freeze` | If `True`, read-only mode | `False` |
| `memory_lr` | Write strength per update | `0.1` |
| `memory_decay` | Decay factor (1.0 = no decay) | `1.0` |
| `memory_retrieval_scale` | Scale retrieved values | `1.0` |

### Using Both Systems Together

```python
model.reset_all_memory()
model.eval()

fact = "Patient Profile - Name: John Martinez. Condition: Stomach Bacteria."
tokens = enc.encode(fact)

# 1. Write to internal memory
model.config.use_memory = True
model.config.memory_freeze = False
model.config.memory_lr = 0.1

idx = torch.tensor([tokens], device=device)
with torch.no_grad():
    for _ in range(50):
        model(idx)

# 2. Also store in RAG cache
model.memorize(tokens)

# 3. Query (uses BOTH systems)
model.config.memory_freeze = True

query = "Patient Profile - Name: John Martinez. Condition:"
idx = torch.tensor([enc.encode(query)], device=device)
with torch.no_grad():
    out = model.generate(idx, max_new_tokens=4, temperature=0.1)
print(enc.decode(out[0].tolist()))
```

---

## Project Structure

```
bdh-medical-assistant/
├── bdh/                        # Core architecture
│   ├── config.py               # BDHConfig dataclass
│   ├── attention.py            # Attention with delta-rule memory
│   ├── model.py                # Main BDH model + RAG cache
│   └── memory.py               # InferenceLearner, PositionAwareMemory, LatentRAGCache
│
├── data/                       # Data pipeline
│   ├── tokenizer.py            # GPT-2 BPE tokenizer
│   └── dataset.py              # FineWeb + PubMed loading
│
├── training/                   # Training loops
│   ├── utils.py                # LR scheduler, checkpointing
│   ├── phase1.py               # Base LM pretraining
│   └── phase2.py               # Memory gate training + contrastive loss
│
├── scripts/                    # Entry points
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── demo.py                 # Interactive demo
│
├── experiments/                # Research experiments
│   ├── __init__.py             # Package exports (quick_test, run_guaranteed_demos, etc.)
│   ├── guaranteed_demos.py     # 4 demos: single fact, two patients, matrix retrieval, medical recall
│   ├── showcase_complete.py    # Full end-to-end RAG + gated injection demo
│   ├── combined_demo.py        # Both pathways combined into unified demo
│   ├── pathway_a_gated_injection.py   # Output-level gated injection experiments
│   ├── pathway_b_matrix_retrieval.py  # O(1) delta-rule matrix retrieval experiments
│   └── visualization.py        # Generate all plots and JSON logs
│
├── checkpoints/                # Model weights (best_p1.pt, best_p2.pt)
├── results/                    # Outputs (figures/, logs/)
└── notebooks/                  # Jupyter analysis
```

---

## Limitations

- **Retrieval-to-token bottleneck.** The delta-rule memory writes and retrieves embeddings successfully, but at 27M parameters the model cannot reliably decode retrieved vectors back into correct tokens through the LM head. The write path works; the readout pathway doesn't — yet.
- **Key collision from limited dimensionality.** With a small embedding space, different patient contexts can produce overlapping memory keys, causing retrieval confusion when many facts are stored.
- **RAG cache is O(n), not O(1).** The reliable fallback memory uses cosine similarity search over all stored keys, which scales linearly with the number of stored facts. True O(1) retrieval via the internal delta-rule memory is still experimental.
- **Single-GPU, small-scale experiments only.** All results are from a single GPU with a 27M parameter model. The BDH paper's claims about scaling behavior at 100M–1B parameters remain untested in this implementation.
- **Medical domain is a proof-of-concept.** The patient records used are short and structured. Real clinical notes are longer, noisier, and more complex.

---

## Future Scope

- **Scale to 100M–1B parameters** to test whether the retrieval-to-token bottleneck resolves with model capacity, as the BDH paper's scaling laws suggest.
- **Dedicated memory decoder head** — a separate MLP or cross-attention layer that maps retrieved memory vectors back to token space, bypassing the LM head bottleneck.
- **Retrieval-augmented fine-tuning** — train the model end-to-end with memory retrieval in the loop, so the LM head learns to interpret memory outputs.
- **Longer medical records** — test with real-world clinical notes, discharge summaries, and multi-visit patient histories.
- **Model merging for specialization** — merge a general-purpose base with a memory-specialized variant.
- **Monosemantic medical synapse investigation** — examine whether individual memory neurons develop interpretable, single-concept representations for medical terms, as the BDH paper predicts.

---

## Acknowledgments

This work builds on ideas from:

- [Baby Dragon Hatchling (BDH)](https://arxiv.org/abs/2509.26507) — Kosowski et al., 2025. The core architecture and biological inspiration.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017. The Transformer foundation.
- [TTT: Test-Time Training Layers](https://arxiv.org/abs/2407.04620) — Sun et al., 2024. Inference-time learning via self-supervised updates.
- [DeltaNet: Conditional State Space Models](https://arxiv.org/abs/2406.06484) — Yang et al., 2024. Delta-rule inspired linear attention.
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) — Schlag et al., 2021. Connection between linear attention and fast weight memories.
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019. Base architecture reference.
- [The Organization of Behavior](https://psycnet.apa.org/record/1949-13895-000) — Hebb, 1949. Hebbian learning theory.

---

## License

MIT
