<div align="center">

# BDH Explainer

**An interactive visual explainer for the Baby Dragon Hatchling (BDH) neural architecture**

[![Live Demo](https://img.shields.io/badge/Live_Demo-bdh--explainer.onrender.com-667eea?style=for-the-badge)](https://bdh-explainer.onrender.com)
[![Backend API](https://img.shields.io/badge/API-bdh--explainer--backend.onrender.com-4ade80?style=for-the-badge)](https://bdh-explainer-backend.onrender.com)
[![Reference](https://img.shields.io/badge/Paper-pathwaycom%2Fbdh-blue?style=for-the-badge&logo=github)](https://github.com/pathwaycom/bdh)

</div>

---

## What We Built

BDH Explainer is a full-stack interactive visualization tool that lets you look inside the Baby Dragon Hatchling (BDH) neural architecture as it processes text in real time. Inspired by projects like [Transformer Explainer](https://poloclub.github.io/transformer-explainer/), it is purpose-built for BDH -- a biologically-grounded language model that replaces standard transformer attention with Hebbian learning and fast weight matrices. The tool provides three modes of interaction: a guided educational walkthrough of BDH theory (Learn), a live forward-pass inspector where you can type a prompt and watch every intermediate tensor flow through each layer (Explore), and a set of hands-on experimental tools for neuron ablation, concept comparison, and Hebbian network visualization (Experiment). The backend runs the real BDH model (from [pathwaycom/bdh](https://github.com/pathwaycom/bdh)) with full trace extraction, and the frontend renders every matrix, activation pattern, and attention score using D3.js heatmaps, force-directed graphs, and animated pipelines -- all in the browser with both light and dark themes.

---

## How to Run Locally

### Prerequisites

- Node.js 18 or higher
- Python 3.10 or higher
- pip (Python package manager)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Pika-pikachu30/BDH_Explainer.git
cd BDH_Explainer

# 2. Install and start the frontend
npm install
npm run dev
# Frontend will be available at http://localhost:3000

# 3. In a separate terminal, install and start the backend
cd backend
pip install -r requirements.txt
python app.py
# Backend API will be available at http://localhost:5000
```

The frontend automatically proxies API calls to the local backend in development mode via the Vite dev server. No additional configuration is needed.

To build for production:

```bash
npm run build
# The output will be in the dist/ directory
```

---

## How to Access the Hosted Demo

The application is deployed on Render and available publicly:

- **Frontend:** [https://bdh-explainer.onrender.com](https://bdh-explainer.onrender.com)
- **Backend API:** [https://bdh-explainer-backend.onrender.com](https://bdh-explainer-backend.onrender.com)

Note: The deployment uses Render's free tier. The first load may take approximately 50 seconds while the backend service starts up. After the initial cold start, subsequent requests will be fast.

---

## Features

### Three Modes of Exploration

The application is organized into three top-level modes, each targeting a different level of engagement:

| Mode | Purpose |
|------|---------|
| **Learn** | Guided educational walkthrough of BDH theory across 3 chapters |
| **Explore (BDH-GPU)** | Live forward-pass inspector with per-layer tensor visualizations |
| **Experiment** | Hands-on tools for neuron ablation, concept comparison, and Hebbian graph visualization |

---

### Learn Mode

#### Chapter 1: The Protocol (BDH Protocol)
- Each round is presented with its corresponding update equation 
- Detailed explanation of the decay mechanism, Hebbian outer-product updates, replicator thresholding, and residual connections

#### Chapter 2: State and Memory (State vs Fast Weights)
- Side-by-side comparison of BDH's fixed-size synaptic state against Transformer's growing KV-cache
- Strengths and trade-offs analysis (fixed memory footprint, Hebbian learning, decay mechanism versus lossy compression)

#### Chapter 3: Attention (Attention as Logic)
- Explanation of how linear attention corresponds to the S combinator from combinatory logic

---

### Explore Mode

#### Layer Explorer (Model Internals)
- Prompt input with instant-run on Enter, plus preset example prompts ("2+2=", "How are you?", "The capital of France is")
- Temperature, max tokens, and top-k controls for generation
- Token strip where you can click or hover over tokens to highlight them across all visualizations
- Embedding heatmap rendered via the reusable MatrixHeatmap component
- Collapsible per-layer blocks showing the full BDH processing pipeline through the RealTimeLayer component
- Output logits bar chart displaying top-k next-token probabilities
- Generated text display with continuation highlighting
- Model config badge showing architecture parameters (layers, dimensions, heads, neuron dimension, vocabulary size)

#### Model Explorer
- Registry view listing all registered models (default pretrained plus custom uploads)
- Model configuration cards showing architecture details: n_layer, n_embd, n_head, vocab_size, neuron_dim, total_params
- Hot-swap active model for all other views without restarting the application
- Delete custom models (the default model is protected)
- Validation of uploaded files for correct BDH checkpoint structure

---

### Experiment Mode

#### Concept Activation Explorer
- Dual-prompt input to enter two different prompts (for example, "2+2=" versus "Hello") and compare their neuron activations
- Neuron fingerprint detail view -- click any neuron to see its per-token activation profile across both prompts

#### Ablation Playground
- Baseline run to execute the model normally and see neuron activations alongside output probabilities
- Token-based ablation -- click a token to ablate all neurons that fire for it
- Impact analysis with a diff visualization showing probability changes between baseline and ablated outputs

#### Hebbian Network Graph
- Real-time D3.js force-directed graph where nodes represent neurons and edges represent fast-weight connections (sigma matrix entries)
- Token step scrubber to play, pause, and step through token timesteps and watch sigma evolve via the outer-product update rule

---

### Additional Features

- **Dark and Light Theme** -- Toggle between dark and light mode with automatic persistence via localStorage, and automatic detection of system preference on first visit
- **Responsive Navigation** -- Top navigation bar with mode tabs and sub-navigation tabs for each mode, with mobile-friendly wrapping
- **Automatic API Routing** -- The frontend automatically detects whether it is running locally (proxies through Vite) or in production (connects directly to the hosted backend)
- **Custom Model Upload** -- Upload your own BDH checkpoints trained on different tasks and switch between them without restarting

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Svelte 4 + TypeScript | Reactive UI with minimal bundle size |
| Visualizations | D3.js + GSAP | Heatmaps, force-directed graphs, animations |
| Math Rendering | KaTeX | Inline and display equations for BDH formulas |
| Backend | FastAPI + Uvicorn | Async Python API serving the real BDH model |
| Model | PyTorch (CPU) | BDH inference with full trace extraction |
| Hosting | Render | Static site (frontend) + Web service (backend) |

---

## Project Structure

```
BDH_Explainer/
|-- src/                                    # Svelte frontend
|   |-- main.ts                             # App entry point
|   |-- api.ts                              # Centralized API config (local/hosted auto-switch)
|   |-- App.svelte                          # Root -- navigation, theming, mode switching
|   |-- app.css                             # Global styles (light/dark theme)
|   +-- components/
|       |-- ModelInternals.svelte            # Full forward pass explorer (Explore mode)
|       |-- RealTimeLayer.svelte             # Per-layer detail view (Explore mode)
|       |-- MatrixHeatmap.svelte             # Reusable D3 heatmap renderer
|       |-- ModelExplorer.svelte             # Checkpoint management (Explore mode)
|       |-- ModelUpload.svelte               # Quick upload widget
|       |-- BDHProtocol.svelte               # Ch.1 -- Protocol walkthrough (Learn mode)
|       |-- StateVsFastWeights.svelte        # Ch.2 -- Memory comparison (Learn mode)
|       |-- AttentionAsLogic.svelte          # Ch.3 -- Attention mechanics (Learn mode)
|       |-- ConceptActivationExplorer.svelte # Prompt comparison (Experiment mode)
|       |-- AblationPlayground.svelte        # Neuron ablation (Experiment mode)
|       +-- HebbianNetworkGraph.svelte       # Live sigma graph (Experiment mode)
|
|-- backend/
|   |-- app.py                              # FastAPI server with BDH trace extraction
|   +-- requirements.txt                    # Python deps (torch CPU, fastapi, uvicorn)
|
|-- bdh/                                    # Official Pathway BDH model
|   |-- bdh.py                              # Model implementation
|   |-- quick_train.py                      # Train a demo checkpoint
|   |-- bdh_trained.pt                      # Trained checkpoint
|   +-- README.md                           # Original BDH documentation
|
|-- vite.config.ts                          # Vite config + dev proxy
|-- index.html                              # SPA entry
+-- package.json                            # Frontend dependencies
```

## Video Demo and Images

### Video Demo

A video walkthrough of the application is available here:

[[Link to video demo](https://youtu.be/J0KXfoDiO3M)]

### Screenshots

**Concept Activation Explorer (Monosemanticity)**

Compare neuron activations between two prompts ("2+2=" vs "Hello"). Neurons are clustered by selectivity (A-only, B-only, Shared) with monosemanticity scoring, per-token filtering, and a sortable neuron fingerprint panel.

<img src="screenshots/concept_activation_explorer.png" alt="Concept Activation Explorer showing dual-prompt comparison with neuron activation map and monosemanticity analysis" width="800"/>

---

**Ablation Playground**

Zero out neurons across layers and run an ablated forward pass. The neuron grid highlights key neurons (top 5) in gold, and the impact analysis shows probability changes between baseline and ablated outputs.

<img src="screenshots/ablation_playground.png" alt="Ablation Playground showing neuron grid with disabled neurons and impact analysis comparing baseline vs ablated output probabilities" width="800"/>

---

**Hebbian Network Graph**

A force-directed D3.js graph where nodes are neurons and edges are fast-weight connections from the sigma matrix. Step through tokens to watch neurons wire together in real time via the Hebbian update rule.

<img src="screenshots/hebbian_network_graph.png" alt="Hebbian Network Graph showing force-directed visualization of neurons wiring together as sigma evolves token by token" width="800"/>

---

## License

This project is open source. The BDH model implementation is from [pathwaycom/bdh](https://github.com/pathwaycom/bdh) -- see [bdh/LICENSE.md](bdh/LICENSE.md) for its license.

---

<div align="center">

Built as an interactive explainer for understanding how the BDH neural architecture thinks.

**[Star this repo](https://github.com/Pika-pikachu30/BDH_Explainer)** if you found it useful.

</div>
