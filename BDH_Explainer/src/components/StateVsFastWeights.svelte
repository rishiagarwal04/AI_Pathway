<script lang="ts">
  /**
   * StateVsFastWeights — BDH's synaptic state σ: how it stores and updates memory.
   * Paper §2.2: BDH has n neurons with state on edges of graph G_s.
   *   State size: O(n + |E(G_s)|) — bounded by graph sparsity.
   *   Parameters: expressed through graphs G_x^e, G_x^i, G_y^e, G_y^i, G_s.
   * Contrasts with Transformer KV-cache at the bottom.
   *
   * Variable conventions (from paper):
   *   n = number of neurons (not embedding dim)
   *   d = embedding dimension (d ≪ n)
   *   T = sequence length (number of tokens)
   */
  import { onMount } from "svelte";
  import katex from "katex";

  let container: HTMLDivElement;
  let nNeurons = 8; // number of neurons for visualization

  const TEX_HEBBIAN = "Y(i),\\; X(j) \\;\\xrightarrow{G_s(i,j)}\\; \\sigma_l(i,j)";
  const TEX_SIZE = "\\sigma_l(i,j),\\; (i,j) \\in E(G_s)";

  $: bdhMemory = nNeurons * nNeurons;

  function renderMath() {
    if (!container) return;
    container.querySelectorAll(".math").forEach((el) => {
      try {
        katex.render(el.getAttribute("data-tex") || "", el as HTMLElement, {
          throwOnError: false,
          displayMode: el.classList.contains("display"),
        });
      } catch {}
    });
  }

  onMount(() => {
    renderMath();
  });

  function cellColor(i: number, j: number): string {
    const v = Math.abs(Math.sin(i * 7.3 + j * 13.7)) * 0.6 + 0.15;
    return `rgba(102, 126, 234, ${v})`;
  }
</script>

<div class="state-comparison" bind:this={container}>
  <div class="section-header">
    <span class="section-icon"></span>
    <h2> State &amp; Memory</h2>
    <p class="subtitle">
      BDH's synaptic state on graph edges: fixed-size, self-updating memory
    </p>
  </div>

  <!-- BDH Synaptic State Detail -->
  <div class="model-detail">
    <div class="model-header">
      <span class="model-icon">σ</span>
      <div>
        <h3>Synaptic State σ — The Heart of BDH Memory</h3>
        <p>
          State variables σ(i,j) live on the <strong>edges</strong> of graph G_s, encoding relationships
          between neuron pairs. Here <em>n</em> = number of neurons, and the total state size is
          O(n + |E(G_s)|). Updated every token via Hebbian learning (Round 2) — the model
          literally <em>rewires itself</em> during inference. Fixed size regardless of sequence length T.
        </p>
      </div>
    </div>

    <div class="eq-box">
      <span class="math display" data-tex={TEX_HEBBIAN}></span>
      <p class="eq-caption">Round 4l+1: Hebbian reweighting — neurons that fire together, wire together</p>
    </div>

    <div class="detail-grid">
      <div class="detail-col pros">
        <h4>✅ Strengths</h4>
        <ul>
          <li><strong>Fixed memory</strong> — O(n + |E(G_s)|) regardless of sequence length T</li>
          <li><strong>Learns during inference</strong> — Round 2 Hebbian updates every token</li>
          <li><strong>Biologically grounded</strong> — mirrors synaptic plasticity</li>
          <li><strong>Rich structure</strong> — state on edges encodes pairwise neuron relationships</li>
          <li><strong>Decay mechanism</strong> — σ ↓<sub>1−u</sub> in Round 1 prevents unbounded growth</li>
        </ul>
      </div>
      <div class="detail-col cons">
        <h4>⚠️ Trade-offs</h4>
        <ul>
          <li>Cannot perfectly recall arbitrary past tokens (lossy compression)</li>
          <li>Graceful degradation of oldest memories via decay factor u</li>
          <li>State size O(|E(G_s)|) can be up to O(n²) if graph is dense</li>
          <li>GPU implementation uses low-rank factorization to manage memory</li>
        </ul>
      </div>
    </div>

    <!-- Interactive σ matrix visualization -->

  </div>

  <!-- ═══ TRANSFORMER CONTRAST ═══ -->
  <div class="tf-contrast">
    <h3><span class="contrast-badge">vs Transformer</span> The KV-Cache Bottleneck</h3>
    <div class="contrast-grid">
      <div class="contrast-card bdh-side">
        <h4>BDH: Fixed Synaptic State σ on Edges</h4>
        <ul class="contrast-list">
          <li>Size: <strong>O(n + |E(G_s)|)</strong> — independent of sequence length T</li>
          <li>Here <em>n</em> = number of neurons, <em>|E(G_s)|</em> = number of edges in the synapse graph</li>
          <li>At T = 1 million tokens, σ is still the same fixed set of edge weights</li>
          <li>Old information decays via σ ↓<sub>1−u</sub> (like biological synaptic forgetting)</li>
          <li>Round 2 updates σ every token — the model <em>consolidates</em> as it reads</li>
        </ul>
      </div>
      <div class="contrast-card tf-side">
        <h4>Transformer: Growing KV-Cache</h4>
        <ul class="contrast-list">
          <li>Size: <strong>O(T · d)</strong> — grows linearly with every new token</li>
          <li>Here <em>T</em> = sequence length, <em>d</em> = embedding dimension per head</li>
          <li>At T = 1 million tokens, KV-cache holds 1M key-value pairs per layer per head</li>
          <li>Nothing is ever forgotten — but memory cost is unbounded</li>
          <li>No learning at inference time — all weights are frozen after training</li>
        </ul>
      </div>
    </div>
    <div class="contrast-takeaway">
      <strong>Why it matters:</strong> Modern LLMs spend the majority of their inference FLOPS on KV-cache
      attention (O(T²) per layer). BDH's fixed σ means state size is <em>constant</em>
      regardless of context length. For BDH-GPU, the attention-form dual uses O(T²) during
      training for parallelism, but the underlying state is still bounded.
    </div>
  </div>

  <!-- ── GPU Implementation Callout ────────────────────────────────── -->
  <div class="gpu-impl">
    <div class="gpu-impl-header">
      <!-- <span class="gpu-badge">⚡</span> -->
      <h3>BDH GPU Implementation</h3>
    </div>
    <p>
      In theory, BDH stores individual edge weights σ(i,j) for every edge in G_s. For dense graphs
      this is O(n²). BDH-GPU solves this with a <strong>low-rank product representation</strong>
      of the parameter graphs, making it fit on a single GPU.
    </p>
    <div class="gpu-detail-grid">
      <div class="gpu-card theory-card">
        <h4>Theory: Explicit Graph Edges</h4>
        <p>Parameters on 5 graphs, state on edges of G_s:</p>
        <div class="gpu-eq"><code>G_x<sup>e</sup>, G_x<sup>i</sup>, G_y<sup>e</sup>, G_y<sup>i</sup>, G_s</code></div>
        <p>State: σ(i,j) for (i,j) ∈ E(G_s). For n = 100K neurons with dense graph → 10B entries per layer.</p>
      </div>
      <div class="gpu-card gpu-card-impl">
        <h4>BDH-GPU: Low-Rank Factorization</h4>
        <p>Graph edges are <strong>implicit</strong> via low-rank product of d×n matrices:</p>
        <div class="gpu-eq"><code>W_enc ∈ ℝ<sup>d×n</sup>, W_enc_v ∈ ℝ<sup>d×n</sup>, W_dec ∈ ℝ<sup>n·h×d</sup></code></div>
        <p>With d ≈ 256 and n up to 1M, this is O(n·d) per layer — orders of magnitude smaller than O(n²).</p>
      </div>
    </div>
    <div class="gpu-detail">
      <strong>Key insight (Paper §3):</strong> The σ matrix is never materialized. Instead, the
      attention mechanism <code>scores = (QR · QRᵀ).tril(-1)</code> computes the cumulative
      outer products implicitly. The ReLU sparse encoding ensures that this low-rank
      approximation preserves the modular, scale-free graph structure of the biological model.
    </div>
    <p class="gpu-note">
      Variables: n = number of neurons per head, d = embedding dimension, h = number of heads.
      Total parameters per layer: (3+o(1))·n·d. Compare to a Transformer layer which also uses O(d²) ≈ O(n·d) parameters but stores O(T·d) state.
    </p>
  </div>
</div>

<style>
  .state-comparison {
    max-width: 920px;
    margin: 0 auto 40px;
    padding: 0 16px;
  }
  .section-header {
    text-align: center;
    margin-bottom: 20px;
  }
  .section-icon {
    font-size: 2rem;
  }
  .section-header h2 {
    margin: 8px 0 4px;
    color: var(--text-heading);
    font-size: 1.5rem;
  }
  .subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
  }

  .model-detail {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 24px;
    box-shadow: var(--shadow-card);
    border: 1px solid var(--border-color);
    border-top: 3px solid #667eea;
    margin-bottom: 20px;
  }
  .model-header {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    margin-bottom: 16px;
  }
  .model-icon {
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    font-size: 1.3rem;
    font-weight: 800;
    color: white;
    background: linear-gradient(135deg, #667eea, #764ba2);
    flex-shrink: 0;
  }
  .model-header h3 {
    margin-bottom: 4px;
    color: var(--text-heading);
  }
  .model-header p {
    font-size: 0.87rem;
    color: var(--text-secondary);
    line-height: 1.55;
  }

  .eq-box {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    margin-bottom: 16px;
  }
  .eq-caption {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 6px;
    font-style: italic;
  }

  .detail-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 16px;
  }
  @media (max-width: 600px) {
    .detail-grid {
      grid-template-columns: 1fr;
    }
  }
  .detail-col {
    padding: 14px;
    border-radius: 8px;
    border: 1px solid var(--border-color);
  }
  .detail-col.pros {
    background: var(--bg-success);
    border-color: var(--border-success);
  }
  .detail-col.cons {
    background: var(--bg-danger);
    border-color: var(--border-danger);
  }
  .detail-col h4 {
    font-size: 0.85rem;
    margin-bottom: 8px;
    color: var(--text-heading);
  }
  .detail-col ul {
    margin: 0;
    padding-left: 18px;
  }
  .detail-col li {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.55;
    margin-bottom: 3px;
  }

  .state-viz-area {
    margin-top: 16px;
  }
  .state-viz-area h4 {
    font-size: 0.9rem;
    margin-bottom: 10px;
    color: var(--text-heading);
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .slider-inline {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .slider-inline label {
    font-size: 0.85rem;
    color: var(--text-secondary);
  }
  .slider-inline input[type="range"] {
    width: 160px;
  }
  .mem-badge {
    font-size: 0.78rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    padding: 3px 10px;
    border-radius: 20px;
  }
  .state-viz {
    padding: 16px;
    background: var(--bg-surface);
    border-radius: 8px;
  }
  .matrix-grid {
    display: grid;
    gap: 3px;
  }
  .cell {
    padding: 6px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: monospace;
    text-align: center;
    min-width: 36px;
    color: var(--text-heading);
  }
  .matrix-cell {
    font-size: 0.65rem;
    padding: 4px;
    transition: transform 0.15s;
  }
  .matrix-cell:hover {
    transform: scale(1.15);
    z-index: 1;
  }
  .state-note {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 10px;
    line-height: 1.55;
  }

  /* ── Transformer Contrast ── */
  .tf-contrast {
    margin-top: 24px;
    padding: 20px 22px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.06), rgba(59, 130, 246, 0.06));
    border: 1px solid rgba(102, 126, 234, 0.18);
  }
  .tf-contrast h3 {
    font-size: 1rem;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .contrast-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    background: linear-gradient(135deg, #667eea, #3b82f6);
    color: #fff;
    padding: 3px 10px;
    border-radius: 20px;
  }
  .contrast-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }
  @media (max-width: 700px) {
    .contrast-grid {
      grid-template-columns: 1fr;
    }
  }
  .contrast-card {
    padding: 16px;
    border-radius: 10px;
    border: 1px solid var(--border-color);
  }
  .contrast-card h4 {
    font-size: 0.88rem;
    margin-bottom: 10px;
  }
  .contrast-list {
    margin: 0;
    padding-left: 18px;
  }
  .contrast-list li {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.6;
    margin-bottom: 4px;
  }
  .bdh-side {
    background: rgba(102, 126, 234, 0.07);
    border-color: rgba(102, 126, 234, 0.2);
  }
  .tf-side {
    background: rgba(59, 130, 246, 0.07);
    border-color: rgba(59, 130, 246, 0.2);
  }
  .contrast-takeaway {
    padding: 14px 18px;
    border-radius: 8px;
    background: rgba(16, 185, 129, 0.07);
    border-left: 4px solid #10b981;
    font-size: 0.84rem;
    line-height: 1.6;
    color: var(--text-primary);
  }

  /* ── GPU Implementation callout ── */
  .gpu-impl {
    background: linear-gradient(135deg, rgba(245,158,11,0.06), rgba(239,68,68,0.06));
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 12px;
    padding: 24px;
    margin-top: 28px;
  }
  .gpu-impl-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
  }
  .gpu-impl-header .gpu-badge { font-size: 1.3rem; }
  .gpu-impl-header h3 {
    font-size: 1.05rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .gpu-impl p {
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.6;
    margin: 0 0 12px;
  }
  .gpu-detail-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }
  .gpu-card {
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px;
    background: var(--bg-card);
  }
  .gpu-card h4 {
    font-size: 0.88rem;
    margin: 0 0 8px;
    color: var(--text-heading);
  }
  .gpu-card p {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin: 0 0 8px;
    line-height: 1.5;
  }
  .gpu-card ul {
    margin: 8px 0 0;
    padding-left: 18px;
  }
  .gpu-card li {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 3px;
  }
  .theory-card { border-left: 3px solid #667eea; }
  .gpu-card-impl { border-left: 3px solid #f59e0b; }
  .gpu-eq {
    background: var(--bg-inset);
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 8px;
    text-align: center;
  }
  .gpu-eq code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-heading);
  }
  .gpu-detail {
    background: var(--bg-surface);
    border-radius: 8px;
    padding: 14px;
    font-size: 0.83rem;
    color: var(--text-secondary);
    line-height: 1.6;
    border-left: 3px solid #f59e0b;
    margin-bottom: 12px;
  }
  .gpu-detail code, .gpu-impl code {
    background: rgba(245,158,11,0.12);
    padding: 1px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82em;
  }
  @media (max-width: 700px) {
    .gpu-detail-grid { grid-template-columns: 1fr; }
  }
</style>