<script lang="ts">
  /**
   * BDHProtocol — The four-round BDH inference protocol.
   * Paper §2.2, Table 1(a): Each layer l performs four rounds of
   * local graph dynamics:
   *   Round 4l:   Inference from state        X(i),σ_l(i,j)→A(j)
   *   Round 4l+1: Reweighting of synapse state  Y(i),X(j)→σ_l(i,j)
   *   Round 4l+2: Replicator dynamics + inference from params  A(i),X(j)→Y(j)
   *   Round 4l+3: Inference from parameters   Y(i)→X(j)
   */
  import { onMount } from "svelte";
  import katex from "katex";

  let container: HTMLDivElement;
  let activeRound = 0; // 0 = overview, 1-4 = detail
  let animProgress = 0; // 0..1 animation within a round
  let autoPlay = false;
  let autoTimer: any;

  const rounds = [
    {
      num: 1,
      title: "Inference from State",
      color: "#667eea",
      icon: "",
      equation: "X(i),\\; \\sigma_l(i,j) \\;\\longrightarrow\\; A(j) \\qquad \\sigma_l(i,j)\\!\\downarrow_{1-u(i,j)}",
      description:
        "Each neuron j accumulates belief A(j) by reading incoming synaptic state σ_l weighted by current activations X(i). Simultaneously, the synaptic state decays by factor (1−u), preventing unbounded growth.",
      detail:
        "This is Rule 1 from Table 1: weighted fact propagation. The decay σ ↓_{1−u} acts as a forgetting mechanism — old implications fade unless reinforced. The variable A(i) is an auxiliary accumulator that collects the inference output.",
      analogy:
        "Like recalling a memory: the brain reads its current synapse strengths to determine what follows from what's currently active, while old unused memories gradually fade.",
    },
    {
      num: 2,
      title: "Reweighting of Synapse State",
      color: "#764ba2",
      icon: "",
      equation:
        "Y(i),\\; X(j) \\;\\xrightarrow{G_s(i,j)}\\; \\sigma_l(i,j) \\qquad Y(i)\\!\\downarrow",
      description:
        "Update synaptic weights between co-active neurons: if pre-synaptic neuron i has output Y(i) and post-synaptic neuron j has activation X(j), strengthen the implication σ_l(i,j). This is the Hebbian rule — neurons that fire together wire together. Y is then reset.",
      detail:
        "The edge weight G_s(i,j) from graph G_s gates whether the update occurs. This is a rank-1 outer product update to the synaptic state matrix: σ += Y·Xᵀ. After the update, Y(i) is reset to zero (↓) to prepare for the next round.",
      analogy:
        "Like learning from experience: each inference strengthens the pathways that were used, making future inferences along the same paths faster and more confident.",
    },
    {
      num: 3,
      title: "Replicator + Params",
      color: "#e11d48",
      icon: "",
      equation:
        "A(i),\\; X(j) \\;\\xrightarrow{G_y^e(i,j)}\\; Y(j) \\qquad A(i)\\!\\downarrow",
      description:
        "Compute value-side activations Y(j) using the accumulated beliefs A(i) and current activations X(j), gated by the learned parameter graph G_y^e. This combines replicator dynamics (selection pressure on neurons) with inference through fixed parameters. A is then reset.",
      detail:
        "The graph G_y^e has fixed edge weights learned during training — these are the model parameters. Unlike σ which changes during inference, G_y^e is static. The local thresholding operation (A(i)−B(i))⁺ acts as the replicator: neurons with fitness above threshold survive, others are suppressed.",
      analogy:
        "Like natural selection: useful neural pathways (those with high A) generate strong Y outputs, while weak ones are pruned. The fixed parameters G_y^e encode the species' evolved knowledge.",
    },
    {
      num: 4,
      title: "Inference from Parameters",
      color: "#059669",
      icon: "",
      equation: "Y(i) \\;\\xrightarrow{G_x^e(i,j)}\\; X(j)",
      description:
        "Decode the value-side activations Y back into the main activation space X via the learned parameter graph G_x^e. This produces the updated neuron activations X(j) that become the input to the next layer.",
      detail:
        "G_x^e is another fixed parameter graph (the decoder). In the GPU implementation, this is the decoder matrix W_dec. Combined with the residual connection and LayerNorm, this completes one layer: x_out = LN(x_in + LN(decode(xy))).",
      analogy:
        "Like passing the baton: the processed information, now enriched by inference and learning, is decoded back into the shared representation space and passed to the next processing stage.",
    },
  ];

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

  function setRound(r: number) {
    activeRound = r;
    animProgress = 0;
    setTimeout(renderMath, 50);
  }

  function nextRound() {
    if (activeRound < 4) setRound(activeRound + 1);
    else setRound(0);
  }
  function prevRound() {
    if (activeRound > 0) setRound(activeRound - 1);
    else setRound(4);
  }

  function toggleAutoPlay() {
    autoPlay = !autoPlay;
    if (autoPlay) {
      setRound(1);
      autoTimer = setInterval(() => {
        if (activeRound >= 4) {
          setRound(0);
          autoPlay = false;
          clearInterval(autoTimer);
        } else setRound(activeRound + 1);
      }, 3000);
    } else {
      clearInterval(autoTimer);
    }
  }

  // SVG pipeline visualization positions
  const pipeW = 700;
  const pipeH = 120;
  const stepW = 140;
  const gap = 20;
  const startX = 30;
  function stepX(i: number) {
    return startX + i * (stepW + gap);
  }
</script>

<div class="bdh-protocol" bind:this={container}>
  <div class="section-header">
    <span class="section-icon"></span>
    <h2>The Four-Round BDH Protocol</h2>
    <p class="subtitle">
      Each layer performs four rounds of local graph
      dynamics: communication on edges and computation on nodes
    </p>
  </div>

  <!-- Pipeline overview SVG -->
  <div class="pipeline-overview">
    <svg viewBox="0 0 {pipeW} {pipeH}" class="pipeline-svg">
      {#each rounds as round, idx}
        {@const x = stepX(idx)}
        <!-- Arrow between steps -->
        {#if idx > 0}
          <line
            x1={stepX(idx - 1) + stepW}
            y1={pipeH / 2}
            x2={x}
            y2={pipeH / 2}
            stroke={activeRound === round.num ? round.color : "#2a2a3a"}
            stroke-width="2"
            marker-end="url(#arrow)"
          />
        {/if}
        <!-- Step box -->
        <g
          class="step-group"
          class:active={activeRound === round.num}
          on:click={() => setRound(round.num)}
          role="button"
          tabindex="0"
        >
          <rect
            {x}
            y={10}
            width={stepW}
            height={pipeH - 20}
            rx="12"
            fill={activeRound === round.num ? round.color : "#0e0e16"}
            stroke={activeRound === round.num ? round.color : "#2a2a3a"}
            stroke-width="2"
            opacity={activeRound === round.num ? 1 : 0.85}
          />
          <text
            x={x + stepW / 2}
            y={pipeH / 2 - 14}
            class="step-icon"
            fill={activeRound === round.num ? "white" : "#94a3b8"}
            >{round.icon}</text
          >
          <text
            x={x + stepW / 2}
            y={pipeH / 2 + 6}
            class="step-num"
            fill={activeRound === round.num
              ? "rgba(255,255,255,0.9)"
              : "#9ca3af"}>Round {round.num}</text
          >
          <text
            x={x + stepW / 2}
            y={pipeH / 2 + 22}
            class="step-title"
            fill={activeRound === round.num ? "white" : "#94a3b8"}
            >{round.title}</text
          >
        </g>
      {/each}
      <defs>
        <marker
          id="arrow"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  </div>

  <!-- Controls -->
  <div class="nav-controls">
    <button class="btn" on:click={prevRound}>← Prev</button>
    <button class="btn primary" on:click={toggleAutoPlay}>
      {autoPlay ? "⏸ Pause" : "▶ Auto-play"}
    </button>
    <button class="btn" on:click={nextRound}>Next →</button>
  </div>

  <!-- Detail panel -->
  {#if activeRound === 0}
    <div class="overview-card">
      <h3>Protocol Overview</h3>
      <p>
        BDH processes each token through a four-round protocol in every layer.
        Click a round above to explore its details, or press Auto-play to walk
        through all four rounds.
      </p>
      <div class="overview-grid">
        {#each rounds as round}
          <button class="overview-item" on:click={() => setRound(round.num)}>
            <span class="ov-icon" style="background:{round.color}"
              >{round.icon}</span
            >
            <span class="ov-text"
              ><strong>Round {round.num}:</strong> {round.title}</span
            >
          </button>
        {/each}
      </div>
      <div class="key-insight">
        <strong>Key Insight:</strong> Unlike transformers that are purely
        feed-forward, BDH's protocol
        <em>updates its own state</em> during each forward pass. Round 1 reads from σ,
        Round 2 modifies σ via Hebbian co-activation, Round 3 computes values through
        learned parameters with replicator selection, and Round 4 decodes back to activations.
        The synaptic state σ is the system's <em>evolving memory</em>.
      </div>
    </div>
  {:else}
    {@const round = rounds[activeRound - 1]}
    <div class="detail-card" style="border-top: 3px solid {round.color}">
      <div class="detail-header">
        <span class="detail-icon" style="background:{round.color}"
          >{round.icon}</span
        >
        <div>
          <h3>Round {round.num}: {round.title}</h3>
          <p class="detail-desc">{round.description}</p>
        </div>
      </div>
      <div class="detail-eq">
        <span class="math display" data-tex={round.equation}></span>
      </div>
      <div class="detail-sections">
        <div class="detail-section">
          <h4>Technical Detail</h4>
          <p>{round.detail}</p>
        </div>
        <div class="detail-section analogy">
          <h4>Brain Analogy</h4>
          <p>{round.analogy}</p>
        </div>
      </div>
      {#if activeRound === 1}
        <!-- Round 1: inference animation -->
        <div class="round-viz">
          <svg viewBox="0 0 500 160" class="mini-viz">
            <text x="20" y="30" class="viz-label">Activations X(i)</text>
            {#each [0, 1, 2, 3] as i}
              <rect
                x={20 + i * 50}
                y={40}
                width="40"
                height="30"
                rx="4"
                fill={i < 2 ? "#667eea" : "#2a2a3a"}
              />
              <text x={40 + i * 50} y={60} class="viz-val"
                >{i < 2 ? "0.8" : "0.0"}</text
              >
            {/each}
            <text x="250" y="30" class="viz-label">Synaptic State σ</text>
            <rect
              x="250"
              y="40"
              width="80"
              height="80"
              rx="4"
              fill="#1a1a24"
              stroke="#2a2a3a"
            />
            <text x="290" y="85" class="viz-val">σ(i,j)</text>
            <text x="370" y="30" class="viz-label">Output A(j)</text>
            {#each [0, 1, 2, 3] as i}
              <rect
                x={370 + i * 30}
                y={40}
                width="24"
                height="30"
                rx="4"
                fill={i === 0 ? "#4ecdc4" : "#2a2a3a"}
              />
            {/each}
            <path
              d="M220,55 L250,55"
              stroke="#667eea"
              stroke-width="2"
              marker-end="url(#arrow)"
            />
            <path
              d="M330,55 L370,55"
              stroke="#4ecdc4"
              stroke-width="2"
              marker-end="url(#arrow)"
            />
          </svg>
        </div>
      {/if}
      {#if activeRound === 3}
        <div class="round-viz">
          <svg viewBox="0 0 500 140" class="mini-viz">
            <text x="30" y="25" class="viz-label">Replicator: A(i), X(j) → Y(j) via G_y^e</text>
            <!-- A(i) neurons -->
            <text x="60" y="50" class="viz-val">A(i)</text>
            {#each [0, 1, 2] as i}
              <circle cx={40 + i * 45} cy={85} r="16" fill={i === 0 ? '#e11d48' : i === 1 ? 'rgba(225,29,72,0.4)' : '#2a2a3a'} />
              <text x={40 + i * 45} y={89} class="viz-val" style="fill:white;font-size:9px">{i === 0 ? '0.9' : i === 1 ? '0.3' : '0.0'}</text>
            {/each}
            <!-- Arrow with G_y^e label -->
            <path d="M180,85 L260,85" stroke="#e11d48" stroke-width="2" marker-end="url(#arrow)" />
            <text x="220" y="75" class="viz-val" style="fill:#e11d48">G_y^e</text>
            <!-- Y(j) outputs -->
            <text x="340" y="50" class="viz-val">Y(j)</text>
            {#each [0, 1, 2] as i}
              <circle cx={310 + i * 45} cy={85} r="16" fill={i === 0 ? '#4ecdc4' : '#2a2a3a'} />
              <text x={310 + i * 45} y={89} class="viz-val" style="fill:white;font-size:9px">{i === 0 ? '0.7' : '0.0'}</text>
            {/each}
            <text x="250" y="130" class="viz-val">Only high-belief facts produce Y outputs (selection pressure)</text>
          </svg>
        </div>
      {/if}
      {#if activeRound === 4}
        <div class="round-viz">
          <svg viewBox="0 0 500 120" class="mini-viz">
            <text x="20" y="20" class="viz-label">Decode: Y(i) → X(j) via G_x^e  (+ residual &amp; LN)</text>
            <rect
              x="20"
              y="35"
              width="80"
              height="50"
              rx="8"
              fill="rgba(78,205,196,0.08)"
              stroke="#4ecdc4"
            />
            <text x="60" y="65" class="viz-val">Y(i)</text>
            <path
              d="M100,60 L150,60"
              stroke="#059669"
              stroke-width="2"
              marker-end="url(#arrow)"
            />
            <rect
              x="150"
              y="35"
              width="80"
              height="50"
              rx="8"
              fill="rgba(34,197,94,0.08)"
              stroke="#059669"
            />
            <text x="190" y="58" class="viz-val">G_x^e</text>
            <text x="190" y="72" class="viz-val">(W_dec)</text>
            <path
              d="M230,60 L280,60"
              stroke="#059669"
              stroke-width="2"
              marker-end="url(#arrow)"
            />
            <rect
              x="280"
              y="35"
              width="80"
              height="50"
              rx="8"
              fill="rgba(147,51,234,0.08)"
              stroke="#7c3aed"
            />
            <text x="320" y="58" class="viz-val">Residual</text>
            <text x="320" y="72" class="viz-val">+ LN</text>
            <path
              d="M360,60 L400,60"
              stroke="#7c3aed"
              stroke-width="2"
              marker-end="url(#arrow)"
            />
            <rect
              x="400"
              y="35"
              width="80"
              height="50"
              rx="8"
              fill="rgba(102,126,234,0.08)"
              stroke="#667eea"
            />
            <text x="440" y="65" class="viz-val">X(j)_new</text>
          </svg>
        </div>
      {/if}
    </div>
  {/if}

  <!-- ── GPU Implementation Callout ────────────────────────────────── -->
  <div class="gpu-impl">
    <div class="gpu-impl-header">
      <h3>BDH GPU Implementation (4 Bio Rounds → GPU Tensor Ops)</h3>
    </div>
    <p>
      The 4-round biological protocol is elegant but inherently <strong>sequential</strong>.
      For GPU training, <code>bdh.py</code> compiles the same dynamics into
      <strong>PyTorch operations</strong>:
    </p>
    <div class="gpu-detail-grid">
      <div class="gpu-card theory-card">
        <h4>Rounds 1 &amp; 2: Attention + Hebbian Update</h4>
        <p>Maps to the <strong>linear attention</strong> block:</p>
        <div class="gpu-eq"><code>x_sparse = ReLU(x · W_enc)</code></div>
        <div class="gpu-eq"><code>scores = (QR · QRᵀ).tril(-1)</code></div>
        <div class="gpu-eq"><code>yKV = LN(scores · x)</code></div>
        <p>Round 1 (read from σ) is implicit in the attention scores — the causal dot-product accumulates outer products X⊗Y from past tokens. Round 2 (Hebbian update) happens automatically as each new token’s outer product is added to the running sum.</p>
      </div>
      <div class="gpu-card gpu-card-impl">
        <h4>Rounds 3 &amp; 4: Value Encode + Gate + Decode</h4>
        <p>Maps to the <strong>gated feedforward</strong> block:</p>
        <div class="gpu-eq"><code>y_sparse = ReLU(yKV · W_enc_v)</code></div>
        <div class="gpu-eq"><code>xy = x_sparse * y_sparse</code></div>
        <div class="gpu-eq"><code>x_out = LN(x + LN(xy · W_dec))</code></div>
        <p>Round 3 (replicator dynamics) is the element-wise gate x*y — only neurons active in BOTH x and y survive (selection pressure). Round 4 (decode to X) is the W_dec projection back to residual stream.</p>
      </div>
    </div>
    <div class="gpu-detail">
      <strong>Key insight:</strong> During training, BDH-GPU never materializes the N×N
      synaptic matrix σ. Instead it uses the <strong>attention-form dual</strong>:
      <code>scores = (QR · QRᵀ).tril(-1)</code> where QR = RoPE(x_sparse).
      The cumulative outer products Σ x_t⊗x_t are computed implicitly via the
      attention mechanism — O(T²) over tokens but fully parallelizable on GPUs.
    </div>
    <p class="gpu-note">
      The biological 4-round protocol and the GPU tensor operations produce identical outputs.
      We design with brain rules, we train with tensor ops.
    </p>
  </div>
</div>

<style>
  .bdh-protocol {
    max-width: 900px;
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
  .pipeline-overview {
    overflow-x: auto;
    margin-bottom: 12px;
  }
  .pipeline-svg {
    width: 100%;
    min-width: 700px;
  }
  .step-group {
    cursor: pointer;
  }
  .step-icon {
    text-anchor: middle;
    font-size: 20px;
  }
  .step-num {
    text-anchor: middle;
    font-size: 10px;
  }
  .step-title {
    text-anchor: middle;
    font-size: 8px;
    font-weight: 600;
  }
  .nav-controls {
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 20px;
  }
  .btn {
    padding: 6px 18px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-card);
    color: var(--text-heading);
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .btn.primary {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  .btn.primary:hover {
    background: var(--accent-hover);
  }
  .overview-card,
  .detail-card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 24px;
    box-shadow: var(--shadow-card);
    border: 1px solid var(--border-color);
  }
  .overview-card h3,
  .detail-card h3 {
    margin-bottom: 10px;
    color: var(--text-heading);
  }
  .overview-card p {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.55;
    margin-bottom: 16px;
  }
  .overview-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 16px;
  }
  @media (max-width: 600px) {
    .overview-grid {
      grid-template-columns: 1fr;
    }
  }
  .overview-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: var(--bg-surface);
    border-radius: 8px;
    border: 1px solid var(--border-color);
    cursor: pointer;
    text-align: left;
  }
  .overview-item:hover {
    border-color: var(--accent);
  }
  .ov-icon {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    font-size: 1rem;
    color: white;
    flex-shrink: 0;
  }
  .ov-text {
    font-size: 0.85rem;
    color: var(--text-secondary);
  }
  .key-insight {
    padding: 14px;
    background: var(--bg-warning);
    border: 1px solid var(--border-warning);
    border-radius: 8px;
    font-size: 0.87rem;
    line-height: 1.55;
    color: #fbbf24;
  }
  .detail-header {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    margin-bottom: 16px;
  }
  .detail-icon {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    font-size: 1.3rem;
    color: white;
    flex-shrink: 0;
  }
  .detail-desc {
    color: var(--text-secondary);
    font-size: 0.88rem;
    line-height: 1.5;
    margin-top: 4px;
  }
  .detail-eq {
    text-align: center;
    padding: 16px;
    background: var(--bg-surface);
    border-radius: 8px;
    margin-bottom: 16px;
    border: 1px solid var(--border-color);
  }
  .detail-sections {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 16px;
  }
  @media (max-width: 600px) {
    .detail-sections {
      grid-template-columns: 1fr;
    }
  }
  .detail-section {
    padding: 14px;
    background: var(--bg-surface);
    border-radius: 8px;
    border: 1px solid var(--border-color);
  }
  .detail-section.analogy {
    background: var(--bg-insight);
    border-color: var(--border-insight);
  }
  .detail-section h4 {
    font-size: 0.85rem;
    margin-bottom: 6px;
    color: var(--text-heading);
  }
  .detail-section p {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
    margin: 0;
  }
  .round-viz {
    margin-top: 8px;
  }
  .mini-viz {
    width: 100%;
    max-height: 160px;
  }
  .viz-label {
    font-size: 11px;
    fill: var(--text-secondary);
    font-weight: 600;
  }
  .viz-val {
    text-anchor: middle;
    font-size: 11px;
    fill: var(--text-secondary);
    font-family: monospace;
  }

  /* ── Transformer Contrast ── */
  .tf-contrast {
    margin-top: 28px;
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
  .contrast-card .round-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 10px;
  }
  .contrast-card .mini-round {
    font-size: 0.8rem;
    padding: 5px 10px;
    background: rgba(255,255,255,0.04);
    border-radius: 6px;
  }
  .contrast-note {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin: 0;
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
  .gpu-note {
    font-size: 0.82rem !important;
    color: var(--text-muted) !important;
    font-style: italic;
    margin-bottom: 0 !important;
  }
  @media (max-width: 700px) {
    .gpu-detail-grid { grid-template-columns: 1fr; }
  }
</style>
