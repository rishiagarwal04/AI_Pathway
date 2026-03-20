<script lang="ts">
  /**
   * AttentionAsLogic — Linear attention as logical implication.
   * Paper §3.1: Linear attention = the S combinator from combinatory logic.
   * σ^(t+1) = σ^(t) + α·x·yᵀ  ↔  SKK = I  (identity combinator)
   *
   * Shows how BDH's synaptic updates mirror logical operations.
   */
  import { onMount } from "svelte";
  import katex from "katex";

  let container: HTMLDivElement;
  let activeTab: "overview" | "combinator" | "interactive" = "overview";

  // Interactive implication demo
  let premises = [
    { id: 0, text: "It is raining", active: true },
    { id: 1, text: "Rain → Wet ground", active: true },
    { id: 2, text: "Wet ground → Slippery", active: false },
  ];
  let conclusions = [
    { id: 0, text: "Wet ground", derived: false, from: [0, 1] },
    { id: 1, text: "Slippery", derived: false, from: [0, 2] },
  ];
  let inferenceStep = 0;

  function stepInference() {
    inferenceStep++;
    if (inferenceStep === 1) {
      // Fact propagation: Rain + (Rain→Wet) = Wet ground
      if (premises[0].active && premises[1].active) {
        conclusions[0].derived = true;
        premises[2].active = true; // now we have Wet ground for next rule
      }
    } else if (inferenceStep === 2) {
      // Wet ground + (Wet→Slippery) = Slippery
      if (conclusions[0].derived && premises[2].active) {
        conclusions[1].derived = true;
      }
    }
    premises = [...premises];
    conclusions = [...conclusions];
  }

  function resetInference() {
    inferenceStep = 0;
    premises = premises.map((p, i) => ({ ...p, active: i <= 1 }));
    conclusions = conclusions.map((c) => ({ ...c, derived: false }));
  }

  function togglePremise(id: number) {
    premises = premises.map((p) =>
      p.id === id ? { ...p, active: !p.active } : p,
    );
    // Reset conclusions when premises change
    conclusions = conclusions.map((c) => ({ ...c, derived: false }));
    inferenceStep = 0;
  }

  // Synaptic matrix demo
  let matrixSize = 4;
  let synapticMatrix: number[][] = Array(matrixSize)
    .fill(0)
    .map(() => Array(matrixSize).fill(0));
  let queryVec = [0.8, 0.2, 0.0, 0.0];
  let valueVec = [0.0, 0.5, 0.9, 0.1];

  function hebbianUpdate() {
    for (let i = 0; i < matrixSize; i++) {
      for (let j = 0; j < matrixSize; j++) {
        synapticMatrix[i][j] += 0.3 * queryVec[i] * valueVec[j];
        synapticMatrix[i][j] = Math.min(1, synapticMatrix[i][j]);
      }
    }
    synapticMatrix = [...synapticMatrix];
  }

  function readFromMatrix() {
    // σ·x = output
    return synapticMatrix.map((row) =>
      row.reduce((sum, val, j) => sum + val * queryVec[j], 0),
    );
  }

  function resetMatrix() {
    synapticMatrix = Array(matrixSize)
      .fill(0)
      .map(() => Array(matrixSize).fill(0));
  }

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

  function switchTab(tab: typeof activeTab) {
    activeTab = tab;
    setTimeout(renderMath, 50);
  }

  // TeX strings with curly braces (Svelte can't parse these in HTML attributes)
  const TEX_ATTN =
    "\\text{Attn}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V";
  const TEX_BDH =
    "y = \\sigma \\cdot x \\quad\\text{where}\\quad \\sigma^{(t+1)} = \\sigma^{(t)} + \\alpha \\cdot x \\cdot y^T";
  const TEX_ALPHA_IJ = "\\alpha_{ij}";
  const TEX_SIGMA_IJ = "\\sigma(i,j)";
  const TEX_S_COMBINATOR =
    "S = \\lambda f.\\lambda g.\\lambda x. (f\\, x)(g\\, x)";
  const TEX_SKK_X = "SKK \\; x";
  const TEX_KX_KX = "= (K\\;x)(K\\;x)";
  const TEX_EQ_X = "= x";
  const TEX_SIGMA_NEW =
    "\\sigma_{\\text{new}} = \\sigma_{\\text{old}} + \\underbrace{x}_{\\text{query}} \\cdot \\underbrace{y^T}_{\\text{value}}";

  const TEX_TF_SOFTMAX = "\\text{Attn} = \\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d}}\\right)V";
  const TEX_BDH_LINEAR = "y = \\sigma \\cdot x";
</script>

<div class="attention-logic" bind:this={container}>
  <div class="section-header">
    <span class="section-icon"></span>
    <h2>Attention as Logical Implication</h2>
    <p class="subtitle">
      Linear attention implements logical operations via synaptic
      state
    </p>
  </div>

  {#if activeTab === "overview"}
    <div class="content-card">
      <h3>The Key Equivalence</h3>
      <p class="intro">
        Traditional attention computes relationships between queries and keys at
        inference time. BDH instead <strong
          >stores implications in synaptic state</strong
        > and reads them back — exactly like looking up logical rules.
      </p>

      <div class="eq-comparison">
        <div class="eq-card">
          <h4><span class="badge tf">Transformer</span> Standard Attention</h4>
          <div class="eq-box">
            <span class="math display" data-tex={TEX_ATTN}></span>
          </div>
          <p>Computes all pairwise relationships at each step. O(T²) cost.</p>
        </div>
        <div class="eq-divider">↔</div>
        <div class="eq-card">
          <h4><span class="badge bdh">BDH</span> Synaptic Read</h4>
          <div class="eq-box">
            <span class="math display" data-tex={TEX_BDH}></span>
          </div>
          <p>
            Reads from accumulated synaptic state. O(n²) cost, independent of T.
          </p>
        </div>
      </div>

      <div class="insight-box">
        <h4>The Logical Interpretation</h4>
        <div class="logic-grid">
          <div class="logic-card">
            <div class="logic-header">Attention Weight</div>
            <div class="logic-body">
              <span class="math" data-tex={TEX_ALPHA_IJ}></span> = "How much does
              token i attend to token j?"
            </div>
          </div>
          <div class="logic-card">
            <div class="logic-header">Logical Implication</div>
            <div class="logic-body">
              <span class="math" data-tex={TEX_SIGMA_IJ}></span> = "To what degree
              does fact i imply fact j?"
            </div>
          </div>
        </div>
        <p class="insight-text">
          When BDH reads from synaptic state (σ·x), it's performing weighted
          fact propagation: "Given active facts x with beliefs X(i) and
          implication strengths σ(i,j), derive new beliefs A(j) = Σ X(i)·σ(i,j)."
          When it updates state (σ += α·x·yᵀ), it's learning new implications.
        </p>
      </div>
    </div>
  {:else if activeTab === "combinator"}
    <div class="content-card">
      <h3>The S Combinator Connection</h3>
      <p class="intro">
        The paper shows that the Hebbian update rule is equivalent to the <strong
          >S combinator</strong
        >
        from combinatory logic — one of the fundamental building blocks of computation.
      </p>

      <div class="combinator-section">
        <div class="eq-box">
          <span class="math display" data-tex={TEX_S_COMBINATOR}></span>
        </div>
        <p>
          The S combinator takes three arguments and applies the first to the
          third, then applies the second to the third, then applies the first
          result to the second.
        </p>

        <div class="derivation">
          <h4>Derivation: SKK = I</h4>
          <div class="deriv-steps">
            <div class="step">
              <span class="step-num">1</span>
              <span class="math" data-tex={TEX_SKK_X}></span>
              <span class="step-note">Apply S to K, K, x</span>
            </div>
            <div class="step">
              <span class="step-num">2</span>
              <span class="math" data-tex={TEX_KX_KX}></span>
              <span class="step-note">S definition: (f x)(g x)</span>
            </div>
            <div class="step">
              <span class="step-num">3</span>
              <span class="math" data-tex={TEX_EQ_X}></span>
              <span class="step-note"
                >K returns its first arg, so K x = λy.x</span
              >
            </div>
          </div>
        </div>

        <div class="bdh-connection">
          <h4>In BDH Terms</h4>
          <div class="eq-box">
            <span class="math display" data-tex={TEX_SIGMA_NEW}></span>
          </div>
          <p>
            The outer product x·yᵀ acts as the S combinator: it takes input (x)
            and output (y) and composes them into a new implication stored in
            synaptic state. Repeated application builds up a complete logical
            reasoning chain.
          </p>
        </div>
      </div>
    </div>
  {:else}
    <div class="content-card">
      <h3>Interactive: Logical Implication Chain</h3>

      <!-- Implication Chain -->
      <div class="demo-section">
        <h4>Logical Implication Chain</h4>
        <p class="demo-desc">
          Click premises to toggle them, then step through inference:
        </p>

        <div class="premise-list">
          {#each premises as p}
            <button
              class="premise"
              class:active={p.active}
              on:click={() => togglePremise(p.id)}
            >
              <span class="check">{p.active ? "✓" : "○"}</span>
              {p.text}
            </button>
          {/each}
        </div>

        <div class="arrow-down">⬇ Inference</div>

        <div class="conclusion-list">
          {#each conclusions as c}
            <div class="conclusion" class:derived={c.derived}>
              <span class="derive-icon">{c.derived ? "✓" : "?"}</span>
              {c.text}
            </div>
          {/each}
        </div>

        <div class="demo-controls">
          <button class="btn primary" on:click={stepInference}>Step →</button>
          <button class="btn" on:click={resetInference}>↺ Reset</button>
          <span class="step-label">Step: {inferenceStep}/2</span>
        </div>
      </div>

      <!-- Synaptic Matrix Write/Read -->
      <div class="demo-section">
        <h4>Synaptic Write & Read</h4>
        <p class="demo-desc">
          Watch how writing x·yᵀ to the synaptic matrix creates implications
          that can be read back:
        </p>

        <div class="matrix-demo">
          <div class="vec-display">
            <div class="vec-label">Query x:</div>
            <div class="vec-cells">
              {#each queryVec as v, i}
                <div
                  class="vec-cell"
                  style="background:rgba(102,126,234,{0.1 + v * 0.8})"
                >
                  {v.toFixed(1)}
                </div>
              {/each}
            </div>
          </div>

          <div class="vec-display">
            <div class="vec-label">Value y:</div>
            <div class="vec-cells">
              {#each valueVec as v, i}
                <div
                  class="vec-cell"
                  style="background:rgba(78,205,196,{0.1 + v * 0.8})"
                >
                  {v.toFixed(1)}
                </div>
              {/each}
            </div>
          </div>

          <div class="matrix-display">
            <div class="vec-label">σ (synaptic state):</div>
            <div class="matrix-grid">
              {#each synapticMatrix as row, i}
                {#each row as val, j}
                  <div
                    class="m-cell"
                    style="background:rgba(118,75,162,{0.05 + val * 0.8})"
                  >
                    {val.toFixed(2)}
                  </div>
                {/each}
              {/each}
            </div>
          </div>

          <div class="vec-display">
            <div class="vec-label">σ·x (read):</div>
            <div class="vec-cells">
              {#each readFromMatrix() as v}
                <div
                  class="vec-cell"
                  style="background:rgba(255,107,107,{0.1 +
                    Math.min(v, 1) * 0.8})"
                >
                  {v.toFixed(2)}
                </div>
              {/each}
            </div>
          </div>
        </div>

        <div class="demo-controls">
          <button class="btn primary" on:click={hebbianUpdate}
            >Write x·yᵀ → σ</button
          >
          <button class="btn" on:click={resetMatrix}>↺ Reset σ</button>
        </div>
        <p class="tip">
          Click "Write" multiple times to accumulate knowledge. Watch the output
          converge toward the value vector.
        </p>
      </div>
    </div>
  {/if}


  <!-- ── GPU Implementation Callout ────────────────────────────────── -->
  <div class="gpu-impl">
    <div class="gpu-impl-header">
      <!-- <span class="gpu-badge">⚙️</span> -->
      <h3>BDH GPU Implementation</h3>
    </div>
    <p>
      BDH's linear attention is mathematically identical whether computed as a
      <strong>recurrent brain model</strong> or as <strong>parallel GPU attention</strong>.
      The secret is the <strong>associativity of matrix multiplication</strong>:
    </p>
    <div class="gpu-detail-grid">
      <div class="gpu-card theory-card">
        <h4> Recurrent Form (Brain)</h4>
        <div class="gpu-eq"><code>σ = X · X<sup>T</sup></code></div>
        <div class="gpu-eq"><code>y = σ · v</code></div>
        <p>Build the full N×N connectome first, then multiply. Sequential, O(N²) memory, O(1) per step. Perfect for real-time inference.</p>
      </div>
      <div class="gpu-card gpu-card-impl">
        <h4> Parallel Form (GPU)</h4>
        <div class="gpu-eq"><code>A = (X @ X<sup>T</sup>).tril()</code></div>
        <div class="gpu-eq"><code>Y = A @ V</code></div>
        <p>Compute pairwise scores as T×T matrix, aggregate values. Parallelizable, O(T²) compute. Perfect for batch training.</p>
      </div>
    </div>
    <div class="gpu-detail">
      <strong>The Mathematical Proof:</strong>
      <div style="display:flex; justify-content:center; align-items:center; gap:20px; margin:12px 0; padding:12px; background:rgba(0,0,0,0.15); border-radius:8px;">
        <span style="text-align:center; color:#a5b4fc;">
          <strong>Brain</strong><br/>
          <code style="font-size:1.1em; background:none;">(X X<sup>T</sup>) V</code>
        </span>
        <span style="font-size:1.5em; color:#64748b;"> = </span>
        <span style="text-align:center; color:#f59e0b;">
          <strong>GPU</strong><br/>
          <code style="font-size:1.1em; background:none;">X (X<sup>T</sup> V)</code>
        </span>
      </div>
      Just moving the parentheses changes the compute path from O(N²) to O(d²).
      The result is bit-for-bit identical — <code>bdh.py</code> uses the right-side form for training.
    </div>
    <p class="gpu-note">
      This duality means BDH is simultaneously a brain model (left form, for interpretability)
      and a high-performance LLM (right form, for training on 100K+ token sequences).
    </p>
  </div>
</div>

<style>
  .attention-logic {
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
    color: var(--text-muted);
    font-size: 0.9rem;
  }

  .tabs {
    display: flex;
    gap: 6px;
    margin-bottom: 20px;
    justify-content: center;
  }
  .tab {
    padding: 8px 20px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-card);
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 600;
    transition: all 0.15s;
  }
  .tab:hover {
    border-color: #667eea;
  }
  .tab.active {
    background: #667eea;
    color: white;
    border-color: #667eea;
  }

  .content-card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 24px;
    box-shadow: var(--shadow-card);
    border: 1px solid var(--border-color);
  }
  .content-card h3 {
    margin-bottom: 12px;
  }
  .intro {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 20px;
  }

  .eq-comparison {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 12px;
    align-items: start;
    margin-bottom: 20px;
  }
  @media (max-width: 700px) {
    .eq-comparison {
      grid-template-columns: 1fr;
    }
    .eq-divider {
      text-align: center;
    }
  }
  .eq-card {
    padding: 16px;
    background: var(--bg-surface);
    border-radius: 10px;
    border: 1px solid var(--border-color);
  }
  .eq-card h4 {
    font-size: 0.88rem;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .eq-card p {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 8px;
  }
  .eq-divider {
    font-size: 1.5rem;
    color: var(--text-muted);
    align-self: center;
    padding-top: 40px;
  }
  .badge {
    font-size: 0.6rem;
    padding: 2px 7px;
    border-radius: 8px;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .badge.tf {
    background: var(--bg-transformer);
    color: var(--text-tf);
  }
  .badge.bdh {
    background: var(--bg-logic);
    color: var(--text-logic);
  }

  .eq-box {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    margin-bottom: 10px;
  }

  .insight-box {
    padding: 18px;
    background: var(--bg-warning);
    border: 1px solid #fcd34d;
    border-radius: 10px;
  }
  .insight-box h4 {
    margin-bottom: 12px;
  }
  .logic-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 12px;
  }
  @media (max-width: 600px) {
    .logic-grid {
      grid-template-columns: 1fr;
    }
  }
  .logic-card {
    padding: 10px;
    background: var(--bg-card);
    border-radius: 8px;
    border: 1px solid #fcd34d;
  }
  .logic-header {
    font-size: 0.75rem;
    color: #92400e;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }
  .logic-body {
    font-size: 0.85rem;
    color: var(--text-secondary);
  }
  .insight-text {
    font-size: 0.85rem;
    color: #92400e;
    line-height: 1.55;
    margin: 0;
  }

  /* Combinator */
  .combinator-section {
  }
  .combinator-section p {
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.55;
    margin-bottom: 16px;
  }
  .derivation {
    padding: 16px;
    background: var(--bg-surface);
    border-radius: 10px;
    margin-bottom: 16px;
  }
  .derivation h4 {
    margin-bottom: 10px;
    font-size: 0.9rem;
  }
  .deriv-steps {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .step {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .step-num {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #667eea;
    color: white;
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
  }
  .step-note {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-left: 8px;
  }
  .bdh-connection {
    padding: 16px;
    background: var(--bg-insight);
    border-radius: 10px;
    border: 1px solid #e9d5ff;
  }
  .bdh-connection h4 {
    margin-bottom: 10px;
  }
  .bdh-connection p {
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.55;
    margin-top: 10px;
  }

  /* Interactive */
  .demo-section {
    margin-bottom: 24px;
  }
  .demo-section h4 {
    font-size: 1rem;
    margin-bottom: 6px;
  }
  .demo-desc {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-bottom: 12px;
  }
  .premise-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 12px;
  }
  .premise {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 8px;
    border: 1px solid var(--border-color);
    background: var(--bg-surface);
    cursor: pointer;
    font-size: 0.88rem;
    text-align: left;
  }
  .premise.active {
    background: var(--bg-logic);
    border-color: #667eea;
  }
  .check {
    font-size: 1rem;
  }
  .arrow-down {
    text-align: center;
    font-size: 1.2rem;
    color: var(--text-muted);
    margin: 8px 0;
  }
  .conclusion-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 14px;
  }
  .conclusion {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 8px;
    border: 1px dashed var(--border-light);
    background: var(--bg-surface);
    font-size: 0.88rem;
  }
  .conclusion.derived {
    background: var(--bg-success);
    border-color: #4ecdc4;
    border-style: solid;
  }
  .derive-icon {
    font-size: 0.9rem;
    color: #059669;
  }
  .demo-controls {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
  }
  .btn {
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid var(--border-light);
    border-radius: 8px;
    background: var(--bg-card);
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn:hover {
    border-color: #667eea;
    color: #667eea;
  }
  .btn.primary {
    background: #667eea;
    color: white;
    border-color: #667eea;
  }
  .step-label {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-left: auto;
  }

  /* Matrix demo */
  .matrix-demo {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 14px;
  }
  .vec-display {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .vec-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-secondary);
    min-width: 100px;
  }
  .vec-cells {
    display: flex;
    gap: 4px;
  }
  .vec-cell {
    width: 50px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: monospace;
    color: var(--text-heading);
  }
  .matrix-display {
  }
  .matrix-grid {
    display: grid;
    grid-template-columns: repeat(4, 50px);
    gap: 3px;
    margin-left: 110px;
  }
  .m-cell {
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 3px;
    font-size: 0.7rem;
    font-family: monospace;
    color: var(--text-heading);
  }
  .tip {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-style: italic;
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
    margin-bottom: 8px;
  }
  .contrast-eq {
    text-align: center;
    margin-bottom: 10px;
    font-size: 0.95rem;
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
