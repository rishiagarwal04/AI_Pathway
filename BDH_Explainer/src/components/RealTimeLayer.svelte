<script lang="ts">
  /**
   * RealTimeLayer — Premium single-layer visualizer.
   *
   * Horizontal pipeline flow (like Transformer Explainer):
   *   x_in → ENCODE → ATTENTION → VALUE ENCODE → GATE+DECODE → x_out
   *
   * Each stage is a card with a clear header and large, readable heatmap.
   */
  import MatrixHeatmap from './MatrixHeatmap.svelte';

  export let index  = 0;
  export let layer: any  = null;
  export let tokens: string[] = [];
  export let expanded = false;
  export let highlightToken = -1;

  $: highlightRow = highlightToken;
  $: T = tokens.length;
  $: deltaMax = layer ? Math.max(...layer.residual_delta, 0.001) : 1;

  // Which sub-section is open inside the expanded layer
  let openSection: 'all' | 'attn' | 'hebb' | 'gate' = 'all';

  // Column labels for heatmaps (neuron indices & dimension indices)
  // Derive all dimensions from actual backend data — no hardcoded sizes
  $: nCols = layer?.x_sparse?.[0]?.length ?? 0;
  $: dCols = layer?.y_kv?.[0]?.length ?? 0;
  $: neuronLabels = Array.from({length: nCols}, (_, i) => `n${i}`);
  $: dimLabels    = Array.from({length: dCols}, (_, i) => `d${i}`);
  $: sigmaCols    = layer?.sigma_before?.[0]?.length ?? 0;
  $: sigmaLabels  = Array.from({length: sigmaCols}, (_, i) => `n${i}`);
  $: outCols      = layer?.x_out?.[0]?.length ?? 0;
  $: outLabels    = Array.from({length: outCols}, (_, i) => `d${i}`);

  function sparsityColor(pct: number): string {
    if (pct < 20) return '#06b6d4';
    if (pct < 50) return '#8b5cf6';
    return '#ef4444';
  }

  function sparsityLabel(pct: number): string {
    if (pct < 20) return 'Very Sparse';
    if (pct < 50) return 'Moderate';
    return 'Dense';
  }
</script>

{#if layer}
  <!-- ── Layer block ──────────────────────────────────────────────────── -->
  <div class="layer-card" class:expanded>

    <!-- ═══════ HEADER (always visible) ═══════ -->
    <button
      class="layer-header"
      on:click={() => expanded = !expanded}
    >
      <div class="header-left">
        <div class="layer-num">L{index}</div>
        <div class="header-info">
          <span class="layer-name">BDH Layer {index + 1}</span>
          <div class="header-stats">
            <span class="stat-chip" style="--chip-color:{sparsityColor(layer.sparsity_x)}">
              x: {layer.sparsity_x}% active · {sparsityLabel(layer.sparsity_x)}
            </span>
            <span class="stat-chip" style="--chip-color:{sparsityColor(layer.sparsity_y)}">
              y: {layer.sparsity_y}% active · {sparsityLabel(layer.sparsity_y)}
            </span>
          </div>
        </div>
      </div>
      <div class="header-right">
        <!-- Residual delta sparkline -->
        <div class="sparkline" title="Per-token residual delta">
          {#each layer.residual_delta as d, i}
            <div
              class="spark-bar"
              class:active={i === highlightToken}
              style="height:{Math.round(20 * d / deltaMax) + 3}px"
              title="{tokens[i] ?? i}: Δ={d.toFixed(3)}"
            ></div>
          {/each}
        </div>
        <span class="expand-icon">{expanded ? '▾' : '▸'}</span>
      </div>
    </button>

    <!-- ═══════ EXPANDED DETAIL ═══════ -->
    {#if expanded}
      <div class="layer-body">

        <!-- Section tabs -->
        <div class="section-tabs">
          <button class="sec-tab" class:active={openSection === 'all'} on:click={() => openSection = 'all'}>All Stages</button>
          <button class="sec-tab" class:active={openSection === 'attn'} on:click={() => openSection = 'attn'}>Attention</button>
          <button class="sec-tab" class:active={openSection === 'hebb'} on:click={() => openSection = 'hebb'}>Hebbian</button>
          <button class="sec-tab" class:active={openSection === 'gate'} on:click={() => openSection = 'gate'}>Gate & Output</button>
        </div>

        <!-- ── STAGE 1: ENCODE ──────────────────────────────────────── -->
        {#if openSection === 'all' || openSection === 'attn'}
        <div class="stage">
          <div class="stage-head">
            <div class="stage-badge encode">ENC</div>
            <div class="stage-info">
              <span class="stage-name">Sparse Encoder</span>
              <span class="stage-math">x_sparse = ReLU(x · W_enc)</span>
            </div>
            <div class="stage-stat">
              {layer.sparsity_x}% active neurons
            </div>
          </div>
          <div class="stage-content">
            <MatrixHeatmap
              matrix={layer.x_sparse}
              rowLabels={tokens}
              colLabels={neuronLabels}
              title="x_sparse  [T × N]"
              width={Math.max(300, T * 10)}
              height={Math.max(60, T * 18)}
              scheme="viridis"
              {highlightRow}
            />
            <div class="stage-desc">
              Each row is a token. Bright cells = active neurons.
              BDH's sparse encoding forces most neurons to zero, creating <strong>interpretable</strong> representations.
            </div>
          </div>
        </div>
        {/if}

        <!-- ── STAGE 2: ATTENTION ───────────────────────────────────── -->
        {#if openSection === 'all' || openSection === 'attn'}
        <div class="flow-arrow">↓</div>
        <div class="stage">
          <div class="stage-head">
            <div class="stage-badge attention">ATN</div>
            <div class="stage-info">
              <span class="stage-name">Linear Causal Attention</span>
              <span class="stage-math">scores = (QR·QRᵀ).tril(-1); yKV = LN(scores · x)</span>
            </div>
          </div>
          <div class="stage-content dual">
            <div class="dual-panel">
              <div class="panel-label">Attention Scores [T × T]</div>
              <MatrixHeatmap
                matrix={layer.attn_scores}
                rowLabels={tokens}
                colLabels={tokens}
                title="Attention scores  [T × T]"
                width={Math.max(120, T * 30)}
                height={Math.max(120, T * 30)}
                scheme="blues"
                grid={true}
                {highlightRow}
              />
              <div class="panel-note">Strictly lower-triangular (d=−1): tokens attend only to previous positions, not themselves. QR = RoPE(x_sparse). Row = query, Col = key.</div>
            </div>
            <div class="dual-panel">
              <div class="panel-label">QR Norms per Token</div>
              <div class="bar-chart">
                {#each layer.attn_qk_norms as v, i}
                  <div class="bar-col" class:highlighted={i === highlightToken}>
                    <div
                      class="bar-fill"
                      style="height:{Math.round(50 * v / (Math.max(...layer.attn_qk_norms) || 1))}px"
                      title="{tokens[i] ?? i}: ‖QR‖ = {v.toFixed(3)}"
                    ></div>
                    <span class="bar-label">{tokens[i] ?? ''}</span>
                  </div>
                {/each}
              </div>
              <div class="panel-label" style="margin-top:16px">y_kv [T × D]</div>
              <MatrixHeatmap
                matrix={layer.y_kv}
                rowLabels={tokens}
                colLabels={dimLabels}
                title="y_kv  [T × D]"
                width={Math.max(260, 64 * 4)}
                height={Math.max(48, T * 16)}
                scheme="blues"
                {highlightRow}
              />
            </div>
          </div>
        </div>
        {/if}

        <!-- ── STAGE 3: HEBBIAN FAST WEIGHTS ────────────────────────── -->
        {#if openSection === 'all' || openSection === 'hebb'}
        <div class="flow-arrow">↓</div>
        <div class="stage">
          <div class="stage-head">
            <div class="stage-badge hebbian">HEB</div>
            <div class="stage-info">
              <span class="stage-name">Value Encoder + Hebbian Co-activation</span>
              <span class="stage-math">y_sparse = ReLU(yKV · W_enc_v)</span>
            </div>
            <div class="stage-stat">{layer.sparsity_y}% active (y)</div>
          </div>
          <div class="stage-content">
            <div class="panel-label">Value Encoder: y_sparse [T × N]</div>
            <MatrixHeatmap
              matrix={layer.y_sparse}
              rowLabels={tokens}
              colLabels={neuronLabels}
              title="y_sparse  [T × N]"
              width={Math.max(300, T * 10)}
              height={Math.max(60, T * 18)}
              scheme="reds"
              {highlightRow}
            />

            <!-- Co-activation analysis σ = Σ x_t⊗y_t (diagnostic, not a model variable) -->
            <div class="panel-label" style="margin-top:12px;font-size:0.7rem;color:var(--text-muted)">Co-activation matrix σ = Σ x_t⊗y_t &nbsp;(analytical — not used in forward pass)</div>
            <div class="sigma-flow">
              <div class="sigma-card">
                <div class="sigma-title">σ_old</div>
                <MatrixHeatmap
                  matrix={layer.sigma_before}
                  rowLabels={sigmaLabels}
                  colLabels={sigmaLabels}
                  title="σ_old  [N × N]"
                  width={140}
                  height={140}
                  scheme="viridis"
                />
              </div>
              <div class="sigma-op">
                <span class="op-symbol">+</span>
              </div>
              <div class="sigma-card delta">
                <div class="sigma-title">Δσ = Σ x⊗y</div>
                <MatrixHeatmap
                  matrix={layer.sigma_delta}
                  rowLabels={sigmaLabels}
                  colLabels={sigmaLabels}
                  title="Δσ  [N × N]"
                  width={140}
                  height={140}
                  scheme="reds"
                />
              </div>
              <div class="sigma-op">
                <span class="op-symbol">=</span>
              </div>
              <div class="sigma-card">
                <div class="sigma-title">σ_new</div>
                <MatrixHeatmap
                  matrix={layer.sigma_after}
                  rowLabels={sigmaLabels}
                  colLabels={sigmaLabels}
                  title="σ_new  [N × N]"
                  width={140}
                  height={140}
                  scheme="viridis"
                />
              </div>
            </div>
            <div class="stage-desc">
              y_sparse encodes the attention output into sparse neuron space via W_enc_v.
              The σ matrix above shows cumulative co-activation (Σ x⊗y) — an <strong>analytical view</strong>
              of which neuron pairs fire together, not a model variable.
            </div>
          </div>
        </div>
        {/if}

        <!-- ── STAGE 4: GATE + OUTPUT ───────────────────────────────── -->
        {#if openSection === 'all' || openSection === 'gate'}
        <div class="flow-arrow">↓</div>
        <div class="stage">
          <div class="stage-head">
            <div class="stage-badge gate">GATE</div>
            <div class="stage-info">
              <span class="stage-name">Hebbian Gate + Decode</span>
              <span class="stage-math">xy = x_sparse * y_sparse → x_out = LN(x + LN(xy · W_dec))</span>
            </div>
          </div>
          <div class="stage-content dual">
            <div class="dual-panel">
              <div class="panel-label">Hebbian Gate x·y [T × N]</div>
              <MatrixHeatmap
                matrix={layer.xy_gate}
                rowLabels={tokens}
                colLabels={neuronLabels}
                title="xy_gate  [T × N]"
                width={Math.max(300, T * 10)}
                height={Math.max(60, T * 18)}
                scheme="reds"
                {highlightRow}
              />
              <div class="panel-note">Co-activation: only active where BOTH x and y fire.</div>
            </div>
            <div class="dual-panel">
              <div class="panel-label">Residual Output x_out [T × D]</div>
              <MatrixHeatmap
                matrix={layer.x_out}
                rowLabels={tokens}
                colLabels={outLabels}
                title="x_out  [T × D]"
                width={Math.max(320, 64 * 5)}
                height={Math.max(48, T * 16)}
                scheme="diverging"
                {highlightRow}
              />
              <div class="delta-bars">
                <span class="delta-label">‖Δ residual‖ per token:</span>
                {#each layer.residual_delta as d, i}
                  <span class="delta-chip" class:hlt={i === highlightToken}>
                    {tokens[i] ?? i}: {d.toFixed(2)}
                  </span>
                {/each}
              </div>
            </div>
          </div>
        </div>
        {/if}

      </div><!-- /layer-body -->
    {/if}
  </div>

  <!-- Connector arrow between layers -->
  <div class="connector">
    <div class="conn-line"></div>
    <div class="conn-dot"></div>
  </div>
{/if}

<style>
  /* ═══════════════════════════════════════════════════════════════════════
   * LAYER CARD
   * ═══════════════════════════════════════════════════════════════════════ */
  .layer-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    overflow: hidden;
    transition: box-shadow 0.2s, border-color 0.2s;
  }
  .layer-card.expanded {
    border-color: #818cf8;
    box-shadow: 0 4px 24px rgba(102, 126, 234, 0.12);
  }

  /* ── HEADER ── */
  .layer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    cursor: pointer;
    background: var(--bg-surface);
    border: none;
    width: 100%;
    text-align: left;
    font: inherit;
    transition: background 0.15s;
  }
  .layer-header:hover { background: var(--bg-inset); }
  .header-left { display: flex; align-items: center; gap: 14px; }
  .header-right { display: flex; align-items: center; gap: 16px; }

  .layer-num {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    font-size: 0.8rem;
    font-weight: 800;
    padding: 4px 10px;
    border-radius: 8px;
    letter-spacing: 0.5px;
    flex-shrink: 0;
  }
  .header-info { display: flex; flex-direction: column; gap: 4px; }
  .layer-name { font-weight: 700; font-size: 0.92rem; color: var(--text-heading); }
  .header-stats { display: flex; gap: 8px; flex-wrap: wrap; }
  .stat-chip {
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    background: color-mix(in srgb, var(--chip-color) 12%, transparent);
    color: var(--chip-color);
    border: 1px solid color-mix(in srgb, var(--chip-color) 25%, transparent);
  }

  /* Sparkline */
  .sparkline {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 26px;
  }
  .spark-bar {
    width: 6px;
    min-height: 3px;
    background: #818cf8;
    border-radius: 2px 2px 0 0;
    opacity: 0.6;
    transition: all 0.2s;
  }
  .spark-bar.active { background: #f59e0b; opacity: 1; }

  .expand-icon {
    font-size: 1.1rem;
    color: var(--text-muted);
    flex-shrink: 0;
  }

  /* ── BODY ── */
  .layer-body {
    padding: 0 20px 24px;
    border-top: 1px solid var(--border-color);
  }

  /* Section tabs */
  .section-tabs {
    display: flex;
    gap: 4px;
    padding: 14px 0 8px;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 16px;
    overflow-x: auto;
  }
  .sec-tab {
    padding: 6px 14px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-card);
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted);
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s;
  }
  .sec-tab:hover { border-color: #818cf8; color: var(--accent); }
  .sec-tab.active { background: var(--accent); color: var(--text-on-accent); border-color: var(--accent); }

  /* ── STAGE CARDS ── */
  .stage {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 4px;
  }
  .stage-head {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border-light);
  }
  .stage-badge {
    font-size: 0.68rem;
    font-weight: 800;
    padding: 3px 8px;
    border-radius: 6px;
    letter-spacing: 0.5px;
    flex-shrink: 0;
  }
  .stage-badge.encode   { background: #ede9fe; color: #7c3aed; }
  .stage-badge.attention { background: #dbeafe; color: #1d4ed8; }
  .stage-badge.hebbian  { background: #fee2e2; color: #b91c1c; }
  .stage-badge.gate     { background: #fef3c7; color: #92400e; }

  .stage-info { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
  .stage-name { font-weight: 700; font-size: 0.85rem; color: var(--text-heading); }
  .stage-math {
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
  }
  .stage-stat {
    margin-left: auto;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    white-space: nowrap;
  }

  .stage-content {
    padding: 16px;
  }
  .stage-content.dual {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  @media (max-width: 800px) {
    .stage-content.dual { grid-template-columns: 1fr; }
  }

  .stage-desc {
    margin-top: 10px;
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.55;
    max-width: 600px;
  }

  /* ── Panel sections ── */
  .dual-panel {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .panel-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .panel-note {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-style: italic;
  }

  /* ── Bar chart ── */
  .bar-chart {
    display: flex;
    align-items: flex-end;
    gap: 5px;
    height: 65px;
    padding: 4px 0;
  }
  .bar-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
  }
  .bar-fill {
    width: 22px;
    min-height: 4px;
    background: #60a5fa;
    border-radius: 3px 3px 0 0;
    transition: height 0.3s, background 0.2s;
  }
  .bar-col.highlighted .bar-fill { background: #f59e0b; }
  .bar-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    font-family: monospace;
    max-width: 22px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* ── Sigma flow ── */
  .sigma-flow {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 16px;
    flex-wrap: wrap;
    justify-content: center;
  }
  .sigma-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    padding: 10px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
  }
  .sigma-card.delta {
    border-color: #fca5a5;
    background: var(--bg-danger);
  }
  .sigma-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
  }
  .sigma-op {
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .op-symbol {
    font-size: 1.6rem;
    font-weight: 300;
    color: var(--text-muted);
  }

  /* ── Delta bars ── */
  .delta-bars {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 8px;
  }
  .delta-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    font-weight: 600;
  }
  .delta-chip {
    font-size: 0.68rem;
    font-family: monospace;
    padding: 2px 8px;
    background: var(--bg-inset);
    border-radius: 4px;
    color: var(--text-secondary);
  }
  .delta-chip.hlt { background: #fef3c7; color: #92400e; font-weight: 700; }

  /* ── Flow arrow ── */
  .flow-arrow {
    text-align: center;
    font-size: 1.1rem;
    color: #818cf8;
    line-height: 1;
    padding: 6px 0;
  }

  /* ── Connector (between layers) ── */
  .connector {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    padding: 3px 0;
  }
  .conn-line {
    width: 2px;
    height: 14px;
    background: linear-gradient(to bottom, #818cf8, #a78bfa);
  }
  .conn-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #a78bfa;
  }
</style>
