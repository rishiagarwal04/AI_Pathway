<script lang="ts">
  /**
   * AblationPlayground — Real-time neuron ablation experiments.
   *
   * Key improvements:
   * - Token-based ablation: click a token to ablate all neurons that fire for it
   * - Impact analysis: diff visualization showing probability changes
   * - Key neuron highlighting: top-5 contributors get golden glow
   * - Sorted neuron grid: most active neurons appear first
   */
  import { onMount } from "svelte";
  import { apiUrl } from '../api';

  // ── State ──────────────────────────────────────────────────────────────
  let prompt = "2+2=";
  let loading = false;
  let error: string | null = null;

  let baseline: any = null;
  let ablated: any = null;
  let modelInfo: any = null;

  let ablationMap: Map<number, Set<number>> = new Map();
  let selectedLayer = 0;
  let temperature = 1.0; // Default: raw/unsharpened
  let topK = 0; // Default: all tokens (no top-k filtering)
  $: nLayers = modelInfo?.n_layer ?? 0;
  $: neuronDim = modelInfo?.neuron_dim ?? 0;
  // Show up to 128 neurons in the grid (backend sends top-128 by activation)
  $: displayNeurons = 128;
  // Total neurons in the model (for display)
  $: totalModelNeurons =
    baseline?.layers?.[selectedLayer]?.total_neurons ?? neuronDim;

  // ── API ────────────────────────────────────────────────────────────────
  async function fetchModelInfo() {
    try {
      const r = await fetch(apiUrl("/api/model/info"));
      if (r.ok) modelInfo = await r.json();
    } catch {}
  }

  async function runBaseline() {
    if (loading) return;
    loading = true;
    error = null;
    baseline = null;
    ablated = null;
    ablationMap = new Map();
    try {
      const res = await fetch(apiUrl("/api/neuron/activations"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, temperature, top_k: topK }),
      });
      if (!res.ok) throw new Error(`Server ${res.status}`);
      baseline = await res.json();
    } catch (e: any) {
      error = e.message ?? "Request failed";
    } finally {
      loading = false;
    }
  }

  async function runAblation() {
    if (loading || !baseline) return;
    loading = true;
    error = null;
    ablated = null;
    try {
      const ablations = Array.from(ablationMap.entries()).map(
        ([layer, neurons]) => ({
          layer,
          neurons: Array.from(neurons),
        }),
      );
      const res = await fetch(apiUrl("/api/ablate"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, ablations, temperature, top_k: topK }),
      });
      if (!res.ok) throw new Error(`Server ${res.status}`);
      ablated = await res.json();
    } catch (e: any) {
      error = e.message ?? "Request failed";
    } finally {
      loading = false;
    }
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter") runBaseline();
  }

  // ── Neuron toggle ─────────────────────────────────────────────────────
  function toggleNeuron(layer: number, neuronId: number) {
    const updated = new Map(ablationMap);
    if (!updated.has(layer)) updated.set(layer, new Set());
    const s = updated.get(layer)!;
    if (s.has(neuronId)) s.delete(neuronId);
    else s.add(neuronId);
    if (s.size === 0) updated.delete(layer);
    ablationMap = updated;
  }

  function isDisabled(layer: number, neuronId: number): boolean {
    return ablationMap.get(layer)?.has(neuronId) ?? false;
  }

  function disableTopK(k: number) {
    if (!baseline) return;
    const layer = baseline.layers[selectedLayer];
    if (!layer) return;
    const neurons = Object.values(layer.neurons as Record<string, any>).sort(
      (a: any, b: any) => b.x_activation - a.x_activation,
    );
    const updated = new Map(ablationMap);
    const s = updated.get(selectedLayer) ?? new Set<number>();
    for (let i = 0; i < Math.min(k, neurons.length); i++) s.add(neurons[i].id);
    updated.set(selectedLayer, s);
    ablationMap = updated;
  }

  function disableLayer(layer: number) {
    if (!baseline) return;
    const layerData = baseline.layers[layer];
    if (!layerData) return;
    const updated = new Map(ablationMap);
    const s = new Set<number>();
    // Extract neuron IDs from the neurons dict (keyed by string ID)
    const ids: number[] = Object.keys(layerData.neurons ?? {}).map(Number);
    for (const id of ids) s.add(id);
    updated.set(layer, s);
    ablationMap = updated;
  }

  function clearAll() {
    ablationMap = new Map();
    ablated = null;
  }

  // Disable top-K most active neurons across ALL layers
  function disableTopKAllLayers(k: number) {
    if (!baseline) return;
    const updated = new Map(ablationMap);
    for (let L = 0; L < nLayers; L++) {
      const layer = baseline.layers[L];
      if (!layer) continue;
      const neurons = Object.values(layer.neurons as Record<string, any>).sort(
        (a: any, b: any) => b.x_activation - a.x_activation,
      );
      const s = updated.get(L) ?? new Set<number>();
      for (let i = 0; i < Math.min(k, neurons.length); i++)
        s.add(neurons[i].id);
      updated.set(L, s);
    }
    ablationMap = updated;
  }

  // Disable all returned neurons across ALL layers
  function disableAllLayers() {
    if (!baseline) return;
    const updated = new Map(ablationMap);
    for (let L = 0; L < nLayers; L++) {
      const layer = baseline.layers[L];
      if (!layer) continue;
      const ids: number[] = Object.keys(layer.neurons ?? {}).map(Number);
      updated.set(L, new Set(ids));
    }
    ablationMap = updated;
  }

  // ── Token-based ablation ──────────────────────────────────────────────
  // Reconstruct per-token data from the neurons dict
  // Backend sends neurons as {"id": {per_token: [...]}} — we rebuild the lookup
  function disableNeuronsForToken(tokIdx: number) {
    if (!baseline) return;
    const layer = baseline.layers[selectedLayer];
    if (!layer?.neurons) return;
    const neuronsDict = layer.neurons as Record<string, any>;
    const updated = new Map(ablationMap);
    const s = updated.get(selectedLayer) ?? new Set<number>();
    for (const [idStr, neuron] of Object.entries(neuronsDict)) {
      const perTok: number[] = neuron.per_token ?? [];
      if (perTok[tokIdx] != null && perTok[tokIdx] > 0.005) {
        s.add(Number(idStr));
      }
    }
    updated.set(selectedLayer, s);
    ablationMap = updated;
  }

  // Per-token active neuron counts (derived from neurons dict)
  $: tokenNeuronCounts = (() => {
    if (!baseline) return [];
    const layer = baseline.layers[selectedLayer];
    if (!layer?.neurons) return [];
    const neuronsArr = Object.values(layer.neurons as Record<string, any>);
    if (!neuronsArr.length || !neuronsArr[0]?.per_token) return [];
    const T = neuronsArr[0].per_token.length;
    const counts: number[] = [];
    for (let t = 0; t < T; t++) {
      let c = 0;
      for (const n of neuronsArr) {
        if ((n.per_token?.[t] ?? 0) > 0.005) c++;
      }
      counts.push(c);
    }
    return counts;
  })();

  $: hasPerToken = tokenNeuronCounts.length > 0;

  // ── Derived data ──────────────────────────────────────────────────────
  $: totalAblated = Array.from(ablationMap.values()).reduce(
    (sum, s) => sum + s.size,
    0,
  );

  // Key neurons: top 5 most active in current layer
  $: keyNeurons = (() => {
    if (!baseline) return new Set<number>();
    const layer = baseline.layers[selectedLayer];
    if (!layer) return new Set<number>();
    const sorted = Object.values(layer.neurons as Record<string, any>).sort(
      (a: any, b: any) => b.x_activation - a.x_activation,
    );
    return new Set(sorted.slice(0, 5).map((n: any) => n.id));
  })();

  // Sorted neurons for grid (most active first)
  $: sortedGridNeurons = (() => {
    if (!baseline) return [];
    const layer = baseline.layers[selectedLayer];
    if (!layer) return [];
    return Object.values(layer.neurons as Record<string, any>)
      .sort((a: any, b: any) => b.x_activation - a.x_activation)
      .slice(0, displayNeurons);
  })();

  // Neuron color based on activation
  function neuronColor(neuron: any, layer: number): string {
    if (isDisabled(layer, neuron.id)) return "#ef4444";
    if (keyNeurons.has(neuron.id) && neuron.x_activation > 0.001)
      return "#f59e0b";
    const act = neuron.x_activation;
    if (act < 0.001) return "var(--neuron-inactive)";
    const t = Math.min(act / 0.1, 1);
    const r = Math.round(102 + (59 - 102) * t);
    const g = Math.round(126 + (130 - 126) * t);
    const b = Math.round(234 + (246 - 234) * t);
    return `rgb(${r},${g},${b})`;
  }

  // ── Diff computation ──────────────────────────────────────────────────
  $: baselineTop = baseline?.prob_bars?.[0];
  $: ablatedTop = ablated?.prob_bars?.[0];
  $: probShift =
    baselineTop && ablatedTop
      ? Math.abs(baselineTop.prob - ablatedTop.prob)
      : 0;

  $: diffData = (() => {
    if (!baseline || !ablated) return [];
    const bBars = baseline.prob_bars as any[];
    const aBars = ablated.prob_bars as any[];
    const tokenMap = new Map<
      string,
      { token: string; baseProb: number; ablProb: number; delta: number }
    >();
    for (const b of bBars) {
      tokenMap.set(b.token, {
        token: b.token,
        baseProb: b.prob,
        ablProb: 0,
        delta: 0,
      });
    }
    for (const a of aBars) {
      const existing = tokenMap.get(a.token);
      if (existing) {
        existing.ablProb = a.prob;
        existing.delta = a.prob - existing.baseProb;
      } else {
        tokenMap.set(a.token, {
          token: a.token,
          baseProb: 0,
          ablProb: a.prob,
          delta: a.prob,
        });
      }
    }
    // Fill in missing ablated entries
    for (const [, v] of tokenMap) {
      if (v.delta === 0 && v.ablProb === 0) v.delta = -v.baseProb;
    }
    return Array.from(tokenMap.values())
      .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
      .slice(0, 10);
  })();

  $: maxDiffProb = (() => {
    if (diffData.length === 0) return 0.5;
    return Math.max(
      ...diffData.map((d) => Math.max(d.baseProb, d.ablProb)),
      0.01,
    );
  })();

  const examples = ["2+2=", "Hello", "The cat sat", "Why is"];

  onMount(() => {
    fetchModelInfo();
  });
</script>

<div class="playground">
  <!-- Model Upload -->

  <div class="pg-header">
    <h2>Ablation Playground</h2>
    <p class="subtitle">
      Disable specific neurons in the real BDH model and observe how outputs
      change
    </p>
  </div>

  {#if error}
    <div class="error-bar">{error} — backend may be waking up, please retry in 30s</div>
  {/if}

  <!-- Prompt + Run -->
  <div class="prompt-section">
    <div class="prompt-wrap">
      <input
        bind:value={prompt}
        on:keydown={onKeyDown}
        placeholder="Enter a prompt..."
        disabled={loading}
      />
      <button class="run-btn" on:click={runBaseline} disabled={loading}>
        Run Baseline
      </button>
    </div>
    <div class="examples">
      {#each examples as ex}
        <button
          class="ex-chip"
          on:click={() => {
            prompt = ex;
            runBaseline();
          }}>{ex}</button
        >
      {/each}
    </div>
    <div class="param-controls">
      <div class="param-group">
        <label class="param-label"
          >Temperature: <strong>{temperature.toFixed(2)}</strong></label
        >
        <input
          type="range"
          min="0.1"
          max="2.0"
          step="0.05"
          bind:value={temperature}
        />
      </div>
      <div class="param-group">
        <label class="param-label">
          Top-K: <strong>{topK === 0 ? "All" : topK}</strong>
        </label>
        <input type="range" min="0" max="50" step="1" bind:value={topK} />
      </div>
    </div>
  </div>

  {#if loading && !baseline}
    <div class="loading">
      <div class="spinner"></div>
       Running BDH baseline...
    </div>
  {/if}

  {#if baseline}
    <!-- Baseline output -->
    <div class="baseline-card">
      <div class="card-head">
        <span class="card-badge base">BASELINE</span>
        <span class="card-title">Normal Forward Pass</span>
      </div>
      <div class="card-body">
        <div class="out-probs">
          {#each baseline.prob_bars.slice(0, 8) as bar, i}
            <div class="prob-item" class:top={i === 0}>
              <span class="prob-tok">{bar.token}</span>
              <div class="prob-bar-track">
                <div
                  class="prob-bar-fill base-fill"
                  style="width:{Math.round(bar.prob * 100)}%"
                ></div>
              </div>
              <span class="prob-val">{(bar.prob * 100).toFixed(1)}%</span>
            </div>
          {/each}
        </div>
      </div>
    </div>

    <!-- Token-based ablation -->
    {#if hasPerToken}
      <div class="token-ablation">
        <div class="ta-header">
          <h4>Concept-Based Ablation</h4>
          <span class="ta-hint"
            >Click a token to ablate all neurons that fire for it in Layer {selectedLayer}</span
          >
        </div>
        <div class="ta-chips">
          {#each baseline.tokens as tok, i}
            <button class="ta-chip" on:click={() => disableNeuronsForToken(i)}>
              <span class="ta-tok">{tok}</span>
              <span class="ta-count">{tokenNeuronCounts[i] ?? 0} neurons</span>
            </button>
          {/each}
        </div>
      </div>
    {/if}

    <!-- Layer selector -->
    <div class="layer-controls">
      <span class="lc-label">Select layer to ablate:</span>
      {#each Array(nLayers) as _, i}
        <button
          class="layer-btn"
          class:active={selectedLayer === i}
          on:click={() => (selectedLayer = i)}
        >
          Layer {i}
          {#if ablationMap.has(i)}
            <span class="abl-count">{ablationMap.get(i)?.size}</span>
          {/if}
        </button>
      {/each}
    </div>

    <!-- Neuron grid -->
    {#if baseline.layers[selectedLayer]}
      {@const layerData = baseline.layers[selectedLayer]}
      <div class="neuron-section">
        <div class="neuron-header">
          <h4>
            Layer {selectedLayer} Neurons — {layerData.sparsity_pct}% active ({layerData
              .neurons.length} of {layerData.total_neurons ?? "?"} shown, sorted
            by activation)
          </h4>
          <div class="neuron-actions">
            <button class="act-btn" on:click={() => disableTopK(10)}
              >Top 10</button
            >
            <button class="act-btn" on:click={() => disableTopK(50)}
              >Top 50</button
            >
            <button class="act-btn" on:click={() => disableTopK(100)}
              >Top 100</button
            >
            <button
              class="act-btn danger"
              on:click={() => disableLayer(selectedLayer)}>All in Layer</button
            >
            <button
              class="act-btn danger"
              on:click={() => disableTopKAllLayers(100)}
              >Top 100 × All Layers</button
            >
            <button
              class="act-btn danger"
              on:click={() => disableTopKAllLayers(500)}
              >Top 500 × All Layers</button
            >
            <button class="act-btn danger" on:click={disableAllLayers}
              >All × All Layers</button
            >
            <button class="act-btn" on:click={clearAll}>Clear</button>
          </div>
        </div>

        <div class="neuron-grid">
          {#each sortedGridNeurons as neuron}
            <button
              class="neuron-cell"
              class:key-neuron={keyNeurons.has(neuron.id) &&
                neuron.x_activation > 0.001}
              class:disabled={isDisabled(selectedLayer, neuron.id)}
              class:active={neuron.x_activation > 0.001}
              style="background:{neuronColor(neuron, selectedLayer)}"
              title="Neuron {neuron.id}: act={neuron.x_activation.toFixed(
                4,
              )} fire={neuron.fire_rate.toFixed(2)}{isDisabled(
                selectedLayer,
                neuron.id,
              )
                ? ' [DISABLED]'
                : ''}{keyNeurons.has(neuron.id) ? ' [KEY]' : ''}"
              on:click={() => toggleNeuron(selectedLayer, neuron.id)}
            >
              {neuron.id}
            </button>
          {/each}
        </div>

        <div class="grid-legend">
          <span class="gl-item"
            ><span
              class="gl-dot"
              style="background:#f59e0b; box-shadow:0 0 6px rgba(245,158,11,0.6)"
            ></span> Key neuron (top 5)</span
          >
          <span class="gl-item"
            ><span class="gl-dot" style="background:var(--accent)"></span> Active</span
          >
          <span class="gl-item"
            ><span class="gl-dot" style="background:var(--neuron-inactive)"
            ></span> Inactive</span
          >
          <span class="gl-item"
            ><span class="gl-dot" style="background:#ef4444"></span> Disabled</span
          >
          <span class="gl-help"
            >Click neurons to toggle | Sorted by activation</span
          >
        </div>
      </div>
    {/if}

    <!-- Ablation summary + Run -->
    <div class="ablation-bar">
      <div class="abl-summary">
        <span class="abl-count-total">{totalAblated}</span> neurons disabled
        {#if ablationMap.size > 0}
          across {ablationMap.size} layer{ablationMap.size > 1 ? "s" : ""}
          ({#each Array.from(ablationMap.entries()) as [layer, neurons], i}
            {#if i > 0},
            {/if}L{layer}:{neurons.size}
          {/each})
        {/if}
      </div>
      <button
        class="ablate-btn"
        on:click={runAblation}
        disabled={loading || totalAblated === 0}
      >
        {loading ? "Running..." : "Run Ablated Forward Pass"}
      </button>
    </div>

    <!-- Impact Analysis (Diff Report) -->
    {#if ablated}
      <div class="damage-report">
        <div class="dr-header">
          <h4>Impact Analysis</h4>
          <span
            class="dr-badge"
            class:severe={probShift > 0.3}
            class:moderate={probShift > 0.1 && probShift <= 0.3}
            class:minimal={probShift <= 0.1}
          >
            {#if probShift > 0.3}SEVERE{:else if probShift > 0.1}MODERATE{:else}MINIMAL{/if}
          </span>
        </div>

        <!-- Diff table -->
        <div class="diff-table">
          <div class="diff-header-row">
            <span class="dh-tok">Token</span>
            <span class="dh-base">Baseline</span>
            <span class="dh-abl">Ablated</span>
            <span class="dh-delta">Change</span>
          </div>
          {#each diffData as d}
            <div class="diff-row">
              <span class="diff-tok">{d.token}</span>
              <div class="diff-bar-cell">
                <div class="diff-bar-track">
                  <div
                    class="diff-bar base-bar"
                    style="width:{Math.round(
                      (100 * d.baseProb) / maxDiffProb,
                    )}%"
                  ></div>
                </div>
                <span class="diff-pct">{(d.baseProb * 100).toFixed(1)}%</span>
              </div>
              <div class="diff-bar-cell">
                <div class="diff-bar-track">
                  <div
                    class="diff-bar abl-bar"
                    style="width:{Math.round((100 * d.ablProb) / maxDiffProb)}%"
                  ></div>
                </div>
                <span class="diff-pct">{(d.ablProb * 100).toFixed(1)}%</span>
              </div>
              <span
                class="diff-delta"
                class:up={d.delta > 0.005}
                class:down={d.delta < -0.005}
              >
                {#if d.delta > 0.005}
                  &#9650; +{(d.delta * 100).toFixed(1)}%
                {:else if d.delta < -0.005}
                  &#9660; {(d.delta * 100).toFixed(1)}%
                {:else}
                  = 0.0%
                {/if}
              </span>
            </div>
          {/each}
        </div>

        <!-- Summary -->
        <div class="dr-summary">
          {#if probShift > 0.3}
            <span class="dmg-icon severe-icon">!!</span>
            <span
              ><strong>Severe disruption.</strong> The disabled neurons are
              <em>causally critical</em>
              for this prediction. Top prediction shifted by {(
                probShift * 100
              ).toFixed(1)}%.</span
            >
          {:else if probShift > 0.1}
            <span class="dmg-icon moderate-icon">!</span>
            <span
              ><strong>Moderate impact.</strong> These neurons contribute meaningfully
              but aren't the sole drivers. Try targeting key neurons (gold) for bigger
              effects.</span
            >
          {:else}
            <span class="dmg-icon minimal-icon">~</span>
            <span
              ><strong>Minimal impact.</strong> These neurons are not critical for
              this specific prediction. Try disabling the gold-highlighted key neurons
              or more of them.</span
            >
          {/if}
        </div>
      </div>

      <!-- Generated text comparison -->
      {#if ablated.continuation}
        <div class="gen-comparison">
          <div class="gen-card">
            <div class="gen-label">Baseline output:</div>
            <div class="gen-text">
              <span class="gen-prompt">{baseline.prompt ?? prompt}</span>
            </div>
          </div>
          <div class="gen-card abl">
            <div class="gen-label">Ablated output:</div>
            <div class="gen-text">
              <span class="gen-prompt">{ablated.prompt ?? prompt}</span><span
                class="gen-cont">{ablated.continuation ?? ""}</span
              >
            </div>
          </div>
        </div>
      {/if}
    {/if}
  {:else if !loading}
    <div class="empty-state">
      <p>Enter a prompt and click <strong>Run Baseline</strong> to start</p>
      <p class="empty-sub">
        Then disable neurons and re-run to see the effect on output
      </p>
    </div>
  {/if}
</div>

<style>
  .playground {
    max-width: 1000px;
    margin: 0 auto;
  }
  .pg-header {
    text-align: center;
    margin-bottom: 20px;
  }
  .pg-header h2 {
    font-size: 1.4rem;
    margin: 0 0 4px;
    color: var(--text-heading);
  }
  .subtitle {
    color: var(--text-secondary);
    font-size: 0.88rem;
  }

  .error-bar {
    background: var(--bg-danger);
    color: #dc2626;
    border: 1px solid var(--border-danger);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.82rem;
    margin-bottom: 12px;
  }

  /* ── Prompt ── */
  .prompt-section {
    margin-bottom: 16px;
  }
  .prompt-wrap {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
  }
  .prompt-wrap input {
    flex: 1;
    padding: 10px 14px;
    border: 2px solid var(--border-color);
    border-radius: 8px;
    font-size: 0.9rem;
    font-family: "JetBrains Mono", monospace;
    outline: none;
    background: var(--bg-card);
    color: var(--text-heading);
    transition: border-color 0.15s;
  }
  .prompt-wrap input:focus {
    border-color: var(--accent);
  }
  .run-btn {
    padding: 10px 20px;
    background: linear-gradient(135deg, var(--accent), #764ba2);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.85rem;
    cursor: pointer;
    white-space: nowrap;
  }
  .run-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .examples {
    display: flex;
    gap: 6px;
  }
  .ex-chip {
    padding: 4px 12px;
    border: 1px solid var(--border-color);
    border-radius: 20px;
    background: var(--bg-card);
    font-size: 0.75rem;
    cursor: pointer;
    color: var(--text-secondary);
    transition: all 0.15s;
  }
  .ex-chip:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  .param-controls {
    display: flex;
    gap: 20px;
    margin-top: 8px;
    flex-wrap: wrap;
  }
  .param-group {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .param-label {
    font-size: 0.78rem;
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .param-group input[type="range"] {
    width: 120px;
    accent-color: var(--accent);
  }

  .loading {
    text-align: center;
    padding: 40px;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  .spinner {
    width: 18px;
    height: 18px;
    border: 2px solid var(--border-color);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }

  /* ── Baseline card ── */
  .baseline-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 16px;
  }
  .card-head {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-color);
  }
  .card-badge {
    font-size: 0.68rem;
    font-weight: 800;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.5px;
  }
  .card-badge.base {
    background: rgba(37, 99, 235, 0.15);
    color: #2563eb;
  }
  .card-title {
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--text-heading);
  }
  .card-body {
    padding: 12px 14px;
  }
  .out-probs {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .prob-item {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .prob-tok {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.82rem;
    width: 28px;
    text-align: center;
    background: var(--bg-inset);
    border-radius: 4px;
    padding: 1px 3px;
    flex-shrink: 0;
    color: var(--text-heading);
  }
  .prob-bar-track {
    flex: 1;
    height: 14px;
    background: var(--bg-inset);
    border-radius: 4px;
    overflow: hidden;
  }
  .prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
  }
  .base-fill {
    background: linear-gradient(90deg, var(--accent), #818cf8);
  }
  .prob-item.top .prob-tok {
    background: var(--accent);
    color: white;
  }
  .prob-val {
    font-family: monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    width: 38px;
    text-align: right;
    flex-shrink: 0;
  }

  /* ── Token-based ablation ── */
  .token-ablation {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 16px;
  }
  .ta-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 10px;
    flex-wrap: wrap;
  }
  .ta-header h4 {
    margin: 0;
    font-size: 0.88rem;
    color: var(--text-heading);
  }
  .ta-hint {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-style: italic;
  }
  .ta-chips {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .ta-chip {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    padding: 8px 14px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-surface);
    cursor: pointer;
    transition: all 0.15s;
  }
  .ta-chip:hover {
    border-color: #ef4444;
    background: rgba(239, 68, 68, 0.06);
  }
  .ta-tok {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--text-heading);
  }
  .ta-count {
    font-size: 0.62rem;
    color: var(--text-muted);
  }

  /* ── Layer controls ── */
  .layer-controls {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 14px;
    flex-wrap: wrap;
  }
  .lc-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-secondary);
  }
  .layer-btn {
    padding: 6px 14px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-card);
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s;
    color: var(--text-primary);
  }
  .layer-btn.active {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  .abl-count {
    font-size: 0.65rem;
    font-weight: 800;
    background: #ef4444;
    color: white;
    border-radius: 50%;
    width: 16px;
    height: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* ── Neuron grid ── */
  .neuron-section {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 14px;
  }
  .neuron-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    flex-wrap: wrap;
    gap: 8px;
  }
  .neuron-header h4 {
    font-size: 0.88rem;
    margin: 0;
    color: var(--text-heading);
  }
  .neuron-actions {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }
  .act-btn {
    padding: 4px 10px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-card);
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    color: var(--text-secondary);
  }
  .act-btn:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .act-btn.danger:hover {
    border-color: #ef4444;
    color: #ef4444;
  }

  .neuron-grid {
    display: grid;
    grid-template-columns: repeat(16, 1fr);
    gap: 3px;
  }
  .neuron-cell {
    aspect-ratio: 1;
    border-radius: 3px;
    border: 2px solid transparent;
    font-size: 0.55rem;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.4);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
    font-family: monospace;
  }
  .neuron-cell:hover {
    transform: scale(1.3);
    z-index: 1;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  }
  .neuron-cell.disabled {
    color: white;
    font-weight: 800;
  }
  .neuron-cell.active {
    color: white;
  }
  .neuron-cell.key-neuron {
    border-color: #f59e0b;
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
  }

  .grid-legend {
    display: flex;
    gap: 14px;
    margin-top: 10px;
    font-size: 0.72rem;
    color: var(--text-secondary);
    flex-wrap: wrap;
    align-items: center;
  }
  .gl-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .gl-dot {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .gl-help {
    margin-left: auto;
    font-style: italic;
    color: var(--text-muted);
    font-size: 0.68rem;
  }

  /* ── Ablation bar ── */
  .ablation-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--bg-surface-alt);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    margin-bottom: 16px;
    flex-wrap: wrap;
    gap: 8px;
  }
  .abl-summary {
    font-size: 0.82rem;
    color: var(--text-secondary);
  }
  .abl-count-total {
    font-weight: 800;
    font-size: 1rem;
    color: #f59e0b;
  }
  .ablate-btn {
    padding: 8px 20px;
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.85rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  .ablate-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* ── Damage Report ── */
  .damage-report {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
  }
  .dr-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
  }
  .dr-header h4 {
    margin: 0;
    font-size: 1rem;
    color: var(--text-heading);
  }
  .dr-badge {
    font-size: 0.68rem;
    font-weight: 800;
    padding: 3px 10px;
    border-radius: 4px;
    letter-spacing: 0.5px;
  }
  .dr-badge.severe {
    background: rgba(239, 68, 68, 0.15);
    color: #dc2626;
  }
  .dr-badge.moderate {
    background: rgba(217, 119, 6, 0.12);
    color: #d97706;
  }
  .dr-badge.minimal {
    background: rgba(34, 197, 94, 0.12);
    color: #059669;
  }

  /* Diff table */
  .diff-table {
    display: flex;
    flex-direction: column;
    gap: 0;
  }
  .diff-header-row {
    display: grid;
    grid-template-columns: 36px 1fr 1fr 90px;
    gap: 8px;
    padding: 6px 8px;
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border-color);
  }
  .diff-row {
    display: grid;
    grid-template-columns: 36px 1fr 1fr 90px;
    gap: 8px;
    padding: 7px 8px;
    align-items: center;
    border-bottom: 1px solid var(--border-light);
    transition: background 0.1s;
  }
  .diff-row:hover {
    background: var(--bg-surface-alt);
  }
  .diff-tok {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.82rem;
    font-weight: 700;
    text-align: center;
    color: var(--text-heading);
  }
  .diff-bar-cell {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .diff-bar-track {
    flex: 1;
    height: 14px;
    background: var(--bg-inset);
    border-radius: 4px;
    overflow: hidden;
  }
  .diff-bar {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
  }
  .base-bar {
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
  }
  .abl-bar {
    background: linear-gradient(90deg, #ef4444, #f87171);
  }
  .diff-pct {
    font-family: monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    width: 38px;
    text-align: right;
    flex-shrink: 0;
  }
  .diff-delta {
    font-family: monospace;
    font-size: 0.75rem;
    font-weight: 700;
    text-align: right;
    color: var(--text-muted);
  }
  .diff-delta.up {
    color: #059669;
  }
  .diff-delta.down {
    color: #dc2626;
  }

  /* Summary */
  .dr-summary {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-top: 14px;
    padding: 12px 14px;
    background: var(--bg-surface);
    border-radius: 8px;
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
  }
  .dmg-icon {
    flex-shrink: 0;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 0.75rem;
  }
  .severe-icon {
    background: rgba(239, 68, 68, 0.15);
    color: #dc2626;
  }
  .moderate-icon {
    background: rgba(217, 119, 6, 0.12);
    color: #d97706;
  }
  .minimal-icon {
    background: rgba(34, 197, 94, 0.12);
    color: #059669;
  }

  /* ── Generated text comparison ── */
  .gen-comparison {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 16px;
  }
  @media (max-width: 600px) {
    .gen-comparison {
      grid-template-columns: 1fr;
    }
  }
  .gen-card {
    padding: 12px 14px;
    border-radius: 8px;
    border: 1px solid var(--border-color);
    background: var(--bg-surface-alt);
  }
  .gen-card.abl {
    border-color: rgba(239, 68, 68, 0.3);
  }
  .gen-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }
  .gen-text {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.85rem;
    color: var(--text-primary);
  }
  .gen-prompt {
    color: var(--text-muted);
  }
  .gen-cont {
    color: #dc2626;
    font-weight: 700;
  }

  /* ── Empty state ── */
  .empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-secondary);
  }
  .empty-state p {
    margin: 6px 0;
  }
  .empty-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
  }
</style>
