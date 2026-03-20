<script lang="ts">
  /**
   * ConceptActivationExplorer — Compare neuron activations between two prompts.
   *
   * Key features:
   * - Token-hover filtering: hover a token to see per-token neuron activations
   * - Clustered neuron grid: neurons grouped by selectivity (A-only, B-only, shared)
   * - Neuron fingerprint: click a neuron to see its per-token activation profile
   * - Colorblind-friendly palette: blue / amber / teal
   */

  // ── Colorblind-friendly palette ────────────────────────────────────────
  import { apiUrl } from '../api';

  const COL_A = "#2563eb";
  const COL_B = "#d97706";
  const COL_SHARED = "#0d9488";

  // ── State ──────────────────────────────────────────────────────────────
  let promptA = "2+2=";
  let promptB = "Hello";
  let loading = false;
  let error: string | null = null;
  let resultA: any = null;
  let resultB: any = null;
  let selectedLayer = 0;
  let sortBy: "difference" | "activation" | "selectivity" | "fire_rate" | "monosemanticity" =
    "difference";
  let showCount = 40;

  // Token-hover state
  let hoveredTokIdx = -1;
  let hoveredSide: "A" | "B" | null = null;

  // Neuron detail state
  let selectedNeuronId = -1;
  let showInactive = false;

  // ── API ────────────────────────────────────────────────────────────────
  async function fetchActivations(prompt: string): Promise<any> {
    const res = await fetch(apiUrl("/api/neuron/activations"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (!res.ok) throw new Error(`Server ${res.status}`);
    return res.json();
  }

  async function runComparison() {
    if (loading) return;
    loading = true;
    error = null;
    resultA = null;
    resultB = null;
    selectedNeuronId = -1;
    try {
      const [a, b] = await Promise.all([
        fetchActivations(promptA),
        fetchActivations(promptB),
      ]);
      resultA = a;
      resultB = b;
      selectedLayer = 0;
    } catch (e: any) {
      error = e.message ?? "Request failed";
    } finally {
      loading = false;
    }
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter") runComparison();
  }

  const examples = [
    { a: "2+2=", b: "Hello", label: "Math vs Greeting" },
    { a: "The cat", b: "The dog", label: "Cat vs Dog" },
    { a: "red blue", b: "123456", label: "Colors vs Numbers" },
    { a: "How are", b: "Because", label: "Question vs Statement" },
  ];

  function selectExample(ex: (typeof examples)[0]) {
    promptA = ex.a;
    promptB = ex.b;
    runComparison();
  }

  // ── Derived data ──────────────────────────────────────────────────────
  $: layerA = resultA?.layers?.[selectedLayer];
  $: layerB = resultB?.layers?.[selectedLayer];
  $: nLayers = resultA?.layers?.length ?? 0;
  $: tokensA = resultA?.tokens ?? [];
  $: tokensB = resultB?.tokens ?? [];
  $: hasPerToken = !!(layerA?.neurons && layerB?.neurons);

  // Merge neurons from both prompts — aligned by neuron ID (dict-keyed)
  $: mergedNeurons = (() => {
    if (!layerA || !layerB) return [];
    const neuronsA: Record<string, any> = layerA.neurons ?? {};
    const neuronsB: Record<string, any> = layerB.neurons ?? {};

    // Union of all neuron IDs from both prompts
    const allIds = new Set([...Object.keys(neuronsA), ...Object.keys(neuronsB)]);

    return Array.from(allIds).map((idStr) => {
      const id = parseInt(idStr);
      const nA = neuronsA[idStr];
      const nB = neuronsB[idStr];

      const actA: number = nA?.x_activation ?? 0;
      const actB: number = nB?.x_activation ?? 0;
      const maxA: number = nA?.x_max ?? 0;
      const maxB: number = nB?.x_max ?? 0;
      const tokActsA: number[] = nA?.per_token ?? [];
      const tokActsB: number[] = nB?.per_token ?? [];
      const fireA: number = nA?.fire_rate ?? 0;
      const fireB: number = nB?.fire_rate ?? 0;

      const diff = Math.abs(actA - actB);
      const total = actA + actB;
      const selectivity = total > 0.001 ? diff / total : 0;

      // Monosemanticity: 1 = fires for only one concept, 0 = fires equally for both
      const peak = Math.max(maxA, maxB);
      const valley = Math.min(maxA, maxB);
      const monosemanticity = peak > 0.001 ? 1 - valley / peak : 0;

      const thresh = 0.005;
      let category: "A-only" | "B-only" | "shared" | "inactive";
      if (actA > thresh && actB <= thresh) category = "A-only";
      else if (actB > thresh && actA <= thresh) category = "B-only";
      else if (actA > thresh && actB > thresh) category = "shared";
      else category = "inactive";

      // Auto-generate concept label from token activations
      let conceptLabel = "";
      if (category === "A-only" && tokActsA.length > 0) {
        const maxI = tokActsA.indexOf(Math.max(...tokActsA));
        conceptLabel = `Fires on '${tokensA[maxI] ?? "?"}' in A, silent in B`;
      } else if (category === "B-only" && tokActsB.length > 0) {
        const maxI = tokActsB.indexOf(Math.max(...tokActsB));
        conceptLabel = `Fires on '${tokensB[maxI] ?? "?"}' in B, silent in A`;
      } else if (category === "shared") {
        const topA = tokActsA.length > 0 ? tokActsA.indexOf(Math.max(...tokActsA)) : 0;
        const topB = tokActsB.length > 0 ? tokActsB.indexOf(Math.max(...tokActsB)) : 0;
        conceptLabel = `Shared — peaks on '${tokensA[topA] ?? "?"}' (A) and '${tokensB[topB] ?? "?"}' (B)`;
      }

      return {
        id,
        actA,
        actB,
        maxA,
        maxB,
        fireA,
        fireB,
        diff,
        selectivity,
        monosemanticity,
        category,
        conceptLabel,
        tokActsA,
        tokActsB,
      };
    });
  })();

  // Cluster neurons by category
  $: clusters = (() => {
    const g: Record<string, any[]> = {
      "A-only": [],
      "B-only": [],
      shared: [],
      inactive: [],
    };
    for (const n of mergedNeurons) g[n.category].push(n);
    g["A-only"].sort((a: any, b: any) => b.actA - a.actA);
    g["B-only"].sort((a: any, b: any) => b.actB - a.actB);
    g["shared"].sort((a: any, b: any) => b.actA + b.actB - (a.actA + a.actB));
    return g;
  })();

  // Monosemanticity stats for the bar visualization
  $: monoStats = (() => {
    const active = mergedNeurons.filter((n) => n.category !== "inactive");
    const total = active.length;
    if (total === 0) return { total: 0, monoA: 0, monoB: 0, shared: 0, pct: 0, pctA: 0, pctShared: 0, pctB: 0 };
    const monoA = clusters["A-only"].length;
    const monoB = clusters["B-only"].length;
    const shared = clusters["shared"].length;
    return {
      total,
      monoA,
      monoB,
      shared,
      pct: Math.round((100 * (monoA + monoB)) / total),
      pctA: Math.round((100 * monoA) / total),
      pctShared: Math.round((100 * shared) / total),
      pctB: Math.round((100 * monoB) / total),
    };
  })();

  // Sorted list for detail panel
  $: sortedNeurons = (() => {
    const arr = mergedNeurons.filter(
      (n) => n.category !== "inactive" || showInactive,
    );
    if (sortBy === "activation")
      arr.sort((a, b) => Math.max(b.actA, b.actB) - Math.max(a.actA, a.actB));
    else if (sortBy === "difference") arr.sort((a, b) => b.diff - a.diff);
    else if (sortBy === "selectivity")
      arr.sort((a, b) => b.selectivity - a.selectivity);
    else if (sortBy === "monosemanticity")
      arr.sort((a, b) => b.monosemanticity - a.monosemanticity);
    else
      arr.sort(
        (a, b) => Math.max(b.fireA, b.fireB) - Math.max(a.fireA, a.fireB),
      );
    return arr.slice(0, showCount);
  })();

  $: maxAct = Math.max(
    ...mergedNeurons.map((n) => Math.max(n.actA, n.actB)),
    0.001,
  );
  $: selectedNeuron =
    selectedNeuronId >= 0
      ? (mergedNeurons.find((n) => n.id === selectedNeuronId) ?? null)
      : null;

  // Grid cell background — adapts when hovering a token
  function gridBg(neuron: any): string {
    if (hoveredTokIdx >= 0 && hoveredSide && hasPerToken) {
      const acts = hoveredSide === "A" ? neuron.tokActsA : neuron.tokActsB;
      const val = acts[hoveredTokIdx] ?? 0;
      if (val < 0.001) return "var(--neuron-inactive)";
      return hoveredSide === "A" ? COL_A : COL_B;
    }
    if (neuron.category === "A-only") return COL_A;
    if (neuron.category === "B-only") return COL_B;
    if (neuron.category === "shared") return COL_SHARED;
    return "var(--neuron-inactive)";
  }

  function gridOpacity(neuron: any): number {
    if (hoveredTokIdx >= 0 && hoveredSide && hasPerToken) {
      const acts = hoveredSide === "A" ? neuron.tokActsA : neuron.tokActsB;
      const val = acts[hoveredTokIdx] ?? 0;
      if (val < 0.001) return 0.25;
      return 0.3 + 0.7 * Math.min(val / maxAct, 1);
    }
    if (neuron.category === "inactive") return 0.25;
    return 0.4 + 0.6 * Math.min(Math.max(neuron.actA, neuron.actB) / maxAct, 1);
  }

  function catColor(cat: string): string {
    if (cat === "A-only") return COL_A;
    if (cat === "B-only") return COL_B;
    if (cat === "shared") return COL_SHARED;
    return "var(--text-muted)";
  }

  function selectNeuron(id: number) {
    selectedNeuronId = selectedNeuronId === id ? -1 : id;
  }

</script>

<div class="explorer">
  <div class="header">
    <h2>Concept Activation Explorer (Monosemanticity)</h2>
    <p class="subtitle">
      Compare neuron activations between two prompts — hover tokens and click
      neurons for deep insights
    </p>
  </div>

  {#if error}
    <div class="error-bar">{error} — backend may be waking up, please retry in 30s</div>
  {/if}

  <!-- Prompt inputs -->
  <div class="prompt-row">
    <div class="prompt-box">
      <label class="prompt-label"
        >Prompt A <span style="color:{COL_A}">&#9679;</span></label
      >
      <input
        bind:value={promptA}
        on:keydown={onKeyDown}
        placeholder="First prompt..."
        disabled={loading}
      />
    </div>
    <button class="compare-btn" on:click={runComparison} disabled={loading}
      >Compare</button
    >
    <div class="prompt-box">
      <label class="prompt-label"
        >Prompt B <span style="color:{COL_B}">&#9679;</span></label
      >
      <input
        bind:value={promptB}
        on:keydown={onKeyDown}
        placeholder="Second prompt..."
        disabled={loading}
      />
    </div>
  </div>

  <!-- Example presets -->
  <div class="examples">
    {#each examples as ex}
      <button class="example-chip" on:click={() => selectExample(ex)}
        >{ex.label}</button
      >
    {/each}
  </div>

  {#if loading}
    <div class="loading">
      <div class="spinner"></div>
       Running BDH on both prompts...
    </div>
  {/if}

  {#if resultA && resultB}
    <!-- Hoverable token chips -->
    <div class="token-section">
      <div class="token-group">
        <span class="tok-header" style="color:{COL_A}"
          >A Tokens
          {#if hasPerToken}<span class="tok-hint">(hover to filter grid)</span
            >{/if}
        </span>
        <div class="tok-chips">
          {#each tokensA as tok, i}
            <button
              class="tok-chip tok-a"
              class:active={hoveredSide === "A" && hoveredTokIdx === i}
              on:mouseenter={() => {
                hoveredTokIdx = i;
                hoveredSide = "A";
              }}
              on:mouseleave={() => {
                hoveredTokIdx = -1;
                hoveredSide = null;
              }}
            >
              <span class="tok-pos">t{i}</span>{tok}
            </button>
          {/each}
        </div>
      </div>
      <div class="token-group">
        <span class="tok-header" style="color:{COL_B}">B Tokens</span>
        <div class="tok-chips">
          {#each tokensB as tok, i}
            <button
              class="tok-chip tok-b"
              class:active={hoveredSide === "B" && hoveredTokIdx === i}
              on:mouseenter={() => {
                hoveredTokIdx = i;
                hoveredSide = "B";
              }}
              on:mouseleave={() => {
                hoveredTokIdx = -1;
                hoveredSide = null;
              }}
            >
              <span class="tok-pos">t{i}</span>{tok}
            </button>
          {/each}
        </div>
      </div>
    </div>

    <!-- Layer selector -->
    <div class="layer-selector">
      <span class="ls-label">Layer:</span>
      {#each Array(nLayers) as _, i}
        <button
          class="layer-btn"
          class:active={selectedLayer === i}
          on:click={() => {
            selectedLayer = i;
            selectedNeuronId = -1;
          }}>L{i}</button
        >
      {/each}
      <span class="sparsity-info"
        >Sparsity: A={layerA?.sparsity_pct}% | B={layerB?.sparsity_pct}% · Active: A={Object.keys(layerA?.neurons ?? {}).length} B={Object.keys(layerB?.neurons ?? {}).length}</span
      >
    </div>

    <!-- Cluster summary bar -->
    <div class="cluster-summary">
      <span class="cs-chip"
        ><span class="cs-dot" style="background:{COL_A}"></span> A-only: {clusters[
          "A-only"
        ].length}</span
      >
      <span class="cs-chip"
        ><span class="cs-dot" style="background:{COL_B}"></span> B-only: {clusters[
          "B-only"
        ].length}</span
      >
      <span class="cs-chip"
        ><span class="cs-dot" style="background:{COL_SHARED}"></span> Shared: {clusters[
          "shared"
        ].length}</span
      >
      <span class="cs-chip"
        ><span class="cs-dot" style="background:var(--neuron-inactive)"></span>
        Inactive: {clusters["inactive"].length}</span
      >
      {#if hoveredSide}
        <span class="cs-filter"
          >Filtering: token '{hoveredSide === "A"
            ? tokensA[hoveredTokIdx]
            : tokensB[hoveredTokIdx]}' from Prompt {hoveredSide}</span
        >
      {/if}
    </div>

    <!-- Monosemanticity Analysis Bar -->
    {#if monoStats.total > 0}
      <div class="mono-section">
        <div class="mono-header">
          <span class="mono-title">🧠 Monosemanticity Analysis</span>
          <span class="mono-score">{monoStats.pct}% concept-specific</span>
          <span class="mono-total">({monoStats.total} active neurons aligned by ID)</span>
        </div>
        <div class="mono-bar">
          {#if monoStats.pctA > 0}
            <div class="mono-seg" style="width:{monoStats.pctA}%; background:{COL_A}" title="{monoStats.monoA} A-specific neurons ({monoStats.pctA}%)"></div>
          {/if}
          {#if monoStats.pctShared > 0}
            <div class="mono-seg" style="width:{monoStats.pctShared}%; background:{COL_SHARED}" title="{monoStats.shared} shared/polysemantic neurons ({monoStats.pctShared}%)"></div>
          {/if}
          {#if monoStats.pctB > 0}
            <div class="mono-seg" style="width:{monoStats.pctB}%; background:{COL_B}" title="{monoStats.monoB} B-specific neurons ({monoStats.pctB}%)"></div>
          {/if}
        </div>
        <div class="mono-legend">
          <span><span class="mono-swatch" style="background:{COL_A}"></span> A-specific: {monoStats.monoA} ({monoStats.pctA}%)</span>
          <span><span class="mono-swatch" style="background:{COL_SHARED}"></span> Polysemantic: {monoStats.shared} ({monoStats.pctShared}%)</span>
          <span><span class="mono-swatch" style="background:{COL_B}"></span> B-specific: {monoStats.monoB} ({monoStats.pctB}%)</span>
        </div>
      </div>
    {/if}

    <div class="results-grid">
      <!-- Left: Clustered neuron grid -->
      <div class="grid-panel">
        <h4>Neuron Activation Map</h4>
        {#each [{ key: "A-only", label: "A-Only Neurons", color: COL_A }, { key: "B-only", label: "B-Only Neurons", color: COL_B }, { key: "shared", label: "Shared Neurons", color: COL_SHARED }] as group}
          {#if clusters[group.key].length > 0}
            <div class="cluster-group">
              <div class="cluster-hdr" style="border-left-color:{group.color}">
                <span class="ch-label">{group.label}</span>
                <span class="ch-count">{clusters[group.key].length}</span>
              </div>
              <div class="neuron-grid">
                {#each clusters[group.key] as neuron}
                  <button
                    class="grid-cell"
                    style="background:{gridBg(neuron)}; opacity:{gridOpacity(
                      neuron,
                    )}"
                    class:selected={selectedNeuronId === neuron.id}
                    on:click={() => selectNeuron(neuron.id)}
                    title="#{neuron.id}: A={neuron.actA.toFixed(
                      3,
                    )} B={neuron.actB.toFixed(3)} | {neuron.conceptLabel ||
                      neuron.category}"
                  ></button>
                {/each}
              </div>
            </div>
          {/if}
        {/each}
        {#if clusters["inactive"].length > 0}
          <button
            class="inactive-toggle"
            on:click={() => (showInactive = !showInactive)}
          >
            {showInactive ? "Hide" : "Show"}
            {clusters["inactive"].length} inactive neurons
          </button>
          {#if showInactive}
            <div class="neuron-grid" style="margin-top:4px">
              {#each clusters["inactive"] as neuron}
                <button
                  class="grid-cell"
                  style="background:var(--neuron-inactive); opacity:0.25"
                  on:click={() => selectNeuron(neuron.id)}
                  title="#{neuron.id}: inactive"
                ></button>
              {/each}
            </div>
          {/if}
        {/if}
      </div>

      <!-- Right: Neuron fingerprint OR sorted list -->
      <div class="detail-panel">
        {#if selectedNeuron}
          <!-- Neuron fingerprint detail -->
          <div class="fingerprint">
            <div class="fp-header">
              <h4>Neuron #{selectedNeuron.id}</h4>
              <span
                class="cat-badge"
                style="background:{catColor(
                  selectedNeuron.category,
                )}18; color:{catColor(
                  selectedNeuron.category,
                )}; border:1px solid {catColor(selectedNeuron.category)}40"
              >
                {selectedNeuron.category}
              </span>
              <button class="fp-close" on:click={() => (selectedNeuronId = -1)}
                >&#10005;</button
              >
            </div>
            {#if selectedNeuron.conceptLabel}
              <p class="fp-concept">{selectedNeuron.conceptLabel}</p>
            {/if}
            <div class="fp-stats">
              <span
                >Mono: <strong
                  >{(selectedNeuron.monosemanticity * 100).toFixed(0)}%</strong
                ></span
              >
              <span>| A: <strong>{selectedNeuron.actA.toFixed(4)}</strong></span
              >
              <span>B: <strong>{selectedNeuron.actB.toFixed(4)}</strong></span>
              <span
                >Delta: <strong>{selectedNeuron.diff.toFixed(4)}</strong></span
              >
            </div>

            <!-- Per-token activation chart -->
            {#if hasPerToken}
              <div class="fp-chart">
                <div class="fp-chart-label" style="color:{COL_A}">
                  Prompt A — per-token activation:
                </div>
                <div class="fp-bars">
                  {#each tokensA as tok, t}
                    {@const val = selectedNeuron.tokActsA[t] ?? 0}
                    {@const pct = Math.round(100 * Math.min(val / maxAct, 1))}
                    <div class="fp-bar-item">
                      <span class="fp-tok">{tok}</span>
                      <div class="fp-bar-track">
                        <div
                          class="fp-bar-fill"
                          style="width:{Math.max(pct, 1)}%; background:{COL_A}"
                        ></div>
                      </div>
                      <span class="fp-bar-val">{val.toFixed(4)}</span>
                    </div>
                  {/each}
                </div>
                <div
                  class="fp-chart-label"
                  style="color:{COL_B}; margin-top:12px"
                >
                  Prompt B — per-token activation:
                </div>
                <div class="fp-bars">
                  {#each tokensB as tok, t}
                    {@const val = selectedNeuron.tokActsB[t] ?? 0}
                    {@const pct = Math.round(100 * Math.min(val / maxAct, 1))}
                    <div class="fp-bar-item">
                      <span class="fp-tok">{tok}</span>
                      <div class="fp-bar-track">
                        <div
                          class="fp-bar-fill"
                          style="width:{Math.max(pct, 1)}%; background:{COL_B}"
                        ></div>
                      </div>
                      <span class="fp-bar-val">{val.toFixed(4)}</span>
                    </div>
                  {/each}
                </div>
              </div>
            {:else}
              <p class="fp-no-data">
                Per-token data not available. Restart the backend to enable.
              </p>
            {/if}
          </div>
        {:else}
          <!-- Sorted neuron comparison list -->
          <div class="sort-controls">
            <span class="sort-label">Sort by:</span>
            <button
              class="sort-btn"
              class:active={sortBy === "difference"}
              on:click={() => (sortBy = "difference")}>Difference</button
            >
            <button
              class="sort-btn"
              class:active={sortBy === "selectivity"}
              on:click={() => (sortBy = "selectivity")}>Selectivity</button
            >
            <button
              class="sort-btn"
              class:active={sortBy === "activation"}
              on:click={() => (sortBy = "activation")}>Activation</button
            >
            <button
              class="sort-btn"
              class:active={sortBy === "fire_rate"}
              on:click={() => (sortBy = "fire_rate")}>Fire Rate</button
            >
            <button
              class="sort-btn"
              class:active={sortBy === "monosemanticity"}
              on:click={() => (sortBy = "monosemanticity")}>Mono Score</button
            >
          </div>
          <p class="list-hint">
            Click any row to see its per-token fingerprint
          </p>
          <div class="neuron-list">
            {#each sortedNeurons as n}
              <button class="neuron-row" on:click={() => selectNeuron(n.id)}>
                <span class="n-id">#{n.id}</span>
                <span
                  class="n-cat-dot"
                  style="background:{catColor(n.category)}"
                ></span>
                <div class="n-bars">
                  <div class="n-bar-row">
                    <span class="n-bar-label" style="color:{COL_A}">A</span>
                    <div class="n-bar-track">
                      <div
                        class="n-bar-fill"
                        style="width:{Math.round(
                          (100 * n.actA) / maxAct,
                        )}%; background:{COL_A}"
                      ></div>
                    </div>
                    <span class="n-val">{n.actA.toFixed(3)}</span>
                  </div>
                  <div class="n-bar-row">
                    <span class="n-bar-label" style="color:{COL_B}">B</span>
                    <div class="n-bar-track">
                      <div
                        class="n-bar-fill"
                        style="width:{Math.round(
                          (100 * n.actB) / maxAct,
                        )}%; background:{COL_B}"
                      ></div>
                    </div>
                    <span class="n-val">{n.actB.toFixed(3)}</span>
                  </div>
                </div>
                <div class="n-meta">
                  <span class="n-mono" class:high={n.monosemanticity > 0.8}
                    >{(n.monosemanticity * 100).toFixed(0)}% mono</span
                  >
                  <span class="n-diff" class:high={n.diff > maxAct * 0.3}
                    >Delta {n.diff.toFixed(3)}</span
                  >
                </div>
              </button>
            {/each}
          </div>
          <div class="show-more">
            <button
              on:click={() =>
                (showCount = Math.min(showCount + 20, mergedNeurons.length))}
            >
              Show more ({showCount}/{mergedNeurons.filter(
                (n) => n.category !== "inactive" || showInactive,
              ).length})
            </button>
          </div>
        {/if}
      </div>
    </div>

    <!-- Output comparison -->
    <div class="output-row">
      <div class="out-card" style="border-left:3px solid {COL_A}">
        <div class="out-label">A predicts:</div>
        <div class="out-probs">
          {#each resultA.prob_bars.slice(0, 5) as bar}
            <span class="out-tok"
              >{bar.token} <em>{(bar.prob * 100).toFixed(1)}%</em></span
            >
          {/each}
        </div>
      </div>
      <div class="out-card" style="border-left:3px solid {COL_B}">
        <div class="out-label">B predicts:</div>
        <div class="out-probs">
          {#each resultB.prob_bars.slice(0, 5) as bar}
            <span class="out-tok"
              >{bar.token} <em>{(bar.prob * 100).toFixed(1)}%</em></span
            >
          {/each}
        </div>
      </div>
    </div>
  {:else if !loading}
    <div class="empty-state">
      <p>
        Enter two prompts and click <strong>Compare</strong> to explore neuron activations
      </p>
      <p class="empty-sub">Or try a preset comparison above</p>
    </div>
  {/if}
</div>

<style>
  .explorer {
    max-width: 1100px;
    margin: 0 auto;
  }
  .header {
    text-align: center;
    margin-bottom: 20px;
  }
  .header h2 {
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

  /* ── Prompts ── */
  .prompt-row {
    display: flex;
    gap: 12px;
    align-items: flex-end;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .prompt-box {
    flex: 1;
    min-width: 180px;
  }
  .prompt-label {
    display: block;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }
  .prompt-box input {
    width: 100%;
    padding: 9px 14px;
    border: 2px solid var(--border-color);
    border-radius: 8px;
    font-size: 0.9rem;
    font-family: "JetBrains Mono", monospace;
    outline: none;
    transition: border-color 0.15s;
    box-sizing: border-box;
    background: var(--bg-card);
    color: var(--text-heading);
  }
  .prompt-box input:focus {
    border-color: var(--accent);
  }
  .compare-btn {
    padding: 9px 22px;
    background: linear-gradient(135deg, var(--accent), #764ba2);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.88rem;
    cursor: pointer;
    white-space: nowrap;
  }
  .compare-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .examples {
    display: flex;
    gap: 6px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }
  .example-chip {
    padding: 5px 12px;
    border: 1px solid var(--border-color);
    border-radius: 20px;
    background: var(--bg-card);
    font-size: 0.78rem;
    cursor: pointer;
    color: var(--text-secondary);
    transition: all 0.15s;
  }
  .example-chip:hover {
    border-color: var(--accent);
    color: var(--accent);
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

  /* ── Token chips (hoverable) ── */
  .token-section {
    display: flex;
    gap: 20px;
    margin-bottom: 14px;
    flex-wrap: wrap;
  }
  .token-group {
    flex: 1;
    min-width: 200px;
  }
  .tok-header {
    font-size: 0.75rem;
    font-weight: 700;
    display: block;
    margin-bottom: 6px;
  }
  .tok-hint {
    font-weight: 400;
    color: var(--text-muted);
    font-style: italic;
    font-size: 0.7rem;
  }
  .tok-chips {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }
  .tok-chip {
    padding: 5px 10px;
    border-radius: 6px;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.82rem;
    font-weight: 600;
    border: 2px solid transparent;
    cursor: pointer;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .tok-pos {
    font-size: 0.58rem;
    opacity: 0.5;
    font-weight: 400;
  }
  .tok-a {
    background: rgba(37, 99, 235, 0.1);
    color: #2563eb;
  }
  .tok-a:hover,
  .tok-a.active {
    background: rgba(37, 99, 235, 0.25);
    border-color: #2563eb;
  }
  .tok-b {
    background: rgba(217, 119, 6, 0.1);
    color: #d97706;
  }
  .tok-b:hover,
  .tok-b.active {
    background: rgba(217, 119, 6, 0.25);
    border-color: #d97706;
  }

  /* ── Layer selector ── */
  .layer-selector {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .ls-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-secondary);
  }
  .layer-btn {
    padding: 5px 12px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-card);
    font-size: 0.78rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s;
    color: var(--text-primary);
  }
  .layer-btn.active {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  .sparsity-info {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-left: auto;
  }

  /* ── Cluster summary bar ── */
  .cluster-summary {
    display: flex;
    gap: 12px;
    margin-bottom: 14px;
    flex-wrap: wrap;
    align-items: center;
  }
  .cs-chip {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
  }
  .cs-dot {
    width: 10px;
    height: 10px;
    border-radius: 3px;
    flex-shrink: 0;
  }
  .cs-filter {
    font-size: 0.75rem;
    font-style: italic;
    color: var(--accent);
    margin-left: auto;
    font-weight: 600;
  }

  /* ── Results grid ── */
  .results-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }
  @media (max-width: 800px) {
    .results-grid {
      grid-template-columns: 1fr;
    }
  }
  .grid-panel,
  .detail-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 16px;
    overflow: hidden;
  }
  .grid-panel h4 {
    font-size: 0.88rem;
    margin: 0 0 12px;
    color: var(--text-heading);
  }

  /* ── Cluster groups ── */
  .cluster-group {
    margin-bottom: 14px;
  }
  .cluster-hdr {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    margin-bottom: 6px;
    border-left: 3px solid;
    border-radius: 0 4px 4px 0;
    background: var(--bg-inset);
  }
  .ch-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-secondary);
  }
  .ch-count {
    font-size: 0.62rem;
    font-weight: 800;
    color: var(--text-muted);
    background: var(--bg-surface-alt);
    padding: 1px 6px;
    border-radius: 10px;
  }

  /* ── Neuron grid ── */
  .neuron-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
  }
  .grid-cell {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1.5px solid transparent;
    cursor: pointer;
    transition: all 0.15s;
  }
  .grid-cell:hover {
    transform: scale(1.6);
    z-index: 2;
    border-color: var(--text-heading);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }
  .grid-cell.selected {
    border-color: #f59e0b;
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
    transform: scale(1.4);
    z-index: 1;
  }

  .inactive-toggle {
    display: block;
    width: 100%;
    margin-top: 10px;
    padding: 6px;
    background: none;
    border: 1px dashed var(--border-color);
    border-radius: 6px;
    font-size: 0.72rem;
    color: var(--text-muted);
    cursor: pointer;
  }
  .inactive-toggle:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  /* ── Fingerprint panel ── */
  .fingerprint {
  }
  .fp-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }
  .fp-header h4 {
    margin: 0;
    font-size: 0.95rem;
    color: var(--text-heading);
  }
  .cat-badge {
    font-size: 0.66rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    white-space: nowrap;
  }
  .fp-close {
    margin-left: auto;
    background: none;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-muted);
    font-size: 0.85rem;
    cursor: pointer;
    padding: 2px 8px;
  }
  .fp-close:hover {
    color: var(--text-heading);
    border-color: var(--text-muted);
  }
  .fp-concept {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-bottom: 10px;
    font-style: italic;
    line-height: 1.4;
  }
  .fp-stats {
    display: flex;
    gap: 14px;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-bottom: 14px;
    flex-wrap: wrap;
  }
  .fp-chart-label {
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 6px;
  }
  .fp-bars {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .fp-bar-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .fp-tok {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.75rem;
    width: 30px;
    text-align: center;
    flex-shrink: 0;
    color: var(--text-secondary);
  }
  .fp-bar-track {
    flex: 1;
    height: 16px;
    background: var(--bg-inset);
    border-radius: 4px;
    overflow: hidden;
  }
  .fp-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
    min-width: 2px;
  }
  .fp-bar-val {
    font-family: monospace;
    font-size: 0.62rem;
    color: var(--text-muted);
    width: 46px;
    text-align: right;
    flex-shrink: 0;
  }
  .fp-no-data {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-style: italic;
  }

  /* ── Sort controls ── */
  .sort-controls {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
  }
  .sort-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-secondary);
  }
  .sort-btn {
    padding: 4px 10px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-card);
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    color: var(--text-primary);
  }
  .sort-btn.active {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  .list-hint {
    font-size: 0.7rem;
    color: var(--text-muted);
    margin: 0 0 8px;
    font-style: italic;
  }

  /* ── Neuron list ── */
  .neuron-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 480px;
    overflow-y: auto;
  }
  .neuron-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    border-radius: 6px;
    cursor: pointer;
    border: none;
    background: none;
    text-align: left;
    width: 100%;
    transition: background 0.15s;
    color: var(--text-primary);
  }
  .neuron-row:hover {
    background: var(--bg-surface-alt);
  }
  .n-id {
    font-family: monospace;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--accent);
    width: 30px;
    flex-shrink: 0;
  }
  .n-cat-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .n-bars {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .n-bar-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .n-bar-label {
    font-size: 0.65rem;
    font-weight: 700;
    width: 12px;
    flex-shrink: 0;
  }
  .n-bar-track {
    flex: 1;
    height: 10px;
    background: var(--bg-inset);
    border-radius: 4px;
    overflow: hidden;
  }
  .n-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
  }
  .n-val {
    font-family: monospace;
    font-size: 0.62rem;
    color: var(--text-muted);
    width: 40px;
    text-align: right;
    flex-shrink: 0;
  }
  .n-meta {
    display: flex;
    flex-direction: column;
    gap: 2px;
    align-items: flex-end;
  }
  .n-diff {
    font-family: monospace;
    font-size: 0.65rem;
    padding: 2px 6px;
    background: var(--bg-inset);
    border-radius: 4px;
    color: var(--text-muted);
    white-space: nowrap;
  }
  .n-diff.high {
    background: var(--bg-warning);
    color: #d97706;
    font-weight: 700;
  }

  .show-more {
    text-align: center;
    margin-top: 10px;
  }
  .show-more button {
    padding: 5px 16px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-card);
    font-size: 0.75rem;
    cursor: pointer;
    color: var(--text-secondary);
  }
  .show-more button:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  /* ── Output row ── */
  .output-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 14px;
  }
  @media (max-width: 600px) {
    .output-row {
      grid-template-columns: 1fr;
    }
  }
  .out-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
  }
  .out-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-secondary);
    margin-bottom: 6px;
  }
  .out-probs {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .out-tok {
    padding: 3px 8px;
    background: var(--bg-inset);
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.78rem;
    color: var(--text-primary);
  }
  .out-tok em {
    font-style: normal;
    font-size: 0.68rem;
    color: var(--text-muted);
  }

  /* ── Insight box ── */
  .insight-box {
    padding: 14px 16px;
    background: var(--bg-surface-alt);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.6;
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

  /* ── Monosemanticity section ── */
  .mono-section {
    margin-bottom: 16px;
    padding: 14px 16px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
  }
  .mono-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    flex-wrap: wrap;
  }
  .mono-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--text-heading);
  }
  .mono-score {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--accent);
    background: var(--bg-surface-alt);
    padding: 2px 10px;
    border-radius: 12px;
  }
  .mono-total {
    font-size: 0.72rem;
    color: var(--text-muted);
  }
  .mono-bar {
    display: flex;
    height: 24px;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 8px;
    background: var(--bg-inset);
  }
  .mono-seg {
    height: 100%;
    transition: width 0.4s ease;
    min-width: 2px;
    position: relative;
  }
  .mono-seg:hover {
    opacity: 0.85;
    filter: brightness(1.1);
  }
  .mono-legend {
    display: flex;
    gap: 18px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    flex-wrap: wrap;
  }
  .mono-swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 3px;
    margin-right: 4px;
    vertical-align: middle;
  }
  .n-mono {
    font-family: monospace;
    font-size: 0.65rem;
    padding: 2px 6px;
    background: var(--bg-inset);
    border-radius: 4px;
    color: var(--text-muted);
    white-space: nowrap;
  }
  .n-mono.high {
    background: rgba(16, 185, 129, 0.15);
    color: #059669;
    font-weight: 700;
  }
</style>
