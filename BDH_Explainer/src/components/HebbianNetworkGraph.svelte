<script lang="ts">
  /**
   * HebbianNetworkGraph — Real-time force-directed graph of Hebbian learning.
   *
   * Nodes  = Neurons in a BDH layer (top-N most active).
   * Edges  = Fast-weight connections (σ matrix).
   * Play   = Scrub through token steps to watch σ evolve via Δσ = x⊗y.
   *
   * All data comes live from /api/run.
   */
  import { onMount, onDestroy } from "svelte";
  import { apiUrl } from '../api';
  import * as d3 from "d3";

  // ── Props ──────────────────────────────────────────────────────────────
  export let width = 820;
  export let height = 560;

  // ── State ──────────────────────────────────────────────────────────────
  let prompt = "2+2=";
  let loading = false;
  let error: string | null = null;

  let apiData: any = null;
  let modelInfo: any = null;
  let selectedLayer = 0;
  let currentStep = 0; // token timestep 0..T-1
  let playing = false;
  let frozen = false;
  let hoveredNode: number | null = null;
  let playTimer: any = null;

  const EDGE_THRESHOLD = 0.005;

  // ── Fetch model info on mount ─────────────────────────────────────────
  onMount(async () => {
    try {
      const r = await fetch(apiUrl('/api/model/info'));
      if (r.ok) modelInfo = await r.json();
    } catch {}
  });

  // ── Derived data ──────────────────────────────────────────────────────
  $: tokens = apiData?.tokens ?? [];
  $: nLayers = modelInfo?.n_layer ?? apiData?.layers?.length ?? 0;
  $: layerData = apiData?.layers?.[selectedLayer] ?? null;
  $: T = tokens.length;

  // Graph neurons come from backend — real neuron IDs, sorted by activity
  $: graphNeuronIds = (layerData?.graph_neuron_ids ?? []) as number[];
  $: totalActive = layerData?.graph_total_active ?? 0;
  $: totalNeurons = layerData?.graph_total_neurons ?? (modelInfo?.neuron_dim ?? 0);
  $: hubInfo = layerData?.hub ?? null;

  // Current sigma = sigma_before + cumulative outer products up to currentStep
  $: currentSigma = computeSigmaAtStep(layerData, currentStep, graphNeuronIds);
  $: currentActivations = getActivationsAtStep(
    layerData,
    currentStep,
    graphNeuronIds,
  );

  function computeSigmaAtStep(
    ld: any,
    step: number,
    neurons: number[],
  ): Map<string, number> {
    const map = new Map<string, number>();
    if (!ld || !neurons.length || !ld.graph_sigma_before) return map;
    const sb: number[][] = ld.graph_sigma_before; // [ng, ng] submatrix
    const xS: number[][] = ld.graph_x_sparse; // [T, ng]
    const yS: number[][] = ld.graph_y_sparse; // [T, ng]
    const ng = neurons.length;

    for (let i = 0; i < ng; i++) {
      for (let j = 0; j < ng; j++) {
        let val = sb[i]?.[j] ?? 0;
        // Accumulate outer products for steps 0..step
        for (let t = 0; t <= step && t < xS.length; t++) {
          val += (xS[t]?.[i] ?? 0) * (yS[t]?.[j] ?? 0);
        }
        if (Math.abs(val) > EDGE_THRESHOLD) {
          // Store using real neuron IDs
          map.set(`${neurons[i]}-${neurons[j]}`, val);
        }
      }
    }
    return map;
  }

  function getActivationsAtStep(
    ld: any,
    step: number,
    neurons: number[],
  ): Map<number, { x: number; y: number }> {
    const map = new Map<number, { x: number; y: number }>();
    if (!ld || !ld.graph_x_sparse) return map;
    const xS: number[][] = ld.graph_x_sparse; // [T, ng]
    const yS: number[][] = ld.graph_y_sparse; // [T, ng]
    for (let idx = 0; idx < neurons.length; idx++) {
      map.set(neurons[idx], {
        x: xS[step]?.[idx] ?? 0,
        y: yS[step]?.[idx] ?? 0,
      });
    }
    return map;
  }

  // ── API ────────────────────────────────────────────────────────────────
  async function runPrompt() {
    if (loading) return;
    loading = true;
    error = null;
    apiData = null;
    stopPlaying();
    currentStep = 0;
    try {
      const res = await fetch(apiUrl("/api/run"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, temperature: 0.5 }),
      });
      if (!res.ok) throw new Error(`Server ${res.status}`);
      apiData = await res.json();
    } catch (e: any) {
      error = e.message ?? "Failed";
    } finally {
      loading = false;
    }
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "Enter") runPrompt();
  }

  // ── Playback ──────────────────────────────────────────────────────────
  function togglePlay() {
    if (playing) {
      stopPlaying();
      return;
    }
    playing = true;
    playTimer = setInterval(() => {
      if (currentStep < T - 1) {
        currentStep++;
      } else {
        stopPlaying();
      }
    }, 800);
  }
  function stopPlaying() {
    playing = false;
    clearInterval(playTimer);
  }
  function resetStep() {
    stopPlaying();
    currentStep = 0;
  }

  // ── D3 Graph ──────────────────────────────────────────────────────────
  let svgEl: SVGSVGElement;
  let simulation: d3.Simulation<any, any> | null = null;

  // Persistent node positions
  let nodePositions = new Map<number, { x: number; y: number }>();

  function buildGraph(
    neurons: number[],
    sigma: Map<string, number>,
    acts: Map<number, { x: number; y: number }>,
    hub: any,
  ) {
    if (!svgEl || !neurons.length) return;

    const svg = d3.select(svgEl);
    const hubId = hub?.neuron_id ?? -1;

    // Build nodes
    const nodes = neurons.map((n) => {
      const prev = nodePositions.get(n);
      return {
        id: n,
        act: acts.get(n) ?? { x: 0, y: 0 },
        isHub: n === hubId,
        x: prev?.x ?? width / 2 + (Math.random() - 0.5) * 200,
        y: prev?.y ?? height / 2 + (Math.random() - 0.5) * 200,
      };
    });

    // Build edges
    const maxSigma = Math.max(
      0.01,
      ...Array.from(sigma.values()).map(Math.abs),
    );
    const links: any[] = [];
    for (const [key, val] of sigma) {
      const [si, sj] = key.split("-").map(Number);
      if (si >= sj) continue; // undirected; skip self-loops and dupes
      const src = nodes.find((n) => n.id === si);
      const tgt = nodes.find((n) => n.id === sj);
      if (src && tgt) {
        links.push({
          source: src,
          target: tgt,
          weight: val,
          normW: Math.abs(val) / maxSigma,
        });
      }
    }

    // Filter to top edges for readability
    links.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
    const visibleLinks = links.slice(0, Math.min(links.length, 150));

    // Simulation
    if (simulation) simulation.stop();

    if (!frozen) {
      simulation = d3
        .forceSimulation(nodes)
        .force("charge", d3.forceManyBody().strength(-40))
        .force("center", d3.forceCenter(width / 2, height / 2).strength(0.04))
        .force(
          "link",
          d3
            .forceLink(visibleLinks)
            .distance((d) => 80 * (1 - d.normW))
            .strength((d) => 0.2 + 0.8 * d.normW),
        )
        .force("collision", d3.forceCollide(14))
        .alpha(0.3)
        .alphaDecay(0.02)
        .on("tick", () => {
          // Save positions
          for (const n of nodes) nodePositions.set(n.id, { x: n.x, y: n.y });
          renderSVG(svg, nodes, visibleLinks, maxSigma, hubId);
        });
    } else {
      renderSVG(svg, nodes, visibleLinks, maxSigma, hubId);
    }
  }

  function renderSVG(
    svg: d3.Selection<any, any, any, any>,
    nodes: any[],
    links: any[],
    maxSigma: number,
    hubId: number,
  ) {
    svg.selectAll("*").remove();

    // Defs for glow filter
    const defs = svg.append("defs");
    const filter = defs.append("filter").attr("id", "glow");
    filter
      .append("feGaussianBlur")
      .attr("stdDeviation", "3")
      .attr("result", "blur");
    const merge = filter.append("feMerge");
    merge.append("feMergeNode").attr("in", "blur");
    merge.append("feMergeNode").attr("in", "SourceGraphic");

    // Hub glow filter (stronger)
    const hubFilter = defs.append("filter").attr("id", "hub-glow");
    hubFilter
      .append("feGaussianBlur")
      .attr("stdDeviation", "5")
      .attr("result", "blur");
    const hubMerge = hubFilter.append("feMerge");
    hubMerge.append("feMergeNode").attr("in", "blur");
    hubMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Edges
    const edgeG = svg.append("g");
    edgeG
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("x1", (d: any) => d.source.x)
      .attr("y1", (d: any) => d.source.y)
      .attr("x2", (d: any) => d.target.x)
      .attr("y2", (d: any) => d.target.y)
      .attr("stroke", (d: any) => {
        if (d.weight < 0) return "#ef4444";
        const srcAct = d.source.act.x;
        const tgtAct = d.target.act.y;
        if (srcAct > 0.01 && tgtAct > 0.01) return "#a78bfa";
        return "#667eea";
      })
      .attr("stroke-width", (d: any) => 0.5 + 3 * d.normW)
      .attr("stroke-opacity", (d: any) => {
        const srcAct = d.source.act.x;
        const tgtAct = d.target.act.y;
        if (srcAct > 0.01 && tgtAct > 0.01) return 0.85;
        return 0.15 + 0.5 * d.normW;
      })
      .style("filter", (d: any) => {
        const srcAct = d.source.act.x;
        const tgtAct = d.target.act.y;
        return srcAct > 0.01 && tgtAct > 0.01 ? "url(#glow)" : "none";
      });

    // Edge weight labels — show on top edges (by normalized weight)
    // const labeledLinks = links.filter((d: any) => d.normW > 0.25);
    // const edgeLabelG = svg.append("g");
    // edgeLabelG
    //   .selectAll("text")
    //   .data(labeledLinks)
    //   .join("text")
    //   .attr("x", (d: any) => (d.source.x + d.target.x) / 2)
    //   .attr("y", (d: any) => (d.source.y + d.target.y) / 2 - 4)
    //   .attr("text-anchor", "middle")
    //   .attr("font-size", "7px")
    //   .attr("font-family", "'JetBrains Mono', monospace")
    //   .attr("font-weight", "600")
    //   .attr("fill", (d: any) => (d.weight < 0 ? "#fca5a5" : "#c4b5fd"))
    //   .attr("opacity", 0.9)
    //   .attr("pointer-events", "none")
    //   .text((d: any) => {
    //     const v = d.weight;
    //     return (v >= 0 ? "+" : "") + v.toFixed(3);
    //   });

    // Nodes
    const maxAct = Math.max(0.001, ...nodes.map((n) => n.act.x));
    const nodeG = svg.append("g");

    // Hub ring (drawn behind nodes)
    const hubNode = nodes.find((n) => n.isHub);
    if (hubNode) {
      nodeG
        .append("circle")
        .attr("cx", hubNode.x)
        .attr("cy", hubNode.y)
        .attr("r", 20)
        .attr("fill", "none")
        .attr("stroke", "#f59e0b")
        .attr("stroke-width", 2.5)
        .attr("stroke-dasharray", "4,3")
        .attr("opacity", 0.7)
        .style("filter", "url(#hub-glow)");
    }

    nodeG
      .selectAll("circle.node")
      .data(nodes)
      .join("circle")
      .classed("node", true)
      .attr("cx", (d: any) => d.x)
      .attr("cy", (d: any) => d.y)
      .attr("r", (d: any) => {
        if (d.isHub) return 12;
        const t = d.act.x / maxAct;
        return 5 + 9 * t;
      })
      .attr("fill", (d: any) => {
        if (d.isHub) return "#f59e0b";
        const t = Math.min(1, d.act.x / maxAct);
        if (d.act.x < 0.001) return "#334155";
        return d3.interpolateRgb("#334155", "#818cf8")(t);
      })
      .attr("stroke", (d: any) => {
        if (hoveredNode === d.id) return "#f59e0b";
        if (d.isHub) return "#fbbf24";
        if (d.act.x > 0.01 && d.act.y > 0.01) return "#a78bfa";
        return "#475569";
      })
      .attr("stroke-width", (d: any) =>
        hoveredNode === d.id || d.isHub ? 3 : 1.5,
      )
      .style("filter", (d: any) =>
        d.act.x > 0.01 || d.isHub ? "url(#glow)" : "none",
      )
      .style("cursor", "pointer")
      .on("mouseenter", (_e: any, d: any) => {
        hoveredNode = d.id;
        buildGraph(graphNeuronIds, currentSigma, currentActivations, hubInfo);
      })
      .on("mouseleave", () => {
        hoveredNode = null;
        buildGraph(graphNeuronIds, currentSigma, currentActivations, hubInfo);
      });

    // Node labels
    nodeG
      .selectAll("text.node-label")
      .data(nodes)
      .join("text")
      .classed("node-label", true)
      .attr("x", (d: any) => d.x)
      .attr("y", (d: any) => d.y + 1)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "central")
      .attr("font-size", (d: any) => (d.isHub ? "9px" : "8px"))
      .attr("font-family", "'JetBrains Mono', monospace")
      .attr("font-weight", "700")
      .attr("fill", (d: any) =>
        d.isHub ? "#1c1917" : d.act.x > 0.01 ? "white" : "#94a3b8",
      )
      .attr("pointer-events", "none")
      .text((d: any) => d.id);

    // Hub label
    if (hubNode) {
      svg
        .append("text")
        .attr("x", hubNode.x)
        .attr("y", hubNode.y - 22)
        .attr("text-anchor", "middle")
        .attr("font-size", "8px")
        .attr("font-family", "'JetBrains Mono', monospace")
        .attr("font-weight", "800")
        .attr("fill", "#fbbf24")
        .attr("pointer-events", "none")
        .text("HUB");
    }

    // Hovered neuron tooltip
    if (hoveredNode !== null) {
      const hn = nodes.find((n) => n.id === hoveredNode);
      if (hn) {
        const connEdges = links.filter(
          (l) => l.source.id === hoveredNode || l.target.id === hoveredNode,
        );
        const connCount = connEdges.length;
        const connStrength = connEdges.reduce(
          (s, l) => s + Math.abs(l.weight),
          0,
        );
        // Top 3 strongest connections
        const topConns = [...connEdges]
          .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
          .slice(0, 3);

        const tipLines = [
          `Neuron ${hoveredNode}${hn.isHub ? " ★ HUB" : ""}`,
          `x_act: ${hn.act.x.toFixed(4)}  y_act: ${hn.act.y.toFixed(4)}`,
          `Edges: ${connCount}  Σ|w|: ${connStrength.toFixed(3)}`,
          ...topConns.map((e) => {
            const other =
              e.source.id === hoveredNode ? e.target.id : e.source.id;
            return `  → N${other}: σ=${e.weight >= 0 ? "+" : ""}${e.weight.toFixed(4)}`;
          }),
        ];

        const tipW = 200;
        const tipH = 14 + tipLines.length * 15;
        // Keep tooltip within SVG bounds
        let tx = hn.x + 18;
        let ty = hn.y - 12;
        if (tx + tipW > width) tx = hn.x - tipW - 8;
        if (ty + tipH > height) ty = height - tipH - 4;
        if (ty < 0) ty = 4;

        const tipG = svg
          .append("g")
          .attr("transform", `translate(${tx}, ${ty})`);
        tipG
          .append("rect")
          .attr("width", tipW)
          .attr("height", tipH)
          .attr("rx", 6)
          .attr("fill", "rgba(10,10,28,0.94)")
          .attr(
            "stroke",
            hn.isHub ? "rgba(251,191,36,0.5)" : "rgba(100,120,200,0.5)",
          )
          .attr("stroke-width", 1);
        tipLines.forEach((txt, i) => {
          tipG
            .append("text")
            .attr("x", 8)
            .attr("y", 14 + i * 15)
            .attr("font-size", i === 0 ? "10px" : "8.5px")
            .attr("font-family", "'JetBrains Mono', monospace")
            .attr("font-weight", i === 0 ? "800" : "500")
            .attr(
              "fill",
              i === 0 ? (hn.isHub ? "#fbbf24" : "#818cf8") : "#cbd5e1",
            )
            .text(txt);
        });
      }
    }
  }

  // Reactive rebuild when data changes
  $: if (svgEl && graphNeuronIds.length) {
    buildGraph(graphNeuronIds, currentSigma, currentActivations, hubInfo);
  }

  onMount(() => {
    runPrompt();
  });
  onDestroy(() => {
    stopPlaying();
    if (simulation) simulation.stop();
  });

  // Examples
  const examples = ["2+2=", "Hello world", "The cat", "Why is", "ABCDE"];
</script>

<div class="hng">
  <!-- Model Upload -->

  <div class="hng-header">
    <h2>Hebbian Network Graph</h2>
    <p class="subtitle">
      Watch neurons wire together in real time as tokens flow through BDH
    </p>
  </div>

  {#if error}
    <div class="error-bar">{error} — backend may be waking up, please retry in 30s</div>
  {/if}

  <!-- Controls -->
  <div class="controls">
    <div class="prompt-row">
      <input
        bind:value={prompt}
        on:keydown={onKey}
        placeholder="Enter prompt…"
        disabled={loading}
      />
      <button class="btn primary" on:click={runPrompt} disabled={loading}>
        {loading ? "" : "▶"} Run
      </button>
    </div>
    <div class="example-row">
      {#each examples as ex}
        <button
          class="chip"
          on:click={() => {
            prompt = ex;
            runPrompt();
          }}>{ex}</button
        >
      {/each}
    </div>
  </div>

  {#if loading}
    <div class="loading">
      <div class="spinner"></div>
       Running BDH forward pass…
    </div>
  {/if}

  {#if apiData}
    <!-- Layer + playback controls -->
    <div class="toolbar">
      <div class="toolbar-section">
        <span class="toolbar-label">Layer:</span>
        {#each Array(nLayers) as _, i}
          <button
            class="btn small"
            class:active={selectedLayer === i}
            on:click={() => {
              selectedLayer = i;
              currentStep = 0;
              stopPlaying();
            }}
          >
            L{i}
          </button>
        {/each}
      </div>

      <div class="toolbar-section">
        <span class="toolbar-label">Step:</span>
        <button class="btn small" on:click={resetStep} title="Reset">⏮</button>
        <button class="btn small" on:click={togglePlay}
          >{playing ? "⏸" : "▶"}</button
        >
        <input
          type="range"
          class="step-slider"
          min="0"
          max={T - 1}
          bind:value={currentStep}
          on:input={() => stopPlaying()}
        />
        <span class="step-display">
          t={currentStep}
          <strong class="step-token">{tokens[currentStep] ?? "?"}</strong>
        </span>
      </div>

      <div class="toolbar-section">
        <button
          class="btn small"
          class:active={frozen}
          on:click={() => {
            frozen = !frozen;
          }}
        >
          {frozen ? "Unfreeze" : "Freeze"}
        </button>
        <span class="neuron-count"
          >{graphNeuronIds.length} of {totalActive} active neurons shown ({totalNeurons}
          total) · {currentSigma.size} edges</span
        >
      </div>
    </div>

    <!-- Token strip -->
    <div class="token-strip">
      {#each tokens as tok, i}
        <button
          class="tok"
          class:past={i < currentStep}
          class:current={i === currentStep}
          class:future={i > currentStep}
          on:click={() => {
            currentStep = i;
            stopPlaying();
          }}
        >
          <span class="tok-idx">t{i}</span>
          <span class="tok-char">{tok}</span>
        </button>
      {/each}
    </div>

    <!-- SVG Graph -->
    <div class="graph-container" style="width:{width}px;height:{height}px;">
      <svg bind:this={svgEl} viewBox="0 0 {width} {height}" {width} {height}
      ></svg>

      <!-- Legend overlay -->
      <div class="legend">
        <div class="legend-item">
          <span class="dot" style="background:#818cf8"></span> Active neuron
        </div>
        <div class="legend-item">
          <span class="dot" style="background:#334155"></span> Inactive neuron
        </div>
        <div class="legend-item">
          <span
            class="dot"
            style="background:#f59e0b; border: 2px dashed #fbbf24; box-sizing: border-box;"
          ></span> Hub (most connected)
        </div>
        <div class="legend-item">
          <span class="line-swatch glow"></span> Active edge (x·y &gt; 0)
        </div>
        <div class="legend-item">
          <span class="line-swatch dim"></span> Stored weight
        </div>
        <div class="legend-item">
          <span class="line-swatch neg"></span> Negative weight
        </div>
        <div class="legend-item" style="font-size:0.6rem;color:#c4b5fd;">
          Edge labels = σ weight values
        </div>
      </div>
    </div>

    <!-- Info panel -->
    <div class="info-panel">
      <div class="info-card">
        <div class="info-label">Current Token</div>
        <div class="info-value">
          {tokens[currentStep] ?? "?"}
          <span class="info-sub"
            >(byte {prompt.charCodeAt(currentStep) ?? "?"})</span
          >
        </div>
      </div>
      <div class="info-card">
        <div class="info-label">σ Edges Above Threshold</div>
        <div class="info-value">{currentSigma.size}</div>
      </div>
      <div class="info-card hub-card">
        <div class="info-label">Hub Neuron</div>
        {#if hubInfo}
          <div class="info-value hub-value">
            N{hubInfo.neuron_id}
            <span class="info-sub"
              >deg={hubInfo.degree} · Σ|w|={hubInfo.total_strength}</span
            >
          </div>
        {:else}
          <div class="info-value">—</div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .hng {
    max-width: 900px;
    margin: 0 auto;
  }
  .hng-header {
    text-align: center;
    margin-bottom: 16px;
  }
  .hng-header h2 {
    font-size: 1.4rem;
    margin: 0 0 4px;
    color: var(--text-heading);
  }
  .subtitle {
    color: var(--text-secondary);
    font-size: 0.88rem;
    margin: 0;
  }

  .error-bar {
    background: #2a1a1a;
    color: #fca5a5;
    border: 1px solid #5a2a2a;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.82rem;
    margin-bottom: 12px;
  }

  /* Controls */
  .controls {
    margin-bottom: 14px;
  }
  .prompt-row {
    display: flex;
    gap: 8px;
    margin-bottom: 6px;
  }
  .prompt-row input {
    flex: 1;
    padding: 9px 14px;
    border: 2px solid var(--border-color);
    border-radius: 8px;
    font-size: 0.9rem;
    font-family: "JetBrains Mono", monospace;
    outline: none;
    background: var(--bg-card);
    color: var(--text-heading);
  }
  .prompt-row input:focus {
    border-color: var(--accent);
  }
  .example-row {
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
  }
  .chip {
    padding: 3px 12px;
    border: 1px solid var(--border-color);
    border-radius: 20px;
    background: var(--bg-card);
    font-size: 0.72rem;
    cursor: pointer;
    color: var(--text-secondary);
  }
  .chip:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  .btn {
    padding: 9px 18px;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.85rem;
    cursor: pointer;
  }
  .btn.primary {
    background: linear-gradient(135deg, var(--accent), #764ba2);
    color: white;
  }
  .btn.primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn.small {
    padding: 5px 11px;
    font-size: 0.75rem;
    background: var(--bg-surface-alt);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-secondary);
  }
  .btn.small.active {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }

  .loading {
    text-align: center;
    padding: 30px;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border-color);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }

  /* Toolbar */
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    padding: 10px 14px;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    margin-bottom: 10px;
  }
  .toolbar-section {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .toolbar-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-secondary);
  }
  .step-slider {
    width: 120px;
    accent-color: var(--accent);
  }
  .step-display {
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-family: "JetBrains Mono", monospace;
  }
  .step-token {
    color: var(--accent);
  }
  .neuron-count {
    font-size: 0.7rem;
    color: var(--text-secondary);
  }

  /* Token strip */
  .token-strip {
    display: flex;
    gap: 4px;
    margin-bottom: 10px;
    flex-wrap: wrap;
  }
  .tok {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1px;
    padding: 4px 10px;
    border-radius: 8px;
    border: 2px solid transparent;
    cursor: pointer;
    background: var(--bg-surface);
    font-size: 0.78rem;
    transition: all 0.2s;
  }
  .tok.current {
    border-color: var(--accent);
    background: var(--bg-surface-alt);
  }
  .tok.past {
    background: var(--bg-surface-alt);
    opacity: 0.7;
  }
  .tok.future {
    opacity: 0.4;
  }
  .tok-idx {
    font-size: 0.55rem;
    color: var(--text-secondary);
  }
  .tok-char {
    font-family: "JetBrains Mono", monospace;
    font-weight: 700;
    color: var(--text-heading);
  }

  /* Graph */
  .graph-container {
    position: relative;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    background: var(--bg-surface);
    overflow: hidden;
    margin-bottom: 12px;
  }
  svg {
    display: block;
  }

  .legend {
    position: absolute;
    bottom: 10px;
    left: 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    background: rgba(10, 10, 28, 0.85);
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid rgba(100, 120, 200, 0.3);
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.65rem;
    color: var(--text-secondary);
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .line-swatch {
    width: 20px;
    height: 3px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .line-swatch.glow {
    background: #a78bfa;
    box-shadow: 0 0 6px #a78bfa;
  }
  .line-swatch.dim {
    background: var(--accent);
    opacity: 0.4;
  }
  .line-swatch.neg {
    background: #ef4444;
    opacity: 0.6;
  }

  /* Info panel */
  .info-panel {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 12px;
  }
  @media (max-width: 700px) {
    .info-panel {
      grid-template-columns: repeat(2, 1fr);
    }
  }
  .info-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
  }
  .info-label {
    font-size: 0.68rem;
    color: var(--text-secondary);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  .info-value {
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--text-heading);
    margin-top: 2px;
  }
  .info-sub {
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-weight: 400;
  }
  .hub-card {
    border-color: rgba(251, 191, 36, 0.3);
  }
  .hub-value {
    color: #f59e0b;
  }

  .insight {
    padding: 12px 16px;
    background: var(--bg-surface-alt);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.55;
  }
  .insight code {
    background: var(--border-color);
    padding: 1px 5px;
    border-radius: 3px;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.78rem;
    color: #a78bfa;
  }
</style>
