<script lang="ts">
  /**
   * MatrixHeatmap — high-performance canvas-based heatmap.
   * Click to expand into a full-screen interactive modal where every cell
   * is large enough to hover and inspect individual values.
   */
  import { onMount, afterUpdate, tick } from 'svelte';

  // ── props ──────────────────────────────────────────────────────────────
  export let matrix: number[][] = [];     // [rows][cols], values 0-1
  export let rowLabels: string[] = [];    // optional token labels on Y axis
  export let colLabels: string[] = [];    // optional column labels on X axis
  export let title  = '';
  export let width  = 256;
  export let height = 256;
  /** Color scheme: 'blues' | 'viridis' | 'reds' | 'diverging' */
  export let scheme: 'blues' | 'viridis' | 'reds' | 'diverging' = 'viridis';
  /** If true, draw a thin grid between cells */
  export let grid   = false;
  /** Highlighted row index (-1 = none) */
  export let highlightRow = -1;
  /** Optional raw (un-normalised) matrix for tooltip display */
  export let rawMatrix: number[][] | null = null;

  // ── canvas refs ────────────────────────────────────────────────────────
  let canvas: HTMLCanvasElement;
  let modalCanvas: HTMLCanvasElement;
  let wrapper: HTMLDivElement;
  let modalContent: HTMLDivElement;

  // ── expanded modal state ───────────────────────────────────────────────
  let expanded = false;

  let tooltip = {
    show: false, x: 0, y: 0,
    row: 0, col: 0,
    rowLabel: '', colLabel: '',
    normVal: 0, rawVal: null as number | null,
    rgb: [0, 0, 0] as [number, number, number],
  };

  // ── color mapping ──────────────────────────────────────────────────────
  function colorFor(v: number): [number, number, number] {
    const t = Math.max(0, Math.min(1, v));
    if (scheme === 'blues') {
      return [
        Math.round(240 - 220 * t),
        Math.round(248 - 200 * t),
        255,
      ];
    }
    if (scheme === 'reds') {
      return [255, Math.round(245 - 230 * t), Math.round(245 - 230 * t)];
    }
    if (scheme === 'diverging') {
      if (t < 0.5) {
        const s = t * 2;
        return [Math.round(30 + 225 * s), Math.round(30 + 225 * s), 255];
      }
      const s = (t - 0.5) * 2;
      return [255, Math.round(255 - 225 * s), Math.round(255 - 225 * s)];
    }
    // viridis-like
    const stops: [number, number, number][] = [
      [68, 1, 84],
      [59, 82, 139],
      [33, 145, 140],
      [94, 201, 98],
      [253, 231, 37],
    ];
    const pos = t * (stops.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, stops.length - 1);
    const f = pos - lo;
    return [
      Math.round(stops[lo][0] + (stops[hi][0] - stops[lo][0]) * f),
      Math.round(stops[lo][1] + (stops[hi][1] - stops[lo][1]) * f),
      Math.round(stops[lo][2] + (stops[hi][2] - stops[lo][2]) * f),
    ];
  }

  // ── draw small thumbnail ──────────────────────────────────────────────
  function draw() {
    if (!canvas || !matrix.length) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rows = matrix.length;
    const cols = matrix[0].length;
    const cw = canvas.width / cols;
    const ch = canvas.height / rows;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = matrix[r][c];
        const [R, G, B] = colorFor(v);
        ctx.fillStyle = `rgb(${R},${G},${B})`;
        ctx.fillRect(c * cw, r * ch, cw, ch);
      }
    }

    if (highlightRow >= 0 && highlightRow < rows) {
      ctx.strokeStyle = 'rgba(255,255,100,0.85)';
      ctx.lineWidth = 2;
      ctx.strokeRect(0, highlightRow * ch, canvas.width, ch);
    }

    if (grid) {
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.lineWidth = 0.5;
      for (let r = 1; r < rows; r++) {
        ctx.beginPath(); ctx.moveTo(0, r * ch); ctx.lineTo(canvas.width, r * ch); ctx.stroke();
      }
      for (let c = 1; c < cols; c++) {
        ctx.beginPath(); ctx.moveTo(c * cw, 0); ctx.lineTo(c * cw, canvas.height); ctx.stroke();
      }
    }
  }

  // ── draw expanded modal canvas ─────────────────────────────────────────
  function drawModal() {
    if (!modalCanvas || !matrix.length) return;
    const ctx = modalCanvas.getContext('2d');
    if (!ctx) return;

    const rows = matrix.length;
    const cols = matrix[0].length;

    // Each cell at least 24px so values are visible; cap at reasonable size
    const CELL_MIN = 24;
    const CELL_MAX = 48;
    const maxW = window.innerWidth - 120;
    const maxH = window.innerHeight - 200;
    const cellW = Math.max(CELL_MIN, Math.min(CELL_MAX, Math.floor(maxW / cols)));
    const cellH = Math.max(CELL_MIN, Math.min(CELL_MAX, Math.floor(maxH / rows)));

    const canvasW = cols * cellW;
    const canvasH = rows * cellH;
    modalCanvas.width = canvasW;
    modalCanvas.height = canvasH;

    ctx.clearRect(0, 0, canvasW, canvasH);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = matrix[r][c];
        const [R, G, B] = colorFor(v);
        ctx.fillStyle = `rgb(${R},${G},${B})`;
        ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
      }
    }

    // Grid lines for expanded view (always on)
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 0.5;
    for (let r = 1; r < rows; r++) {
      ctx.beginPath(); ctx.moveTo(0, r * cellH); ctx.lineTo(canvasW, r * cellH); ctx.stroke();
    }
    for (let c = 1; c < cols; c++) {
      ctx.beginPath(); ctx.moveTo(c * cellW, 0); ctx.lineTo(c * cellW, canvasH); ctx.stroke();
    }

    // Highlight row
    if (highlightRow >= 0 && highlightRow < rows) {
      ctx.strokeStyle = 'rgba(255,255,100,0.85)';
      ctx.lineWidth = 2;
      ctx.strokeRect(0, highlightRow * cellH, canvasW, cellH);
    }

    // Row labels on left
    if (rowLabels.length) {
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let r = 0; r < Math.min(rows, rowLabels.length); r++) {
        const lbl = rowLabels[r] ?? `${r}`;
        const [R, G, B] = colorFor(matrix[r][0]);
        const bright = 0.299 * R + 0.587 * G + 0.114 * B;
        ctx.fillStyle = bright > 128 ? '#111' : '#eee';
        ctx.fillText(lbl.slice(0, 6), cellW - 2, r * cellH + cellH / 2);
      }
    }

    // Col labels on top
    if (colLabels.length) {
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillStyle = '#94a3b8';
    }
  }

  onMount(draw);
  afterUpdate(() => {
    draw();
    if (expanded) drawModal();
  });

  // ── open / close modal ─────────────────────────────────────────────────
  async function openExpanded() {
    expanded = true;
    tooltip = { ...tooltip, show: false };
    await tick();
    drawModal();
  }
  function closeExpanded() {
    expanded = false;
    tooltip = { ...tooltip, show: false };
  }
  function onBackdropClick(e: MouseEvent) {
    if (e.target === e.currentTarget) closeExpanded();
  }
  function onKeyDown(e: KeyboardEvent) {
    if (e.key === 'Escape') closeExpanded();
  }

  // ── hover tooltip (works for both small + modal) ───────────────────────
  function makeTooltip(e: MouseEvent, cvs: HTMLCanvasElement, container: HTMLElement) {
    if (!matrix.length) return;
    const rect = cvs.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const rows = matrix.length;
    const cols = matrix[0].length;
    const r = Math.floor((my / rect.height) * rows);
    const c = Math.floor((mx / rect.width) * cols);
    if (r >= 0 && r < rows && c >= 0 && c < cols) {
      const normVal = matrix[r][c];
      const raw = rawMatrix ? rawMatrix[r]?.[c] ?? null : null;
      const rgb = colorFor(normVal);
      const wrapRect = container.getBoundingClientRect();
      let tx = e.clientX - wrapRect.left + 14;
      let ty = e.clientY - wrapRect.top - 10;
      if (tx + 240 > wrapRect.width) tx = tx - 260;
      if (ty + 120 > wrapRect.height) ty = ty - 120;
      if (ty < 4) ty = 4;
      tooltip = {
        show: true, x: tx, y: ty,
        row: r, col: c,
        rowLabel: rowLabels[r] ?? `row ${r}`,
        colLabel: colLabels[c] ?? `col ${c}`,
        normVal, rawVal: raw,
        rgb: rgb as [number, number, number],
      };
    }
  }

  function onMouseMove(e: MouseEvent) {
    if (!wrapper) return;
    makeTooltip(e, canvas, wrapper);
  }
  function onModalMouseMove(e: MouseEvent) {
    if (!modalContent) return;
    makeTooltip(e, modalCanvas, modalContent);
  }
  function onMouseLeave() { tooltip = { ...tooltip, show: false }; }
</script>

<svelte:window on:keydown={onKeyDown} />

<div class="heatmap-wrap" style="width:{width}px" bind:this={wrapper}>
  {#if title}
    <div class="hm-title">{title}</div>
  {/if}
  <div class="canvas-container" style="width:{width}px; height:{height}px">
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <canvas
      bind:this={canvas}
      {width}
      {height}
      on:mousemove={onMouseMove}
      on:mouseleave={onMouseLeave}
      on:click={openExpanded}
    ></canvas>
    <div class="click-hint">click to expand</div>
    {#if tooltip.show && !expanded}
      <div class="tip" style="left:{tooltip.x}px; top:{tooltip.y}px">
        {#if title}
          <div class="tip-head">{title}</div>
        {/if}
        <div class="tip-pos">
          <span class="tip-label">Row</span>
          <span class="tip-val">[{tooltip.row}] {tooltip.rowLabel}</span>
        </div>
        <div class="tip-pos">
          <span class="tip-label">Col</span>
          <span class="tip-val">[{tooltip.col}] {tooltip.colLabel}</span>
        </div>
        <div class="tip-divider"></div>
        <div class="tip-value-row">
          <span class="tip-swatch" style="background:rgb({tooltip.rgb[0]},{tooltip.rgb[1]},{tooltip.rgb[2]})"></span>
          <span class="tip-label">Norm</span>
          <span class="tip-num">{tooltip.normVal.toFixed(4)}</span>
        </div>
        {#if tooltip.rawVal !== null}
          <div class="tip-value-row">
            <span class="tip-label">Raw</span>
            <span class="tip-num">{tooltip.rawVal.toFixed(5)}</span>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<!-- ═══ EXPANDED MODAL ═══ -->
{#if expanded}
  <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
  <div class="modal-backdrop" on:click={onBackdropClick}>
    <div class="modal-content" bind:this={modalContent}>
      <div class="modal-header">
        <h3>{title || 'Matrix'}</h3>
        <span class="modal-dims">
          {matrix.length} × {matrix[0]?.length || 0}
        </span>
        <button class="modal-close" on:click={closeExpanded}>✕</button>
      </div>
      <!-- Labels bar -->
      {#if colLabels.length && matrix[0]}
        <div class="col-labels-bar" style="padding-left:52px">
          {#each matrix[0] as _, c}
            <span class="col-lbl" style="width:{Math.max(24, Math.min(48, Math.floor((window.innerWidth - 120) / (matrix[0]?.length || 1))))}px">
              {colLabels[c] ?? c}
            </span>
          {/each}
        </div>
      {/if}
      <div class="modal-body">
        <!-- Row labels -->
        {#if rowLabels.length}
          <div class="row-labels-col">
            {#each matrix as _, r}
              <span class="row-lbl" style="height:{Math.max(24, Math.min(48, Math.floor((window.innerHeight - 200) / matrix.length)))}px">
                {rowLabels[r] ?? r}
              </span>
            {/each}
          </div>
        {/if}
        <div class="modal-canvas-wrap">
          <canvas
            bind:this={modalCanvas}
            on:mousemove={onModalMouseMove}
            on:mouseleave={onMouseLeave}
          ></canvas>
        </div>
      </div>
      <!-- Tooltip in modal -->
      {#if tooltip.show}
        <div class="tip modal-tip" style="left:{tooltip.x}px; top:{tooltip.y}px">
          {#if title}
            <div class="tip-head">{title}</div>
          {/if}
          <div class="tip-pos">
            <span class="tip-label">Row</span>
            <span class="tip-val">[{tooltip.row}] {tooltip.rowLabel}</span>
          </div>
          <div class="tip-pos">
            <span class="tip-label">Col</span>
            <span class="tip-val">[{tooltip.col}] {tooltip.colLabel}</span>
          </div>
          <div class="tip-divider"></div>
          <div class="tip-value-row">
            <span class="tip-swatch" style="background:rgb({tooltip.rgb[0]},{tooltip.rgb[1]},{tooltip.rgb[2]})"></span>
            <span class="tip-label">Norm</span>
            <span class="tip-num">{tooltip.normVal.toFixed(4)}</span>
          </div>
          {#if tooltip.rawVal !== null}
            <div class="tip-value-row">
              <span class="tip-label">Raw</span>
              <span class="tip-num">{tooltip.rawVal.toFixed(5)}</span>
            </div>
          {/if}
        </div>
      {/if}
      <!-- Color legend -->
      <div class="modal-legend">
        <span class="legend-lo">0.0</span>
        <div class="legend-gradient {scheme}"></div>
        <span class="legend-hi">1.0</span>
      </div>
    </div>
  </div>
{/if}

<style>
  .heatmap-wrap {
    display: inline-flex;
    flex-direction: column;
    gap: 4px;
    position: relative;
  }
  .hm-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-align: center;
    letter-spacing: 0.4px;
    text-transform: uppercase;
  }
  .canvas-container {
    position: relative;
    overflow: visible;
    border-radius: 4px;
  }
  canvas {
    display: block;
    width: 100%;
    height: 100%;
    cursor: pointer;
    image-rendering: pixelated;
  }
  .click-hint {
    position: absolute;
    bottom: 4px;
    right: 6px;
    font-size: 0.58rem;
    color: rgba(148, 163, 184, 0.6);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s;
  }
  .canvas-container:hover .click-hint {
    opacity: 1;
  }
  /* ── Rich tooltip ── */
  .tip {
    position: absolute;
    background: rgba(10, 10, 28, 0.96);
    color: #e2e8f0;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 8px 10px;
    border-radius: 6px;
    pointer-events: none;
    white-space: nowrap;
    z-index: 9999;
    border: 1px solid rgba(100,120,200,0.5);
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    display: flex;
    flex-direction: column;
    gap: 3px;
    min-width: 170px;
  }
  .modal-tip {
    z-index: 100001;
  }
  .tip-head {
    font-size: 0.65rem;
    font-weight: 700;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding-bottom: 3px;
    border-bottom: 1px solid rgba(100,120,200,0.25);
    margin-bottom: 1px;
  }
  .tip-pos {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .tip-label {
    font-size: 0.62rem;
    color: #94a3b8;
    min-width: 28px;
  }
  .tip-val {
    color: #cbd5e1;
  }
  .tip-divider {
    height: 1px;
    background: rgba(100,120,200,0.2);
    margin: 2px 0;
  }
  .tip-value-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .tip-swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    border: 1px solid rgba(255,255,255,0.2);
    flex-shrink: 0;
  }
  .tip-num {
    font-weight: 700;
    color: #f1f5f9;
    font-size: 0.78rem;
  }

  /* ═══ MODAL ═══ */
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.82);
    z-index: 99999;
    display: flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(6px);
    animation: fadeIn 0.18s ease-out;
  }
  @keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  .modal-content {
    position: relative;
    background: #0f1129;
    border: 1px solid rgba(100,120,200,0.35);
    border-radius: 12px;
    padding: 16px;
    max-width: 95vw;
    max-height: 92vh;
    display: flex;
    flex-direction: column;
    gap: 8px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    animation: scaleIn 0.18s ease-out;
  }
  @keyframes scaleIn {
    from { transform: scale(0.92); opacity: 0; }
    to   { transform: scale(1);    opacity: 1; }
  }
  .modal-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(100,120,200,0.2);
  }
  .modal-header h3 {
    margin: 0;
    font-size: 1rem;
    color: #e2e8f0;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .modal-dims {
    font-size: 0.75rem;
    color: #818cf8;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(129,140,248,0.12);
    padding: 2px 8px;
    border-radius: 4px;
  }
  .modal-close {
    margin-left: auto;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.3);
    color: #f87171;
    font-size: 1.1rem;
    width: 32px;
    height: 32px;
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
  }
  .modal-close:hover {
    background: rgba(239,68,68,0.3);
    color: #fca5a5;
  }
  .col-labels-bar {
    display: flex;
    gap: 0;
    overflow: hidden;
  }
  .col-lbl {
    font-size: 0.55rem;
    color: #64748b;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .modal-body {
    display: flex;
    gap: 0;
    overflow: auto;
    max-height: calc(92vh - 140px);
  }
  .row-labels-col {
    display: flex;
    flex-direction: column;
    gap: 0;
    flex-shrink: 0;
    padding-right: 4px;
    min-width: 48px;
  }
  .row-lbl {
    font-size: 0.58rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .modal-canvas-wrap {
    overflow: auto;
    border: 1px solid rgba(100,120,200,0.2);
    border-radius: 4px;
  }
  .modal-canvas-wrap canvas {
    cursor: crosshair;
    image-rendering: pixelated;
    display: block;
  }
  .modal-legend {
    display: flex;
    align-items: center;
    gap: 8px;
    justify-content: center;
    padding-top: 6px;
  }
  .legend-lo, .legend-hi {
    font-size: 0.65rem;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
  }
  .legend-gradient {
    width: 180px;
    height: 12px;
    border-radius: 3px;
    border: 1px solid rgba(255,255,255,0.1);
  }
  .legend-gradient.viridis {
    background: linear-gradient(to right, rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37));
  }
  .legend-gradient.blues {
    background: linear-gradient(to right, rgb(240,248,255), rgb(20,48,255));
  }
  .legend-gradient.reds {
    background: linear-gradient(to right, rgb(255,245,245), rgb(255,15,15));
  }
  .legend-gradient.diverging {
    background: linear-gradient(to right, rgb(30,30,255), rgb(255,255,255), rgb(255,30,30));
  }
</style>
