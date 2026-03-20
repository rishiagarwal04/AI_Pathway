<script lang="ts">
  /**
   * ModelInternals — The main "Explore" view.
   *
   * Inspired by Transformer Explainer:
   *   1. Prompt input with instant-run on Enter
   *   2. Token strip (click to highlight)
   *   3. Embedding heatmap [T × D]
   *   4. Layer blocks (collapsible, showing dual BDH mechanism)
   *   5. Output logits bar chart + generated text
   */
  import { onMount } from 'svelte';
  import { apiUrl } from '../api';
  import MatrixHeatmap from './MatrixHeatmap.svelte';
  import RealTimeLayer from './RealTimeLayer.svelte';

  // ── state ──────────────────────────────────────────────────────────────
  let promptText   = '2+2=';
  let temperature  = 0.7;
  let maxNewTokens = 6;
  let topK         = 10;

  let loading      = false;
  let error: string | null = null;
  let result: any  = null;
  let modelInfo: any = null;

  let highlightToken = -1;   // token index hovered/clicked in token strip
  let expandedLayers: boolean[] = [];

  // ── init ──────────────────────────────────────────────────────────────
  onMount(async () => {
    try {
      const url = apiUrl('/api/model/info');
      console.log('Fetching model info from:', url);
      const r = await fetch(url);
      if (r.ok) {
        modelInfo = await r.json();
      } else {
        console.error('Model info failed:', r.status, r.statusText);
      }
    } catch (e) {
      console.error('Model info fetch error:', e);
    }
  });

  // ── run ───────────────────────────────────────────────────────────────
  async function run() {
    if (loading) return;
    loading = true;
    error = null;
    result = null;
    try {
      const res = await fetch(apiUrl('/api/run'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: promptText,
          temperature,
          max_new_tokens: maxNewTokens,
          top_k: topK,
        }),
      });
      if (!res.ok) throw new Error(`Server ${res.status}`);
      result = await res.json();
      // default: expand first layer only
      expandedLayers = result.layers.map((_: any, i: number) => i === 0);
    } catch (e: any) {
      error = e.message ?? 'Request failed';
    } finally {
      loading = false;
    }
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run(); }
  }

  function selectExample(p: string) {
    const shortcuts: Record<string, string> = {
      '2+2=4': '2+2=', 'How are you ?': 'How are', 'The capital of France is Paris': 'The capital of France is',
    };
    promptText = shortcuts[p] ?? p;
    run();
  }

  // ── output logit bars ─────────────────────────────────────────────────
  $: maxProb = result ? Math.max(...result.prob_bars.map((b: any) => b.prob), 0.01) : 1;

  // Format config
  $: cfgStr = modelInfo
    ? `${modelInfo.n_layer}L · d=${modelInfo.n_embd} · ${modelInfo.n_head}H · N=${modelInfo.neuron_dim}`
    : '';

</script>

<div class="mi-root">
  <!-- ── Top bar: prompt + controls ──────────────────────────────────── -->
  <div class="prompt-bar">
    <div class="prompt-left">
      <div class="prompt-input-wrap">
        <span class="prompt-icon"></span>
        <input
          class="prompt-input"
          bind:value={promptText}
          on:keydown={onKeyDown}
          placeholder="Type a prompt…"
          disabled={loading}
        />
        <button class="run-btn" on:click={run} disabled={loading}>
          {loading ? '…' : '▶ Run'}
        </button>
      </div>
      {#if modelInfo?.prompts}
        <div class="examples">
          {#each modelInfo.prompts as p}
            <button class="example-chip" on:click={() => selectExample(p)}>{p}</button>
          {/each}
        </div>
      {/if}
    </div>
    <div class="controls">
      <label class="ctrl-label">
        T={temperature.toFixed(1)}
        <input type="range" min="0.2" max="2" step="0.1" bind:value={temperature} />
      </label>
      <label class="ctrl-label">
        max={maxNewTokens}
        <input type="range" min="1" max="20" step="1" bind:value={maxNewTokens} />
      </label>
    </div>
    {#if cfgStr}
      <div class="model-badge">{cfgStr}</div>
    {/if}
  </div>

  {#if error}
    <div class="error-bar">⚠ {error} — backend may be waking up, please retry in 30s</div>
  {/if}

  <!-- ── Pipeline ─────────────────────────────────────────────────────── -->
  {#if result}
  <div class="pipeline">

    <!-- ── INPUT TOKENS ──────────────────────────────────────────────── -->
    <div class="stage-card">
      <div class="stage-header">
        <span class="stage-icon input-icon">➜</span>
        <span class="stage-title">Input Tokens</span>
        <span class="stage-dim">[T = {result.n_tokens}]</span>
      </div>
      <div class="token-strip">
        {#each result.tokens as tok, i}
          <button
            class="token-chip"
            class:active={highlightToken === i}
            on:click={() => highlightToken = highlightToken === i ? -1 : i}
          >{tok}</button>
        {/each}
      </div>
    </div>

    <!-- connector -->
    <div class="pipe-arrow">↓</div>

    <!-- ── EMBEDDING ─────────────────────────────────────────────────── -->
    <div class="stage-card">
      <div class="stage-header">
        <span class="stage-icon embed-icon">E</span>
        <span class="stage-title">Token Embedding + LayerNorm</span>
        <span class="stage-dim">[T × D] → [T × {result.config.d_show}] shown</span>
        <span class="stage-eq">x = LN(embed(token))</span>
      </div>
      <div class="embed-body">
        <MatrixHeatmap
          matrix={result.embedding.matrix}
          rowLabels={result.tokens}
          title="Embedding matrix  [T × D]"
          width={Math.min(520, result.config.d_show * 7)}
          height={Math.max(48, result.n_tokens * 20)}
          scheme="viridis"
          highlightRow={highlightToken}
        />
        <div class="embed-norms">
          <div class="norm-label">‖x‖ per token</div>
          {#each result.embedding.norms as n, i}
            <div class="norm-row" class:hlt={i === highlightToken}>
              <span class="norm-tok">{result.tokens[i]}</span>
              <div class="norm-bar-wrap">
                <div class="norm-bar" style="width:{Math.round(120 * n / (Math.max(...result.embedding.norms) || 1))}px"></div>
              </div>
              <span class="norm-val">{n.toFixed(2)}</span>
            </div>
          {/each}
        </div>
      </div>
    </div>

    <!-- ── LAYERS ─────────────────────────────────────────────────────── -->
    {#each result.layers as layer, i}
      <div class="pipe-arrow">↓</div>
      <RealTimeLayer
        index={i}
        {layer}
        tokens={result.tokens}
        bind:expanded={expandedLayers[i]}
        highlightToken={highlightToken}
      />
    {/each}

    <!-- connector -->
    <div class="pipe-arrow">↓</div>

    <!-- ── OUTPUT LOGITS ─────────────────────────────────────────────── -->
    <div class="stage-card">
      <div class="stage-header">
        <span class="stage-icon output-icon">→</span>
        <span class="stage-title">Output Logits</span>
        <span class="stage-dim">x · W_lm_head → top-10 tokens</span>
      </div>
      <div class="output-body">
        <div class="logit-bars">
          {#each result.prob_bars as bar}
            <div class="logit-row">
              <span class="logit-tok">{bar.token}</span>
              <div class="logit-bar-wrap">
                <div
                  class="logit-bar"
                  style="width:{Math.round(240 * bar.prob / maxProb)}px"
                  class:top={bar === result.prob_bars[0]}
                ></div>
              </div>
              <span class="logit-prob">{(bar.prob * 100).toFixed(1)}%</span>
            </div>
          {/each}
        </div>
        <div class="generated-box">
          <div class="gen-label">Generated</div>
          <div class="gen-text">
            <span class="gen-prompt">{result.prompt}</span><span class="gen-cont">{result.continuation}</span>
          </div>
        </div>
      </div>
    </div>

  </div><!-- /pipeline -->

  {:else if !loading}
  <div class="empty-state">
    <div class="empty-icon"></div>
    <p>Enter a prompt and press <strong>▶ Run</strong> to visualize the BDH forward pass</p>
    <p class="empty-sub">Every matrix operation at every layer will be shown in real time</p>
    {#if !modelInfo}
      <p class="backend-warn">
        ⚠ Backend not reachable.
        <button class="retry-btn" on:click={() => location.reload()}>Retry</button>
      </p>
    {/if}
  </div>
  {/if}

  {#if loading}
  <div class="loading-state">
    <div class="spinner"></div>
    <span>Running BDH forward pass…</span>
  </div>
  {/if}

</div>

<style>
  .mi-root {
    max-width: 1100px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  }

  /* ── prompt bar ── */
  .prompt-bar {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    padding: 18px 0 16px;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 20px;
  }
  .prompt-left { display: flex; flex-direction: column; gap: 8px; flex: 1; min-width: 260px; }
  .prompt-input-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-card);
    border: 2px solid var(--border-color);
    border-radius: 10px;
    padding: 6px 10px;
    transition: border-color 0.15s;
  }
  .prompt-input-wrap:focus-within { border-color: #667eea; }
  .prompt-icon { font-size: 1.1rem; }
  .prompt-input {
    flex: 1;
    border: none;
    outline: none;
    font-size: 0.95rem;
    font-family: 'JetBrains Mono', monospace;
    background: transparent;
    color: var(--text-heading);
  }
  .run-btn {
    padding: 6px 16px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 7px;
    font-size: 0.85rem;
    font-weight: 700;
    cursor: pointer;
    white-space: nowrap;
    transition: opacity 0.15s;
  }
  .run-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .examples {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .example-chip {
    padding: 3px 10px;
    border: 1px solid var(--border-light);
    border-radius: 20px;
    background: var(--bg-card);
    font-size: 0.72rem;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }
  .example-chip:hover { border-color: #667eea; color: #667eea; }

  .controls {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .ctrl-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.72rem;
    color: var(--text-muted);
    white-space: nowrap;
  }
  .ctrl-label input[type=range] { width: 80px; accent-color: #667eea; }
  .model-badge {
    font-size: 0.7rem;
    font-family: monospace;
    color: var(--text-muted);
    align-self: center;
    background: var(--bg-inset);
    padding: 4px 10px;
    border-radius: 20px;
    white-space: nowrap;
  }

  /* ── error ── */
  .error-bar {
    background: #fff3f3;
    color: #b91c1c;
    border: 1px solid #fca5a5;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.82rem;
    margin-bottom: 12px;
  }

  /* ── pipeline ── */
  .pipeline {
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
  .pipe-arrow {
    text-align: center;
    font-size: 1.1rem;
    color: #667eea;
    line-height: 1.2;
    margin: 4px 0;
  }

  /* ── stage cards ── */
  .stage-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    overflow: hidden;
  }
  .stage-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-color);
    flex-wrap: wrap;
  }
  .stage-icon {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 800;
    flex-shrink: 0;
  }
  .input-icon  { background: #dbeafe; color: #1d4ed8; }
  .embed-icon  { background: #ede9fe; color: #7c3aed; }
  .output-icon { background: #d1fae5; color: #065f46; }
  .stage-title { font-weight: 700; font-size: 0.88rem; color: var(--text-heading); }
  .stage-dim   { font-size: 0.7rem; color: var(--text-muted); font-family: monospace; }
  .stage-eq    { font-size: 0.7rem; color: var(--text-muted); font-style: italic; margin-left: auto; }

  /* ── tokens ── */
  .token-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 14px 16px;
  }
  .token-chip {
    padding: 5px 11px;
    background: var(--bg-inset);
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    color: var(--text-secondary);
  }
  .token-chip:hover { border-color: #667eea; color: #667eea; }
  .token-chip.active { background: #667eea; color: white; border-color: #667eea; }

  /* ── embedding ── */
  .embed-body {
    display: flex;
    align-items: flex-start;
    gap: 20px;
    padding: 14px 16px;
    flex-wrap: wrap;
  }
  .embed-norms {
    display: flex;
    flex-direction: column;
    gap: 5px;
    min-width: 160px;
  }
  .norm-label { font-size: 0.7rem; color: var(--text-muted); margin-bottom: 4px; font-weight: 600; }
  .norm-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .norm-row.hlt .norm-bar { background: #fbbf24; }
  .norm-tok {
    font-family: monospace;
    font-size: 0.75rem;
    color: var(--text-secondary);
    width: 16px;
    text-align: center;
    flex-shrink: 0;
  }
  .norm-bar-wrap { flex: 1; }
  .norm-bar {
    height: 10px;
    background: #667eea;
    border-radius: 3px;
    transition: width 0.3s;
  }
  .norm-val { font-family: monospace; font-size: 0.68rem; color: var(--text-muted); }

  /* ── output logits ── */
  .output-body {
    display: flex;
    gap: 24px;
    padding: 14px 16px;
    flex-wrap: wrap;
    align-items: flex-start;
  }
  .logit-bars {
    display: flex;
    flex-direction: column;
    gap: 5px;
    min-width: 280px;
  }
  .logit-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .logit-tok {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    width: 28px;
    text-align: center;
    background: var(--bg-inset);
    border-radius: 4px;
    padding: 1px 3px;
    flex-shrink: 0;
    color: var(--text-secondary);
  }
  .logit-bar-wrap { flex: 1; }
  .logit-bar {
    height: 14px;
    background: var(--border-light);
    border-radius: 3px;
    transition: width 0.4s;
  }
  .logit-bar.top { background: linear-gradient(90deg, #667eea, #764ba2); }
  .logit-prob {
    font-family: monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    width: 38px;
    text-align: right;
    flex-shrink: 0;
  }

  .generated-box {
    background: var(--bg-surface);
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    min-width: 180px;
  }
  .gen-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
  }
  .gen-text { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; }
  .gen-prompt { color: var(--text-secondary); }
  .gen-cont { color: #059669; font-weight: 700; }

  /* ── empty / loading ── */
  .empty-state, .loading-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-muted);
  }
  .empty-icon { font-size: 3rem; margin-bottom: 12px; }
  .empty-state p { margin: 6px 0; font-size: 0.92rem; }
  .empty-sub { color: var(--text-muted); font-size: 0.82rem; }
  .backend-warn {
    margin-top: 16px;
    font-size: 0.78rem;
    color: #b45309;
    background: #fef3c7;
    border: 1px solid #fcd34d;
    border-radius: 8px;
    display: inline-block;
    padding: 8px 16px;
  }
  .backend-warn code { font-family: monospace; background: transparent; }
  .loading-state {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    font-size: 0.88rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    width: 20px;
    height: 20px;
    border: 2px solid var(--border-color);
    border-top-color: #667eea;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  @media (max-width: 700px) {
    .output-body { flex-direction: column; }
    .embed-body  { flex-direction: column; }
  }
  .retry-btn { margin-left:8px; padding:2px 6px; border-radius:4px; border:1px solid #777; background:#333; color:#eee; cursor:pointer }
  .retry-btn:hover { background:#444; }
</style>
