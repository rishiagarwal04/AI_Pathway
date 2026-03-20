<script lang="ts">
  /**
   * ModelExplorer — Manage multiple BDH checkpoints trained on different tasks.
   *
   * Features:
   * - List all registered BDH models (default + custom uploads)
   * - Upload new .pt checkpoints with name, task type, and description
   * - Drag-and-drop upload zone
   * - View model configuration details
   * - Switch the active model
   * - Delete custom models
   */
  import { onMount, createEventDispatcher } from 'svelte';
  import { apiUrl } from '../api';

  const dispatch = createEventDispatcher<{
    modelChanged: { activeId: string };
  }>();

  // ── State ──
  interface ModelEntry {
    id: string;
    name: string;
    task: string;
    description: string;
    created_at: string;
    deletable: boolean;
    active: boolean;
    config: {
      n_layer: number;
      n_embd: number;
      n_head: number;
      vocab_size: number;
      mlp_internal_dim_multiplier: number;
      neuron_dim: number;
      total_params: number;
    } | null;
  }

  let models: ModelEntry[] = [];
  let selectedId: string | null = null;
  let loading = false;
  let error: string | null = null;
  let successMsg: string | null = null;

  // Upload form state
  let uploading = false;
  let dragOver = false;
  let fileInput: HTMLInputElement;

  const taskTypes = ['Arithmetic', 'Language', 'Game', 'Code', 'Music', 'Vision', 'General', 'Other'];

  // ── Lifecycle ──
  onMount(() => {
    fetchModels();
  });

  // ── API calls ──
  async function fetchModels() {
    loading = true;
    error = null;
    try {
      const res = await fetch(apiUrl('/api/models'));
      if (!res.ok) throw new Error(`Failed to fetch models (${res.status})`);
      models = await res.json();
      // Auto-select active model
      const active = models.find(m => m.active);
      if (active) selectedId = active.id;
    } catch (e: any) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  async function selectModel(id: string) {
    if (models.find(m => m.id === id)?.active) return; // already active
    loading = true;
    error = null;
    successMsg = null;
    try {
      const res = await fetch(apiUrl('/api/models/select'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: 'Failed' }));
        throw new Error(data.detail || `Switch failed (${res.status})`);
      }
      const data = await res.json();
      successMsg = data.message;
      dispatch('modelChanged', { activeId: id });
      await fetchModels();
    } catch (e: any) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  async function deleteModel(id: string) {
    if (!confirm('Delete this model? This cannot be undone.')) return;
    error = null;
    successMsg = null;
    try {
      const res = await fetch(apiUrl(`/api/models/${id}`), { method: 'DELETE' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: 'Failed' }));
        throw new Error(data.detail || `Delete failed (${res.status})`);
      }
      successMsg = 'Model deleted';
      if (selectedId === id) selectedId = 'default';
      await fetchModels();
    } catch (e: any) {
      error = e.message;
    }
  }

  async function uploadFile(file: File) {
    if (!file.name.endsWith('.pt') && !file.name.endsWith('.pth')) {
      error = 'Only .pt or .pth files are accepted';
      return;
    }
    uploading = true;
    error = null;
    successMsg = null;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', file.name.replace(/\.(pt|pth)$/, ''));
    formData.append('task', 'General');
    formData.append('description', '');

    try {
      const res = await fetch(apiUrl('/api/models/upload'), {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: `Upload failed (${res.status})` }));
        throw new Error(data.detail || `Upload failed (${res.status})`);
      }
      const data = await res.json();
      successMsg = data.message;
      await fetchModels();
    } catch (e: any) {
      error = e.message;
    } finally {
      uploading = false;
      if (fileInput) fileInput.value = '';
    }
  }

  // ── Drag & drop ──
  function handleDrop(e: DragEvent) {
    e.preventDefault();
    dragOver = false;
    const file = e.dataTransfer?.files?.[0];
    if (file) uploadFile(file);
  }
  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    dragOver = true;
  }
  function handleDragLeave() { dragOver = false; }
  function handleFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) uploadFile(file);
  }

  // ── Helpers ──
  $: selected = models.find(m => m.id === selectedId) ?? null;

  function formatParams(n: number): string {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
    return n.toString();
  }

  function formatDate(iso: string): string {
    try {
      return new Date(iso).toLocaleDateString('en-US', {
        year: 'numeric', month: 'short', day: 'numeric',
      });
    } catch { return iso; }
  }

  function taskColor(task: string): string {
    const colors: Record<string, string> = {
      'Arithmetic': '#f59e0b',
      'Language': '#6366f1',
      'Game': '#e11d48',
      'Code': '#059669',
      'Music': '#a855f7',
      'Vision': '#06b6d4',
      'General': '#64748b',
      'Other': '#78716c',
    };
    return colors[task] || '#64748b';
  }
</script>

<div class="model-explorer">
  <!-- Header -->
  <div class="explorer-header">
    <h2>BDH Model Explorer</h2>
    <p class="subtitle">Manage BDH checkpoints trained on different tasks</p>
  </div>

  {#if error}
    <div class="msg error-msg">⚠ {error}
      <button class="dismiss" on:click={() => error = null}>×</button>
    </div>
  {/if}
  {#if successMsg}
    <div class="msg success-msg">✓ {successMsg}
      <button class="dismiss" on:click={() => successMsg = null}>×</button>
    </div>
  {/if}

  <div class="explorer-body">
    <!-- Left: Model List -->
    <div class="model-list-panel">
      <h3 class="panel-title">Available Models</h3>

      {#if loading && models.length === 0}
        <div class="loading-placeholder">Loading models…</div>
      {:else}
        <div class="model-cards">
          {#each models as m (m.id)}
            <button
              class="model-card"
              class:selected={selectedId === m.id}
              class:active-model={m.active}
              on:click={() => (selectedId = m.id)}
            >
              <div class="card-top-row">
                {#if m.active}
                  <span class="active-badge">● Active</span>
                {/if}
              </div>
              <div class="card-name">{m.name}</div>
              {#if m.config}
                <div class="card-meta">
                  {m.config.n_layer}L · {m.config.n_head}H · {m.config.n_embd}D · {formatParams(m.config.total_params)} params
                </div>
              {/if}
            </button>
          {/each}
        </div>
      {/if}

      <!-- Upload zone -->
      <div class="upload-section">
        <h3 class="panel-title">Upload New Checkpoint</h3>
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <div
          class="drop-zone"
          class:drag-over={dragOver}
          class:uploading
          on:drop={handleDrop}
          on:dragover={handleDragOver}
          on:dragleave={handleDragLeave}
        >
          {#if uploading}
            <span class="spinner"></span>
            <span>Validating & uploading…</span>
          {:else}
            <span class="drop-icon">📂</span>
            <span>Drag & drop a <strong>.pt</strong> file here</span>
            <span class="drop-or">or</span>
            <label class="browse-btn">
              Browse files
              <input type="file" accept=".pt,.pth" on:change={handleFileSelect} bind:this={fileInput} />
            </label>
          {/if}
        </div>

        <p class="upload-note">
          <strong>Tip:</strong> For best visualization results, use <strong>NLP-based BDH models. </strong>
          Non-text models (e.g. Game of Life) can be inspected but may not support text prompting.
        </p>
      </div>
    </div>

    <!-- Right: Detail panel -->
    <div class="detail-panel">
      {#if selected}
        <div class="detail-header">
          <div class="detail-title-row">
            <h3>{selected.name}</h3>
          </div>
          {#if selected.description}
            <p class="detail-desc">{selected.description}</p>
          {/if}
        </div>

        {#if selected.config}
          <div class="config-grid">
            <div class="config-item">
              <span class="config-label">Layers</span>
              <span class="config-value">{selected.config.n_layer}</span>
            </div>
            <div class="config-item">
              <span class="config-label">Heads</span>
              <span class="config-value">{selected.config.n_head}</span>
            </div>
            <div class="config-item">
              <span class="config-label">Embed Dim</span>
              <span class="config-value">{selected.config.n_embd}</span>
            </div>
            <div class="config-item">
              <span class="config-label">Vocab Size</span>
              <span class="config-value">{selected.config.vocab_size}</span>
            </div>
            <div class="config-item">
              <span class="config-label">MLP Multiplier</span>
              <span class="config-value">{selected.config.mlp_internal_dim_multiplier}×</span>
            </div>
            <div class="config-item">
              <span class="config-label">Neuron Dim (N)</span>
              <span class="config-value">{selected.config.neuron_dim}</span>
            </div>
            <div class="config-item span-2">
              <span class="config-label">Total Parameters</span>
              <span class="config-value big">{formatParams(selected.config.total_params)}</span>
            </div>
          </div>

          <!-- Architecture diagram -->
          <div class="arch-diagram">
            <h4>Architecture</h4>
            <div class="arch-flow">
              <div class="arch-block embed">Embed<br/><span class="arch-dim">[{selected.config.vocab_size} → {selected.config.n_embd}]</span></div>
              <div class="arch-arrow">→</div>
              {#each Array(selected.config.n_layer) as _, i}
                <div class="arch-block layer">
                  Layer {i}
                  <span class="arch-dim">
                    Enc → RoPE Attn → Hebb → Dec
                  </span>
                </div>
                {#if i < selected.config.n_layer - 1}
                  <div class="arch-arrow">→</div>
                {/if}
              {/each}
              <div class="arch-arrow">→</div>
              <div class="arch-block head">LM Head<br/><span class="arch-dim">[{selected.config.n_embd} → {selected.config.vocab_size}]</span></div>
            </div>
          </div>
        {:else}
          <div class="no-config">
            <p>⚠ Could not read config from checkpoint file.</p>
          </div>
        {/if}

        <!-- Actions -->
        <div class="detail-actions">
          {#if !selected.active}
            <button class="btn-load" on:click={() => selectModel(selected.id)} disabled={loading}>
              {#if loading}
                <span class="spinner sm"></span> Switching…
              {:else}
                ⚡ Load This Model
              {/if}
            </button>
          {:else}
            <button class="btn-active" disabled>
              ✓ Currently Active
            </button>
          {/if}

          {#if selected.deletable}
            <button class="btn-delete" on:click={() => deleteModel(selected.id)} disabled={loading || selected.active}>
              🗑 Delete
            </button>
          {/if}
        </div>
      {:else}
        <div class="detail-empty">
          <span class="empty-icon">🔍</span>
          <p>Select a model to view details</p>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .model-explorer {
    max-width: 1100px;
    margin: 0 auto;
  }

  .explorer-header {
    margin-bottom: 20px;
  }
  .explorer-header h2 {
    margin: 0 0 4px;
    font-size: 1.4rem;
    color: var(--text-heading);
  }
  .subtitle {
    margin: 0;
    font-size: 0.88rem;
    color: var(--text-muted);
  }

  /* Messages */
  .msg {
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .error-msg {
    background: rgba(225, 29, 72, 0.08);
    color: #e11d48;
    border: 1px solid rgba(225, 29, 72, 0.2);
  }
  .success-msg {
    background: rgba(16, 185, 129, 0.08);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.2);
  }
  .dismiss {
    background: none; border: none; color: inherit;
    font-size: 1.2rem; cursor: pointer; padding: 0 4px;
  }

  /* Layout */
  .explorer-body {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 20px;
    align-items: start;
  }

  /* Model List Panel */
  .model-list-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .panel-title {
    margin: 0;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--text-heading);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .model-cards {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .model-card {
    text-align: left;
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid var(--border-color);
    background: var(--bg-card);
    cursor: pointer;
    transition: all 0.15s;
  }
  .model-card:hover {
    border-color: var(--accent);
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.08);
  }
  .model-card.selected {
    border-color: var(--accent);
    background: rgba(99, 102, 241, 0.06);
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15);
  }
  .model-card.active-model {
    border-left: 3px solid #059669;
  }

  .card-top-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }

  .task-badge {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 10px;
    background: color-mix(in srgb, var(--badge-color) 12%, transparent);
    color: var(--badge-color);
    border: 1px solid color-mix(in srgb, var(--badge-color) 25%, transparent);
  }
  .task-badge.large {
    font-size: 0.75rem;
    padding: 3px 10px;
  }

  .active-badge {
    font-size: 0.7rem;
    font-weight: 700;
    color: #059669;
  }

  .card-name {
    font-weight: 600;
    font-size: 0.92rem;
    color: var(--text-heading);
    margin-bottom: 2px;
  }
  .card-meta {
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  /* Upload */
  .upload-section {
    border-top: 1px solid var(--border-color);
    padding-top: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .drop-zone {
    border: 2px dashed var(--border-color);
    border-radius: 12px;
    padding: 24px 16px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    font-size: 0.85rem;
    color: var(--text-muted);
    transition: all 0.2s;
    cursor: default;
  }
  .drop-zone.drag-over {
    border-color: var(--accent);
    background: rgba(99, 102, 241, 0.05);
    color: var(--accent);
  }
  .drop-zone.uploading {
    opacity: 0.7;
    pointer-events: none;
  }
  .drop-icon { font-size: 1.5rem; }
  .drop-or {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin: 2px 0;
  }
  .browse-btn {
    display: inline-block;
    padding: 6px 16px;
    border: 1px solid var(--accent);
    border-radius: 6px;
    color: var(--accent);
    font-weight: 600;
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.15s;
  }
  .browse-btn:hover {
    background: rgba(99, 102, 241, 0.08);
  }
  .browse-btn input { display: none; }

  .upload-note {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin: 8px 0 0 0;
    line-height: 1.5;
    padding: 10px 12px;
    background: rgba(99, 102, 241, 0.06);
    border-radius: 8px;
    border-left: 3px solid var(--accent);
  }
  .upload-note strong {
    color: var(--accent);
  }

  /* Detail Panel */
  .detail-panel {
    border: 1px solid var(--border-color);
    border-radius: 12px;
    background: var(--bg-card);
    padding: 20px;
    min-height: 300px;
  }

  .detail-header {
    margin-bottom: 20px;
  }
  .detail-title-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
  }
  .detail-title-row h3 {
    margin: 0;
    font-size: 1.2rem;
    color: var(--text-heading);
  }
  .detail-desc {
    font-size: 0.88rem;
    color: var(--text-secondary);
    margin: 0 0 4px;
    line-height: 1.5;
  }
  .detail-date {
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  /* Config grid */
  .config-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 10px;
    margin-bottom: 20px;
  }
  .config-item {
    padding: 10px;
    background: var(--bg-primary);
    border-radius: 8px;
    border: 1px solid var(--border-color);
    text-align: center;
  }
  .config-item.span-2 { grid-column: span 2; }
  .config-label {
    display: block;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: var(--text-muted);
    margin-bottom: 4px;
  }
  .config-value {
    display: block;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-heading);
  }
  .config-value.big {
    font-size: 1.4rem;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* Architecture diagram */
  .arch-diagram {
    margin-bottom: 20px;
  }
  .arch-diagram h4 {
    margin: 0 0 10px;
    font-size: 0.85rem;
    color: var(--text-heading);
  }
  .arch-flow {
    display: flex;
    align-items: center;
    gap: 6px;
    overflow-x: auto;
    padding: 10px 0;
  }
  .arch-block {
    flex-shrink: 0;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    text-align: center;
    color: white;
    min-width: 70px;
  }
  .arch-block.embed { background: linear-gradient(135deg, #3b82f6, #06b6d4); }
  .arch-block.layer { background: linear-gradient(135deg, #667eea, #764ba2); }
  .arch-block.head  { background: linear-gradient(135deg, #059669, #34d399); }
  .arch-dim {
    display: block;
    font-size: 0.65rem;
    font-weight: 400;
    opacity: 0.85;
    margin-top: 2px;
  }
  .arch-arrow {
    color: var(--text-muted);
    font-size: 1rem;
    flex-shrink: 0;
  }

  .no-config {
    padding: 20px;
    text-align: center;
    color: var(--text-muted);
  }

  /* Actions */
  .detail-actions {
    display: flex;
    gap: 10px;
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid var(--border-color);
  }
  .btn-load {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 10px 24px;
    border: none;
    border-radius: 8px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    font-size: 0.9rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-load:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3); }
  .btn-load:disabled { opacity: 0.6; cursor: wait; }

  .btn-active {
    padding: 10px 24px;
    border: 1px solid #059669;
    border-radius: 8px;
    background: rgba(16, 185, 129, 0.08);
    color: #059669;
    font-size: 0.9rem;
    font-weight: 700;
    cursor: default;
  }

  .btn-delete {
    padding: 10px 18px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-card);
    color: var(--text-muted);
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-delete:hover:not(:disabled) {
    border-color: #e11d48;
    color: #e11d48;
    background: rgba(225, 29, 72, 0.06);
  }
  .btn-delete:disabled { opacity: 0.4; cursor: not-allowed; }

  .detail-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 250px;
    color: var(--text-muted);
  }
  .empty-icon { font-size: 2rem; margin-bottom: 8px; }

  .loading-placeholder {
    padding: 20px;
    text-align: center;
    color: var(--text-muted);
    font-size: 0.85rem;
  }

  /* Spinners */
  .spinner {
    display: inline-block;
    width: 18px; height: 18px;
    border: 2px solid rgba(99, 102, 241, 0.3);
    border-top-color: #6366f1;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }
  .spinner.sm { width: 14px; height: 14px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  @media (max-width: 800px) {
    .explorer-body {
      grid-template-columns: 1fr;
    }
  }
</style>
