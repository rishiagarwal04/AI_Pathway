<script lang="ts">
  /**
   * ModelUpload — Upload a custom .pt checkpoint to visualize.
   *
   * Used in Explore and Experiment modes. Sends the file to
   * POST /api/upload_model, then triggers a callback so the
   * parent view can refresh its data.
   */
  import { createEventDispatcher } from 'svelte';
  import { apiUrl } from '../api';

  const dispatch = createEventDispatcher<{
    modelChanged: { source: string; info: any };
  }>();

  let uploading = false;
  let error: string | null = null;
  let successMsg: string | null = null;
  let modelSource: 'default' | 'custom' = 'default';
  let fileInput: HTMLInputElement;

  // Check current model source on mount
  async function checkModelSource() {
    try {
      const r = await fetch(apiUrl('/api/model/source'));
      if (r.ok) {
        const data = await r.json();
        modelSource = data.source;
      }
    } catch {}
  }

  // Initial check
  checkModelSource();

  async function handleFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.pt') && !file.name.endsWith('.pth')) {
      error = 'Please select a .pt or .pth file';
      return;
    }

    uploading = true;
    error = null;
    successMsg = null;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(apiUrl('/api/upload_model'), {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: `Server error ${res.status}` }));
        throw new Error(data.detail || `Upload failed (${res.status})`);
      }

      const data = await res.json();
      modelSource = 'custom';
      successMsg = data.message || 'Model loaded!';
      dispatch('modelChanged', { source: 'custom', info: data.model_info });
    } catch (e: any) {
      error = e.message ?? 'Upload failed';
    } finally {
      uploading = false;
      // Reset file input so same file can be re-selected
      if (fileInput) fileInput.value = '';
    }
  }

  async function resetModel() {
    uploading = true;
    error = null;
    successMsg = null;

    try {
      const res = await fetch(apiUrl('/api/reset_model'), { method: 'POST' });
      if (!res.ok) throw new Error(`Reset failed (${res.status})`);
      const data = await res.json();
      modelSource = 'default';
      successMsg = 'Reset to default model';
      dispatch('modelChanged', { source: 'default', info: data.model_info });
    } catch (e: any) {
      error = e.message ?? 'Reset failed';
    } finally {
      uploading = false;
    }
  }
</script>

<div class="upload-widget">
  <div class="upload-row">
    <div class="upload-label">
      <span class="upload-icon">📦</span>
      <span class="label-text">Model:</span>
      <span class="source-badge" class:custom={modelSource === 'custom'}>
        {modelSource === 'custom' ? 'Custom' : 'Default'}
      </span>
    </div>

    <label class="upload-btn" class:disabled={uploading}>
      <input
        type="file"
        accept=".pt,.pth"
        on:change={handleFileSelect}
        bind:this={fileInput}
        disabled={uploading}
      />
      {#if uploading}
        <span class="spinner"></span> Loading…
      {:else}
        ⬆ Upload .pt
      {/if}
    </label>

    {#if modelSource === 'custom'}
      <button class="reset-btn" on:click={resetModel} disabled={uploading}>
        ↩ Reset
      </button>
    {/if}
  </div>

  {#if error}
    <div class="upload-msg error-msg">⚠ {error}</div>
  {/if}
  {#if successMsg}
    <div class="upload-msg success-msg">✓ {successMsg}</div>
  {/if}
</div>

<style>
  .upload-widget {
    margin-bottom: 12px;
  }

  .upload-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }

  .upload-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85rem;
    color: var(--text-secondary);
  }

  .upload-icon {
    font-size: 1rem;
  }

  .label-text {
    font-weight: 600;
  }

  .source-badge {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(16, 185, 129, 0.12);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.25);
  }
  .source-badge.custom {
    background: rgba(99, 102, 241, 0.12);
    color: #6366f1;
    border-color: rgba(99, 102, 241, 0.25);
  }

  .upload-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 8px;
    border: 1px solid var(--border-color);
    background: var(--bg-card);
    color: var(--text-secondary);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .upload-btn:hover:not(.disabled) {
    border-color: #6366f1;
    background: rgba(99, 102, 241, 0.06);
    color: #6366f1;
  }
  .upload-btn.disabled {
    opacity: 0.6;
    cursor: wait;
  }
  .upload-btn input[type="file"] {
    display: none;
  }

  .reset-btn {
    padding: 6px 12px;
    border-radius: 8px;
    border: 1px solid var(--border-color);
    background: var(--bg-card);
    color: var(--text-muted);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .reset-btn:hover:not(:disabled) {
    border-color: #e11d48;
    color: #e11d48;
    background: rgba(225, 29, 72, 0.06);
  }
  .reset-btn:disabled {
    opacity: 0.5;
    cursor: wait;
  }

  .upload-msg {
    margin-top: 6px;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 500;
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

  .spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid rgba(99, 102, 241, 0.3);
    border-top-color: #6366f1;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
