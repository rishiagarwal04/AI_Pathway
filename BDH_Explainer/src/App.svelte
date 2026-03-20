<script lang="ts">
  import { onMount } from "svelte";

  // ── Explore mode ──
  import ModelInternals from "./components/ModelInternals.svelte";
  import ModelExplorer from "./components/ModelExplorer.svelte";

  import BDHProtocol from "./components/BDHProtocol.svelte";
  import StateVsFastWeights from "./components/StateVsFastWeights.svelte";
  import AttentionAsLogic from "./components/AttentionAsLogic.svelte";


  // ── New Tier 3: Interactive Tools ──
  import ConceptActivationExplorer from "./components/ConceptActivationExplorer.svelte";
  import AblationPlayground from "./components/AblationPlayground.svelte";
  import HebbianNetworkGraph from "./components/HebbianNetworkGraph.svelte";

  // -------- Theme toggle --------
  let darkMode = false;

  onMount(() => {
    const stored = localStorage.getItem("bdh-theme");
    if (stored === "dark") {
      darkMode = true;
    } else if (stored === "light") {
      darkMode = false;
    } else {
      darkMode = window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    applyTheme();
  });

  function toggleTheme() {
    darkMode = !darkMode;
    localStorage.setItem("bdh-theme", darkMode ? "dark" : "light");
    applyTheme();
  }

  function applyTheme() {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }

  // -------- Navigation modes --------
  type Mode = "learn" | "explore" | "experiment";
  type LearnPage =
    | "welcome"
    | "modus-ponens"
    | "protocol"
    | "state-comparison"
    | "attention-logic"
    | "brain-mapping";
  type ExploreTab = "layer-explorer" | "model-explorer";
  type ExperimentPage = "concept-explorer" | "ablation" | "hebbian-graph";

  let mode: Mode = "learn";
  let learnPage: LearnPage = "welcome";
  let exploreTab: ExploreTab = "layer-explorer";
  let experimentPage: ExperimentPage = "concept-explorer";

  const exploreTabs: { id: ExploreTab; label: string }[] = [
    { id: "layer-explorer", label: "Layer Explorer" },
    { id: "model-explorer", label: "Model Explorer" },
  ];

  // Learn-mode pages in order — systematic syllabus
  const learnPages: { id: LearnPage; label: string }[] = [
    { id: "welcome", label: "Welcome" },
    // { id: "modus-ponens", label: "1. Foundations" },
    { id: "protocol", label: "1. The Protocol" },
    { id: "state-comparison", label: "2. State & Memory" },
    { id: "attention-logic", label: "3. Attention" },
    // { id: "brain-mapping", label: "4. Biology" },
  ];
  const experimentPages: { id: ExperimentPage; label: string }[] = [
    { id: "concept-explorer", label: "Concept Explorer" },
    { id: "ablation", label: "Ablation Playground" },
    { id: "hebbian-graph", label: "Hebbian Graph" },
  ];

  // -------- Learn mode navigation --------
  function nextLearnPage() {
    const idx = learnPages.findIndex((p) => p.id === learnPage);
    if (idx < learnPages.length - 1) learnPage = learnPages[idx + 1].id;
  }
  function prevLearnPage() {
    const idx = learnPages.findIndex((p) => p.id === learnPage);
    if (idx > 0) learnPage = learnPages[idx - 1].id;
  }
  $: learnPageIdx = learnPages.findIndex((p) => p.id === learnPage);
</script>

<div class="app">
  <!-- Top navigation bar -->
  <nav class="top-nav">
    <div class="nav-brand">
      <span class="brand-text">BDH Explainer</span>
    </div>
    <div class="mode-tabs">
      <button
        class="mode-tab"
        class:active={mode === "learn"}
        on:click={() => (mode = "learn")}
      >
        Learn
      </button>
      <button
        class="mode-tab"
        class:active={mode === "explore"}
        on:click={() => (mode = "explore")}
      >
        Explore BDH-GPU
      </button>
      <button
        class="mode-tab"
        class:active={mode === "experiment"}
        on:click={() => (mode = "experiment")}
      >
        Experiment
      </button>
    </div>
    <div class="nav-spacer"></div>
    <button
      class="theme-toggle"
      on:click={toggleTheme}
      title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
    >
      {#if darkMode}
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          ><circle cx="12" cy="12" r="5" /><line
            x1="12"
            y1="1"
            x2="12"
            y2="3"
          /><line x1="12" y1="21" x2="12" y2="23" /><line
            x1="4.22"
            y1="4.22"
            x2="5.64"
            y2="5.64"
          /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" /><line
            x1="1"
            y1="12"
            x2="3"
            y2="12"
          /><line x1="21" y1="12" x2="23" y2="12" /><line
            x1="4.22"
            y1="19.78"
            x2="5.64"
            y2="18.36"
          /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" /></svg
        >
      {:else}
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          ><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg
        >
      {/if}
    </button>
  </nav>

  <!-- Sub-navigation -->
  <div class="sub-nav">
    {#if mode === "learn"}
      {#each learnPages as page}
        <button
          class="sub-tab"
          class:active={learnPage === page.id}
          on:click={() => (learnPage = page.id)}>{page.label}</button
        >
      {/each}
    {:else if mode === "explore"}
      {#each exploreTabs as tab}
        <button
          class="sub-tab"
          class:active={exploreTab === tab.id}
          on:click={() => (exploreTab = tab.id)}>{tab.label}</button
        >
      {/each}
    {:else}
      {#each experimentPages as page}
        <button
          class="sub-tab"
          class:active={experimentPage === page.id}
          on:click={() => (experimentPage = page.id)}>{page.label}</button
        >
      {/each}
    {/if}
  </div>

  <main class="main">
    <!-- ═══════ LEARN MODE ═══════ -->
    {#if mode === "learn"}
      {#if learnPage === "welcome"}
        <div class="splash">
          <h2 class="splash-title">
            Welcome to the <span class="gradient-text">BDH Explainer</span>
          </h2>
          <p>
            The <strong class="accent-bdh">Baby Dragon Hatchling (BDH)</strong>
            is a biologically-grounded language model that combines
            <span class="accent-hebb">Hebbian learning</span>,
            <span class="accent-evo">evolutionary game theory</span>, and
            <span class="accent-logic">logical inference</span> into a single architecture.
            Each chapter below teaches a key concept and contrasts it with the standard
            Transformer approach.
          </p>
          <div class="splash-features">
            <div
              class="feat feat-2"
              on:click={() => (learnPage = "protocol")}
              role="button"
              tabindex="0"
            >
              <span class="ch-num ch-2">1</span>
              <span class="ch-text"
                ><strong>Protocol</strong> — 4-Round Update vs Feed-Forward Block</span
              >
            </div>
            <div
              class="feat feat-3"
              on:click={() => (learnPage = "state-comparison")}
              role="button"
              tabindex="0"
            >
              <span class="ch-num ch-3">2</span>
              <span class="ch-text"
                ><strong>Memory</strong> — Synaptic State O(N²) vs KV-Cache O(T)</span
              >
            </div>
            <div
              class="feat feat-4"
              on:click={() => (learnPage = "attention-logic")}
              role="button"
              tabindex="0"
            >
              <span class="ch-num ch-4">3</span>
              <span class="ch-text"
                ><strong>Attention</strong> — Linear Hebbian vs Softmax QKV</span
              >
            </div>
            <!-- <div
              class="feat feat-5"
              on:click={() => (learnPage = "brain-mapping")}
              role="button"
              tabindex="0"
            >
              <span class="ch-num ch-5">4</span>
              <span class="ch-text"
                ><strong>Biology</strong> — Brain ↔ BDH ↔ Transformer</span
              >
            </div> -->
          </div>
          <button
            class="btn-start"
            on:click={() => (learnPage = "protocol")}
          >
            Start Learning →
          </button>
        </div>
      {:else if learnPage === "protocol"}
        <BDHProtocol />
      {:else if learnPage === "state-comparison"}
        <StateVsFastWeights />
      {:else if learnPage === "attention-logic"}
        <AttentionAsLogic />
      {/if}

      <!-- Learn navigation footer -->
      {#if learnPage !== "welcome"}
        <div class="learn-nav-footer">
          <button
            class="btn-nav"
            on:click={prevLearnPage}
            disabled={learnPageIdx <= 0}
          >
            ← Previous
          </button>
          <span class="page-indicator">
            {learnPageIdx} / {learnPages.length -1}
          </span>
          <button
            class="btn-nav"
            on:click={nextLearnPage}
            disabled={learnPageIdx >= learnPages.length - 1}
          >
            Next →
          </button>
        </div>
      {/if}

      <!-- ═══════ EXPLORE MODE ═══════ -->
    {:else if mode === "explore"}
      {#if exploreTab === "layer-explorer"}
        <ModelInternals />
      {:else if exploreTab === "model-explorer"}
        <ModelExplorer on:modelChanged={() => { /* force re-mount if switching back */ exploreTab = 'layer-explorer'; setTimeout(() => { exploreTab = 'model-explorer'; }, 0); }} />
      {/if}

      <!-- ═══════ EXPERIMENT MODE ═══════ -->
    {:else if mode === "experiment"}
      {#if experimentPage === "concept-explorer"}
        <ConceptActivationExplorer />
      {:else if experimentPage === "ablation"}
        <AblationPlayground />
      {:else if experimentPage === "hebbian-graph"}
        <HebbianNetworkGraph />
      {/if}
    {/if}
  </main>
</div>

<style>
  :global(body) {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
  }

  .app {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    transition: background 0.3s;
  }

  /* ── Top navigation ── */
  .top-nav {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 24px;
    background: var(--nav-bg);
    color: var(--nav-text);
    box-shadow: var(--shadow-nav);
    position: sticky;
    top: 0;
    z-index: 100;
    border-bottom: 1px solid var(--nav-border);
  }
  .nav-brand {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .brand-text {
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
    color: var(--nav-text);
  }
  .nav-spacer {
    flex: 1;
  }

  .mode-tabs {
    display: flex;
    gap: 4px;
    background: var(--nav-pill-bg);
    border-radius: 10px;
    padding: 3px;
  }
  .mode-tab {
    padding: 7px 18px;
    border: none;
    border-radius: 8px;
    background: transparent;
    color: var(--nav-text-dim);
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .mode-tab:hover {
    color: var(--nav-text);
    background: rgba(255, 255, 255, 0.1);
  }
  .mode-tab.active {
    background: rgba(102, 126, 234, 0.75);
    color: white;
  }

  /* Theme toggle */
  .theme-toggle {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.08);
    color: rgba(255, 255, 255, 0.8);
    cursor: pointer;
    transition: all 0.2s;
    flex-shrink: 0;
  }
  .theme-toggle:hover {
    background: rgba(255, 255, 255, 0.15);
    color: white;
  }

  /* ── Sub-navigation ── */
  .sub-nav {
    display: flex;
    gap: 4px;
    padding: 8px 24px;
    background: var(--subnav-bg);
    border-bottom: 1px solid var(--subnav-border);
    overflow-x: auto;
    flex-wrap: nowrap;
    transition: background 0.3s;
  }
  .sub-tab {
    padding: 6px 14px;
    border: 1px solid var(--subtab-border);
    border-radius: 8px;
    background: var(--subtab-bg);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s;
    color: var(--subtab-text);
  }
  .sub-tab:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .sub-tab.active {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }

  .main {
    flex: 1;
    padding: 24px 24px 80px;
  }

  /* Splash / welcome */
  .splash {
    text-align: center;
    max-width: 720px;
    margin: 40px auto 0;
  }
  .splash-title {
    margin-bottom: 10px;
    font-size: 1.7rem;
    color: var(--text-heading);
  }
  .gradient-text {
    background: linear-gradient(135deg, #667eea, #764ba2, #e11d48);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .accent-bdh {
    color: #667eea;
  }
  .accent-hebb {
    color: #a78bfa;
  }
  .accent-evo {
    color: #f472b6;
  }
  .accent-logic {
    color: #34d399;
  }
  .splash p {
    color: var(--text-secondary);
    line-height: 1.7;
    margin-bottom: 28px;
    font-size: 1rem;
  }
  .splash-features {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    text-align: left;
    margin-bottom: 28px;
  }
  .feat {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.9em;
    color: var(--text-secondary);
    padding: 12px 16px;
    background: var(--bg-card);
    border-radius: 12px;
    border: 1px solid var(--border-color);
    cursor: pointer;
    transition: all 0.2s;
  }
  .feat:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
  }
  .ch-num {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    font-weight: 800;
    font-size: 0.85rem;
    color: #fff;
    flex-shrink: 0;
  }
  .ch-1 {
    background: linear-gradient(135deg, #667eea, #764ba2);
  }
  .ch-2 {
    background: linear-gradient(135deg, #764ba2, #e11d48);
  }
  .ch-3 {
    background: linear-gradient(135deg, #059669, #34d399);
  }
  .ch-4 {
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
  }
  .ch-5 {
    background: linear-gradient(135deg, #d97706, #f59e0b);
  }
  .feat-1 {
    border-left: 3px solid #667eea;
  }
  .feat-2 {
    border-left: 3px solid #e11d48;
  }
  .feat-3 {
    border-left: 3px solid #059669;
  }
  .feat-4 {
    border-left: 3px solid #3b82f6;
  }
  .feat-5 {
    border-left: 3px solid #d97706;
  }
  .ch-text strong {
    color: var(--text-heading);
  }

  .btn-start {
    padding: 12px 32px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    transition: transform 0.15s;
  }
  .btn-start:hover {
    transform: scale(1.03);
  }

  /* Learn nav footer */
  .learn-nav-footer {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    margin-top: 32px;
    padding-top: 20px;
    border-top: 1px solid var(--border-color);
  }
  .btn-nav {
    padding: 8px 20px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-card);
    color: var(--text-secondary);
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-nav:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--accent);
  }
  .btn-nav:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .page-indicator {
    font-size: 0.85rem;
    color: var(--text-muted);
  }

  @media (max-width: 700px) {
    .top-nav {
      flex-wrap: wrap;
      gap: 8px;
    }
    .splash-features {
      grid-template-columns: 1fr;
    }
    .sub-nav {
      padding: 6px 12px;
    }
  }
</style>
