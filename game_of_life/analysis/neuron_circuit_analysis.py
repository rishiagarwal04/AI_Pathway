"""
Neuron Circuit Analysis for BDH Game of Life Model
====================================================
Identifies which neurons (and groups) decode the logic of Conway's Game of Life.

Strategy:
  1. Profile: Find the ~5-10% of active neurons per layer using many random inputs.
  2. Ablate: Zero out active neurons (singles, pairs, triples) and measure accuracy
     on controlled test cases (0-8 neighbors).
  3. Visualize: Heatmaps showing which neurons are critical for which GoL rules.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from itertools import combinations
import dataclasses
import os
import sys
import json
from collections import defaultdict

# Import model
try:
    from bdh_life import BDH_Life, BDHConfig
except ImportError:
    print("Error: 'bdh_life.py' not found in current directory.")
    sys.exit(1)

# ==========================================
# CONFIG
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Which checkpoint to load — adjust this to your best model
CHECKPOINT_PATH = "bdh_life_12_padded_3_layer_1_head_16_d_4_mlp_100_accuracy.pth"

# How many random boards to use for profiling active neurons
NUM_PROFILE_SAMPLES = 2000

# Activation threshold: a neuron is "active" if its mean activation > this
ACTIVE_THRESHOLD_PERCENTILE = 90  # Top 10% most active neurons

# How many test boards per neighbor-count scenario
NUM_TEST_BOARDS_PER_SCENARIO = 200

# Max group size for ablation (1=singles, 2=pairs, 3=triples)
MAX_ABLATION_GROUP_SIZE = 3

# Max number of ablation combinations to test per group size (to keep runtime sane)
MAX_COMBOS_PER_SIZE = {1: None, 2: 200, 3: 300}

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")


# ==========================================
# 1. LOAD MODEL
# ==========================================
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    if 'config' in checkpoint:
        cfg_dict = checkpoint['config']
        config = BDHConfig(**cfg_dict)
    else:
        # Fallback: try to infer from state dict
        print("WARNING: No config in checkpoint, using default BDHConfig")
        config = BDHConfig()
    
    print(f"Model config: n_layer={config.n_layer}, n_head={config.n_head}, "
          f"n_embd={config.n_embd}, mlp_mult={config.mlp_internal_dim_multiplier}, "
          f"grid_size={config.grid_size}")
    
    model = BDH_Life(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Compute neuron counts
    nh = config.n_head
    D = config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    print(f"Neurons per head: {N}, Heads: {nh}, Total sparse neurons per layer: {N * nh}")
    print(f"Layers: {config.n_layer}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())/1e6:.4f}M")
    
    return model, config


# ==========================================
# 2. CONTROLLED TEST DATA GENERATION
# ==========================================
def generate_padded_batch(batch_size, device, density=0.3):
    """Generate random padded 12x12 boards and their GoL next-states."""
    inner_grid = 10
    inner = (torch.rand(batch_size, 1, inner_grid, inner_grid, device=device) < density).float()
    boards = F.pad(inner, (1, 1, 1, 1), mode='constant', value=0.0)
    
    kernel = torch.tensor([[[[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]]], device=device)
    boards_calc = F.pad(boards, pad=(1, 1, 1, 1), mode='constant', value=0)
    neighbors = F.conv2d(boards_calc, kernel)
    
    survival = (boards == 1.0) & ((neighbors == 2.0) | (neighbors == 3.0))
    birth = (boards == 0.0) & (neighbors == 3.0)
    next_boards = (survival | birth).float()
    
    mask = torch.zeros_like(next_boards)
    mask[:, :, 1:-1, 1:-1] = 1.0
    next_boards = next_boards * mask
    
    x = boards.view(batch_size, -1)
    y = next_boards.view(batch_size, -1)
    return x, y


def create_neighbor_count_scenarios(num_per_scenario, device):
    """
    Create controlled test cases where a specific center cell has exactly N neighbors.
    Returns dict: neighbor_count -> (x_batch, y_batch, center_state_alive)
    
    We test both:
      - Dead cell with N neighbors (birth test: should birth iff N==3)
      - Alive cell with N neighbors (survival test: should survive iff N==2 or N==3)
    
    All cells placed in the INNER region (rows/cols 1-10 of the 12x12 grid) to avoid walls.
    """
    grid_size = 12
    scenarios = {}
    
    # All 8 neighbor offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),          (0, 1),
               (1, -1),  (1, 0), (1, 1)]
    
    for n_neighbors in range(9):  # 0 through 8
        for center_alive in [False, True]:
            boards_list = []
            
            for _ in range(num_per_scenario):
                board = torch.zeros(1, 1, grid_size, grid_size, device=device)
                
                # Pick a random center in the safe inner region (rows 2-9, cols 2-9)
                # so all 8 neighbors are also in the inner region
                cr = torch.randint(2, 10, (1,)).item()
                cc = torch.randint(2, 10, (1,)).item()
                
                if center_alive:
                    board[0, 0, cr, cc] = 1.0
                
                # Randomly choose which N offsets are alive
                if n_neighbors > 0:
                    chosen = torch.randperm(8)[:n_neighbors]
                    for idx in chosen:
                        dr, dc = offsets[idx.item()]
                        board[0, 0, cr + dr, cc + dc] = 1.0
                
                boards_list.append(board)
            
            boards = torch.cat(boards_list, dim=0)
            
            # Compute ground truth for the FULL board
            kernel = torch.tensor([[[[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]]], device=device)
            boards_calc = F.pad(boards, pad=(1, 1, 1, 1), mode='constant', value=0)
            neighbors_map = F.conv2d(boards_calc, kernel)
            
            survival = (boards == 1.0) & ((neighbors_map == 2.0) | (neighbors_map == 3.0))
            birth = (boards == 0.0) & (neighbors_map == 3.0)
            next_boards = (survival | birth).float()
            
            mask_t = torch.zeros_like(next_boards)
            mask_t[:, :, 1:-1, 1:-1] = 1.0
            next_boards = next_boards * mask_t
            
            x = boards.view(num_per_scenario, -1)
            y = next_boards.view(num_per_scenario, -1)
            
            key = (n_neighbors, center_alive)
            scenarios[key] = (x, y)
    
    return scenarios


# ==========================================
# 3. ACTIVATION PROFILING (Find Active Neurons)
# ==========================================
def profile_activations(model, config, num_samples=2000, batch_size=200):
    """
    Run many random boards through the model and record mean activation
    of every sparse neuron in every layer.
    
    In BDH, the sparse activations are the ReLU outputs:
      x_sparse = F.relu(x @ encoder)        -> shape (B, nh, T, N)
      y_sparse = F.relu(yKV @ encoder_v)     -> shape (B, nh, T, N)
      xy_sparse = x_sparse * y_sparse        -> shape (B, nh, T, N)  <-- the brain_state
    
    brain_states[layer] has shape (B, n_head, T, N)
    """
    nh = config.n_head
    D = config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    T = config.grid_size ** 2  # 144 for 12x12
    
    n_layers = config.n_layer
    
    # Accumulators: mean activation per neuron (head, neuron_idx) per layer
    # Also track activation frequency (how often > 0)
    activation_sums = [torch.zeros(nh, N, device=DEVICE) for _ in range(n_layers)]
    activation_counts = [torch.zeros(nh, N, device=DEVICE) for _ in range(n_layers)]
    total_tokens = 0
    
    # We also need per-token activations for finer analysis — but that's too much memory
    # Instead, sum over tokens
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for b_idx in range(num_batches):
            current_bs = min(batch_size, num_samples - b_idx * batch_size)
            if current_bs <= 0:
                break
            
            x_batch, _ = generate_padded_batch(current_bs, DEVICE)
            _, _, brain_states = model(x_batch)
            
            for layer_idx, bs in enumerate(brain_states):
                # bs shape: (B, nh, T, N)
                # Mean over batch and tokens
                act_mean_per_neuron = bs.abs().mean(dim=(0, 2))  # (nh, N)
                act_active_per_neuron = (bs.abs() > 1e-6).float().mean(dim=(0, 2))  # (nh, N)
                
                activation_sums[layer_idx] += act_mean_per_neuron * current_bs
                activation_counts[layer_idx] += act_active_per_neuron * current_bs
            
            total_tokens += current_bs
    
    # Normalize
    activation_means = [s / total_tokens for s in activation_sums]
    activation_freqs = [c / total_tokens for c in activation_counts]  # fraction of time active
    
    return activation_means, activation_freqs


def find_active_neurons(activation_means, activation_freqs, config, percentile=90):
    """
    Identify neurons that are consistently active (above percentile threshold).
    Returns dict: layer_idx -> list of (head_idx, neuron_idx) tuples
    """
    active_neurons = {}
    
    for layer_idx in range(config.n_layer):
        means = activation_means[layer_idx]  # (nh, N)
        freqs = activation_freqs[layer_idx]  # (nh, N)
        
        # Flatten to rank all neurons in this layer
        flat_means = means.flatten().cpu().numpy()
        flat_freqs = freqs.flatten().cpu().numpy()
        
        # Use frequency of activation as the primary criterion
        threshold = np.percentile(flat_freqs, percentile)
        
        nh, N = means.shape
        active = []
        for h in range(nh):
            for n in range(N):
                if freqs[h, n].item() >= threshold and means[h, n].item() > 1e-6:
                    active.append((h, n))
        
        active_neurons[layer_idx] = active
        print(f"  Layer {layer_idx}: {len(active)}/{nh * N} neurons active "
              f"(threshold freq >= {threshold:.4f})")
    
    return active_neurons


# ==========================================
# 4. ABLATION ENGINE
# ==========================================
class AblationHook:
    """
    Hook that zeros out specific neurons in the brain_states during forward pass.
    
    We need to intercept the computation inside the forward() loop.
    Since brain_states are computed inside forward(), we'll use a modified forward.
    """
    def __init__(self, ablation_mask_per_layer):
        """
        ablation_mask_per_layer: dict of layer_idx -> list of (head_idx, neuron_idx)
        """
        self.ablation_mask_per_layer = ablation_mask_per_layer


def forward_with_ablation(model, inputs, ablation_spec, targets=None):
    """
    Modified forward pass that zeros out specified neurons at specified layers.
    
    ablation_spec: dict of layer_idx -> list of (head_idx, neuron_idx) to zero out
    """
    C = model.config
    B, T = inputs.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.input_proj(inputs.unsqueeze(-1)).unsqueeze(1)
    x = model.ln(x)
    brain_states = []

    for level in range(C.n_layer):
        x_latent = x @ model.encoder
        x_sparse = F.relu(x_latent)

        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = model.ln(yKV)

        y_latent = yKV @ model.encoder_v
        y_sparse = F.relu(y_latent)

        xy_sparse = x_sparse * y_sparse
        xy_sparse = model.drop(xy_sparse)
        
        # *** ABLATION: Zero out specified neurons ***
        if level in ablation_spec:
            for (h_idx, n_idx) in ablation_spec[level]:
                xy_sparse[:, h_idx, :, n_idx] = 0.0
        
        brain_states.append(xy_sparse.detach())

        yMLP = (xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder)
        y = model.ln(yMLP)
        x = model.ln(x + y)

    logits = model.output_head(x.squeeze(1))

    loss = None
    if targets is not None:
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits.view(-1), targets.view(-1).float())

    return logits, loss, brain_states


def measure_accuracy(model, x_batch, y_batch, ablation_spec=None):
    """Measure pixel-wise accuracy, optionally with ablation."""
    with torch.no_grad():
        if ablation_spec is not None:
            logits, _, _ = forward_with_ablation(model, x_batch, ablation_spec)
        else:
            logits, _, _ = model(x_batch)
        
        preds = (torch.sigmoid(logits) > 0.5).float().squeeze(-1)
        correct = (preds == y_batch).float()
        
        # Overall accuracy
        overall_acc = correct.mean().item()
        
        return overall_acc


def measure_scenario_accuracies(model, scenarios, ablation_spec=None):
    """
    Measure accuracy for each (neighbor_count, center_alive) scenario.
    Returns dict: (n_neighbors, center_alive) -> accuracy
    """
    results = {}
    for key, (x, y) in scenarios.items():
        acc = measure_accuracy(model, x, y, ablation_spec)
        results[key] = acc
    return results


# ==========================================
# 5. MAIN ANALYSIS PIPELINE
# ==========================================
def run_analysis():
    print("=" * 70)
    print("NEURON CIRCUIT ANALYSIS FOR BDH GAME OF LIFE")
    print("=" * 70)
    
    # --- Load Model ---
    print("\n[1/5] Loading model...")
    model, config = load_model(CHECKPOINT_PATH)
    
    nh = config.n_head
    D = config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    
    # --- Verify baseline accuracy ---
    print("\n[2/5] Verifying baseline accuracy...")
    x_test, y_test = generate_padded_batch(1000, DEVICE)
    baseline_acc = measure_accuracy(model, x_test, y_test)
    print(f"  Baseline accuracy: {baseline_acc * 100:.4f}%")
    
    # --- Profile activations ---
    print(f"\n[3/5] Profiling activations over {NUM_PROFILE_SAMPLES} random boards...")
    activation_means, activation_freqs = profile_activations(
        model, config, num_samples=NUM_PROFILE_SAMPLES
    )
    
    print(f"\n  Finding active neurons (top {100 - ACTIVE_THRESHOLD_PERCENTILE}%)...")
    active_neurons = find_active_neurons(
        activation_means, activation_freqs, config,
        percentile=ACTIVE_THRESHOLD_PERCENTILE
    )
    
    # --- Generate controlled test scenarios ---
    print(f"\n[4/5] Generating controlled neighbor-count test scenarios...")
    scenarios = create_neighbor_count_scenarios(NUM_TEST_BOARDS_PER_SCENARIO, DEVICE)
    
    # Baseline per-scenario accuracy
    baseline_scenario_acc = measure_scenario_accuracies(model, scenarios)
    print("  Baseline per-scenario accuracy:")
    for n_neigh in range(9):
        for alive in [False, True]:
            key = (n_neigh, alive)
            state_str = "ALIVE" if alive else "DEAD "
            rule = ""
            if alive:
                rule = "→ SURVIVE" if n_neigh in [2, 3] else "→ DIE"
            else:
                rule = "→ BIRTH" if n_neigh == 3 else "→ STAY DEAD"
            print(f"    {state_str} + {n_neigh} neighbors {rule}: {baseline_scenario_acc[key]*100:.2f}%")
    
    # --- Run ablation experiments ---
    print(f"\n[5/5] Running ablation experiments on active neurons...")
    
    all_ablation_results = {}  # (layer, group_tuple) -> scenario_accuracies
    
    for layer_idx in range(config.n_layer):
        neurons = active_neurons[layer_idx]
        if not neurons:
            print(f"\n  Layer {layer_idx}: No active neurons found, skipping.")
            continue
        
        print(f"\n  === Layer {layer_idx} ({len(neurons)} active neurons) ===")
        
        for group_size in range(1, min(MAX_ABLATION_GROUP_SIZE + 1, len(neurons) + 1)):
            all_combos = list(combinations(neurons, group_size))
            max_combos = MAX_COMBOS_PER_SIZE.get(group_size, 100)
            
            if max_combos is not None and len(all_combos) > max_combos:
                # Randomly sample combos
                indices = np.random.choice(len(all_combos), max_combos, replace=False)
                combos = [all_combos[i] for i in indices]
                print(f"    Group size {group_size}: testing {len(combos)}/{len(all_combos)} combinations")
            else:
                combos = all_combos
                print(f"    Group size {group_size}: testing all {len(combos)} combinations")
            
            for combo in combos:
                ablation_spec = {layer_idx: list(combo)}
                scenario_acc = measure_scenario_accuracies(model, scenarios, ablation_spec)
                all_ablation_results[(layer_idx, combo)] = scenario_acc
    
    # --- Compute impact scores ---
    print("\n" + "=" * 70)
    print("RESULTS: Computing impact scores...")
    print("=" * 70)
    
    impact_data = compute_impact_scores(
        all_ablation_results, baseline_scenario_acc, active_neurons, config
    )
    
    # --- Visualize ---
    print("\nGenerating visualizations...")
    visualize_results(
        impact_data, activation_means, activation_freqs, 
        active_neurons, baseline_scenario_acc, config
    )
    
    # --- Save results ---
    save_results(impact_data, active_neurons, baseline_scenario_acc, config)
    
    print("\n✅ Analysis complete! Check the generated PNG files.")


# ==========================================
# 6. IMPACT SCORING
# ==========================================
def compute_impact_scores(all_ablation_results, baseline_acc, active_neurons, config):
    """
    Compute how much each neuron (and group) matters for each GoL rule.
    
    Impact = baseline_accuracy - ablated_accuracy  (positive = neuron was helpful)
    """
    # Per-neuron impact on each scenario
    single_impacts = {}  # (layer, head, neuron) -> {scenario_key: impact}
    
    # Per-group impact
    group_impacts = {}  # (layer, combo) -> {scenario_key: impact}
    
    # GoL rule categories
    rule_categories = {
        'birth': [(3, False)],  # Dead cell + 3 neighbors -> birth
        'survival_2': [(2, True)],  # Alive + 2 neighbors -> survive
        'survival_3': [(3, True)],  # Alive + 3 neighbors -> survive
        'death_underpop': [(0, True), (1, True)],  # Alive + 0-1 neighbors -> die
        'death_overpop': [(4, True), (5, True), (6, True), (7, True), (8, True)],
        'stay_dead': [(0, False), (1, False), (2, False), (4, False), 
                      (5, False), (6, False), (7, False), (8, False)],
    }
    
    for (layer_idx, combo), scenario_acc in all_ablation_results.items():
        impacts = {}
        for key in scenario_acc:
            impacts[key] = baseline_acc[key] - scenario_acc[key]
        
        # Aggregate by rule category
        rule_impacts = {}
        for rule_name, rule_keys in rule_categories.items():
            rule_impact_vals = [impacts.get(k, 0.0) for k in rule_keys if k in impacts]
            if rule_impact_vals:
                rule_impacts[rule_name] = np.mean(rule_impact_vals)
            else:
                rule_impacts[rule_name] = 0.0
        
        group_impacts[(layer_idx, combo)] = {
            'per_scenario': impacts,
            'per_rule': rule_impacts,
            'total_impact': np.mean(list(impacts.values()))
        }
        
        # For single neuron ablations, store separately
        if len(combo) == 1:
            h, n = combo[0]
            single_impacts[(layer_idx, h, n)] = {
                'per_scenario': impacts,
                'per_rule': rule_impacts,
                'total_impact': np.mean(list(impacts.values()))
            }
    
    # Find top neurons per rule
    print("\n--- Top 10 Most Impactful Single Neurons Per Rule ---")
    for rule_name in rule_categories:
        ranked = sorted(
            single_impacts.items(),
            key=lambda item: abs(item[1]['per_rule'].get(rule_name, 0)),
            reverse=True
        )[:10]
        
        print(f"\n  Rule: {rule_name}")
        for (layer, h, n), data in ranked:
            imp = data['per_rule'].get(rule_name, 0)
            print(f"    L{layer} H{h} N{n:02d}: impact={imp*100:+.3f}%")
    
    # Find synergistic pairs (pairs whose combined impact > sum of individual impacts)
    print("\n--- Top Synergistic Pairs ---")
    synergies = []
    for (layer_idx, combo), data in group_impacts.items():
        if len(combo) == 2:
            n1, n2 = combo
            ind1 = single_impacts.get((layer_idx, n1[0], n1[1]), None)
            ind2 = single_impacts.get((layer_idx, n2[0], n2[1]), None)
            
            if ind1 and ind2:
                expected = ind1['total_impact'] + ind2['total_impact']
                actual = data['total_impact']
                synergy = actual - expected
                synergies.append((layer_idx, combo, synergy, actual, expected))
    
    synergies.sort(key=lambda x: abs(x[2]), reverse=True)
    for layer, combo, syn, actual, expected in synergies[:15]:
        n1, n2 = combo
        direction = "SYNERGY" if syn > 0 else "REDUNDANCY"
        print(f"  L{layer} ({n1}, {n2}): {direction} "
              f"actual={actual*100:+.3f}% expected={expected*100:+.3f}% "
              f"synergy={syn*100:+.3f}%")
    
    return {
        'single_impacts': single_impacts,
        'group_impacts': group_impacts,
        'synergies': synergies,
        'rule_categories': rule_categories,
    }


# ==========================================
# 7. VISUALIZATION
# ==========================================
def visualize_results(impact_data, activation_means, activation_freqs,
                      active_neurons, baseline_acc, config):
    """Generate comprehensive visualizations."""
    
    single_impacts = impact_data['single_impacts']
    rule_categories = impact_data['rule_categories']
    synergies = impact_data['synergies']
    
    nh = config.n_head
    D = config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    
    # ==============================
    # FIG 1: Activation Heatmaps per Layer
    # ==============================
    fig1, axes1 = plt.subplots(config.n_layer, 2, figsize=(16, 4 * config.n_layer))
    if config.n_layer == 1:
        axes1 = axes1.reshape(1, -1)
    
    fig1.suptitle("Neuron Activation Profile (Mean Magnitude & Frequency)", fontsize=14, fontweight='bold')
    
    for layer_idx in range(config.n_layer):
        means = activation_means[layer_idx].cpu().numpy()  # (nh, N)
        freqs = activation_freqs[layer_idx].cpu().numpy()
        
        im1 = axes1[layer_idx, 0].imshow(means, aspect='auto', cmap='hot', interpolation='nearest')
        axes1[layer_idx, 0].set_title(f"Layer {layer_idx} - Mean Activation")
        axes1[layer_idx, 0].set_ylabel("Head")
        axes1[layer_idx, 0].set_xlabel("Neuron Index")
        plt.colorbar(im1, ax=axes1[layer_idx, 0])
        
        im2 = axes1[layer_idx, 1].imshow(freqs, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        axes1[layer_idx, 1].set_title(f"Layer {layer_idx} - Activation Frequency")
        axes1[layer_idx, 1].set_ylabel("Head")
        axes1[layer_idx, 1].set_xlabel("Neuron Index")
        plt.colorbar(im2, ax=axes1[layer_idx, 1])
        
        # Mark active neurons
        if layer_idx in active_neurons:
            for h, n in active_neurons[layer_idx]:
                axes1[layer_idx, 0].plot(n, h, 'c^', markersize=4)
                axes1[layer_idx, 1].plot(n, h, 'c^', markersize=4)
    
    fig1.tight_layout()
    fig1.savefig("fig1_activation_profile.png", dpi=150, bbox_inches='tight')
    print("  Saved fig1_activation_profile.png")
    
    # ==============================
    # FIG 2: Single Neuron Impact Heatmap (Neuron × Rule)
    # ==============================
    rule_names = list(rule_categories.keys())
    
    # Collect all single neurons
    single_keys = sorted(single_impacts.keys())
    if not single_keys:
        print("  WARNING: No single-neuron ablation results to plot.")
    else:
        # Create impact matrix
        impact_matrix = np.zeros((len(single_keys), len(rule_names)))
        neuron_labels = []
        
        for i, (layer, h, n) in enumerate(single_keys):
            neuron_labels.append(f"L{layer}H{h}N{n}")
            for j, rule in enumerate(rule_names):
                impact_matrix[i, j] = single_impacts[(layer, h, n)]['per_rule'].get(rule, 0) * 100
        
        # Sort by total absolute impact
        total_abs = np.abs(impact_matrix).sum(axis=1)
        sort_idx = np.argsort(total_abs)[::-1]
        
        # Show top 40 most impactful neurons
        top_k = min(40, len(sort_idx))
        sort_idx = sort_idx[:top_k]
        
        fig2, ax2 = plt.subplots(figsize=(12, max(6, top_k * 0.35)))
        
        im = ax2.imshow(impact_matrix[sort_idx], aspect='auto', cmap='RdBu_r', 
                         interpolation='nearest',
                         norm=Normalize(vmin=-np.percentile(np.abs(impact_matrix), 95),
                                       vmax=np.percentile(np.abs(impact_matrix), 95)))
        
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels([neuron_labels[i] for i in sort_idx], fontsize=8)
        ax2.set_xticks(range(len(rule_names)))
        ax2.set_xticklabels(rule_names, rotation=45, ha='right', fontsize=9)
        ax2.set_title("Single Neuron Ablation Impact (% accuracy drop)", fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax2, label="Accuracy Impact (%)")
        
        # Annotate values
        for i in range(top_k):
            for j in range(len(rule_names)):
                val = impact_matrix[sort_idx[i], j]
                if abs(val) > 0.01:  # Only annotate non-trivial
                    ax2.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6,
                            color='white' if abs(val) > np.percentile(np.abs(impact_matrix), 80) else 'black')
        
        fig2.tight_layout()
        fig2.savefig("fig2_single_neuron_impact.png", dpi=150, bbox_inches='tight')
        print("  Saved fig2_single_neuron_impact.png")
    
    # ==============================
    # FIG 3: Per-Scenario Detailed Impact (for top neurons)
    # ==============================
    if single_keys:
        top_neurons = [single_keys[i] for i in sort_idx[:min(20, len(sort_idx))]]
        
        scenario_keys = sorted(baseline_acc.keys())
        scenario_labels = [f"{'A' if alive else 'D'}+{n}n" for (n, alive) in scenario_keys]
        
        detail_matrix = np.zeros((len(top_neurons), len(scenario_keys)))
        
        for i, nk in enumerate(top_neurons):
            for j, sk in enumerate(scenario_keys):
                detail_matrix[i, j] = single_impacts[nk]['per_scenario'].get(sk, 0) * 100
        
        fig3, ax3 = plt.subplots(figsize=(16, max(6, len(top_neurons) * 0.4)))
        
        vmax = np.percentile(np.abs(detail_matrix), 95) if detail_matrix.size > 0 else 1
        im3 = ax3.imshow(detail_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                          norm=Normalize(vmin=-vmax, vmax=vmax))
        
        ax3.set_yticks(range(len(top_neurons)))
        ax3.set_yticklabels([f"L{l}H{h}N{n}" for l, h, n in top_neurons], fontsize=8)
        ax3.set_xticks(range(len(scenario_labels)))
        ax3.set_xticklabels(scenario_labels, rotation=45, ha='right', fontsize=8)
        ax3.set_title("Detailed: Neuron Ablation Impact per Scenario (D=Dead center, A=Alive center, Nn=neighbors)", 
                       fontsize=11, fontweight='bold')
        plt.colorbar(im3, ax=ax3, label="Accuracy Impact (%)")
        
        # Add GoL rule annotations on x-axis
        for j, (n_neigh, alive) in enumerate(scenario_keys):
            if alive:
                rule = "survive" if n_neigh in [2, 3] else "die"
            else:
                rule = "birth" if n_neigh == 3 else "dead"
            ax3.text(j, len(top_neurons) + 0.3, rule, ha='center', va='top', fontsize=6,
                    color='green' if rule in ['survive', 'birth'] else 'red', rotation=45)
        
        fig3.tight_layout()
        fig3.savefig("fig3_detailed_scenario_impact.png", dpi=150, bbox_inches='tight')
        print("  Saved fig3_detailed_scenario_impact.png")
    
    # ==============================
    # FIG 4: Circuit Diagram — Which neurons encode which rules
    # ==============================
    fig4, axes4 = plt.subplots(1, config.n_layer, figsize=(6 * config.n_layer, 8))
    if config.n_layer == 1:
        axes4 = [axes4]
    
    fig4.suptitle("Circuit Map: Neurons Colored by Their Primary GoL Rule", fontsize=14, fontweight='bold')
    
    rule_colors = {
        'birth': '#2ecc71',       # Green
        'survival_2': '#3498db',   # Blue
        'survival_3': '#9b59b6',   # Purple
        'death_underpop': '#e74c3c',  # Red
        'death_overpop': '#e67e22',   # Orange
        'stay_dead': '#95a5a6',    # Gray
        'none': '#ecf0f1',        # Light gray
    }
    
    for layer_idx in range(config.n_layer):
        ax = axes4[layer_idx]
        ax.set_title(f"Layer {layer_idx}", fontsize=12)
        
        # Create grid for this layer
        grid_data = np.zeros((nh, N))
        grid_colors = np.full((nh, N), '#ecf0f1', dtype=object)
        
        for h in range(nh):
            for n in range(N):
                key = (layer_idx, h, n)
                if key in single_impacts:
                    impacts = single_impacts[key]['per_rule']
                    # Find which rule this neuron impacts most
                    best_rule = max(impacts, key=lambda r: abs(impacts[r]))
                    best_val = abs(impacts[best_rule])
                    grid_data[h, n] = best_val * 100
                    if best_val > 0.001:  # Only color if meaningful
                        grid_colors[h, n] = rule_colors.get(best_rule, '#ecf0f1')
        
        # Draw colored rectangles
        for h in range(nh):
            for n in range(N):
                rect = plt.Rectangle((n - 0.5, h - 0.5), 1, 1,
                                    facecolor=grid_colors[h, n],
                                    edgecolor='white', linewidth=0.5,
                                    alpha=min(1.0, 0.3 + grid_data[h, n] * 5))
                ax.add_patch(rect)
                
                if grid_data[h, n] > 0.01:
                    ax.text(n, h, f"{grid_data[h, n]:.1f}", ha='center', va='center',
                           fontsize=5, fontweight='bold')
        
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(-0.5, nh - 0.5)
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Head")
        ax.set_aspect('equal')
        ax.invert_yaxis()
    
    # Legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c, label=r)
                       for r, c in rule_colors.items() if r != 'none']
    fig4.legend(handles=legend_elements, loc='lower center', ncol=len(rule_colors) - 1,
               fontsize=9, frameon=True)
    
    fig4.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig4.savefig("fig4_circuit_map.png", dpi=150, bbox_inches='tight')
    print("  Saved fig4_circuit_map.png")
    
    # ==============================
    # FIG 5: Synergy Network
    # ==============================
    if synergies:
        fig5, ax5 = plt.subplots(figsize=(12, 8))
        ax5.set_title("Neuron Pair Synergies (Red=Synergy, Blue=Redundancy)", fontsize=13, fontweight='bold')
        
        top_syn = synergies[:min(30, len(synergies))]
        
        # Arrange neurons in a circle
        all_neurons_in_pairs = set()
        for layer, combo, syn, actual, expected in top_syn:
            for h, n in combo:
                all_neurons_in_pairs.add((layer, h, n))
        
        neuron_list = sorted(all_neurons_in_pairs)
        n_nodes = len(neuron_list)
        
        if n_nodes > 0:
            angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
            pos = {nk: (np.cos(a) * 3, np.sin(a) * 3) for nk, a in zip(neuron_list, angles)}
            
            # Draw edges
            max_syn = max(abs(s[2]) for s in top_syn) if top_syn else 1
            for layer, combo, syn, actual, expected in top_syn:
                n1 = (layer, combo[0][0], combo[0][1])
                n2 = (layer, combo[1][0], combo[1][1])
                if n1 in pos and n2 in pos:
                    color = 'red' if syn > 0 else 'blue'
                    width = abs(syn) / max_syn * 3 + 0.5
                    ax5.plot([pos[n1][0], pos[n2][0]], [pos[n1][1], pos[n2][1]],
                            color=color, linewidth=width, alpha=0.6)
            
            # Draw nodes
            for nk in neuron_list:
                x, y = pos[nk]
                ax5.plot(x, y, 'ko', markersize=10, zorder=5)
                ax5.text(x, y + 0.3, f"L{nk[0]}H{nk[1]}N{nk[2]}", ha='center', fontsize=7, zorder=6)
        
        ax5.set_aspect('equal')
        ax5.axis('off')
        fig5.tight_layout()
        fig5.savefig("fig5_synergy_network.png", dpi=150, bbox_inches='tight')
        print("  Saved fig5_synergy_network.png")
    
    plt.close('all')


# ==========================================
# 8. SAVE RESULTS
# ==========================================
def save_results(impact_data, active_neurons, baseline_acc, config):
    """Save analysis results to JSON for later inspection."""
    
    results = {
        'config': dataclasses.asdict(config),
        'baseline_accuracy': {f"{k[0]}_{k[1]}": v for k, v in baseline_acc.items()},
        'active_neurons': {
            str(layer): [(h, n) for h, n in neurons]
            for layer, neurons in active_neurons.items()
        },
        'top_single_impacts': {},
    }
    
    # Top 20 most impactful single neurons
    for (layer, h, n), data in sorted(
        impact_data['single_impacts'].items(),
        key=lambda item: abs(item[1]['total_impact']),
        reverse=True
    )[:20]:
        key = f"L{layer}_H{h}_N{n}"
        results['top_single_impacts'][key] = {
            'total_impact': float(data['total_impact']),
            'per_rule': {r: float(v) for r, v in data['per_rule'].items()},
        }
    
    # Top synergies
    results['top_synergies'] = []
    for layer, combo, syn, actual, expected in impact_data['synergies'][:20]:
        results['top_synergies'].append({
            'layer': layer,
            'neurons': [(h, n) for h, n in combo],
            'synergy': float(syn),
            'actual_impact': float(actual),
            'expected_sum': float(expected),
        })
    
    with open('neuron_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved neuron_analysis_results.json")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    run_analysis()