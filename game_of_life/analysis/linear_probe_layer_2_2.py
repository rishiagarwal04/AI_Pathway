import torch
import torch.nn.functional as F
import numpy as np
import itertools
from bdh_life import BDH_Life, BDHConfig
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP
# ==========================================
WEIGHT_PATH = "bdh_life_12_padded_3_layer_1_head_16_d_4_mlp_100_accuracy.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(WEIGHT_PATH, map_location=device)
config = BDHConfig(**checkpoint['config']) if 'config' in checkpoint else BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, grid_size=12)

model = BDH_Life(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

CENTER_IDX = 78

# ==========================================
# 2. PROPER DATA GENERATOR
# ==========================================
def create_scenario(center_state, num_neighbors):
    grid = torch.zeros(1, 1, 12, 12, device=device)
    grid[0, 0, 6, 6] = center_state
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    for idx in np.random.choice(len(offsets), num_neighbors, replace=False):
        dr, dc = offsets[idx]
        grid[0, 0, 6+dr, 6+dc] = 1.0
    # Returns exactly (1, 144) to match B, T = inputs.size()
    return grid.view(1, 144)

def get_data():
    print("Generating robust dataset...")
    X_list, Y_list = [], []
    with torch.no_grad():
        for center in [0, 1]:
            for neighbors in range(9):
                for _ in range(30): # 540 variations total
                    grid = create_scenario(center, neighbors)
                    _, _, hubs = model(grid)
                    
                    # Extract Layer 2 brain states
                    X_list.append(hubs[2][0, 0, CENTER_IDX].unsqueeze(0))
                    
                    is_alive = 1.0 if (center == 0 and neighbors == 3) or (center == 1 and neighbors in [2, 3]) else 0.0
                    Y_list.append(is_alive)
                    
    return torch.cat(X_list), torch.tensor(Y_list, device=device).view(-1, 1)

X_all, Y_all = get_data()

# Filter down to only active neurons
active_indices = torch.where(X_all.max(dim=0)[0] > 1e-5)[0]
X_active = X_all[:, active_indices]
F_dim = X_active.size(1)
print(f"Found {F_dim} active neurons in Layer 2.")

# ==========================================
# 3. VRAM-SAFE BATCHED SEARCH
# ==========================================
BATCH_SIZE = 5000  # Adjust down to 2000 if your RTX 5060 complains about VRAM

for dim in range(1, 7):
    combos = list(itertools.combinations(range(F_dim), dim))
    total_combos = len(combos)
    print(f"\n--- Testing {dim}D (Total Combos: {total_combos}) ---")
    
    best_acc_global = 0
    best_combo_global = None

    for i in range(0, total_combos, BATCH_SIZE):
        batch_combos = combos[i : i + BATCH_SIZE]
        C = len(batch_combos)
        
        # Load only this chunk into VRAM
        combo_tensor = torch.tensor(batch_combos, device=device)
        X_batch = X_active[:, combo_tensor].transpose(0, 1) # Shape: (C, Samples, Dim)
        Y_expanded = Y_all.unsqueeze(0).expand(C, -1, -1)
        
        # Initialize parallel models
        W = torch.randn(C, dim, 1, device=device, requires_grad=True)
        B = torch.zeros(C, 1, 1, device=device, requires_grad=True)
        opt = torch.optim.Adam([W, B], lr=0.3)
        
        # Rapid GPU Optimization
        for epoch in range(800):
            opt.zero_grad()
            logits = torch.bmm(X_batch, W) + B
            loss = F.binary_cross_entropy_with_logits(logits, Y_expanded)
            loss.backward()
            opt.step()
            
        with torch.no_grad():
            acc = ((torch.bmm(X_batch, W) + B > 0) == Y_expanded).float().mean(dim=1)
            batch_max_acc, batch_max_idx = acc.max(0)
            
            if batch_max_acc > best_acc_global:
                best_acc_global = batch_max_acc.item()
                best_combo_global = [active_indices[j].item() for j in batch_combos[batch_max_idx]]
        
        if (i > 0) and (i % 20000 == 0):
            print(f"  Processed {i}/{total_combos} combinations...")

    print(f"Result for {dim}D: Best Acc {best_acc_global*100:.2f}% with Neurons {best_combo_global}")
    
    # Early exit if we hit perfect logic separation
    if best_acc_global >= 0.999:
        print(f"\n[!] BREAKTHROUGH: Layer 2 perfectly separates the logic using {dim} dimensions!")
        break
