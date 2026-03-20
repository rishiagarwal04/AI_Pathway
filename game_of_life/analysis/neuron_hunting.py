import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os

# 1. IMPORT YOUR EXACT ARCHITECTURE
from bdh_life import BDH_Life, BDHConfig

torch.manual_seed(42)
np.random.seed(42)

WEIGHT_PATH = "bdh_life_12_padded_3_layer_1_head_16_d_4_mlp_100_accuracy.pth"

# ==========================================
# 2. LOAD YOUR TRAINED BRAIN
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device.type.upper()}")

config = BDHConfig() # Pulls your exact 16-D, 4-MLP, 12-Grid config
model = BDH_Life(config).to(device)

if os.path.exists(WEIGHT_PATH):
    print(f"Loading trained weights from: {WEIGHT_PATH}")
    checkpoint = torch.load(WEIGHT_PATH, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("Weights successfully loaded!")
else:
    print(f"CRITICAL ERROR: Could not find {WEIGHT_PATH}.")
    exit()

model.eval()

# ==========================================
# 3. Generate the 12x12 Scenarios
# ==========================================
# We place the cell at Row 6, Col 6 (Index 78) in a 12x12 grid (144 tokens)
CENTER_IDX = 6 * 12 + 6
NEIGHBOR_OFFSETS = [-13, -12, -11, -1, 1, 11, 12, 13]

def create_12x12_scenario(center_state, num_neighbors):
    grid = torch.zeros(144)
    grid[CENTER_IDX] = center_state
    
    # Pick random neighbors around the center
    chosen_offsets = np.random.choice(NEIGHBOR_OFFSETS, num_neighbors, replace=False)
    for offset in chosen_offsets:
        grid[CENTER_IDX + offset] = 1.0
    return grid

print("\nGenerating strict 12x12 Game of Life logic scenarios...")
scenarios = []
labels = []

for _ in range(400):
    # 0. Underpopulation (Dies): Center=1, Neighbors < 2
    scenarios.append(create_12x12_scenario(1, np.random.choice([0, 1])))
    labels.append(0)
    
    # 1. Survival (Lives): Center=1, Neighbors = 2 or 3
    scenarios.append(create_12x12_scenario(1, np.random.choice([2, 3])))
    labels.append(1)
    
    # 2. Overpopulation (Dies): Center=1, Neighbors > 3
    scenarios.append(create_12x12_scenario(1, np.random.choice([4, 5, 6, 7, 8])))
    labels.append(2)
    
    # 3. Birth (Lives): Center=0, Neighbors = 3
    scenarios.append(create_12x12_scenario(0, 3))
    labels.append(3)

X = torch.stack(scenarios).to(device)
y = np.array(labels)

# ==========================================
# 4. Intercept the Brain
# ==========================================
print("Extracting intermediate Hub activations from Layer 3...")
with torch.no_grad():
    # Pass inputs through the model. It gracefully returns the brain_states!
    logits, loss, brain_states = model(X)
    
    # brain_states is a list of layers. We want the last layer [-1]
    # Shape of xy_sparse: (Batch, Heads, Tokens, Dim) -> (400, 1, 144, 64)
    final_layer_hubs = brain_states[-1]
    
    # We ONLY want to look at the "brain" of the Center Cell (Index 78)
    center_cell_hub = final_layer_hubs[:, 0, CENTER_IDX, :].cpu().numpy() 

# ==========================================
# 5. Find the "Bunches" (Linear Probing)
# ==========================================
print("\n" + "="*50)
print(" PROBING THE NEURON BUNCHES (Birth Logic)")
print("="*50)

is_birth = (y == 3).astype(int)
probe = LogisticRegression(max_iter=1000)
probe.fit(center_cell_hub, is_birth)

birth_bunch = probe.coef_[0]
top_neurons = np.argsort(np.abs(birth_bunch))[-5:][::-1]

print(f"The 'Birth' Logic Vector is primarily driven by these Hub Neurons:")
for neuron_idx in top_neurons:
    weight = birth_bunch[neuron_idx]
    direction = "POSITIVE (Triggers Birth)" if weight > 0 else "NEGATIVE (Prevents Birth)"
    print(f"  -> Hub Neuron {neuron_idx:02d} (Weight: {weight:+.4f}) - Pushes {direction}")

# ==========================================
# 6. Visualizing the Logic Space (PCA)
# ==========================================
print("\nMapping the 64-Dimensional Hub into 2D Space...")
pca = PCA(n_components=2)
hub_2d = pca.fit_transform(center_cell_hub)

plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue', 'orange']
names = ["Underpopulation (Die)", "Survival (Live)", "Overpopulation (Die)", "Birth (Live)"]

for i in range(4):
    mask = (y == i)
    plt.scatter(hub_2d[mask, 0], hub_2d[mask, 1], c=colors[i], label=names[i], alpha=0.6, edgecolors='w')

plt.title("Game of Life Logic Clusters in the Trained BDH Hub Space")
plt.xlabel("Principal Component 1 (Main Logic Direction)")
plt.ylabel("Principal Component 2 (Secondary Logic Direction)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("trained_gol_hub_logic.png", dpi=300)
print("Saved logic visualization to trained_gol_hub_logic.png!")
