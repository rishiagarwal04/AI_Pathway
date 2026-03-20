import torch
import numpy as np
import pandas as pd
import plotly.express as px
from bdh_life import BDH_Life, BDHConfig

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

def create_scenario(center_state, num_neighbors):
    grid = torch.zeros(1, 1, 12, 12, device=device)
    grid[0, 0, 6, 6] = center_state
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    for idx in np.random.choice(len(offsets), num_neighbors, replace=False):
        dr, dc = offsets[idx]
        grid[0, 0, 6+dr, 6+dc] = 1.0
    return grid.view(1, 144)

# ==========================================
# 2. EXTRACT DATA FOR ALL CANDIDATES
# ==========================================
print("Extracting coordinates for N17, N36, and Z-axis candidates (N50, N28, N19)...")

data = []

with torch.no_grad():
    for center in [0, 1]:
        for neighbors in range(9):
            for _ in range(40):
                grid = create_scenario(center, neighbors)
                logits, _, hubs = model(grid)
                l1 = hubs[1][0, 0, CENTER_IDX]
                
                # Ground Truth Logic
                is_alive = 1 if (center == 0 and neighbors == 3) or (center == 1 and neighbors in [2, 3]) else 0
                rule_name = "Birth" if (center==0 and neighbors==3) else \
                            "Survival" if (center==1 and neighbors in [2,3]) else \
                            "Overpopulation" if (center==1 and neighbors>3) else \
                            "Underpopulation" if (center==1 and neighbors<2) else "Stay Dead"

                data.append({
                    "N17 (X)": l1[17].item(),
                    "N36 (Y)": l1[36].item(),
                    "N50 (Z1)": l1[50].item(),
                    "N28 (Z2)": l1[28].item(),
                    "N19 (Z3)": l1[19].item(),
                    "Neighbors": str(neighbors),
                    "Center_State": "Alive" if center == 1 else "Dead",
                    "Rule": rule_name,
                    "Target": "Alive" if is_alive else "Dead"
                })

df = pd.DataFrame(data)

# Sort so the legend is in order 0-8
df = df.sort_values(by=["Neighbors", "Center_State"])

# ==========================================
# 3. GENERATE INTERACTIVE HTML PLOTS
# ==========================================
print("Generating interactive HTML files...")

z_candidates = {"N50 (Z1)": "interactive_3D_N50.html", 
                "N28 (Z2)": "interactive_3D_N28.html", 
                "N19 (Z3)": "interactive_3D_N19.html"}

for z_col, filename in z_candidates.items():
    fig = px.scatter_3d(
        df, 
        x="N17 (X)", 
        y="N36 (Y)", 
        z=z_col,
        color="Neighbors",
        symbol="Center_State",
        hover_name="Rule",
        hover_data=["Target"],
        color_discrete_sequence=px.colors.qualitative.Set1,
        title=f"3D Logic Manifold (Z-Axis: {z_col})"
    )
    
    # Make the markers a bit bigger so they are easy to see
    fig.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(scene=dict(xaxis_title='N17', yaxis_title='N36', zaxis_title=z_col))
    
    fig.write_html(filename)
    print(f"Saved: {filename}")

print("\nDone! Open these .html files in your web browser (Chrome, Firefox, Safari) to interact with them.")
