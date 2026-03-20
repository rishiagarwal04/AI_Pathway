import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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
# 2. EXTRACT DATA FOR BOTH LAYERS
# ==========================================
print("Extracting coordinates for Layer 1 (3D) and Layer 0 (2D)...")
X_L1 = [] # [N17, N36, N50]
X_L0 = [] # [N19, N40]
Y_labels = []
colors_map = []
markers_map = []
cmap = cm.get_cmap('tab10')

with torch.no_grad():
    for center in [0, 1]:
        for neighbors in range(9):
            for _ in range(40):
                grid = create_scenario(center, neighbors)
                logits, _, hubs = model(grid)
                
                # Grab Layer 0 and Layer 1 hubs
                l0 = hubs[0][0, 0, CENTER_IDX]
                l1 = hubs[1][0, 0, CENTER_IDX]
                
                X_L1.append([l1[17].item(), l1[36].item(), l1[50].item()])
                X_L0.append([l0[19].item(), l0[40].item()])
                
                is_alive = 1 if (center == 0 and neighbors == 3) or (center == 1 and neighbors in [2, 3]) else 0
                
                Y_labels.append(is_alive)
                colors_map.append(cmap(neighbors))
                markers_map.append('X' if center == 1 else 'o')

X_L1 = np.array(X_L1)
X_L0 = np.array(X_L0)
Y_labels = np.array(Y_labels)

# ==========================================
# 3. LAYER 1: 3D MANIFOLD & HYPERPLANE
# ==========================================
print("Calculating 3D Hyperplane for Layer 1...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# We use a Linear Support Vector Machine to find the perfect 3D separating plane
clf3d = SVC(kernel='linear', C=1000)
clf3d.fit(X_L1, Y_labels)
w = clf3d.coef_[0]
b = clf3d.intercept_[0]

for i in range(len(X_L1)):
    ax.scatter(X_L1[i, 0], X_L1[i, 1], X_L1[i, 2], 
               color=colors_map[i], marker=markers_map[i], 
               s=60 if markers_map[i] == 'o' else 80, 
               edgecolor='white' if markers_map[i] == 'o' else 'black', alpha=0.8)

# Draw the separating plane: z = -(w0*x + w1*y + b) / w2
if abs(w[2]) > 1e-5:
    xx_plane, yy_plane = np.meshgrid(np.linspace(X_L1[:,0].min(), X_L1[:,0].max(), 10),
                                     np.linspace(X_L1[:,1].min(), X_L1[:,1].max(), 10))
    zz_plane = -(w[0] * xx_plane + w[1] * yy_plane + b) / w[2]
    ax.plot_surface(xx_plane, yy_plane, zz_plane, color='cyan', alpha=0.2, edgecolor='none')

ax.set_xlabel('L1H0N17')
ax.set_ylabel('L1H0N36')
ax.set_zlabel('L1H0N50')
ax.set_title("Layer 1 (3D): Untangling the Traffic Jam", fontsize=14)

# Set viewing angle for best separation
ax.view_init(elev=20, azim=45) 
plt.savefig("layer1_3d_manifold.png", dpi=300)
print("Saved 3D map to 'layer1_3d_manifold.png'!")

# ==========================================
# 4. LAYER 0: 2D DECISION MAP (L0H0N19 vs L0H0N40)
# ==========================================
print("Calculating 2D Map for Layer 0...")
plt.figure(figsize=(12, 9))
clf2d = LogisticRegression(C=1e5, class_weight='balanced')
clf2d.fit(X_L0, Y_labels)

x_min, x_max = X_L0[:, 0].min() - 0.01, X_L0[:, 0].max() + 0.01
y_min, y_max = X_L0[:, 1].min() - 0.01, X_L0[:, 1].max() + 0.01
xx2, yy2 = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = clf2d.predict(np.c_[xx2.ravel(), yy2.ravel()])
Z = Z.reshape(xx2.shape)

plt.contourf(xx2, yy2, Z, alpha=0.2, levels=[-0.5, 0.5, 1.5], colors=['red', 'lime'])
plt.contour(xx2, yy2, Z, colors='black', linewidths=2, levels=[0.5], linestyles='--')

for i in range(len(X_L0)):
    plt.scatter(X_L0[i, 0], X_L0[i, 1], color=colors_map[i], marker=markers_map[i], 
                s=80 if markers_map[i] == 'o' else 100, edgecolor='white' if markers_map[i] == 'o' else 'black', alpha=0.8)

plt.axvline(x=0, color='black', linewidth=1)
plt.axhline(y=0, color='black', linewidth=1)
plt.xlabel("Activation of L0H0N19", fontsize=12)
plt.ylabel("Activation of L0H0N40", fontsize=12)
plt.title("Layer 0 Logic: How L0H0N19 & L0H0N40 see the Game", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

plt.savefig("layer0_2d_manifold.png", dpi=300)
print("Saved Layer 0 map to 'layer0_2d_manifold.png'!")
