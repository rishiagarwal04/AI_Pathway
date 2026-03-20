import pygame
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import math

# Import your model
try:
    from bdh_life import BDH_Life, BDHConfig
except ImportError:
    print("Error: 'bdh_life.py' not found.")
    sys.exit(1)

# ==========================================
# 1. Configuration
# ==========================================
GRID_SIZE = 12      
HEADS = 1           
LAYERS = 3          
EMBED_DIM = 16      
MLP_MULT = 4        

# Colors
BLACK = (5, 5, 8) 
WHITE = (220, 220, 220)

pygame.init()
screen = pygame.display.set_mode((1400, 900))
pygame.display.set_caption(f"BDH Logic: The Parliament of Neurons")

# ==========================================
# 2. Model Wrapper
# ==========================================
class BDH_MRI(BDH_Life):
    pass 

# ==========================================
# 3. Circuit Analysis Engine (Logit Lens)
# ==========================================
def analyze_neuron_contributions(model, device):
    """
    1. Calculate Tuning Curves (Activations vs Neighbors)
    2. Calculate Effective Weights (Neuron -> Logit)
    3. Multiply them to get "Contribution Curves"
    """
    
    # --- A. GET TUNING CURVES (The Activations) ---
    print("Generating data...")
    boards = []
    # Using specific densities to ensure we see all neighbor counts (0-8)
    for density in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]:
        b = (torch.rand(30, GRID_SIZE, GRID_SIZE) < density).float()
        boards.append(b)
    inputs = torch.cat(boards, dim=0).view(-1, GRID_SIZE*GRID_SIZE).to(device)
    
    # Calculate Ground Truth Neighbors for Binning
    padded = inputs.view(-1, 1, GRID_SIZE, GRID_SIZE)
    padded = F.pad(padded, (1,1,1,1), mode='constant', value=0)
    kernel = torch.tensor([[[[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]]], device=device)
    neighbor_counts = F.conv2d(padded, kernel).view(-1, 144).detach().cpu().numpy().flatten()
    
    print("Running model...")
    with torch.no_grad():
        C = model.config
        x = model.input_proj(inputs.unsqueeze(-1)).unsqueeze(1)
        x = model.ln(x)
        
        # We need to know the Hidden Dimension (N)
        # N = 16 * 4 = 64
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh 

        # Fast forward to Layer 3
        for layer_idx in range(C.n_layer):
            # Attention Part
            x_latent = x @ model.encoder
            x_sparse = F.relu(x_latent)
            Q=x_sparse; K=x_sparse; V=x
            scores = (Q @ K.mT) / math.sqrt(Q.size(-1))
            attn = F.softmax(scores, dim=-1)
            x = model.ln(x + (attn @ V))
            
            # MLP Part
            y_latent = x @ model.encoder_v
            L3_neurons_raw = F.relu(y_latent) # [B, 1, T, 64]
            
            # Standard update
            xy_sparse = x_sparse * F.relu(y_latent)
            
            # Fix: Reshape to [..., 64] for multiplication
            yMLP = (xy_sparse.transpose(1, 2).reshape(-1, 1, 144, N) @ model.decoder)
            x = model.ln(x + yMLP)
            
            if layer_idx == 2: # Capture Layer 3
                neurons = L3_neurons_raw.squeeze(1).view(-1, N).detach().cpu().numpy()

    # Bin activations by Neighbor Count
    tuning_curves = np.zeros((64, 9))
    for n in range(9):
        mask = (neighbor_counts == n)
        if np.sum(mask) > 0:
            tuning_curves[:, n] = neurons[mask].mean(axis=0)

    # --- B. CALCULATE EFFECTIVE WEIGHTS (The Votes) ---
    print("Calculating effective weights...")
    
    # Helper to get weight tensor safely (handles Linear Layer vs Raw Parameter)
    def get_weight(layer_or_tensor):
        if hasattr(layer_or_tensor, 'weight'):
            return layer_or_tensor.weight.detach()
        else:
            return layer_or_tensor.detach()

    # 1. MLP Down Projection (64 -> 16)
    W_down = get_weight(model.decoder)
    # Ensure shape is [16, 64] (Output x Input) for matrix mult
    # If it is [64, 16], transpose it.
    if W_down.shape[0] == 64 and W_down.shape[1] == 16:
        W_down = W_down.T
    
    # 2. Output Head (16 -> 1)
    W_head = get_weight(model.output_head)
    # Ensure shape is [1, 16]
    if W_head.shape[0] == 16 and W_head.shape[1] == 1:
        W_head = W_head.T
        
    print(f"Decoder Shape: {W_down.shape}, Head Shape: {W_head.shape}")
    
    # 3. Effective Weight = W_head @ W_down (Result: 1 x 64)
    # "If Neuron X fires, how much does the final Logit change?"
    W_effective = (W_head @ W_down).squeeze(0).cpu().numpy() # [64]

    # --- C. CALCULATE CONTRIBUTIONS ---
    # Contribution = Activation * Weight
    # This shows the actual impact on the final decision
    contribution_curves = tuning_curves * W_effective[:, np.newaxis] 
    
    return tuning_curves, W_effective, contribution_curves

# ==========================================
# 4. Plotting
# ==========================================
def plot_circuit_mechanism(tuning, weights, contributions):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#050508')
    fig.suptitle("Logic Decoded: The 'Parliament of Neurons' (Layer 3)", color='white', fontsize=20)
    
    # Plot 1: The Votes (Effective Weights)
    ax = axes[0]
    ax.set_facecolor('#050508')
    
    # Sort neurons by their vote weight
    sorted_indices = np.argsort(weights)
    sorted_weights = weights[sorted_indices]
    
    # Color bars by sign
    colors = ['#FF4444' if w < 0 else '#44FF44' for w in sorted_weights]
    
    ax.bar(range(64), sorted_weights, color=colors)
    ax.set_title("The Voting Power (Effective Weights)", color='white', fontsize=14)
    ax.set_xlabel("Neurons (Sorted from Killer to Helper)", color='white')
    ax.set_ylabel("Vote Strength (Negative=Dead, Positive=Alive)", color='white')
    ax.grid(color='#333333', axis='y')
    ax.tick_params(colors='white')

    # Plot 2: The Logic Assembly (Contributions)
    ax = axes[1]
    ax.set_facecolor('#050508')
    ax.set_title("Contribution to Decision (Activation × Weight)", color='white', fontsize=14)
    
    # Identify key players
    # "The Helpers" (Top 3 Positive Contribution at 3 neighbors)
    score_at_3 = contributions[:, 3]
    top_helpers = np.argsort(score_at_3)[-3:]
    
    # "The Killers" (Top 3 Negative Contribution at 4 neighbors)
    score_at_4 = contributions[:, 4]
    top_killers = np.argsort(score_at_4)[:3]
    
    # Plot Helpers
    for idx in top_helpers:
        ax.plot(range(9), contributions[idx], color='#44FF44', linewidth=3, label=f"Helper #{idx}", marker='o')

    # Plot Killers
    for idx in top_killers:
        ax.plot(range(9), contributions[idx], color='#FF4444', linewidth=3, label=f"Killer #{idx}", marker='x')

    # Plot Sum (The Total Logic)
    total_logic = contributions.sum(axis=0)
    ax.plot(range(9), total_logic, color='#00FFFF', linewidth=5, label="TOTAL SUM", alpha=0.5)

    ax.set_xlabel("Neighbor Count", color='white')
    ax.set_ylabel("Logit Contribution", color='white')
    ax.grid(color='#333333')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#151520', labelcolor='white')

    # Render
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    raw_data = canvas.buffer_rgba()
    size = int(canvas.get_width_height()[0]), int(canvas.get_width_height()[1])
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    plt.close(fig)
    return surf

# ==========================================
# 5. Main
# ==========================================
device = torch.device("cpu")
config = BDHConfig()
config.n_layer = 3; config.n_embd = 16; config.n_head = 1; config.grid_size = 12; config.mlp_internal_dim_multiplier = 4 

model = BDH_MRI(config).to(device)
weights_path = 'bdh_life_12_padded_3_layer_1_head_16_d_4_mlp_100_accuracy.pth'
if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
else:
    print("WARNING: Using random weights.")
model.eval()

# Run Analysis
tuning, weights, contributions = analyze_neuron_contributions(model, device)
plot_surf = plot_circuit_mechanism(tuning, weights, contributions)

running = True
clock = pygame.time.Clock()

while running:
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
            
    screen.blit(plot_surf, (50, 50))
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
