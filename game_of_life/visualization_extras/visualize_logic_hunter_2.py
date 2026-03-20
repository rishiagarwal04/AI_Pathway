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

# Simulation Settings
BATCH_SIZE = 64 

# Colors
BLACK = (5, 5, 8) 
WHITE = (220, 220, 220)
CYAN_HEX = '#00FFFF'

pygame.init()
screen = pygame.display.set_mode((1200, 800))
pygame.display.set_caption(f"BDH Logic: The Final Decision")

# ==========================================
# 2. Model Wrapper
# ==========================================
class BDH_MRI(BDH_Life):
    pass 

# ==========================================
# 3. Data Gathering Engine
# ==========================================
def calculate_decision_curve(model, device):
    print("Generating random boards...")
    boards = []
    # Use many densities to get good data for 0-8 neighbors
    for density in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        b = (torch.rand(50, GRID_SIZE, GRID_SIZE) < density).float()
        boards.append(b)
    
    inputs = torch.cat(boards, dim=0).view(-1, GRID_SIZE*GRID_SIZE).to(device) 
    
    print("Calculating neighbor counts...")
    padded = inputs.view(-1, 1, GRID_SIZE, GRID_SIZE)
    padded = F.pad(padded, (1,1,1,1), mode='constant', value=0)
    kernel = torch.tensor([[[[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]]], device=device)
    neighbor_counts = F.conv2d(padded, kernel).view(-1, 144) 
    
    print("Running model prediction...")
    with torch.no_grad(): 
        # FIX: Handle tuple return (logits, loss)
        output = model(inputs)
        
        # Safe unpacking
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
            
        probs = torch.sigmoid(logits) # Convert to 0.0 - 1.0 probability
        
    print("Binning data...")
    flat_neighbors = neighbor_counts.view(-1).detach().cpu().numpy() 
    flat_probs = probs.view(-1).detach().cpu().numpy()
    
    decision_curve = np.zeros(9)
    
    for n in range(9):
        mask = (flat_neighbors == n)
        if np.sum(mask) > 0:
            decision_curve[n] = flat_probs[mask].mean()
            
    return decision_curve

# ==========================================
# 4. Plotting
# ==========================================
def plot_decision_curve(curve):
    fig = plt.figure(figsize=(10, 6), facecolor='#050508')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#050508')
    
    # Plot the curve
    ax.plot(range(9), curve, color=CYAN_HEX, linewidth=4, marker='o', markersize=10, label='Model Prediction')
    
    # Draw the "Ideal" Game of Life Rules for comparison
    # Born at 3, Survives at 2 or 3. 
    # Average expected value at 2 is 0.5 (since 50% are alive/dead in random noise)
    # Average expected value at 3 is 1.0
    ideal_x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ideal_y = [0, 0, 0.5, 1, 0, 0, 0, 0, 0] 
    
    ax.plot(ideal_x, ideal_y, color='white', linestyle='--', alpha=0.5, label='Ideal Rule (Approx)')

    ax.set_title("The Final Decision: Probability of Life", color='white', fontsize=20)
    ax.set_xlabel("Number of Neighbors", color='white', fontsize=14)
    ax.set_ylabel("Probability of Life (0-1)", color='white', fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(colors='white', labelsize=12)
    
    ax.grid(color='#333333', linestyle='--')
    ax.legend(facecolor='#151520', edgecolor='white', labelcolor='white', fontsize=12)
    
    # Annotate the Peak
    if curve[3] > 0.5:
        ax.annotate('The "Birth" Spike', xy=(3, curve[3]), xytext=(4, 0.8),
                    arrowprops=dict(facecolor='white', shrink=0.05),
                    color='white', fontsize=12)
    
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

# Run
decision_curve = calculate_decision_curve(model, device)
plot_surf = plot_decision_curve(decision_curve)

running = True
clock = pygame.time.Clock()

while running:
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
            
    screen.blit(plot_surf, (100, 100))
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
