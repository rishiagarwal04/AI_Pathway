import pygame
import torch
import torch.nn.functional as F
import torch.nn as nn
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
BATCH_SIZE = 64 
BLACK = (5, 5, 8) 

pygame.init()
screen = pygame.display.set_mode((1200, 900))
pygame.display.set_caption(f"BDH Logic: Universal Wiretap")

# ==========================================
# 2. Model Wrapper
# ==========================================
class BDH_MRI(BDH_Life):
    pass 

# ==========================================
# 3. The Universal Wiretap
# ==========================================
# We store every single tensor that passes through the choke point
captured_streams = []

def universal_hook(module, input, output):
    # 'output' is the tensor leaving the layer
    if isinstance(output, tuple):
        val = output[0]
    else:
        val = output
    captured_streams.append(val.detach())

def attach_universal_hook(model):
    """
    Finds the busiest layer (usually LayerNorm) and hooks it.
    """
    print("Searching for a valid choke point (LayerNorm)...")
    
    target_layer = None
    target_name = ""
    
    # Priority 1: 'ln' (Common in your previous snippets)
    if hasattr(model, 'ln') and isinstance(model.ln, nn.Module):
        target_layer = model.ln
        target_name = "model.ln"
    # Priority 2: 'ln1' or 'ln_1'
    elif hasattr(model, 'ln1') and isinstance(model.ln1, nn.Module):
        target_layer = model.ln1
        target_name = "model.ln1"
    # Priority 3: Any LayerNorm
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                target_layer = module
                target_name = name
                break
    
    if target_layer:
        print(f" -> Wiretap attached to: {target_name}")
        target_layer.register_forward_hook(universal_hook)
        return True
    else:
        print("ERROR: No LayerNorm found to hook.")
        return False

def analyze_logic_flow(model, device):
    print("Generating data...")
    boards = []
    # Use many densities to get good data for 0-8 neighbors
    for density in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]:
        b = (torch.rand(30, GRID_SIZE, GRID_SIZE) < density).float()
        boards.append(b)
    inputs = torch.cat(boards, dim=0).view(-1, GRID_SIZE*GRID_SIZE).to(device)
    
    # Ground Truth Neighbors
    padded = inputs.view(-1, 1, GRID_SIZE, GRID_SIZE)
    padded = F.pad(padded, (1,1,1,1), mode='constant', value=0)
    kernel = torch.tensor([[[[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]]], device=device)
    neighbor_counts = F.conv2d(padded, kernel).view(-1, 144).detach().cpu().numpy().flatten()
    
    print("Running REAL Model Forward Pass...")
    
    # Clear previous captures
    global captured_streams
    captured_streams = []
    
    with torch.no_grad():
        # 1. Run the model
        # The hook will fire multiple times and fill 'captured_streams'
        _ = model(inputs)
        
        print(f"Captured {len(captured_streams)} snapshots during forward pass.")
        
        # 2. Decode each snapshot
        results = {}
        
        # We need the Output Head weights
        if hasattr(model.output_head, 'weight'):
            h_w = model.output_head.weight
            if h_w.shape[0] == 1: h_w = h_w.T
            bias = model.output_head.bias if model.output_head.bias is not None else 0
        else:
            h_w = model.output_head.T # Raw parameter fallback
            bias = 0

        for i, stream in enumerate(captured_streams):
            # Stream shape: [B, 144, D] or [B, 1, 144, D]
            if stream.dim() == 4: stream = stream.squeeze(1)
            
            # Apply Logit Lens: stream @ Head
            logits = (stream @ h_w).squeeze() + bias
            results[i] = logits.view(-1).cpu().numpy()

    # Binning
    curves = {}
    for step_idx in results:
        curve = np.zeros(9)
        logits = results[step_idx]
        probs = 1.0 / (1.0 + np.exp(-logits)) 
        
        for n in range(9):
            mask = (neighbor_counts == n)
            if np.sum(mask) > 0:
                curve[n] = probs[mask].mean()
        curves[step_idx] = curve
            
    return curves

# ==========================================
# 4. Plotting
# ==========================================
def plot_logic_evolution(curves):
    fig = plt.figure(figsize=(10, 6), facecolor='#050508')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#050508')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(curves)))
    
    # Plot layers
    for i in sorted(curves.keys()):
        width = 4 if i == len(curves)-1 else 2
        alpha = 1.0 if i == len(curves)-1 else 0.7
        label = f"Step {i} (Input)" if i == 0 else f"Step {i}"
        
        ax.plot(range(9), curves[i], color=colors[i], linewidth=width, 
                label=label, marker='o', alpha=alpha)

    # Reference Rule
    ideal_x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ideal_y = [0, 0, 0.5, 1, 0, 0, 0, 0, 0]
    ax.plot(ideal_x, ideal_y, color='white', alpha=0.3, label='Ideal Rule', linestyle=':')

    ax.set_title("Evolution of Logic: Universal Wiretap", color='white', fontsize=20)
    ax.set_xlabel("Number of Neighbors", color='white', fontsize=14)
    ax.set_ylabel("Predicted Probability", color='white', fontsize=14)
    ax.grid(color='#333333')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#151520', labelcolor='white', fontsize=12)
    
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

# 1. Attach Hook
success = attach_universal_hook(model)

if success:
    # 2. Run Real Forward Pass
    curves = analyze_logic_flow(model, device)
    plot_surf = plot_logic_evolution(curves)

    running = True
    while running:
        screen.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
                
        screen.blit(plot_surf, (100, 100))
        pygame.display.flip()

pygame.quit()
