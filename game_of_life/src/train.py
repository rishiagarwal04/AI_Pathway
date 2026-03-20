import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
import dataclasses

# Import the model
try:
    from bdh_life import BDH_Life, BDHConfig
except ImportError:
    print("Error: 'bdh_life.py' not found.")
    exit()

# ==========================================
# 1. Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Training Padded BDH (12x12 with Dead Walls) ---")
print(f"Device: {device}")

# Configure the 'Starving Student' Model (Small & Efficient)
config = BDHConfig()
config.n_layer = 3            
config.n_embd = 16           
config.n_head = 1           
config.mlp_internal_dim_multiplier = 4

# CRITICAL CHANGE: Grid is now 12x12 (10x10 + 1 pixel border on all sides)
config.grid_size = 12 

model = BDH_Life(config).to(device)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

# ==========================================
# 2. Padded Data Generation
# ==========================================
def generate_padded_batch(batch_size, device):
    # 1. Generate the inner 10x10 random world
    inner_grid = 10
    inner = (torch.rand(batch_size, 1, inner_grid, inner_grid, device=device) < 0.3).float()
    
    # 2. Pad it with Zeros to make it 12x12
    # This creates the "Wall" that the model can see
    boards = F.pad(inner, (1, 1, 1, 1), mode='constant', value=0.0)
    
    # 3. Calculate Physics on the full 12x12
    kernel = torch.tensor([[[[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]]], device=device)
    
    # We pad AGAIN for the physics calculation (to handle the edge of the 12x12 world)
    # But since the 12x12 edge is already 0, this is just for the convolution math
    boards_calc = F.pad(boards, pad=(1, 1, 1, 1), mode='constant', value=0)
    neighbors = F.conv2d(boards_calc, kernel)
    
    survival = (boards == 1.0) & ((neighbors == 2.0) | (neighbors == 3.0))
    birth = (boards == 0.0) & (neighbors == 3.0)
    next_boards = (survival | birth).float()
    
    # 4. FORCE THE WALLS TO STAY DEAD
    # (Physics usually handles this, but we ensure it to be safe)
    mask = torch.zeros_like(next_boards)
    mask[:, :, 1:-1, 1:-1] = 1.0 # Only the inner 10x10 can be alive
    next_boards = next_boards * mask

    x = boards.view(batch_size, -1)
    y = next_boards.view(batch_size, -1)
    return x, y

# ==========================================
# 3. Training Loop
# ==========================================
BATCH_SIZE = 512    
EPOCHS = 50
SAMPLES_PER_EPOCH = 50000 
NUM_BATCHES = SAMPLES_PER_EPOCH // BATCH_SIZE

def train_model():
    model.train()
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        correct_pixels = 0
        total_pixels = 0
        
        for _ in range(NUM_BATCHES):
            optimizer.zero_grad()
            x_batch, y_batch = generate_padded_batch(BATCH_SIZE, device)
            
            logits, loss, _ = model(x_batch, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float().squeeze(-1)
            correct_pixels += (preds == y_batch).sum().item()
            total_pixels += y_batch.numel()
            
        scheduler.step()
        
        avg_loss = total_loss / NUM_BATCHES
        accuracy = (correct_pixels / total_pixels) * 100.0
        elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}% | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
        
        if accuracy > 99.998:
            print("Perfect Padded Accuracy. Stopping.")
            break

    print("Saving Padded Model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': dataclasses.asdict(config)
    }, 'bdh_life_padded.pth')
    print("Saved to 'bdh_life_padded.pth'")

if __name__ == "__main__":
    train_model()
