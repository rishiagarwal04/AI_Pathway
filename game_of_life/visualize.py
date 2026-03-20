import pygame
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import dataclasses
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

# Layout Settings (GIANT MODE)
CELL_SIZE = 40      
GRID_BOX_SIZE = 450 # <--- INCREASED TO 450px
GAP = 40
SECTION_GAP = 60
VIRTUAL_W = 1200    
VIRTUAL_H = 2600    # <--- TALLER FOR SCROLLING

# Colors
BLACK = (5, 5, 8) 
GRID_COLOR = (40, 40, 45)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)        
NEON_GREEN = (50, 255, 100) 
ORANGE = (255, 140, 0)      
RED_BORDER = (255, 50, 50)
PREDICT_BIRTH = (50, 255, 50)
PREDICT_DEATH = (255, 50, 50)
PREDICT_STABLE = (200, 200, 200)

# Heatmap
C_LOW = (30, 0, 60); C_MID = (200, 0, 80); C_HIGH = (255, 200, 0); C_PEAK = (255, 255, 255)

pygame.init()
display_info = pygame.display.Info()
WINDOW_W = min(1300, display_info.current_w - 50)
WINDOW_H = min(1000, display_info.current_h - 100)
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption(f"BDH Deep Probe: GIANT 64-NEURON EDITION")

virtual_surf = pygame.Surface((VIRTUAL_W, VIRTUAL_H))

# Fonts (Scaled Up)
font_title = pygame.font.SysFont("Arial", 28, bold=True)
font_label = pygame.font.SysFont("Arial", 18, bold=True)
font_val = pygame.font.SysFont("Arial", 12, bold=True)

# ==========================================
# 2. Model Wrapper
# ==========================================
class BDH_MRI(BDH_Life):
    def forward_mri(self, inputs):
        C = self.config
        B, T = inputs.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh 

        x = self.input_proj(inputs.unsqueeze(-1)).unsqueeze(1)
        x = self.ln(x)
        history = []

        for layer_idx in range(C.n_layer):
            layer_snapshot = {}
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)

            Q = x_sparse; K = x_sparse; V = x
            scores = (Q @ K.mT) / math.sqrt(Q.size(-1))
            attn_weights = F.softmax(scores, dim=-1)
            
            layer_snapshot['attn_matrix'] = attn_weights[0].detach() 
            head_out = attn_weights @ V 
            layer_snapshot['heads_spatial'] = head_out[0].detach() 

            yKV = self.ln(self.attn(Q=x_sparse, K=x_sparse, V=x))
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            
            # [144, 64]
            layer_snapshot['neurons_full'] = xy_sparse[0].detach() 
            history.append(layer_snapshot)

            yMLP = (xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder)
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = self.output_head(x.squeeze(1))
        return logits, history

# ==========================================
# 3. Drawing Helpers
# ==========================================
def get_heat_color(val):
    if val < 0.02: return (10, 10, 15)
    if val < 0.33:
        t = val / 0.33
        return (int(C_LOW[0] + (C_MID[0]-C_LOW[0])*t), int(C_LOW[1] + (C_MID[1]-C_LOW[1])*t), int(C_LOW[2] + (C_MID[2]-C_LOW[2])*t))
    elif val < 0.66:
        t = (val - 0.33) / 0.33
        return (int(C_MID[0] + (C_HIGH[0]-C_MID[0])*t), int(C_MID[1] + (C_HIGH[1]-C_MID[1])*t), int(C_MID[2] + (C_HIGH[2]-C_MID[2])*t))
    else:
        t = (val - 0.66) / 0.34
        return (int(C_HIGH[0] + (C_PEAK[0]-C_HIGH[0])*t), int(C_HIGH[1] + (C_PEAK[1]-C_HIGH[1])*t), int(C_HIGH[2] + (C_PEAK[2]-C_HIGH[2])*t))

def draw_spatial_12x12(surface, data_flat, x, y, w, h, scroll_y, mouse_pos, highlight_idx=None, color_override=None):
    """ STRICT 12x12 GRID for Spatial Data """
    mx, my = mouse_pos; mx_virt = mx - 540; my_virt = my + scroll_y
    rows, cols = 12, 12
    cell_w = w / cols; cell_h = h / rows
    
    pygame.draw.rect(surface, (10, 10, 15), (x, y, w, h))
    pygame.draw.rect(surface, GRID_COLOR, (x, y, w, h), 1)
    
    if hasattr(data_flat, 'flatten'): data_flat = data_flat.flatten()
    min_val, max_val = data_flat.min(), data_flat.max()
    rng = max_val - min_val if (max_val - min_val) > 0.0001 else 1.0
    norm_data = (data_flat - min_val) / rng
    
    hover_idx = -1
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            cx = x + c*cell_w; cy = y + r*cell_h
            if cx <= mx_virt < cx + cell_w and cy <= my_virt < cy + cell_h: hover_idx = idx

            if idx < len(norm_data):
                val = float(norm_data[idx]); raw_val = float(data_flat[idx])
                color = get_heat_color(val)
                if color_override is not None and val > 0.02:
                    c_val = int(255 * val)
                    if color_override == CYAN: color = (0, c_val, c_val)
                    else: color = (c_val, int(c_val*0.6), 0)

                if val > 0.02: pygame.draw.rect(surface, color, (cx, cy, cell_w-1, cell_h-1))
                if highlight_idx == idx: pygame.draw.rect(surface, WHITE, (cx, cy, cell_w-1, cell_h-1), 2)
                
                if val > 0.05:
                    val_str = f"{raw_val:.2f}".lstrip('0')
                    if val_str == ".00": val_str = "0"
                    txt_col = BLACK if val > 0.6 else WHITE
                    ts = font_val.render(val_str, True, txt_col)
                    tr = ts.get_rect(center=(cx + cell_w//2, cy + cell_h//2))
                    surface.blit(ts, tr)
    return hover_idx

def draw_concepts_8x8(surface, data_flat, x, y, w, h, scroll_y, mouse_pos, highlight_idx=None):
    """ STRICT 8x8 GRID for 64 Neurons """
    mx, my = mouse_pos; mx_virt = mx - 540; my_virt = my + scroll_y
    rows, cols = 8, 8 # 64 Neurons
    cell_w = w / cols; cell_h = h / rows
    
    pygame.draw.rect(surface, (10, 10, 15), (x, y, w, h))
    pygame.draw.rect(surface, GRID_COLOR, (x, y, w, h), 1)
    
    if hasattr(data_flat, 'flatten'): data_flat = data_flat.flatten()
    min_val, max_val = data_flat.min(), data_flat.max()
    rng = max_val - min_val if (max_val - min_val) > 0.0001 else 1.0
    norm_data = (data_flat - min_val) / rng
    
    hover_idx = -1
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            cx = x + c*cell_w; cy = y + r*cell_h
            if cx <= mx_virt < cx + cell_w and cy <= my_virt < cy + cell_h: hover_idx = idx

            if idx < len(norm_data):
                val = float(norm_data[idx]); raw_val = float(data_flat[idx])
                color = get_heat_color(val)
                if val > 0.02: pygame.draw.rect(surface, color, (cx, cy, cell_w-2, cell_h-2))
                if highlight_idx == idx: pygame.draw.rect(surface, WHITE, (cx, cy, cell_w-2, cell_h-2), 3)

                if val > 0.05:
                    val_str = f"{raw_val:.2f}".lstrip('0')
                    txt_col = BLACK if val > 0.6 else WHITE
                    ts = font_val.render(val_str, True, txt_col)
                    tr = ts.get_rect(center=(cx + cell_w//2, cy + cell_h//2))
                    surface.blit(ts, tr)
    return hover_idx

# ==========================================
# 4. Main Loop
# ==========================================
print("Initializing Giant Probe...")
device = torch.device("cpu")
config = BDHConfig()
config.n_layer = 3; config.n_embd = 16; config.n_head = 1; config.grid_size = 12; config.mlp_internal_dim_multiplier = 4 

model = BDH_MRI(config).to(device)
weights_path = 'bdh_life_12_padded_3_layer_1_head_16_d_4_mlp_100_accuracy.pth'
if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    print("Weights loaded.")
else:
    print("WARNING: Using random weights.")
model.eval()

board = torch.zeros(1, GRID_SIZE * GRID_SIZE)
future_board_flat = torch.zeros(GRID_SIZE * GRID_SIZE)
future_probs_flat = torch.zeros(GRID_SIZE * GRID_SIZE)
last_history = None

clock = pygame.time.Clock()
paused = True
scroll_y = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: pygame.quit(); sys.exit()
        if event.type == pygame.MOUSEWHEEL:
            scroll_y -= event.y * 50 # Faster Scroll
            scroll_y = max(0, min(scroll_y, VIRTUAL_H - WINDOW_H))
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if mx < 540: 
                bx, by = 40, 50; bw = GRID_SIZE * CELL_SIZE
                if bx <= mx < bx + bw and by <= my < by + bw:
                    c = (mx - bx) // CELL_SIZE; r = (my - by) // CELL_SIZE
                    if 0 < c < GRID_SIZE-1 and 0 < r < GRID_SIZE-1:
                        idx = r * GRID_SIZE + c; board[0, idx] = 1.0 - board[0, idx]
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE: paused = not paused
            if event.key == pygame.K_c: board = torch.zeros(1, GRID_SIZE * GRID_SIZE)
            if event.key == pygame.K_r:
                new_board = (torch.rand(1, GRID_SIZE, GRID_SIZE) < 0.2).float()
                mask = torch.zeros_like(new_board); mask[:, 1:-1, 1:-1] = 1.0
                board = (new_board * mask).view(1, -1)
            if event.key == pygame.K_s:
                if paused: board = future_board_flat.view(1, -1)

    with torch.no_grad():
        logits, history = model.forward_mri(board)
        last_history = history
        probs = torch.sigmoid(logits).squeeze(-1)
        probs_grid = probs.view(GRID_SIZE, GRID_SIZE)
        mask = torch.zeros_like(probs_grid); mask[1:-1, 1:-1] = 1.0
        future_probs_flat = (probs_grid * mask).view(-1)
        future_board_flat = (future_probs_flat > 0.5).float()
        if not paused: board = future_board_flat.view(1, -1); pygame.time.delay(80)

    screen.fill(BLACK)

    # --- LEFT PANEL ---
    main_x, main_y = 40, 50
    pygame.draw.rect(screen, RED_BORDER, (main_x+CELL_SIZE-2, main_y+CELL_SIZE-2, (GRID_SIZE-2)*CELL_SIZE+4, (GRID_SIZE-2)*CELL_SIZE+4), 2)
    cur = board.view(-1).numpy()
    for i, val in enumerate(cur):
        r, c = i // GRID_SIZE, i % GRID_SIZE
        pygame.draw.rect(screen, (30,30,30), (main_x+c*CELL_SIZE, main_y+r*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
        if val > 0.5: pygame.draw.rect(screen, WHITE, (main_x+c*CELL_SIZE+1, main_y+r*CELL_SIZE+1, CELL_SIZE-2, CELL_SIZE-2))
    screen.blit(font_title.render("INPUT BOARD", True, WHITE), (main_x, main_y - 30))
    
    future_y = main_y + GRID_SIZE * CELL_SIZE + 50
    pygame.draw.rect(screen, (50, 50, 200), (main_x+CELL_SIZE-2, future_y+CELL_SIZE-2, (GRID_SIZE-2)*CELL_SIZE+4, (GRID_SIZE-2)*CELL_SIZE+4), 2)
    fut = future_probs_flat.numpy()
    for i, val in enumerate(fut):
        r, c = i // GRID_SIZE, i % GRID_SIZE
        col = None
        if cur[i]>0.5 and val>0.5: col=PREDICT_STABLE
        elif cur[i]<0.5 and val>0.5: col=PREDICT_BIRTH
        elif cur[i]>0.5 and val<0.5: pygame.draw.rect(screen, PREDICT_DEATH, (main_x+c*CELL_SIZE+4, future_y+r*CELL_SIZE+4, CELL_SIZE-8, CELL_SIZE-8), 2)
        if col: pygame.draw.rect(screen, col, (main_x+c*CELL_SIZE+1, future_y+r*CELL_SIZE+1, CELL_SIZE-2, CELL_SIZE-2))
    screen.blit(font_title.render("PREDICTION", True, (150, 150, 255)), (main_x, future_y - 30))

    # --- RIGHT PANEL ---
    virtual_surf.fill(BLACK)
    spatial_x = 20
    concept_x = spatial_x + GRID_BOX_SIZE + GAP + SECTION_GAP
    start_y = 50
    
    virtual_surf.blit(font_title.render("SPATIAL (Head 1)", True, CYAN), (spatial_x, start_y - 30))
    virtual_surf.blit(font_title.render("CONCEPTS (64 Neurons)", True, ORANGE), (concept_x, start_y - 30))

    global_hover_idx = None
    global_hover_type = None 
    mouse_pos_real = pygame.mouse.get_pos()

    if last_history:
        # DETECT HOVER
        for l_idx, layer_data in enumerate(last_history):
            y_pos = start_y + l_idx * (GRID_BOX_SIZE + 80)
            
            s_data = layer_data['heads_spatial'].norm(dim=-1).numpy()
            h_spatial = draw_spatial_12x12(virtual_surf, s_data[0], spatial_x, y_pos, GRID_BOX_SIZE, GRID_BOX_SIZE, scroll_y, mouse_pos_real)
            if h_spatial != -1: 
                global_hover_idx = h_spatial
                global_hover_type = 'spatial'

            n_data = layer_data['neurons_full'].mean(dim=0).numpy() # [64]
            h_concept = draw_concepts_8x8(virtual_surf, n_data, concept_x, y_pos, GRID_BOX_SIZE, GRID_BOX_SIZE, scroll_y, mouse_pos_real)
            if h_concept != -1:
                global_hover_idx = h_concept
                global_hover_type = 'concept'

        # DRAW FINAL
        virtual_surf.fill(BLACK) 
        virtual_surf.blit(font_title.render("SPATIAL (Head 1)", True, CYAN), (spatial_x, start_y - 30))
        virtual_surf.blit(font_title.render("CONCEPTS (64 Neurons)", True, ORANGE), (concept_x, start_y - 30))

        for l_idx, layer_data in enumerate(last_history):
            y_pos = start_y + l_idx * (GRID_BOX_SIZE + 80)
            virtual_surf.blit(font_title.render(f"LAYER {l_idx+1}", True, WHITE), (spatial_x - 100, y_pos + GRID_BOX_SIZE//2))
            
            s_data = layer_data['heads_spatial'].norm(dim=-1).numpy()[0]
            n_mean = layer_data['neurons_full'].mean(dim=0).numpy()
            
            spatial_to_draw = s_data
            spatial_color = None
            concept_hl = None

            if global_hover_type == 'spatial':
                attn = layer_data['attn_matrix'][0]
                if global_hover_idx < 144:
                    spatial_to_draw = attn[global_hover_idx].cpu().numpy()
                    spatial_color = CYAN
            elif global_hover_type == 'concept':
                n_full = layer_data['neurons_full']
                if global_hover_idx < 64:
                    spatial_to_draw = n_full[:, global_hover_idx].cpu().numpy()
                    spatial_color = ORANGE
                    concept_hl = global_hover_idx

            draw_spatial_12x12(virtual_surf, spatial_to_draw, spatial_x, y_pos, GRID_BOX_SIZE, GRID_BOX_SIZE, scroll_y, mouse_pos_real, 
                               highlight_idx=(global_hover_idx if global_hover_type=='spatial' else None), 
                               color_override=spatial_color)

            draw_concepts_8x8(virtual_surf, n_mean, concept_x, y_pos, GRID_BOX_SIZE, GRID_BOX_SIZE, scroll_y, mouse_pos_real, 
                              highlight_idx=concept_hl)

    screen.blit(virtual_surf, (540, -scroll_y))
    
    scroll_pct = scroll_y / VIRTUAL_H
    bar_h = (WINDOW_H / VIRTUAL_H) * WINDOW_H
    pygame.draw.rect(screen, (50, 50, 50), (WINDOW_W-8, scroll_pct * WINDOW_H, 8, bar_h))

    pygame.display.flip()
    clock.tick(60)
