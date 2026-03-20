import dataclasses
import math
import torch
import torch.nn.functional as F
from torch import nn

# ==========================================
# 1. Configuration (Upgraded Size)
# ==========================================
@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 3              # INCREASED: Deeper thought process
    n_embd: int = 16             # INCREASED: Wider capacity
    n_head: int = 1              # INCREASED: More attention heads
    dropout: float = 0.0
    mlp_internal_dim_multiplier: int = 4 
    grid_size: int = 12           # Fixed 10x10 Grid

# ==========================================
# 2. Simplified 2D RoPE (Fixed Size)
# ==========================================
def get_freqs_2d_fixed(h, w, dim, theta=10000.0):
    """
    Generates a 2D grid of frequencies for a FIXED size.
    Returns flattened shape: (H*W, Dim/2) directly.
    """
    # Note: dim is the HEAD dimension (e.g., 128 / 8 = 16)
    dim_half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim_half, 2).float() / dim_half))
    
    x = torch.arange(w, dtype=torch.float32)
    y = torch.arange(h, dtype=torch.float32)
    
    freqs_x = torch.outer(x, freqs) 
    freqs_y = torch.outer(y, freqs) 
    
    freqs_x_complex = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_y_complex = torch.polar(torch.ones_like(freqs_y), freqs_y)
    
    freqs_x_grid = freqs_x_complex.unsqueeze(0).expand(h, -1, -1)
    freqs_y_grid = freqs_y_complex.unsqueeze(1).expand(-1, w, -1)
    
    freqs_2d = torch.cat([freqs_x_grid, freqs_y_grid], dim=-1)
    
    # Flatten immediately so it is ready for the transformer
    return freqs_2d.view(-1, freqs_2d.shape[-1])

def apply_rotary_emb(x, freqs):
    # View x as complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Reshape freqs for broadcasting: (T, D/2) -> (1, 1, T, D/2)
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    
    # Rotate
    x_rotated = x_complex * freqs
    
    # View back as real
    x_out = torch.view_as_real(x_rotated).flatten(3)
    return x_out.type_as(x)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # RoPE applies to the Keys/Queries, which live in the SPARSE dimension in BDH
        # Calculation: (128 * 16) // 8 = 256 dim per head
        sparse_head_dim = (config.n_embd * config.mlp_internal_dim_multiplier) // config.n_head
        
        # Precompute frequencies for the exact 10x10 grid
        self.freqs = get_freqs_2d_fixed(config.grid_size, config.grid_size, sparse_head_dim)
        self.register_buffer('freqs_buffer', self.freqs)

    def forward(self, Q, K, V):
        # Q shape: (Batch, Heads, T, Dim)
        # Apply Fixed RoPE
        QR = apply_rotary_emb(Q, self.freqs_buffer)
        KR = apply_rotary_emb(K, self.freqs_buffer)

        scores = (QR @ KR.mT) / math.sqrt(QR.size(-1))
        attn = F.softmax(scores, dim=-1)
        
        return attn @ V

# ==========================================
# 3. The BDH Spatial Brain
# ==========================================
class BDH_Life(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        # N is the sparse dimension width
        N = config.mlp_internal_dim_multiplier * D // nh

        self.input_proj = nn.Linear(1, D)
        
        # Main Concept Matrices
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))

        self.attn = Attention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)
        
        self.output_head = nn.Linear(D, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs, targets=None):
        C = self.config
        B, T = inputs.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.input_proj(inputs.unsqueeze(-1)).unsqueeze(1)
        x = self.ln(x)  
        brain_states = []

        for level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)  

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            
            xy_sparse = x_sparse * y_sparse  
            xy_sparse = self.drop(xy_sparse)
            brain_states.append(xy_sparse.detach())

            yMLP = (xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder) 
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = self.output_head(x.squeeze(1)) 

        loss = None
        if targets is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.view(-1), targets.view(-1).float())

        return logits, loss, brain_states
