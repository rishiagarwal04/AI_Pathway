"""
BDH Visualizer Backend (FastAPI)
Runs the real BDH model and returns full, layer-exact visualization data.

Every layer returns:
  x_in          – residual stream entering this layer  [T, D_show]
  x_sparse      – encoder projections relu(x @ W_enc)  [T, N_show] head-avg
  attn_scores   – causal attention matrix               [T, T]  head-avg, norm
  attn_qk_norms – per-token QK vector norms             [T]
  y_kv          – attention output (V=x weighted)       [T, D_show]
  sigma_before  – fast-weight matrix before Hebbian     [N_show, N_show] head-0
  sigma_delta   – outer-product update sum_t x_t⊗y_t   [N_show, N_show]
  sigma_after   – fast-weight matrix after Hebbian      [N_show, N_show]
  y_sparse      – value projections relu(yKV @ W_enc_v) [T, N_show] head-avg
  xy_gate       – Hebbian gate x*y                      [T, N_show]
  x_out         – residual stream leaving this layer    [T, D_show]
  residual_delta – |x_out - x_in| per token             [T]
  sparsity_x    – % neurons active in x_sparse
  sparsity_y    – % neurons active in y_sparse
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import gc
import sys
import os
import json
import uuid
import shutil
from datetime import datetime, timezone
from typing import Optional

# ── paths ──────────────────────────────────────────────────────────────────
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, 'bdh'))

from bdh import BDH, BDHConfig

# ── app ────────────────────────────────────────────────────────────────────
app = FastAPI(title="BDH Visualizer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── health check (Render pings GET / ) ────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "ok", "service": "BDH Visualizer API"}

# ── helpers ────────────────────────────────────────────────────────────────
def _to_list(t: torch.Tensor) -> list:
    return t.detach().cpu().tolist()

def _norm_01(t: torch.Tensor) -> torch.Tensor:
    mn, mx = t.min(), t.max()
    return (t - mn) / (mx - mn).clamp(min=1e-8)

def _sample_matrix(mat: torch.Tensor, max_dim: int) -> list:
    """Uniformly subsample a 2D [R, C] tensor to [<=max_dim, <=max_dim]."""
    R, C = mat.shape
    rs = torch.linspace(0, R - 1, min(R, max_dim)).long()
    cs = torch.linspace(0, C - 1, min(C, max_dim)).long()
    return _to_list(_norm_01(mat[rs][:, cs]))

def _sample_vec(v: torch.Tensor, max_len: int = 64) -> list:
    if v.numel() <= max_len:
        return _to_list(v)
    idx = torch.linspace(0, v.numel() - 1, max_len).long()
    return _to_list(v[idx])


# ── uploads dir ────────────────────────────────────────────────────────────
UPLOADS_DIR = os.path.join(PARENT_DIR, 'uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)
REGISTRY_PATH = os.path.join(UPLOADS_DIR, '_registry.json')
DEFAULT_CKPT = os.path.join(PARENT_DIR, 'bdh', 'bdh_trained.pt')

# Required keys in a valid BDH state_dict (core layers needed for visualization)
_BDH_REQUIRED_KEYS = {'decoder', 'encoder', 'encoder_v'}
# Optional/alternative keys for Input/Output layers
_BDH_IO_KEYS = {('lm_head', 'output_head'), ('embed.weight', 'input_proj.weight')}


def _validate_bdh_checkpoint(path: str) -> dict:
    """Load a .pt file and verify it is a valid BDH checkpoint.
    Returns the checkpoint dict on success, raises ValueError on failure."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise ValueError(f"Cannot load .pt file: {e}")

    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a dict")
    
    # Check for config (allow dict or object)
    if 'config' not in ckpt:
        raise ValueError("Missing 'config' key – not a BDH checkpoint")
    
    cfg = ckpt['config']
    is_dict = isinstance(cfg, dict)
    
    # Check required config attributes/keys
    # Relaxed check: we just need to be able to read them
    required_attrs = ['n_layer', 'n_embd', 'n_head']
    for attr in required_attrs:
        val = cfg.get(attr) if is_dict else getattr(cfg, attr, None)
        if val is None:
            # It's okay if some are missing, we can default them in load_model, 
            # but ideally a BDH model should have architecture params.
            # We'll be lenient here to allow "Inspect" in UI.
            pass

    if 'model_state_dict' not in ckpt:
        raise ValueError("Missing 'model_state_dict' key – not a BDH checkpoint")

    # Relaxed key check
    # We no longer enforce specific layer names like 'embed' or 'lm_head'
    # to support non-NLP variants (Game of Life, etc).
    
    # Only check that it *looks* like a torch state dict
    if not isinstance(ckpt['model_state_dict'], dict):
         raise ValueError("model_state_dict is not a dictionary")

    return ckpt


class ModelRegistry:
    """Manages a collection of BDH checkpoints stored on disk."""

    def __init__(self):
        self._entries: list[dict] = []
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(REGISTRY_PATH):
            try:
                with open(REGISTRY_PATH, 'r') as f:
                    self._entries = json.load(f)
            except Exception:
                self._entries = []
        
        # Always force-update the default model path to the current environment's path
        # This fixes issues when deploying to a new environment (e.g. Render) where absolute paths differ
        default_entry = next((e for e in self._entries if e['id'] == 'default'), None)
        if default_entry:
            default_entry['path'] = DEFAULT_CKPT
        else:
            # Create default entry if missing
            self._entries.insert(0, {
                'id': 'default',
                'name': 'BDH Default',
                'task': 'Arithmetic',
                'description': 'Default pre-trained BDH checkpoint (byte-level, math-focused)',
                'path': DEFAULT_CKPT,
                'created_at': '2025-01-01T00:00:00Z',
                'deletable': False,
            })
        
        self._save()

    def _save(self):
        with open(REGISTRY_PATH, 'w') as f:
            json.dump(self._entries, f, indent=2)

    def list_all(self) -> list[dict]:
        return self._entries

    def get(self, model_id: str) -> dict | None:
        return next((e for e in self._entries if e['id'] == model_id), None)

    def add(self, name: str, task: str, description: str, filepath: str) -> dict:
        entry = {
            'id': str(uuid.uuid4())[:8],
            'name': name,
            'task': task,
            'description': description,
            'path': filepath,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'deletable': True,
        }
        self._entries.append(entry)
        self._save()
        return entry

    def remove(self, model_id: str) -> bool:
        entry = self.get(model_id)
        if not entry or not entry.get('deletable', True):
            return False
        # Remove file
        if os.path.exists(entry['path']):
            os.remove(entry['path'])
        self._entries = [e for e in self._entries if e['id'] != model_id]
        self._save()
        return True


registry = ModelRegistry()

# Track which model ID is currently loaded
active_model_id: str = 'default'


def _config_to_dict(cfg: BDHConfig) -> dict:
    N = cfg.n_embd * cfg.mlp_internal_dim_multiplier // cfg.n_head
    return {
        'n_layer': cfg.n_layer,
        'n_embd': cfg.n_embd,
        'n_head': cfg.n_head,
        'dropout': cfg.dropout,
        'mlp_internal_dim_multiplier': cfg.mlp_internal_dim_multiplier,
        'vocab_size': cfg.vocab_size,
        'neuron_dim': N,
        'total_params': sum(
            p.numel() for p in
            (torch.zeros(1),)  # placeholder – real count done at load time
        ),
    }


def _compute_prob_bars(logits_2d: torch.Tensor, temperature: float = 0.5,
                       top_k: int = 10, n_bars: int = 10) -> list:
    """Shared helper: temperature-scale, top-k filter, softmax, return prob bars.
    logits_2d: 1-D tensor [vocab_size] (last-token logits).
    Returns list of {token, prob, byte} dicts.
    """
    scaled = logits_2d / max(temperature, 1e-8)
    if top_k and top_k > 0:
        vals, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
        scaled[scaled < vals[-1]] = float('-inf')
    probs = F.softmax(scaled, dim=-1)
    topk_p, topk_i = torch.topk(probs, min(n_bars, probs.size(-1)))
    return [
        {'token': chr(i) if 32 <= i < 127 else f'[{i}]',
         'prob': round(float(p), 4), 'byte': int(i)}
        for p, i in zip(topk_p.tolist(), topk_i.tolist())
    ]


# ── BDH_Explainer ──────────────────────────────────────────────────────────
class BDH_Explainer(BDH):
    """
    Extends BDH with a forward_trace() method that captures every
    intermediate tensor at each layer for precise visualization.
    """
    N_SHOW: int = 32   # neuron matrix side length for heatmaps
    D_SHOW: int = 64   # residual-stream dims to ship to frontend

    def forward_trace(self, idx: torch.Tensor):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        NS = self.N_SHOW
        DS = self.D_SHOW

        # ── embedding ──────────────────────────────────────────────────
        x = self.ln(self.embed(idx).unsqueeze(1))   # B, 1, T, D
        x2 = x.squeeze(0).squeeze(0)               # T, D  (view for snapshots)

        embedding_data = {
            'matrix': _sample_matrix(x2, DS),        # T × DS
            'norms':  _to_list(x2.norm(dim=-1)),     # [T]
        }

        # Running per-head fast-weight (sigma) state – starts at zero
        sigma = [torch.zeros(N, N) for _ in range(nh)]

        layers_data = []
        for level in range(C.n_layer):
            ld = {}

            # ── x_in snapshot ─────────────────────────────────────────
            x2 = x.squeeze(0).squeeze(0)
            x_in_snap = x2.detach().cpu()
            ld['x_in'] = _sample_matrix(x_in_snap, DS)

            # ── sparse encode  x_sparse = relu(x @ W_enc) ─────────────
            x_latent = x @ self.encoder            # B, nh, T, N
            x_sparse = F.relu(x_latent)

            xs_avg = x_sparse[0].mean(dim=0)       # T, N  (head-averaged)
            ld['x_sparse']  = _sample_matrix(xs_avg, NS)
            ld['sparsity_x'] = round(100.0 * (xs_avg > 0).float().mean().item(), 1)

            # ── attention: replicate rope + scores ────────────────────
            r_phases = (
                torch.arange(0, T, device=self.attn.freqs.device,
                             dtype=self.attn.freqs.dtype).view(1, 1, -1, 1)
            ) * self.attn.freqs
            QR = self.attn.rope(r_phases, x_sparse)   # B, nh, T, N
            scores = (QR @ QR.mT).tril(diagonal=-1)   # B, nh, T, T  causal

            attn_avg = scores[0].mean(dim=0).detach().cpu()  # T, T
            ld['attn_scores']   = _to_list(_norm_01(attn_avg.clamp(min=0)))
            ld['attn_qk_norms'] = _to_list(QR[0].norm(dim=-1).mean(dim=0))  # [T]

            yKV = scores @ x      # B, nh, T, D
            yKV = self.ln(yKV)
            ld['y_kv'] = _sample_matrix(yKV[0].mean(dim=0), DS)

            # ── value encode  y_sparse = relu(yKV @ W_enc_v) ──────────
            y_latent = yKV @ self.encoder_v        # B, nh, T, N
            y_sparse = F.relu(y_latent)

            ys_avg = y_sparse[0].mean(dim=0)       # T, N
            ld['y_sparse']   = _sample_matrix(ys_avg, NS)
            ld['sparsity_y'] = round(100.0 * (ys_avg > 0).float().mean().item(), 1)

            # ── Hebbian fast-weight snapshots (head 0) ─────────────────
            #  sigma_new = sigma_old + sum_t  x_t ⊗ y_t
            sig_h = sigma[0].clone()
            ld['sigma_before'] = _sample_matrix(sig_h, NS)

            xs_h0 = x_sparse[0, 0].detach().cpu()   # T, N
            ys_h0 = y_sparse[0, 0].detach().cpu()   # T, N
            delta  = torch.zeros(N, N)
            for t_idx in range(T):
                delta += torch.outer(xs_h0[t_idx], ys_h0[t_idx])

            ld['sigma_delta'] = _sample_matrix(_norm_01(delta.abs()), NS)
            sigma[0] = sigma[0] + delta
            ld['sigma_after'] = _sample_matrix(_norm_01(sigma[0].abs()), NS)

            # ── Graph data: top active neurons with real IDs ───────────
            N_GRAPH = 64  # top neurons for the Hebbian graph
            # Union of active neurons at any token step (head 0)
            any_active = (xs_h0 > 0.001).any(dim=0) | (ys_h0 > 0.001).any(dim=0)
            # Rank by last-token activation strength
            last_act = xs_h0[-1, :]
            # Among active neurons, pick top N_GRAPH
            last_act_masked = last_act.clone()
            last_act_masked[~any_active] = -1
            top_graph_idx = last_act_masked.argsort(descending=True)[:N_GRAPH]
            # Filter to only those actually active somewhere
            top_graph_idx = top_graph_idx[any_active[top_graph_idx]]
            gids = top_graph_idx.tolist()

            ng = len(gids)
            # Per-token activations for these neurons
            ld['graph_neuron_ids'] = gids
            ld['graph_total_neurons'] = int(N)
            ld['graph_total_active'] = int(any_active.sum().item())
            ld['graph_x_sparse'] = xs_h0[:, top_graph_idx].tolist()  # [T, ng]
            ld['graph_y_sparse'] = ys_h0[:, top_graph_idx].tolist()  # [T, ng]
            # Sigma submatrix for these neurons
            sig_sub = sigma[0][top_graph_idx][:, top_graph_idx].detach().cpu()
            ld['graph_sigma'] = sig_sub.tolist()  # [ng, ng]
            # Sigma-before submatrix
            sig_before_sub = sig_h[top_graph_idx][:, top_graph_idx].detach().cpu()
            ld['graph_sigma_before'] = sig_before_sub.tolist()  # [ng, ng]

            # Hub detection: neuron with highest total connection strength
            sig_full = sigma[0].detach().cpu()
            conn_strength = sig_full.abs().sum(dim=1) + sig_full.abs().sum(dim=0)
            hub_idx = int(conn_strength.argmax().item())
            hub_degree = int((sig_full[hub_idx].abs() > 0.01).sum().item())
            hub_strength = float(conn_strength[hub_idx].item())
            ld['hub'] = {
                'neuron_id': hub_idx,
                'degree': hub_degree,
                'total_strength': round(hub_strength, 4),
                'in_graph': hub_idx in gids,
            }
            # If hub not in graph, add it
            if hub_idx not in gids:
                gids.append(hub_idx)
                ld['graph_neuron_ids'] = gids
                # Recompute submatrices with hub included
                all_idx = torch.tensor(gids, dtype=torch.long)
                ld['graph_x_sparse'] = xs_h0[:, all_idx].tolist()
                ld['graph_y_sparse'] = ys_h0[:, all_idx].tolist()
                sig_sub2 = sigma[0][all_idx][:, all_idx].detach().cpu()
                ld['graph_sigma'] = sig_sub2.tolist()
                sig_before_sub2 = sig_h[all_idx][:, all_idx].detach().cpu()
                ld['graph_sigma_before'] = sig_before_sub2.tolist()

            # ── Hebbian gate x * y ─────────────────────────────────────
            xy_sparse = x_sparse * y_sparse
            ld['xy_gate'] = _sample_matrix(xy_sparse[0].mean(dim=0), NS)

            # ── feedforward decode & residual add ──────────────────────
            xy_drop = self.drop(xy_sparse)
            yMLP = (
                xy_drop.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )
            y = self.ln(yMLP)
            x = self.ln(x + y)

            # ── x_out snapshot ────────────────────────────────────────
            x_out_snap = x.squeeze(0).squeeze(0).detach().cpu()
            ld['x_out'] = _sample_matrix(x_out_snap, DS)
            ld['residual_delta'] = _to_list(
                (x_out_snap - x_in_snap).norm(dim=-1)
            )

            layers_data.append(ld)

        logits = x.view(B, T, D) @ self.lm_head
        return logits, embedding_data, layers_data


# ── global model ───────────────────────────────────────────────────────────
model: BDH_Explainer | None = None
config: BDHConfig | None = None

PROMPTS = ["2+2=4", "How are you ?", "The capital of France is Paris"]


def _load_model(model_id: str = 'default'):
    """Load a BDH checkpoint by registry ID."""
    global model, config, active_model_id

    entry = registry.get(model_id)
    if not entry:
        raise ValueError(f"Unknown model id: {model_id}")

    ckpt_path = entry['path']
    if os.path.exists(ckpt_path):
        print(f"[backend] Loading checkpoint '{entry['name']}' from {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            sd = ckpt['model_state_dict']
            raw_config = ckpt['config']

            # 1. Normalize Config to BDHConfig object
            if isinstance(raw_config, dict):
                # Infer vocab_size if missing (common in custom models)
                if 'vocab_size' not in raw_config:
                    # Try to infer from output head weight
                    if 'output_head.weight' in sd:
                        raw_config['vocab_size'] = sd['output_head.weight'].shape[0]
                    # Or input embeddings
                    elif 'embed.weight' in sd:
                        raw_config['vocab_size'] = sd['embed.weight'].shape[0]
                    else:
                        print("[backend] Warning: vocab_size missing, defaulting to 256")
                        raw_config['vocab_size'] = 256

                # Keep only valid BDHConfig fields to avoid errors
                # This explicitly filters out 'grid_size', 'input_shape', etc.
                valid_fields = set(BDHConfig.__dataclass_fields__.keys())
                filtered_cfg = {k: v for k, v in raw_config.items() if k in valid_fields}
                
                # Ensure defaults for strict fields if missing
                defaults = {'n_layer': 4, 'n_embd': 128, 'n_head': 4, 
                            'mlp_internal_dim_multiplier': 4}
                for k, v in defaults.items():
                    if k not in filtered_cfg:
                        filtered_cfg[k] = v
                
                config = BDHConfig(**filtered_cfg)
            else:
                config = raw_config

            # 2. Initialize Model
            model = BDH_Explainer(config)

            # 3. Load State Dict (Relaxed)
            # This handles models with DIFFERENT layer names (e.g. input_proj vs embed)
            # The layers that match will load. The ones that don't will restart random.
            # This is "safer" than crashing, though functionality might be impaired.
            keys = model.load_state_dict(sd, strict=False)
            if keys.missing_keys:
                print(f"[backend] Partial load. Missing keys: {len(keys.missing_keys)} (e.g. {keys.missing_keys[:3]})")
            if keys.unexpected_keys:
                print(f"[backend] Unexpected keys: {len(keys.unexpected_keys)} (e.g. {keys.unexpected_keys[:3]})")

        except Exception as e:
            print(f"[backend] Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    else:
        print("[backend] No checkpoint – using fresh (untrained) model")
        config = BDHConfig(
            n_layer=4, n_embd=128, dropout=0.1,
            n_head=2, mlp_internal_dim_multiplier=8, vocab_size=256,
        )
        model = BDH_Explainer(config)

    model.eval()
    active_model_id = model_id
    print(f"[backend] Ready  layers={config.n_layer}  d={config.n_embd}  heads={config.n_head}")


# ── request schemas ────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    prompt: str = "2+2="
    max_new_tokens: int = 6
    temperature: float = 0.5
    top_k: int = 10


class NeuronActivationsRequest(BaseModel):
    prompt: str = "2+2="
    temperature: float = 0.5
    top_k: int = 10


class AblateRequest(BaseModel):
    prompt: str = "2+2="
    ablations: list = []  # list of {"layer": int, "neurons": [int, ...]}
    temperature: float = 0.5
    top_k: int = 10


# ── helper: run forward + get neuron activations per layer ─────────────────
def _get_neuron_activations(prompt_str: str, temperature: float = 0.5,
                            top_k: int = 10) -> dict:
    """Run the model on a prompt and return raw neuron activations per layer.
    Memory-optimized: caps returned neurons and cleans up tensors eagerly."""
    byte_tokens = list(bytearray(prompt_str, 'utf-8'))
    token_chars = [chr(b) if 32 <= b < 127 else f'[{b}]' for b in byte_tokens]
    tokens_t = torch.tensor([byte_tokens], dtype=torch.long)

    C = config
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    MAX_NEURONS_RETURNED = 128  # cap to save memory/bandwidth

    with torch.no_grad():
        x = model.ln(model.embed(tokens_t).unsqueeze(1))  # B,1,T,D
        T = len(byte_tokens)

        layers_activations = []
        for level in range(C.n_layer):
            x_latent = x @ model.encoder
            x_sparse = F.relu(x_latent)  # B, nh, T, N

            xs_avg = x_sparse[0].mean(dim=0)  # T, N  (head-averaged)

            neuron_mean = xs_avg.mean(dim=0)  # [N]
            neuron_max  = xs_avg.max(dim=0).values  # [N]

            active_mask = (xs_avg > 0).float()
            per_neuron_frac = active_mask.mean(dim=0)  # fraction of tokens each neuron fires for

            # Attention
            r_phases = (
                torch.arange(0, T, device=model.attn.freqs.device,
                             dtype=model.attn.freqs.dtype).view(1, 1, -1, 1)
            ) * model.attn.freqs
            QR = model.attn.rope(r_phases, x_sparse)
            scores = (QR @ QR.mT).tril(diagonal=-1)
            yKV = scores @ x
            yKV = model.ln(yKV)

            y_latent = yKV @ model.encoder_v
            y_sparse = F.relu(y_latent)
            ys_avg = y_sparse[0].mean(dim=0)  # T, N
            y_neuron_mean = ys_avg.mean(dim=0)  # [N]

            # Select top neurons by max activation (capped)
            ACTIVATION_THRESHOLD = 0.001
            sig_mask = neuron_max > ACTIVATION_THRESHOLD
            sig_ids_all = torch.where(sig_mask)[0]
            active_count = int(sig_ids_all.numel())

            # Sort by max activation descending, take top MAX_NEURONS_RETURNED
            if sig_ids_all.numel() > MAX_NEURONS_RETURNED:
                topk_vals, topk_idx = torch.topk(
                    neuron_max[sig_ids_all], MAX_NEURONS_RETURNED
                )
                sig_ids = sig_ids_all[topk_idx].tolist()
            else:
                sig_ids = sig_ids_all.tolist()

            # Move to CPU lists once, then build dict from Python scalars
            nm_cpu = neuron_mean.detach().cpu()
            nmax_cpu = neuron_max.detach().cpu()
            ynm_cpu = y_neuron_mean.detach().cpu()
            pnf_cpu = per_neuron_frac.detach().cpu()
            pt_cpu = xs_avg.detach().cpu()  # [T, N]

            neurons_dict = {}
            for n_idx in sig_ids:
                neurons_dict[str(n_idx)] = {
                    'id': int(n_idx),
                    'layer': level,
                    'x_activation': round(float(nm_cpu[n_idx]), 5),
                    'x_max': round(float(nmax_cpu[n_idx]), 5),
                    'y_activation': round(float(ynm_cpu[n_idx]), 5),
                    'fire_rate': round(float(pnf_cpu[n_idx]), 4),
                    'per_token': [round(float(pt_cpu[t, n_idx]), 5) for t in range(T)],
                }

            # Heatmap: [T x N_show] (sampled overview)
            NS = min(N, 64)
            heatmap = _sample_matrix(pt_cpu, NS)

            # Clean up large intermediates
            del xs_avg, neuron_mean, neuron_max, active_mask, per_neuron_frac
            del ys_avg, y_neuron_mean, nm_cpu, nmax_cpu, ynm_cpu, pnf_cpu, pt_cpu
            del x_latent, x_sparse, y_latent, y_sparse, QR, scores

            layers_activations.append({
                'layer': level,
                'neurons': neurons_dict,
                'total_neurons': N,
                'active_count': active_count,
                'heatmap': heatmap,
                'sparsity_pct': round(100.0 * active_count / max(N, 1), 1),
            })

            # Continue forward pass (recompute needed tensors cheaply)
            x_latent2 = x @ model.encoder
            x_sparse2 = F.relu(x_latent2)
            r_phases2 = (
                torch.arange(0, T, device=model.attn.freqs.device,
                             dtype=model.attn.freqs.dtype).view(1, 1, -1, 1)
            ) * model.attn.freqs
            QR2 = model.attn.rope(r_phases2, x_sparse2)
            scores2 = (QR2 @ QR2.mT).tril(diagonal=-1)
            yKV2 = scores2 @ x
            yKV2 = model.ln(yKV2)
            y_latent2 = yKV2 @ model.encoder_v
            y_sparse2 = F.relu(y_latent2)

            xy_sparse = x_sparse2 * y_sparse2
            xy_drop = model.drop(xy_sparse)
            B = 1
            yMLP = (
                xy_drop.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
            )
            y = model.ln(yMLP)
            x = model.ln(x + y)
            del x_latent2, x_sparse2, y_latent2, y_sparse2, QR2, scores2, yKV2

        # Final logits
        logits = x.view(1, T, D) @ model.lm_head
        prob_bars = _compute_prob_bars(logits[0, -1, :], temperature, top_k)
        del logits

    gc.collect()
    return {
        'prompt': prompt_str,
        'tokens': token_chars,
        'n_tokens': T,
        'layers': layers_activations,
        'prob_bars': prob_bars,
    }


# ── helper: forward pass with ablation ─────────────────────────────────────
def _run_with_ablation(prompt_str: str, ablations: list,
                       temperature: float = 0.5, top_k: int = 10) -> dict:
    """
    Run the model on a prompt with specific neurons zeroed out.
    ablations: list of {"layer": int, "neurons": [int,...]}
    """
    byte_tokens = list(bytearray(prompt_str, 'utf-8'))
    token_chars = [chr(b) if 32 <= b < 127 else f'[{b}]' for b in byte_tokens]
    tokens_t = torch.tensor([byte_tokens], dtype=torch.long)

    C = config
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    T = len(byte_tokens)

    # Build ablation lookup: layer -> set of neuron indices
    ablation_map: dict[int, set] = {}
    for abl in ablations:
        layer_idx = abl.get('layer', -1)
        neuron_ids = abl.get('neurons', [])
        if layer_idx >= 0:
            ablation_map.setdefault(layer_idx, set()).update(neuron_ids)

    with torch.no_grad():
        x = model.ln(model.embed(tokens_t).unsqueeze(1))  # B,1,T,D

        per_layer_sparsity = []
        for level in range(C.n_layer):
            x_latent = x @ model.encoder
            x_sparse = F.relu(x_latent)  # B, nh, T, N

            # Apply ablation: zero out specific neurons
            if level in ablation_map:
                for n_idx in ablation_map[level]:
                    if n_idx < N:
                        x_sparse[:, :, :, n_idx] = 0.0

            xs_avg = x_sparse[0].mean(dim=0)
            per_layer_sparsity.append(
                round(100.0 * (xs_avg > 0).float().mean().item(), 1)
            )

            # Attention
            r_phases = (
                torch.arange(0, T, device=model.attn.freqs.device,
                             dtype=model.attn.freqs.dtype).view(1, 1, -1, 1)
            ) * model.attn.freqs
            QR = model.attn.rope(r_phases, x_sparse)
            scores = (QR @ QR.mT).tril(diagonal=-1)
            yKV = scores @ x
            yKV = model.ln(yKV)

            # Value encode
            y_latent = yKV @ model.encoder_v
            y_sparse = F.relu(y_latent)

            # Hebbian gate + decode
            xy_sparse = x_sparse * y_sparse
            xy_drop = model.drop(xy_sparse)
            B = 1
            yMLP = (
                xy_drop.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
            )
            y = model.ln(yMLP)
            x = model.ln(x + y)

        # Final logits
        logits = x.view(1, T, D) @ model.lm_head
        prob_bars = _compute_prob_bars(logits[0, -1, :], temperature, top_k)

        # Sample next token from logits (instead of full model.generate to save RAM)
        scaled = logits[0, -1, :] / max(temperature, 1e-8)
        if top_k and top_k > 0:
            vals, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
            scaled[scaled < vals[-1]] = float('-inf')
        next_probs = F.softmax(scaled, dim=-1)
        next_token = int(torch.multinomial(next_probs, 1).item())
        continuation_char = chr(next_token) if 32 <= next_token < 127 else f'[{next_token}]'

        del logits, scaled, next_probs

    gc.collect()
    return {
        'prompt': prompt_str,
        'tokens': token_chars,
        'prob_bars': prob_bars,
        'generated': prompt_str + continuation_char,
        'continuation': continuation_char,
        'sparsity_per_layer': per_layer_sparsity,
        'ablations_applied': [
            {'layer': k, 'neurons': sorted(v)} for k, v in ablation_map.items()
        ],
    }


# ── routes ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/api/model/info")
def model_info():
    if config is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    N = config.n_embd * config.mlp_internal_dim_multiplier // config.n_head
    return {
        "n_layer":   config.n_layer,
        "n_embd":    config.n_embd,
        "n_head":    config.n_head,
        "neuron_dim": N,
        "vocab_size": config.vocab_size,
        "n_show":    BDH_Explainer.N_SHOW,
        "d_show":    BDH_Explainer.D_SHOW,
        "prompts":   PROMPTS,
    }


@app.post("/api/run")
def run_prompt(req: RunRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Safety check: Validating vocab size against byte tokens (0-255)
    # Text prompting requires a vocabulary that covers byte range.
    if config and config.vocab_size < 256:
         raise HTTPException(status_code=400, detail=f"Model vocab_size ({config.vocab_size}) is too small for text prompting (needs >= 256). Use a compatible specialized viewer.")

    prompt = req.prompt
    byte_tokens = list(bytearray(prompt, 'utf-8'))
    token_chars = [chr(b) if 32 <= b < 127 else f'[{b}]' for b in byte_tokens]
    tokens_t = torch.tensor([byte_tokens], dtype=torch.long)

    try:
        with torch.no_grad():
            logits, embedding_data, layers_data = model.forward_trace(tokens_t)
    except IndexError:
         # Likely token ID > vocab_size
         raise HTTPException(status_code=400, detail="Token index out of bounds. Model vocabulary is smaller than the input characters used.")
    except Exception as e:
         print(f"Run error: {e}")
         raise HTTPException(status_code=500, detail=f"Model execution failed: {str(e)}")

    # top-k next-token probabilities
    last_logits = logits[0, -1, :] / max(req.temperature, 1e-8)
    if req.top_k:
        vals, _ = torch.topk(last_logits, min(req.top_k, last_logits.size(-1)))
        last_logits[last_logits < vals[-1]] = float('-inf')
    probs = F.softmax(last_logits, dim=-1)
    topk_p, topk_i = torch.topk(probs, min(10, probs.size(-1)))
    prob_bars = [
        {'token': chr(i) if 32 <= i < 127 else f'[{i}]',
         'prob': round(float(p), 4), 'byte': int(i)}
        for p, i in zip(topk_p.tolist(), topk_i.tolist())
    ]

    # generate continuation
    gen = tokens_t.clone()
    with torch.no_grad():
        gen = model.generate(gen, max_new_tokens=req.max_new_tokens,
                             temperature=req.temperature,
                             top_k=req.top_k if req.top_k and req.top_k > 0 else None)
    generated_text = bytes(gen[0].tolist()).decode(errors='replace')
    continuation = generated_text[len(prompt):]

    # Free intermediate tensors
    del logits, tokens_t, gen
    gc.collect()

    return {
        'prompt':       prompt,
        'tokens':       token_chars,
        'n_tokens':     len(byte_tokens),
        'embedding':    embedding_data,
        'layers':       layers_data,
        'prob_bars':    prob_bars,
        'generated':    generated_text,
        'continuation': continuation,
        'config': {
            'n_layer': config.n_layer,
            'n_embd':  config.n_embd,
            'n_head':  config.n_head,
            'n_show':  BDH_Explainer.N_SHOW,
            'd_show':  BDH_Explainer.D_SHOW,
        },
    }


@app.post("/api/neuron/activations")
def neuron_activations(req: NeuronActivationsRequest):
    """Get per-layer neuron activations for a prompt. Used by Concept Explorer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = _get_neuron_activations(req.prompt, req.temperature, req.top_k)
    gc.collect()
    return result


@app.post("/api/ablate")
def ablate(req: AblateRequest):
    """Run forward pass with specified neurons ablated. Used by Ablation Playground."""
    gc.collect()  # pre-clean before heavy operation
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _run_with_ablation(req.prompt, req.ablations,
                              req.temperature, req.top_k)


# ── Model Explorer endpoints ──────────────────────────────────────────────

@app.get("/api/models")
def list_models():
    """List all registered BDH checkpoints with their metadata."""
    models = []
    for entry in registry.list_all():
        info = {'id': entry['id'], 'name': entry['name'],
                'task': entry['task'], 'description': entry['description'],
                'created_at': entry['created_at'],
                'deletable': entry.get('deletable', True),
                'active': entry['id'] == active_model_id}

        # Peek into the checkpoint to get config info (lightweight)
        if os.path.exists(entry['path']):
            try:
                ckpt = torch.load(entry['path'], map_location='cpu', weights_only=False)
                cfg = ckpt.get('config', {})
                sd = ckpt.get('model_state_dict', {})
                
                # Helper to get attributes safely from object or dict
                def get_cfg_val(key, default=0):
                    if isinstance(cfg, dict):
                        return cfg.get(key, default)
                    return getattr(cfg, key, default)

                n_layer = get_cfg_val('n_layer', 4)
                n_embd = get_cfg_val('n_embd', 128)
                n_head = get_cfg_val('n_head', 2)
                vocab_size = get_cfg_val('vocab_size', 256) or 256 # guard None
                mlp_mul = get_cfg_val('mlp_internal_dim_multiplier', 8)
                
                total_params = sum(v.numel() for v in sd.values())
                
                info['config'] = {
                    'n_layer': int(n_layer),
                    'n_embd': int(n_embd),
                    'n_head': int(n_head),
                    'vocab_size': int(vocab_size),
                    'mlp_internal_dim_multiplier': int(mlp_mul),
                    'neuron_dim': int(n_embd * mlp_mul / max(n_head, 1)),
                    'total_params': total_params,
                }
                del ckpt, sd, cfg
            except Exception as e:
                # print(f"Error reading config for {entry['name']}: {e}")
                info['config'] = None
        else:
            info['config'] = None

        models.append(info)
    gc.collect()
    return models


@app.post("/api/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form("Untitled BDH"),
    task: str = Form("General"),
    description: str = Form(""),
):
    """Upload a new BDH checkpoint. Validates it is a real BDH model."""
    if not file.filename or not (file.filename.endswith('.pt') or file.filename.endswith('.pth')):
        raise HTTPException(status_code=400, detail="File must be .pt or .pth")

    # Save to temp location first
    uid = str(uuid.uuid4())[:8]
    filename = f"bdh_{uid}.pt"
    filepath = os.path.join(UPLOADS_DIR, filename)

    try:
        with open(filepath, 'wb') as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Validate it's a real BDH checkpoint
    try:
        _validate_bdh_checkpoint(filepath)
    except ValueError as e:
        os.remove(filepath)
        raise HTTPException(status_code=400, detail=str(e))

    # Register it
    entry = registry.add(name=name, task=task, description=description, filepath=filepath)
    return {"message": f"Model '{name}' uploaded successfully", "model": entry}


@app.post("/api/models/select")
def select_model(body: dict):
    """Switch the active model to a different BDH checkpoint."""
    model_id = body.get('id')
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing 'id'")

    entry = registry.get(model_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    try:
        _load_model(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    gc.collect()
    return {"message": f"Switched to '{entry['name']}'", "active_id": model_id,
            "config": {
                'n_layer': config.n_layer,
                'n_embd': config.n_embd,
                'n_head': config.n_head,
                'vocab_size': config.vocab_size,
            }}


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str):
    """Delete a custom BDH checkpoint (cannot delete the default)."""
    if model_id == 'default':
        raise HTTPException(status_code=403, detail="Cannot delete the default model")
    if model_id == active_model_id:
        raise HTTPException(status_code=400,
                            detail="Cannot delete the active model. Switch to another model first.")
    if not registry.remove(model_id):
        raise HTTPException(status_code=404, detail="Model not found or cannot be deleted")
    return {"message": "Deleted", "id": model_id}


@app.get("/api/model/source")
def model_source():
    """Return whether the active model is default or custom."""
    return {"source": "default" if active_model_id == "default" else "custom",
            "active_id": active_model_id}


@app.post("/api/upload_model")
async def legacy_upload_model(file: UploadFile = File(...)):
    """Legacy upload endpoint used by ModelUpload.svelte – wraps the new flow."""
    if not file.filename or not (file.filename.endswith('.pt') or file.filename.endswith('.pth')):
        raise HTTPException(status_code=400, detail="File must be .pt or .pth")

    uid = str(uuid.uuid4())[:8]
    filename = f"bdh_{uid}.pt"
    filepath = os.path.join(UPLOADS_DIR, filename)

    try:
        with open(filepath, 'wb') as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        _validate_bdh_checkpoint(filepath)
    except ValueError as e:
        os.remove(filepath)
        raise HTTPException(status_code=400, detail=str(e))

    entry = registry.add(name="Custom Upload", task="General",
                         description="Uploaded via quick upload", filepath=filepath)
    _load_model(entry['id'])
    gc.collect()

    return {"message": "Model loaded!", "model_info": {
        'n_layer': config.n_layer, 'n_embd': config.n_embd,
        'n_head': config.n_head, 'vocab_size': config.vocab_size,
    }}


@app.post("/api/reset_model")
def reset_model():
    """Reset to the default BDH model."""
    _load_model('default')
    gc.collect()
    return {"message": "Reset to default model", "model_info": {
        'n_layer': config.n_layer, 'n_embd': config.n_embd,
        'n_head': config.n_head, 'vocab_size': config.vocab_size,
    }}


# ── startup ────────────────────────────────────────────────────────────────
_load_model()

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 5000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)
