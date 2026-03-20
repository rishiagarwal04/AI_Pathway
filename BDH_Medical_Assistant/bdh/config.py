"""
BDH Configuration
"""
import dataclasses


@dataclasses.dataclass
class BDHConfig:
    """Configuration for BDH model with TTT/DeltaNet-style memory."""
    
    # Architecture
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 6
    mlp_internal_dim_multiplier: int = 12  # N = 12*384/6 = 768 per head
    vocab_size: int = 50257
    block_size: int = 512
    dropout: float = 0.1

    # Memory config
    use_memory: bool = False
    memory_lr: float = 0.1
    memory_decay: float = 1.0
    memory_max_norm: float = 100.0
    memory_train_prob: float = 0.3

    # Inference controls
    memory_freeze: bool = False
    memory_retrieval_scale: float = 1.0

    @property
    def N(self):
        """Dimension per head for sparse encoding."""
        return self.mlp_internal_dim_multiplier * self.n_embd // self.n_head
