"""
Memory utilities for BDH

Contains:
- InferenceLearner: Learn facts at inference time
- PositionAwareMemory: External memory with position-weighted keys
- LatentRAGCache: Simple retrieval-augmented cache
"""
import torch
import torch.nn.functional as F


class InferenceLearner:
    """
    Handles learning facts at inference time and querying memory.

    Usage:
        learner = InferenceLearner(model, enc, device)
        learner.enable()
        learner.learn("Patient John has diabetes.")
        response = learner.ask("What condition does John have?")
    """
    
    def __init__(self, model, tokenizer, device: str = 'cuda'):
        self.model = model
        self.enc = tokenizer
        self.device = device
        self.facts = []

    def enable(self, memory_lr: float = 0.1, memory_decay: float = 1.0, 
               retrieval_scale: float = 0.5):
        """Enable memory system with specified parameters."""
        self.model.config.use_memory = True
        self.model.config.memory_freeze = False
        self.model.config.memory_lr = memory_lr
        self.model.config.memory_decay = memory_decay
        self.model.config.memory_retrieval_scale = retrieval_scale

    def disable(self):
        """Disable memory system."""
        self.model.config.use_memory = False

    def learn(self, text: str, repetitions: int = 50):
        """
        Learn a fact by processing it multiple times.
        
        Args:
            text: The fact to learn
            repetitions: Number of times to process (more = stronger memory)
        """
        self.facts.append(text)
        tokens = self.enc.encode(text)
        idx = torch.tensor([tokens], device=self.device)
        
        self.model.eval()
        self.model.config.memory_freeze = False
        
        with torch.no_grad():
            for _ in range(repetitions):
                self.model(idx)

    def freeze(self):
        """Freeze memory (read-only mode for querying)."""
        self.model.config.memory_freeze = True

    def ask(self, query: str, max_tokens: int = 20, temperature: float = 0.7,
            top_k: int = 40) -> str:
        """
        Query the model with memory active.
        
        Args:
            query: The question to ask
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Generated response string
        """
        self.freeze()
        tokens = self.enc.encode(query)
        idx = torch.tensor([tokens], device=self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                idx, 
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        return self.enc.decode(out[0].tolist())

    def reset(self):
        """Clear all memories and facts."""
        self.model.reset_all_memory()
        self.facts.clear()


class PositionAwareMemory:
    """
    External memory using position-weighted context keys.
    
    Creates keys by weighting recent tokens exponentially,
    making the last token most important for matching.
    """
    
    def __init__(self, model, n_context: int = 5, device: str = 'cuda'):
        self.model = model
        self.n_context = n_context
        self.device = device
        self.entries = []  # List of (key, value_embed, next_tok, context_str)

    def _get_context_key(self, token_ids: torch.Tensor | list) -> torch.Tensor:
        """Create a position-weighted key from the last N tokens."""
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, device=self.device)

        context_tokens = token_ids[-self.n_context:]
        n = len(context_tokens)

        embeds = self.model.embed.weight[context_tokens]  # [N, D]

        # Position weights: exponentially increasing (last token = most important)
        weights = torch.tensor([2.0 ** i for i in range(n)], device=self.device)
        weights = weights / weights.sum()

        # Weighted sum
        key = (embeds * weights.unsqueeze(1)).sum(dim=0)  # [D]
        key = F.normalize(key, p=2, dim=0)
        return key

    def store(self, context_tokens: list | torch.Tensor, 
              next_token_id: int, context_str: str = ""):
        """Store a context -> next_token association."""
        if isinstance(context_tokens, list):
            context_tokens = torch.tensor(context_tokens, device=self.device)

        key = self._get_context_key(context_tokens)
        value = self.model.embed.weight[next_token_id].detach().clone()

        self.entries.append((key, value, next_token_id, context_str))

    def store_sequence(self, tokens: list, tokenizer=None):
        """Store all context->next associations from a token sequence."""
        for i in range(self.n_context - 1, len(tokens) - 1):
            context = tokens[:i+1]
            next_tok = tokens[i + 1]
            ctx_str = tokenizer.decode(context[-self.n_context:]) if tokenizer else ""
            self.store(context, next_tok, ctx_str)

    def retrieve(self, query_tokens: list | torch.Tensor, 
                 debug: bool = False) -> tuple:
        """
        Retrieve best matching entry.
        
        Returns:
            (value_embed, next_token_id, similarity, context_str) or (None, None, None, None)
        """
        if len(self.entries) == 0:
            return None, None, None, None

        if isinstance(query_tokens, list):
            query_tokens = torch.tensor(query_tokens, device=self.device)

        query_key = self._get_context_key(query_tokens)

        best_sim = -1
        best_entry = None

        if debug:
            print(f"    Query key (last 3 dims): {query_key[:3].tolist()}")

        for i, (key, value, next_tok, ctx) in enumerate(self.entries):
            sim = F.cosine_similarity(query_key.unsqueeze(0), key.unsqueeze(0)).item()
            if debug and sim > 0.8:
                print(f"    Entry {i}: sim={sim:.4f} ctx='{ctx}' -> tok {next_tok}")
            if sim > best_sim:
                best_sim = sim
                best_entry = (value, next_tok, sim, ctx)

        return best_entry

    def clear(self):
        """Clear all entries."""
        self.entries.clear()

    def __len__(self):
        return len(self.entries)


class LatentRAGCache:
    """
    Simple Latent RAG cache for O(1)-style retrieval.
    
    Stores (context_embedding, next_token_embedding) pairs
    and retrieves via cosine similarity.
    """
    
    def __init__(self, model, n_context: int = 5, 
                 threshold: float = 0.85, device: str = 'cuda'):
        self.model = model
        self.n_context = n_context
        self.threshold = threshold
        self.device = device
        
        self.keys = []    # Context embeddings
        self.values = []  # Next token embeddings
        self.tokens = []  # Next token IDs (for debugging)

    def _compute_key(self, token_ids: list) -> torch.Tensor:
        """Compute exponentially-weighted context embedding."""
        ctx = token_ids[-self.n_context:]
        ctx_tensor = torch.tensor(ctx, device=self.device)
        embeds = self.model.embed.weight[ctx_tensor]
        
        weights = torch.tensor([2.0 ** i for i in range(len(ctx))], device=self.device)
        weights = weights / weights.sum()
        
        key = (embeds * weights.unsqueeze(1)).sum(dim=0)
        return F.normalize(key, p=2, dim=0)

    def memorize(self, tokens: list):
        """Store all context->next associations from token sequence."""
        for i in range(self.n_context - 1, len(tokens) - 1):
            ctx = tokens[:i+1]
            next_tok = tokens[i + 1]
            
            key = self._compute_key(ctx)
            value = self.model.embed.weight[next_tok].detach().clone()
            
            self.keys.append(key)
            self.values.append(value)
            self.tokens.append(next_tok)

    def retrieve(self, query_tokens: list) -> tuple:
        """
        Retrieve if similarity exceeds threshold.
        
        Returns:
            (value_embedding, token_id, similarity) or (None, None, 0.0)
        """
        if len(self.keys) == 0:
            return None, None, 0.0

        query_key = self._compute_key(query_tokens)
        keys_tensor = torch.stack(self.keys)
        
        sims = F.cosine_similarity(query_key.unsqueeze(0), keys_tensor, dim=1)
        best_idx = sims.argmax().item()
        best_sim = sims[best_idx].item()

        if best_sim >= self.threshold:
            return self.values[best_idx], self.tokens[best_idx], best_sim
        return None, None, best_sim

    def clear(self):
        """Clear cache."""
        self.keys.clear()
        self.values.clear()
        self.tokens.clear()

    def __len__(self):
        return len(self.keys)
