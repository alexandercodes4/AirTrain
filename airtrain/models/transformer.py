"""GPT-2 style transformer model implemented in MLX.

Provides small transformer models for distributed training demos.
Falls back to a numpy-based stub on non-macOS platforms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from airtrain.compat import MLX_AVAILABLE

if MLX_AVAILABLE:
    import mlx.core as mx
    import mlx.nn as nn


@dataclass
class TransformerConfig:
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    vocab_size: int = 50257
    max_seq_len: int = 1024
    dropout: float = 0.1


# Presets
PRESETS = {
    "gpt2-tiny": TransformerConfig(n_layers=4, n_heads=4, d_model=256, d_ff=1024, max_seq_len=512),
    "gpt2-small": TransformerConfig(n_layers=12, n_heads=12, d_model=768, d_ff=3072),
    "gpt2-medium": TransformerConfig(n_layers=24, n_heads=16, d_model=1024, d_ff=4096),
}


if MLX_AVAILABLE:

    class MultiHeadAttention(nn.Module):
        def __init__(self, config: TransformerConfig):
            super().__init__()
            self.n_heads = config.n_heads
            self.d_model = config.d_model
            self.head_dim = config.d_model // config.n_heads

            self.q_proj = nn.Linear(config.d_model, config.d_model)
            self.k_proj = nn.Linear(config.d_model, config.d_model)
            self.v_proj = nn.Linear(config.d_model, config.d_model)
            self.out_proj = nn.Linear(config.d_model, config.d_model)

        def __call__(self, x):
            B, T, C = x.shape

            q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

            scale = math.sqrt(self.head_dim)
            scores = (q @ k.transpose(0, 1, 3, 2)) / scale

            # Causal mask
            mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
            scores = scores + mask

            weights = mx.softmax(scores, axis=-1)
            out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
            return self.out_proj(out)

    class FeedForward(nn.Module):
        def __init__(self, config: TransformerConfig):
            super().__init__()
            self.fc1 = nn.Linear(config.d_model, config.d_ff)
            self.fc2 = nn.Linear(config.d_ff, config.d_model)

        def __call__(self, x):
            return self.fc2(nn.gelu(self.fc1(x)))

    class TransformerBlock(nn.Module):
        def __init__(self, config: TransformerConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(config.d_model)
            self.attn = MultiHeadAttention(config)
            self.ln2 = nn.LayerNorm(config.d_model)
            self.ffn = FeedForward(config)

        def __call__(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x

    class GPT2Model(nn.Module):
        """GPT-2 style transformer language model."""

        def __init__(self, config: TransformerConfig):
            super().__init__()
            self.config = config
            self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
            self.blocks = [TransformerBlock(config) for _ in range(config.n_layers)]
            self.ln_f = nn.LayerNorm(config.d_model)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        def __call__(self, x):
            B, T = x.shape
            positions = mx.arange(T)

            h = self.token_emb(x) + self.pos_emb(positions)
            for block in self.blocks:
                h = block(h)
            h = self.ln_f(h)
            logits = self.lm_head(h)
            return logits

    def cross_entropy_loss(logits, targets):
        """Compute cross-entropy loss for language modeling."""
        B, T, V = logits.shape
        logits = logits.reshape(B * T, V)
        targets = targets.reshape(B * T)
        return mx.mean(nn.losses.cross_entropy(logits, targets))

    def create_model(name: str = "gpt2-small") -> GPT2Model:
        """Create a GPT-2 model from a preset name."""
        if name not in PRESETS:
            raise ValueError(f"Unknown model: {name}. Available: {list(PRESETS.keys())}")
        config = PRESETS[name]
        model = GPT2Model(config)
        mx.eval(model.parameters())
        return model

else:
    # Stub for non-macOS platforms
    def create_model(name: str = "gpt2-small"):
        from airtrain.compat import require_mlx
        require_mlx()

    def cross_entropy_loss(logits, targets):
        from airtrain.compat import require_mlx
        require_mlx()
