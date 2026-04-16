"""
Transformer Encoder — stack of pre-norm encoder layers.

Each layer:
  x = x + Dropout(SelfAttention(LayerNorm(x)))
  x = x + Dropout(FFN(LayerNorm(x)))

Pre-norm (LayerNorm before sublayer) trains more stably than post-norm,
especially important when training from scratch.
"""

import torch.nn as nn
from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise FFN: Linear → ReLU → Dropout → Linear."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Pre-norm self-attention + residual
        normed = self.norm1(x)
        x = x + self.dropout1(self.self_attn(normed, normed, normed, src_mask))

        # Pre-norm FFN + residual
        normed = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed))

        return x


class Encoder(nn.Module):
    """Full encoder: N layers + final LayerNorm."""

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
