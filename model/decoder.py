"""
Transformer Decoder — stack of pre-norm decoder layers.

Each layer has three sublayers:
  x = x + Dropout(MaskedSelfAttention(LayerNorm(x)))
  x = x + Dropout(CrossAttention(LayerNorm(x), encoder_output))
  x = x + Dropout(FFN(LayerNorm(x)))

Masked self-attention uses a causal mask so position i can only
attend to positions 0..i (autoregressive generation).
Cross-attention reads from the encoder output.
"""

import torch.nn as nn
from model.attention import MultiHeadAttention
from model.encoder import FeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        """
        Args:
            x:        (batch, tgt_len, d_model) — decoder input
            memory:   (batch, src_len, d_model) — encoder output
            tgt_mask: causal + padding mask for decoder self-attention
            src_mask: padding mask for encoder output
        """
        # Masked self-attention + residual
        normed = self.norm1(x)
        x = x + self.dropout1(self.self_attn(normed, normed, normed, tgt_mask))

        # Cross-attention to encoder output + residual
        normed = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(normed, memory, memory, src_mask))

        # FFN + residual
        normed = self.norm3(x)
        x = x + self.dropout3(self.ffn(normed))

        return x


class Decoder(nn.Module):
    """Full decoder: N layers + final LayerNorm."""

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)
