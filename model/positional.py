"""
Fixed sinusoidal positional encoding.

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Generalizes to sequence lengths not seen during training because
the encoding is a deterministic function of position, not a learned
lookup table.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=2048):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model) — broadcastable over batch dim
        pe = pe.unsqueeze(0)

        # Buffer: moves with .to(device), not a learnable parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model) → adds positional encoding."""
        return x + self.pe[:, :x.size(1)]
