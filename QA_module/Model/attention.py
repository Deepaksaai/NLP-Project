"""
Multi-Head Scaled Dot-Product Attention — implemented from scratch.

No nn.MultiheadAttention used. Every projection is an explicit nn.Linear.

Three variants share this same module:
  - Encoder self-attention       (query=key=value=encoder states)
  - Decoder masked self-attention (query=key=value=decoder states, causal mask)
  - Decoder cross-attention       (query=decoder, key=value=encoder output)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        # Linear projections for Q, K, V (no bias — standard for attention)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query:  (batch, q_len, d_model)
            key:    (batch, k_len, d_model)
            value:  (batch, k_len, d_model)
            mask:   bool tensor, True = attend, False = mask out
                    Shape: (batch, 1, q_len, k_len) for causal+pad
                       or: (batch, 1, 1, k_len)     for padding only

        Returns:
            output: (batch, q_len, d_model)
        """
        batch_size = query.size(0)

        # Project and reshape: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_head)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        # scores: (batch, n_heads, q_len, k_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask before softmax
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        # (batch, n_heads, q_len, d_head)
        context = torch.matmul(attn_weights, V)

        # Concatenate heads: (batch, q_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(context)
