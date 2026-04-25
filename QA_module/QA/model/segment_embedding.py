"""
Segment embedding: a 2-row embedding that adds a role signal
(0 = question segment, 1 = context segment) to the token + positional
embeddings before the encoder stack.

Randomly initialized. Trained from scratch during QA fine-tuning.
"""

import torch
import torch.nn as nn


class SegmentEmbedding(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(2, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        """segment_ids: (batch, seq_len) long — returns (batch, seq_len, d_model)."""
        return self.emb(segment_ids)
