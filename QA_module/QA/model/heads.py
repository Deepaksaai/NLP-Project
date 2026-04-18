"""
Three QA-specific heads attached on top of the reused summarizer encoder:

  StartHead     — Linear(d_model -> 1) over every token position.
  EndHead       — Linear(d_model -> 1) over every token position.
  HasAnswerHead — Linear(d_model -> 1) over the [CLS] hidden state only.

All three are randomly initialized. No pretrained weights.
"""

import torch
import torch.nn as nn


class SpanHead(nn.Module):
    """Produces a single logit per token position."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) -> (batch, seq_len, 1) -> (batch, seq_len)
        return self.proj(encoder_output).squeeze(-1)


class HasAnswerHead(nn.Module):
    """Reads only the [CLS] position and produces one scalar logit."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        # (batch, d_model) -> (batch, 1) -> (batch,)
        return self.proj(cls_hidden).squeeze(-1)
