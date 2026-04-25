"""
Task-specific heads for the Legal QA model.

Each head is a standalone nn.Module wrapping a single Linear layer so
they can be swapped out, inspected, or frozen independently.

Heads:
    SpanHead       — Projects every token's hidden state (1024-d) to a
                     single logit.  Used twice: once for start_logits and
                     once for end_logits.
                     Input:  (batch, seq_len, 1024)
                     Output: (batch, seq_len)

    HasAnswerHead  — Binary classifier on the [CLS] token representation
                     that predicts whether the context contains an answer.
                     Input:  (batch, 1024)
                     Output: (batch,)
"""

import torch
import torch.nn as nn


class SpanHead(nn.Module):
    """
    Single-logit projection for every token in the sequence.

    Intended usage:
        head = SpanHead(hidden_size=1024)
        logits = head(encoder_output)  # (B, L, 1024) → (B, L)

    The Linear layer maps 1024 → 1 and the trailing dimension is squeezed.
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

        # Xavier-uniform weight, zero bias
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            logits: (batch, seq_len)
        """
        return self.linear(hidden_states).squeeze(-1)


class HasAnswerHead(nn.Module):
    """
    Binary classifier on the [CLS] token hidden state.

    Intended usage:
        head = HasAnswerHead(hidden_size=1024)
        logit = head(cls_hidden)  # (B, 1024) → (B,)

    The Linear layer maps 1024 → 1 and the trailing dimension is squeezed.
    The output is a raw logit (pass through sigmoid for probability).
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

        # Xavier-uniform weight, zero bias
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_hidden: (batch, hidden_size) — the [CLS] representation.

        Returns:
            logit: (batch,)
        """
        return self.linear(cls_hidden).squeeze(-1)
