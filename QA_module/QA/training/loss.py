"""
Combined QA loss: span loss + has-answer loss.

    span_loss       = (CE(start) + CE(end)) / 2
    has_answer_loss = BCE(sigmoid(has_answer_logit), label)
    total           = span_loss + 0.5 * has_answer_loss

SQuAD 2.0 convention for unanswerable examples:
    start_position = end_position = 0  (points to [CLS])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from QA.qa_config import HAS_ANSWER_LOSS_WEIGHT


def compute_loss(
    start_logits: torch.Tensor,   # (B, L)
    end_logits:   torch.Tensor,   # (B, L)
    has_answer_logits: torch.Tensor,  # (B,)
    start_positions: torch.Tensor,    # (B,) long
    end_positions:   torch.Tensor,    # (B,) long
    has_answer_labels: torch.Tensor,  # (B,) float 0/1
    has_answer_weight: float = HAS_ANSWER_LOSS_WEIGHT,
):
    """Returns (total_loss, components_dict)."""
    # Cross-entropy on the (masked) logits. Positions with -1e9
    # contribute ~0 probability mass so gradients stay well-behaved.
    start_loss = F.cross_entropy(start_logits, start_positions)
    end_loss   = F.cross_entropy(end_logits,   end_positions)
    span_loss  = 0.5 * (start_loss + end_loss)

    has_answer_loss = F.binary_cross_entropy_with_logits(
        has_answer_logits, has_answer_labels.float()
    )

    total = span_loss + has_answer_weight * has_answer_loss

    return total, {
        "total":      total.detach(),
        "span":       span_loss.detach(),
        "start":      start_loss.detach(),
        "end":        end_loss.detach(),
        "has_answer": has_answer_loss.detach(),
    }
