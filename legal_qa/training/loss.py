"""
Combined QA Loss Function.

Includes CrossEntropyLoss for start_logits and end_logits, and
BCEWithLogitsLoss for has_answer_logit.
"""

import torch
import torch.nn as nn

class QALoss(nn.Module):
    """
    Combined loss for Extractive QA with an answerability head.
    """

    def __init__(self, has_answer_weight: float = 0.5):
        super().__init__()
        self.has_answer_weight = has_answer_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        has_answer_logit: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        is_answerable: torch.Tensor,
    ) -> dict:
        """
        Computes the combined loss.

        Args:
            start_logits: (batch, seq_len)
            end_logits: (batch, seq_len)
            has_answer_logit: (batch,)
            start_positions: (batch,) indices
            end_positions: (batch,) indices
            is_answerable: (batch,) float values (1.0 or 0.0)

        Returns:
            A dictionary containing:
                total_loss: The final combined scalar loss
                span_loss: The average of start and end losses
                has_answer_loss: The loss from the answerability head
        """
        # Clamp positions to valid range — guards against dataset edge cases
        seq_len = start_logits.size(1)
        start_positions = start_positions.clamp(0, seq_len - 1)
        end_positions = end_positions.clamp(0, seq_len - 1)

        # Span Loss
        start_loss = self.ce_loss(start_logits, start_positions)
        end_loss = self.ce_loss(end_logits, end_positions)
        span_loss = (start_loss + end_loss) / 2.0

        # Has Answer Loss
        has_answer_loss = self.bce_loss(has_answer_logit, is_answerable)

        # Total Loss
        total_loss = span_loss + self.has_answer_weight * has_answer_loss

        return {
            "total_loss": total_loss,
            "span_loss": span_loss,
            "has_answer_loss": has_answer_loss,
        }
