"""
Label-smoothed cross-entropy loss.

Instead of putting 100% probability on the correct token, we spread
10% of the probability mass across all other tokens. This prevents
overconfident predictions and improves generalization.

Padding positions (target == pad_id) are excluded from the loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothedCrossEntropy(nn.Module):

    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size

    def forward(self, logits, target):
        """
        Args:
            logits: (batch, seq_len, vocab_size)
            target: (batch, seq_len)
        Returns:
            scalar loss (mean over non-padding tokens)
        """
        batch, seq_len, vocab = logits.shape
        logits = logits.reshape(-1, vocab)
        target = target.reshape(-1)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(
                logits, self.smoothing / (self.vocab_size - 1)
            )
            smooth_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Mask padding positions
        pad_mask = (target != self.ignore_index)

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_dist * log_probs).sum(dim=-1)

        # Mean over non-padding tokens only
        loss = loss[pad_mask].mean()

        return loss
