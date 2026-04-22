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


class CoverageLoss(nn.Module):
    """
    Coverage loss from See et al. (2017) "Get To The Point".

    Penalizes the decoder for repeatedly attending to the same source
    positions — the core fix for cross-attention collapse / hallucination.

    At each decode step t the coverage vector c_t is the sum of all
    previous cross-attention distributions:
        c_t = Σ_{t'=0}^{t-1}  α_{t'}

    The loss penalizes overlap between the current attention α_t and
    the accumulated coverage c_t:
        L_cov = Σ_t Σ_i  min(α_{t,i}, c_{t,i})

    We compute this efficiently in parallel over the full target sequence
    using a cumsum trick rather than a Python loop.

    Args:
        attn_weights: (batch, tgt_len, src_len) — cross-attention
                      weights averaged over heads and layers, as returned
                      by model.forward(src, tgt, return_coverage=True).
    Returns:
        scalar coverage loss (mean over batch and tgt positions)
    """

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        # cumulative[t] = Σ_{t'=0}^{t} α_{t'}
        cumulative = attn_weights.cumsum(dim=1)          # (B, T, S)
        # coverage BEFORE step t = cumsum up to t-1
        prev_coverage = cumulative - attn_weights        # (B, T, S)
        # penalty: min(current attention, coverage so far)
        penalty = torch.min(attn_weights, prev_coverage) # (B, T, S)
        # mean over batch and target positions; sum over source positions
        return penalty.sum(dim=-1).mean()


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
