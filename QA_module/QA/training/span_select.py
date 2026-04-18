"""
Joint span selection at inference time.

For a pair of (start_logits, end_logits) over a single sequence,
pick the (i, j) that maximizes start_logits[i] + end_logits[j]
subject to:
    i <= j
    j - i + 1 <= max_answer_len

Implementation uses a vectorized upper-triangular score matrix.
Invalid positions are expected to already be at -1e9 in the logits.
"""

import torch

from QA.qa_config import MAX_ANSWER_LEN


def joint_span_select(
    start_logits: torch.Tensor,  # (B, L) or (L,)
    end_logits:   torch.Tensor,  # (B, L) or (L,)
    max_answer_len: int = MAX_ANSWER_LEN,
    length_penalty: float = 0.0,
):
    """
    Returns (start_idx, end_idx) as long tensors shaped (B,).
    Accepts (L,) inputs for batch_size=1 convenience.

    length_penalty > 0 ADDS `length_penalty * (j - i)` to each
    (i, j) score to bias toward LONGER spans. This is critical for
    legal answers which are full clauses, not short facts.
    length_penalty < 0 would bias toward shorter spans.
    """
    squeeze = False
    if start_logits.dim() == 1:
        start_logits = start_logits.unsqueeze(0)
        end_logits   = end_logits.unsqueeze(0)
        squeeze = True

    B, L = start_logits.shape
    # score[b, i, j] = start[b, i] + end[b, j]
    scores = start_logits.unsqueeze(2) + end_logits.unsqueeze(1)  # (B, L, L)

    # Build a (L, L) mask: allow i <= j and j - i + 1 <= max_len
    idx_i = torch.arange(L, device=start_logits.device).unsqueeze(1)  # (L, 1)
    idx_j = torch.arange(L, device=start_logits.device).unsqueeze(0)  # (1, L)
    valid = (idx_j >= idx_i) & ((idx_j - idx_i + 1) <= max_answer_len)

    if length_penalty != 0.0:
        length = (idx_j - idx_i).clamp(min=0).to(scores.dtype)       # (L, L)
        scores = scores + length_penalty * length.unsqueeze(0)

    scores = scores.masked_fill(~valid.unsqueeze(0), -1e9)

    flat = scores.view(B, L * L)
    best = flat.argmax(dim=-1)
    start_idx = best // L
    end_idx   = best %  L

    if squeeze:
        return start_idx.item(), end_idx.item()
    return start_idx, end_idx
