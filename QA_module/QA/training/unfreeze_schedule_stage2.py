"""
Stage 2 optimizer builder — all layers trainable from epoch 1 with
discriminative learning rates.

Groups:
    encoder_bot  layers 0,1       -> 5e-6
    encoder_mid  layers 2,3       -> 1e-5
    encoder_top  layers 4,5+norm  -> 2e-5
    segment      segment_embedding+ embedding -> 3e-5
    heads        start/end/has_answer         -> 5e-5
"""

from typing import List
import torch


STAGE2_LRS = {
    "encoder_bot":  5e-6,
    "encoder_mid":  1e-5,
    "encoder_top":  2e-5,
    "segment":      3e-5,
    "heads":        5e-5,
}


def _enc_layer_params(model, idxs: List[int]):
    out = []
    for i in idxs:
        out.extend(list(model.encoder.layers[i].parameters()))
    return out


def build_stage2_optimizer(model, weight_decay: float = 0.01, eps: float = 1e-9) -> torch.optim.Optimizer:
    # All parameters trainable — Stage 2 adapts the whole model.
    for p in model.parameters():
        p.requires_grad = True

    groups = [
        {"name": "encoder_bot",
         "params": _enc_layer_params(model, [0, 1]),
         "lr": STAGE2_LRS["encoder_bot"]},
        {"name": "encoder_mid",
         "params": _enc_layer_params(model, [2, 3]),
         "lr": STAGE2_LRS["encoder_mid"]},
        {"name": "encoder_top",
         "params": _enc_layer_params(model, [4, 5]) + list(model.encoder.norm.parameters()),
         "lr": STAGE2_LRS["encoder_top"]},
        {"name": "segment",
         "params": list(model.segment_embedding.parameters()) + list(model.embedding.parameters()),
         "lr": STAGE2_LRS["segment"]},
        {"name": "heads",
         "params": list(model.start_head.parameters())
                 + list(model.end_head.parameters())
                 + list(model.has_answer_head.parameters()),
         "lr": STAGE2_LRS["heads"]},
    ]

    optim = torch.optim.AdamW(
        groups,
        lr=STAGE2_LRS["heads"],
        betas=(0.9, 0.98),
        weight_decay=weight_decay,
        eps=eps,
    )
    return optim
