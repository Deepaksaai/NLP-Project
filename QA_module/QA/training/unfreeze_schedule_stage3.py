"""
Stage 3 optimizer + epoch LR multiplier schedule for legal fine-tuning.

Discriminative base LRs (applied to all layers from epoch 1):
    encoder_bot  layers 0,1        -> 1e-6
    encoder_mid  layers 2,3        -> 3e-6
    encoder_top  layers 4,5+norm   -> 8e-6
    segment      segment_emb+embed -> 1e-5
    heads        start/end/hasA    -> 3e-5

Epoch multiplier (applied to ENCODER groups only — heads stay at
full LR throughout):
    epoch 1-2 : 0.10
    epoch 3-4 : 0.50
    epoch 5   : 0.75
    epoch 6+  : 1.00
"""

from typing import Dict
import torch


STAGE3_LRS = {
    "encoder_bot":  1e-6,
    "encoder_mid":  3e-6,
    "encoder_top":  8e-6,
    "segment":      1e-5,
    "heads":        3e-5,
}

_ENCODER_GROUP_NAMES = ("encoder_bot", "encoder_mid", "encoder_top")


def _enc_layers(model, idxs):
    out = []
    for i in idxs:
        out.extend(list(model.encoder.layers[i].parameters()))
    return out


def build_stage3_optimizer(model, weight_decay: float = 0.01, eps: float = 1e-9) -> torch.optim.Optimizer:
    for p in model.parameters():
        p.requires_grad = True

    groups = [
        {"name": "encoder_bot",
         "params": _enc_layers(model, [0, 1]),
         "lr": STAGE3_LRS["encoder_bot"]},
        {"name": "encoder_mid",
         "params": _enc_layers(model, [2, 3]),
         "lr": STAGE3_LRS["encoder_mid"]},
        {"name": "encoder_top",
         "params": _enc_layers(model, [4, 5]) + list(model.encoder.norm.parameters()),
         "lr": STAGE3_LRS["encoder_top"]},
        {"name": "segment",
         "params": list(model.segment_embedding.parameters()) + list(model.embedding.parameters()),
         "lr": STAGE3_LRS["segment"]},
        {"name": "heads",
         "params": list(model.start_head.parameters())
                 + list(model.end_head.parameters())
                 + list(model.has_answer_head.parameters()),
         "lr": STAGE3_LRS["heads"]},
    ]
    return torch.optim.AdamW(
        groups, lr=STAGE3_LRS["heads"],
        betas=(0.9, 0.98), weight_decay=weight_decay, eps=eps,
    )


def epoch_encoder_multiplier(epoch: int) -> float:
    if epoch <= 2: return 0.10
    if epoch <= 4: return 0.50
    if epoch == 5: return 0.75
    return 1.00


def apply_stage3_schedule(optimizer: torch.optim.Optimizer, epoch: int) -> Dict:
    """
    Reset each param-group LR to its Stage-3 base, then apply the
    epoch multiplier to encoder groups only. Call at epoch start,
    BEFORE the WarmupCosine scheduler runs its first step of the
    epoch — the scheduler multiplies on top of whatever we set here.
    """
    mult = epoch_encoder_multiplier(epoch)
    actions = {}
    for g in optimizer.param_groups:
        name = g.get("name", "?")
        base = STAGE3_LRS.get(name, g["lr"])
        if name in _ENCODER_GROUP_NAMES:
            g["lr"] = base * mult
        else:
            g["lr"] = base
        actions[name] = g["lr"]
    return {"epoch": epoch, "multiplier": mult, "lrs": actions}
