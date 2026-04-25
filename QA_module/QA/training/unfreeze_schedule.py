"""
Gradual unfreezing + discriminative learning rates for QA Stage 1.

Stage 0 (epochs 1-3):
    heads + segment_emb + embedding  @ 3e-4
    encoder                          frozen

Stage 1 (epochs 4-5):
    heads                            @ 3e-4
    encoder.layers.{4,5}             @ 3e-5

Stage 2 (epochs 6-7):
    heads                            @ 3e-4
    encoder.layers.{4,5}             @ 3e-5
    encoder.layers.{2,3}             @ 1e-5

Stage 3 (epochs 8+):
    heads                            @ 3e-4
    encoder.layers.{4,5}             @ 3e-5
    encoder.layers.{2,3}             @ 1e-5
    encoder.layers.{0,1}             @ 5e-6
    encoder.norm                     @ 5e-6

Implementation: param group names match the keys below. The trainer
calls apply_unfreeze_schedule(model, optimizer, epoch) at the start
of each epoch. New groups are added to the optimizer (preserving
momentum on groups that already existed); frozen params are kept out
of the optimizer entirely.
"""

from typing import Dict, List
import torch


HEAD_LR         = 3e-4
ENC_TOP_LR      = 3e-5   # layers 4, 5
ENC_MID_LR      = 1e-5   # layers 2, 3
ENC_BOT_LR      = 5e-6   # layers 0, 1


def _encoder_layer_params(model, layer_idxs: List[int]):
    out = []
    for idx in layer_idxs:
        out.extend(list(model.encoder.layers[idx].parameters()))
    return out


def _stage_for_epoch(epoch: int) -> int:
    """1-indexed epoch -> unfreezing stage index."""
    if epoch <= 3: return 0
    if epoch <= 5: return 1
    if epoch <= 7: return 2
    return 3


def build_initial_optimizer(model) -> torch.optim.Optimizer:
    """Stage-0 optimizer: only heads + segment emb + embedding are trainable."""
    # Freeze everything, then re-enable the pieces we want.
    for p in model.parameters():
        p.requires_grad = False

    trainable = []
    trainable += list(model.start_head.parameters())
    trainable += list(model.end_head.parameters())
    trainable += list(model.has_answer_head.parameters())
    trainable += list(model.segment_embedding.parameters())
    trainable += list(model.embedding.parameters())
    for p in trainable:
        p.requires_grad = True

    optim = torch.optim.AdamW(
        [{"name": "heads", "params": trainable, "lr": HEAD_LR}],
        lr=HEAD_LR, betas=(0.9, 0.98), weight_decay=0.01, eps=1e-9,
    )
    return optim


def apply_unfreeze_schedule(model, optimizer: torch.optim.Optimizer, epoch: int) -> Dict:
    """
    Mutate `model.requires_grad_` and extend `optimizer.param_groups`
    to match the stage for this epoch. Safe to call at every epoch
    start — no-op when the stage hasn't changed.
    """
    stage = _stage_for_epoch(epoch)
    existing_names = {g.get("name") for g in optimizer.param_groups}
    actions = []

    def _add_group(name, params, lr):
        if name in existing_names:
            return
        for p in params:
            p.requires_grad = True
        optimizer.add_param_group({"name": name, "params": params, "lr": lr})
        existing_names.add(name)
        actions.append(f"+{name}@{lr}")

    if stage >= 1:
        _add_group("encoder_top",
                   _encoder_layer_params(model, [4, 5]) + list(model.encoder.norm.parameters()),
                   ENC_TOP_LR)
    if stage >= 2:
        _add_group("encoder_mid",
                   _encoder_layer_params(model, [2, 3]),
                   ENC_MID_LR)
    if stage >= 3:
        _add_group("encoder_bot",
                   _encoder_layer_params(model, [0, 1]),
                   ENC_BOT_LR)

    return {
        "epoch": epoch,
        "stage": stage,
        "added": actions,
        "group_names": [g.get("name") for g in optimizer.param_groups],
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def lrs_by_group(optimizer) -> Dict[str, float]:
    return {g.get("name", f"g{i}"): g["lr"] for i, g in enumerate(optimizer.param_groups)}
