"""
Discriminative optimizer setup for DeBERTa QA fine-tuning.
"""

from torch.optim import AdamW

def get_discriminative_optimizer(model, stage: int) -> AdamW:
    """
    Sets up an AdamW optimizer with layer-wise learning rates (discriminative fine-tuning)
    for DeBERTa-v3-large.

    Args:
        model: The LegalQAModel instance.
        stage: The training stage (1, 2, or 3).

    Returns:
        AdamW optimizer with configured parameter groups.
    """
    # Define learning rates based on stage
    if stage == 1:
        lr_bottom = 1e-5
        lr_middle = 2e-5
        lr_heads = 1e-4
    elif stage == 2:
        lr_bottom = 5e-6
        lr_middle = 1e-5
        lr_heads = 5e-5
    elif stage == 3:
        lr_bottom = 1e-6
        lr_middle = 3e-6
        lr_heads = 2e-5
    else:
        raise ValueError(f"Invalid stage {stage}")

    # Separate parameters into groups with and without weight decay
    weight_decay = 0.01

    def is_no_decay(name):
        return any(nd in name for nd in ["bias", "LayerNorm.weight"])

    # Base sets for no decay and decay
    optimizer_grouped_parameters = []

    # Layers 0-7 (Bottom)
    bottom_params = [(n, p) for n, p in model.named_parameters() if "encoder.encoder.layer." in n and int(n.split("encoder.encoder.layer.")[1].split(".")[0]) <= 7]
    optimizer_grouped_parameters.extend([
        {
            "params": [p for n, p in bottom_params if not is_no_decay(n)],
            "weight_decay": weight_decay,
            "lr": lr_bottom,
        },
        {
            "params": [p for n, p in bottom_params if is_no_decay(n)],
            "weight_decay": 0.0,
            "lr": lr_bottom,
        }
    ])

    # Layers 8-23 (Middle)
    middle_params = [(n, p) for n, p in model.named_parameters() if "encoder.encoder.layer." in n and int(n.split("encoder.encoder.layer.")[1].split(".")[0]) > 7]
    optimizer_grouped_parameters.extend([
        {
            "params": [p for n, p in middle_params if not is_no_decay(n)],
            "weight_decay": weight_decay,
            "lr": lr_middle,
        },
        {
            "params": [p for n, p in middle_params if is_no_decay(n)],
            "weight_decay": 0.0,
            "lr": lr_middle,
        }
    ])

    # Non-layer encoder components (embeddings, relative attention bias, etc)
    other_encoder_params = [(n, p) for n, p in model.named_parameters() if n.startswith("encoder.") and "encoder.layer." not in n]
    optimizer_grouped_parameters.extend([
        {
            "params": [p for n, p in other_encoder_params if not is_no_decay(n)],
            "weight_decay": weight_decay,
            "lr": lr_bottom,  # tying to bottom LR
        },
        {
            "params": [p for n, p in other_encoder_params if is_no_decay(n)],
            "weight_decay": 0.0,
            "lr": lr_bottom,
        }
    ])

    # Heads (Top)
    head_params = [(n, p) for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer_grouped_parameters.extend([
        {
            "params": [p for n, p in head_params if not is_no_decay(n)],
            "weight_decay": weight_decay,
            "lr": lr_heads,
        },
        {
            "params": [p for n, p in head_params if is_no_decay(n)],
            "weight_decay": 0.0,
            "lr": lr_heads,
        }
    ])

    optimizer = AdamW(optimizer_grouped_parameters)
    return optimizer
