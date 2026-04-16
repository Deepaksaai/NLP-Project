"""
Learning rate schedule: linear warmup → cosine decay.

No warm restarts — simple and stable for curriculum pre-training.
The warmup prevents early instability when gradients are noisy.
The cosine decay smoothly reduces LR to min_lr by the end of training.
"""

import math


class WarmupCosineSchedule:
    """
    Step-level LR scheduler.

    Phase 1 (step 0..warmup_steps):
        LR = max_lr * step / warmup_steps

    Phase 2 (step warmup_steps..total_steps):
        LR = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))

    Call .step() after every optimizer step, then apply the LR.
    """

    def __init__(self, optimizer, total_steps, warmup_steps=8000,
                 max_lr=5e-4, min_lr=1e-5):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Advance one step and update optimizer LR. Returns the new LR."""
        self.current_step += 1
        lr = self.get_lr(self.current_step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_lr(self, step=None):
        """Compute LR for a given step."""
        if step is None:
            step = self.current_step

        if step == 0:
            return self.min_lr

        if step < self.warmup_steps:
            # Linear warmup: 0 → max_lr
            return self.max_lr * step / self.warmup_steps

        # Cosine decay: max_lr → min_lr
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)  # clamp at end
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine

    def current_lr(self):
        """Get the current LR from the optimizer."""
        return self.optimizer.param_groups[0]["lr"]
