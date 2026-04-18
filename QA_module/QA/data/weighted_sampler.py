"""
Balanced batch sampler — guarantees every training batch contains
the target mix of answerable / unanswerable examples regardless of
the underlying class ratio in the dataset.

Target per batch (default batch_size=32):
    21 answerable   (~65%)
    11 unanswerable (~35%)

If the minority class runs out before the epoch ends, it is cycled
(upsampling by repetition) rather than truncating the batch. This
matches the spec's 'upsample minority' rule.
"""

import random
from typing import List, Iterator
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler[List[int]]):

    def __init__(
        self,
        answerable_indices: List[int],
        unanswerable_indices: List[int],
        batch_size: int = 32,
        answerable_frac: float = 0.65,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if not answerable_indices or not unanswerable_indices:
            raise ValueError(
                "BalancedBatchSampler needs both classes present "
                f"(ans={len(answerable_indices)}, unans={len(unanswerable_indices)})"
            )

        self.answerable   = list(answerable_indices)
        self.unanswerable = list(unanswerable_indices)
        self.batch_size   = batch_size
        self.n_ans_per_batch = max(1, round(batch_size * answerable_frac))
        self.n_unans_per_batch = batch_size - self.n_ans_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self._rng = random.Random(seed)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random((self.epoch + 1) * 10_000 + id(self) % 1000)

        ans   = list(self.answerable)
        unans = list(self.unanswerable)
        if self.shuffle:
            rng.shuffle(ans)
            rng.shuffle(unans)

        # Base epoch length = however many batches we can fill from the
        # majority class; the minority class gets cycled if needed.
        majority_size = max(len(ans), len(unans))
        num_batches = majority_size // max(self.n_ans_per_batch, self.n_unans_per_batch)

        ans_ptr = unans_ptr = 0
        for _ in range(num_batches):
            batch = []
            for _ in range(self.n_ans_per_batch):
                if ans_ptr >= len(ans):
                    if self.shuffle:
                        rng.shuffle(ans)
                    ans_ptr = 0
                batch.append(ans[ans_ptr]); ans_ptr += 1
            for _ in range(self.n_unans_per_batch):
                if unans_ptr >= len(unans):
                    if self.shuffle:
                        rng.shuffle(unans)
                    unans_ptr = 0
                batch.append(unans[unans_ptr]); unans_ptr += 1

            if self.shuffle:
                rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        majority_size = max(len(self.answerable), len(self.unanswerable))
        return majority_size // max(self.n_ans_per_batch, self.n_unans_per_batch)
