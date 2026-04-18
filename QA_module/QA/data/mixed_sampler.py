"""
MixedBatchSampler — composes each training batch from three pools
with a target ratio.

Default for Stage 2 (batch_size=16):
    40% quality   -> 6 per batch  (rounded)
    30% qasper    -> 5 per batch
    30% squad     -> 5 per batch
  (rounded shares are re-adjusted to sum exactly to batch_size.)

If any pool is empty the sampler raises. If one pool is smaller than
needed it is cycled (upsampled by repetition) rather than exhausting
the epoch early — the epoch length is set by whichever pool would be
consumed first at its own rate.
"""

import random
from typing import Dict, List, Iterator
from torch.utils.data import Sampler


class MixedBatchSampler(Sampler[List[int]]):

    def __init__(
        self,
        indices_by_source: Dict[str, List[int]],
        mix: Dict[str, float],
        batch_size: int = 16,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if abs(sum(mix.values()) - 1.0) > 1e-6:
            raise ValueError(f"mix must sum to 1.0, got {sum(mix.values())}")
        for k in mix:
            if k not in indices_by_source or not indices_by_source[k]:
                raise ValueError(f"MixedBatchSampler: pool {k!r} is empty")

        self.pools: Dict[str, List[int]] = {
            k: list(indices_by_source[k]) for k in mix
        }
        self.mix = mix
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Per-batch counts (rounded, then tweaked to sum to batch_size)
        counts: Dict[str, int] = {k: max(1, round(batch_size * mix[k])) for k in mix}
        diff = batch_size - sum(counts.values())
        if diff != 0:
            # Push the slack onto the largest-share source
            biggest = max(mix.keys(), key=lambda k: mix[k])
            counts[biggest] += diff
        self.counts = counts

        # Epoch length: bounded by the fastest-depleting pool at its rate
        per_epoch = min(len(self.pools[k]) // max(1, counts[k]) for k in mix)
        self._len = max(1, per_epoch)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + 7919 * (self.epoch + 1))
        pools = {k: list(v) for k, v in self.pools.items()}
        if self.shuffle:
            for v in pools.values():
                rng.shuffle(v)
        ptrs = {k: 0 for k in pools}

        for _ in range(self._len):
            batch = []
            for k, n in self.counts.items():
                pool = pools[k]
                for _ in range(n):
                    if ptrs[k] >= len(pool):
                        if self.shuffle:
                            rng.shuffle(pool)
                        ptrs[k] = 0
                    batch.append(pool[ptrs[k]])
                    ptrs[k] += 1
            if self.shuffle:
                rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self._len
