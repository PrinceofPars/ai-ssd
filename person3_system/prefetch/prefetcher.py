"""
Speculative Prefetcher: Asynchronous DRAM staging buffer to pre-stage cold blocks from SSD.
"""

from typing import Set, List
from person3_system.prefetch.predictor import NextLayerPredictor
from person3_system.prefetch.history import AccessHistory


class SpeculativePrefetcher:
    def __init__(self, buffer_capacity_blocks: int = 512):
        self.buffer_capacity = buffer_capacity_blocks
        self._staged_blocks: Set[int] = set()
        self.predictor = NextLayerPredictor()
        self.history = AccessHistory()
        self.hits = 0
        self.misses = 0

    def stage_blocks(self, block_ids: List[int]) -> None:
        for bid in block_ids:
            if len(self._staged_blocks) < self.buffer_capacity:
                self._staged_blocks.add(bid)

    def is_staged(self, block_ids: List[int]) -> bool:
        if not block_ids:
            return True
        all_hit = all(bid in self._staged_blocks for bid in block_ids)
        if all_hit:
            self.hits += 1
        else:
            self.misses += 1
        return all_hit

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / float(total)) if total > 0 else 0.0

    def clear(self) -> None:
        self._staged_blocks.clear()
        self.hits = 0
        self.misses = 0
