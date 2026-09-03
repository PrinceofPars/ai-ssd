"""
Top-K Selector: Selects top-k candidate blocks with highest attention scores.
"""

from typing import List, Dict
from common.schemas.kv_block import KVBlock


class TopKSelector:
    def __init__(self, default_k: int = 16):
        self.default_k = default_k

    def select(
        self,
        scores: Dict[int, float],
        blocks: List[KVBlock],
        k: int,
    ) -> List[KVBlock]:
        """
        Sorts blocks by score descending and returns top-k blocks.
        """
        block_map = {b.block_id: b for b in blocks}
        sorted_ids = sorted(scores.keys(), key=lambda bid: scores.get(bid, 0.0), reverse=True)
        top_ids = sorted_ids[:k]
        return [block_map[bid] for bid in top_ids if bid in block_map]
