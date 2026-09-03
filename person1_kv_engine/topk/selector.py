"""Top-K Selector: Selects top-k candidate blocks with highest attention scores."""

from typing import List, Dict, Optional
from common.schemas.kv_block import KVBlock


class TopKSelector:
    def __init__(self, default_k: int = 16):
        self.default_k = default_k

    def select(
        self,
        scores: Dict[int, float],
        blocks: List[KVBlock],
        k: Optional[int] = None,
    ) -> List[KVBlock]:
        """Sorts blocks by score descending and returns top-k blocks."""
        k_val = k if k is not None else self.default_k
        block_map = {b.block_id: b for b in blocks}
        sorted_ids = sorted(scores.keys(), key=lambda bid: scores.get(bid, 0.0), reverse=True)
        top_ids = sorted_ids[:k_val]
        return [block_map[bid] for bid in top_ids if bid in block_map]
