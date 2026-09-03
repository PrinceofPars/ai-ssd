"""
Attention computation layer abstraction.
"""

from typing import List, Dict, Any
from common.schemas.kv_block import KVBlock
from person1_kv_engine.attention.scoring import AttentionScorer


class AttentionEngine:
    def __init__(self, head_dim: int = 128):
        self.scorer = AttentionScorer(head_dim=head_dim)

    def compute_attention(
        self,
        query: Any,
        blocks: List[KVBlock],
    ) -> Dict[int, float]:
        return self.scorer.score_blocks(blocks)
