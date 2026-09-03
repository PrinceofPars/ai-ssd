"""
Attention Scorer: Computes block-level importance scores.
"""

from typing import List, Dict, Optional, Any
from common.schemas.kv_block import KVBlock


class AttentionScorer:
    def __init__(self, head_dim: int = 128):
        self.head_dim = head_dim

    def score_blocks(
        self,
        candidate_blocks: List[KVBlock],
        query_vector: Optional[Any] = None,
    ) -> Dict[int, float]:
        """
        Computes attention relevance score for each candidate block.
        If synthetic (query is None), uses block hotness and decay profile.
        """
        scores: Dict[int, float] = {}
        for block in candidate_blocks:
            if query_vector is not None:
                try:
                    import numpy as np
                    pseudo_key = np.full(self.head_dim, block.hotness)
                    score = float(np.dot(query_vector, pseudo_key) / (self.head_dim ** 0.5))
                except ImportError:
                    score = block.hotness
            else:
                score = block.hotness
            scores[block.block_id] = score
        return scores
