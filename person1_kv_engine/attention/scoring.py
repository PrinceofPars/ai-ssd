"""Attention Scorer: Computes block-level importance scores.

Computes dot-product relevance scores between the active Query vector
and stored Key blocks.
"""

from typing import List, Dict, Optional, Any
import numpy as np
from common.schemas.kv_block import KVBlock


class AttentionScorer:
    def __init__(self, head_dim: int = 128):
        self.head_dim = head_dim
        self.scale = 1.0 / np.sqrt(head_dim, dtype=np.float32)

    def score_blocks(
        self,
        candidate_blocks: List[KVBlock],
        query_vector: Optional[Any] = None,
    ) -> Dict[int, float]:
        """Computes attention relevance score for each candidate block.
        
        If query_vector is provided, calculates dot-product relevance.
        Otherwise, uses the block's hotness attribute.
        """
        scores: Dict[int, float] = {}

        for block in candidate_blocks:
            if query_vector is not None:
                # If block has real tensor data attached
                if hasattr(block, "k_data") and block.k_data is not None:
                    # block.k_data: [tokens, heads, head_dim] or [tokens, head_dim]
                    k = block.k_data
                    if k.ndim == 3 and query_vector.ndim == 2:
                        dots = np.einsum("hd,thd->th", query_vector, k) * self.scale
                        score = float(np.max(dots))
                    elif k.ndim == 2 and query_vector.ndim == 1:
                        dots = np.dot(k, query_vector) * self.scale
                        score = float(np.max(dots))
                    else:
                        score = float(block.hotness)
                else:
                    # Synthetic scoring fallback
                    pseudo_key = np.full(self.head_dim, block.hotness, dtype=np.float32)
                    q = np.asarray(query_vector, dtype=np.float32)
                    if q.ndim > 1:
                        q = q[0]
                    score = float(np.dot(q, pseudo_key) / (self.head_dim ** 0.5))
            else:
                score = float(block.hotness)

            scores[block.block_id] = score

        return scores
