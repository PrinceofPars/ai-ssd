"""
Request factory and validation helpers for AI-SSD.
"""

from typing import List, Optional
from common.schemas.request import KVRequest, KVOperation


class RequestFactory:
    _counter = 0

    @classmethod
    def create_topk_request(
        cls,
        layer_id: int,
        head_id: int,
        candidate_blocks: List[int],
        top_k: int,
        query_id: Optional[int] = None,
    ) -> KVRequest:
        cls._counter += 1
        return KVRequest(
            request_id=f"req_{cls._counter:06d}",
            operation=KVOperation.KV_TOPK,
            layer_id=layer_id,
            head_id=head_id,
            query_id=query_id,
            candidate_blocks=candidate_blocks,
            top_k=top_k,
        )

    @classmethod
    def create_prefetch_request(
        cls,
        layer_id: int,
        candidate_blocks: List[int],
    ) -> KVRequest:
        cls._counter += 1
        return KVRequest(
            request_id=f"req_{cls._counter:06d}",
            operation=KVOperation.KV_PREFETCH,
            layer_id=layer_id,
            head_id=0,
            candidate_blocks=candidate_blocks,
        )
