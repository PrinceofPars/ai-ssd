"""
Response factory helpers for standardizing subsystem outputs.
"""

from typing import List, Optional
from common.schemas.result import KVResponse, OperationStatus


class ResponseFactory:
    @staticmethod
    def success(
        request_id: str,
        selected_blocks: List[int],
        bytes_requested: int,
        bytes_transferred: int,
        latency_us: float,
        cache_hit: bool = False,
    ) -> KVResponse:
        return KVResponse(
            request_id=request_id,
            status=OperationStatus.SUCCESS,
            selected_blocks=selected_blocks,
            bytes_requested=bytes_requested,
            bytes_transferred=bytes_transferred,
            latency_us=latency_us,
            cache_hit=cache_hit,
        )

    @staticmethod
    def error(request_id: str, details: str) -> KVResponse:
        return KVResponse(
            request_id=request_id,
            status=OperationStatus.ERROR,
            details=details,
        )
