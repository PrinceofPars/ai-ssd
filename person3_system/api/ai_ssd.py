"""
AI-SSD Unified API Gateway: Single entry point handling KV cache storage, retrieval,
top-k filtering, and prefetch dispatching.
"""

from typing import Any, Optional
from common.schemas.request import KVRequest, KVOperation
from common.schemas.result import KVResponse, OperationStatus
from person3_system.api.responses import ResponseFactory


class AISSDSystem:
    def __init__(self, kv_engine: Any, ssd_engine: Any, prefetcher: Optional[Any] = None):
        self.kv_engine = kv_engine
        self.ssd_engine = ssd_engine
        self.prefetcher = prefetcher

    def execute(self, request: KVRequest) -> KVResponse:
        """Route request to appropriate subsystem."""
        if request.operation == KVOperation.KV_TOPK:
            return self._handle_topk(request)
        elif request.operation == KVOperation.KV_READ:
            return self._handle_read(request)
        elif request.operation == KVOperation.KV_PREFETCH:
            return self._handle_prefetch(request)
        else:
            return ResponseFactory.error(request.request_id, f"Unsupported operation {request.operation}")

    def _handle_topk(self, request: KVRequest) -> KVResponse:
        # Check prefetch hit
        cache_hit = False
        if self.prefetcher and self.prefetcher.is_staged(request.candidate_blocks):
            cache_hit = True

        # Perform Top-k selection
        k = request.top_k or min(len(request.candidate_blocks), 16)
        selected = request.candidate_blocks[:k]

        # Calculate bytes and latency
        bytes_req = len(request.candidate_blocks) * 4096
        bytes_transferred = len(selected) * 4096
        latency_us = 5.0 if cache_hit else self.ssd_engine.estimate_read_latency(selected)

        return ResponseFactory.success(
            request_id=request.request_id,
            selected_blocks=selected,
            bytes_requested=bytes_req,
            bytes_transferred=bytes_transferred,
            latency_us=latency_us,
            cache_hit=cache_hit,
        )

    def _handle_read(self, request: KVRequest) -> KVResponse:
        latency = self.ssd_engine.estimate_read_latency(request.candidate_blocks)
        bytes_total = len(request.candidate_blocks) * 4096
        return ResponseFactory.success(
            request_id=request.request_id,
            selected_blocks=request.candidate_blocks,
            bytes_requested=bytes_total,
            bytes_transferred=bytes_total,
            latency_us=latency,
            cache_hit=False,
        )

    def _handle_prefetch(self, request: KVRequest) -> KVResponse:
        if self.prefetcher:
            self.prefetcher.stage_blocks(request.candidate_blocks)
        return ResponseFactory.success(
            request_id=request.request_id,
            selected_blocks=request.candidate_blocks,
            bytes_requested=len(request.candidate_blocks) * 4096,
            bytes_transferred=0,
            latency_us=1.0,
            cache_hit=True,
        )
