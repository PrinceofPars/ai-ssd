"""
AI-SSD Unified API Gateway: Single entry point handling KV cache storage, retrieval,
top-k filtering, and prefetch dispatching.
"""

from typing import Any, Optional, List
from common.schemas.request import KVRequest, KVOperation
from common.schemas.result import KVResponse, OperationStatus
from person3_system.api.responses import ResponseFactory


class AISSDSystem:
    """
    Standardized API Gateway facade unifying KV cache operations across
    Host Memory, Storage Controller, and Physical Flash Tiers.
    """

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
        elif request.operation == KVOperation.KV_WRITE:
            return self._handle_write(request)
        elif request.operation == KVOperation.KV_PREFETCH:
            return self._handle_prefetch(request)
        elif request.operation == KVOperation.KV_EVICT:
            return self._handle_evict(request)
        else:
            return ResponseFactory.error(request.request_id, f"Unsupported operation {request.operation}")

    def _handle_topk(self, request: KVRequest) -> KVResponse:
        candidate_blocks = request.candidate_blocks or []
        k = request.top_k or min(len(candidate_blocks), 16)
        k = min(k, len(candidate_blocks)) if candidate_blocks else 0

        # Check if blocks are pre-staged in host DRAM buffer
        cache_hit = False
        if self.prefetcher and candidate_blocks:
            cache_hit = self.prefetcher.is_staged(candidate_blocks, layer_id=request.layer_id)

        # In-storage top-k selection
        selected = candidate_blocks[:k]

        bytes_req = len(candidate_blocks) * 4096
        bytes_transferred = len(selected) * 4096

        if cache_hit:
            # Latency hidden in host DRAM (near zero flash latency)
            latency_us = 1.0
        else:
            if hasattr(self.ssd_engine, "estimate_read_latency"):
                latency_us = float(self.ssd_engine.estimate_read_latency(selected))
            else:
                latency_us = len(selected) * 25.0

        return ResponseFactory.success(
            request_id=request.request_id,
            selected_blocks=selected,
            bytes_requested=bytes_req,
            bytes_transferred=bytes_transferred,
            latency_us=latency_us,
            cache_hit=cache_hit,
        )

    def _handle_read(self, request: KVRequest) -> KVResponse:
        candidate_blocks = request.candidate_blocks or []
        if hasattr(self.ssd_engine, "estimate_read_latency"):
            latency = float(self.ssd_engine.estimate_read_latency(candidate_blocks))
        else:
            latency = len(candidate_blocks) * 25.0

        bytes_total = len(candidate_blocks) * 4096
        return ResponseFactory.success(
            request_id=request.request_id,
            selected_blocks=candidate_blocks,
            bytes_requested=bytes_total,
            bytes_transferred=bytes_total,
            latency_us=latency,
            cache_hit=False,
        )

    def _handle_write(self, request: KVRequest) -> KVResponse:
        blocks_to_write = request.blocks_to_write or request.candidate_blocks or []
        bytes_total = len(blocks_to_write) * 4096
        latency_us = len(blocks_to_write) * 200.0  # tPROG approx

        return ResponseFactory.success(
            request_id=request.request_id,
            selected_blocks=blocks_to_write,
            bytes_requested=bytes_total,
            bytes_transferred=bytes_total,
            latency_us=latency_us,
            cache_hit=False,
        )

    def _handle_prefetch(self, request: KVRequest) -> KVResponse:
        candidate_blocks = request.candidate_blocks or []
        if self.prefetcher and candidate_blocks:
            self.prefetcher.stage_blocks(candidate_blocks, layer_id=request.layer_id)

        if hasattr(self.ssd_engine, "prefetch_kv"):
            try:
                self.ssd_engine.prefetch_kv(candidate_blocks, layer_id=request.layer_id)
            except Exception:
                pass

        return ResponseFactory.success(
            request_id=request.request_id,
            selected_blocks=candidate_blocks,
            bytes_requested=len(candidate_blocks) * 4096,
            bytes_transferred=0,
            latency_us=1.0,
            cache_hit=True,
        )

    def _handle_evict(self, request: KVRequest) -> KVResponse:
        candidate_blocks = request.candidate_blocks or []
        if hasattr(self.ssd_engine, "evict_kv"):
            for bid in candidate_blocks:
                self.ssd_engine.evict_kv(bid, request.layer_id)

        return ResponseFactory.success(
            request_id=request.request_id,
            selected_blocks=candidate_blocks,
            bytes_requested=0,
            bytes_transferred=0,
            latency_us=5.0,
            cache_hit=False,
        )
