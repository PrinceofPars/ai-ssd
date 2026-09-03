"""
Speculative Prefetcher: Asynchronous DRAM staging buffer to pre-stage cold blocks from SSD.
Overlaps GPU computation of Layer L with flash retrieval of Layer L+1 blocks.
"""

from collections import OrderedDict
from typing import Set, List, Dict, Any, Optional, Tuple
from person3_system.prefetch.predictor import NextLayerPredictor
from person3_system.prefetch.history import AccessHistory


class SpeculativePrefetcher:
    """
    Host DRAM Staging Buffer and Speculative Prefetch Engine.
    Simulates asynchronous prefetching of Layer L+1 cold blocks while Layer L executes on GPU.
    """

    def __init__(
        self,
        buffer_capacity_blocks: int = 512,
        bytes_per_block: int = 4096,
        gpu_compute_time_per_layer_us: float = 65.0,
    ):
        self.buffer_capacity = buffer_capacity_blocks
        self.bytes_per_block = bytes_per_block
        self.gpu_compute_time_per_layer_us = gpu_compute_time_per_layer_us

        # LRU Staging buffer: maps (layer_id, block_id) -> timestamp/payload
        self._staged_buffer: OrderedDict[Tuple[int, int], bool] = OrderedDict()
        # Legacy fallback set for backwards-compatibility when layer_id is omitted
        self._legacy_staged_blocks: Set[int] = set()

        self.predictor = NextLayerPredictor()
        self.history = AccessHistory()

        # Performance counters
        self.hits: int = 0
        self.misses: int = 0
        self.blocks_hit: int = 0
        self.blocks_requested: int = 0
        self.pipeline_stalls: int = 0
        self.total_stall_penalty_us: float = 0.0
        self.total_prefetched_bytes: int = 0

    @property
    def used_capacity_blocks(self) -> int:
        return len(self._staged_buffer) or len(self._legacy_staged_blocks)

    @property
    def used_capacity_bytes(self) -> int:
        return self.used_capacity_blocks * self.bytes_per_block

    def stage_blocks(self, block_ids: List[int], layer_id: int = 0) -> None:
        """Stages blocks into the host DRAM buffer with LRU eviction."""
        for bid in block_ids:
            key = (layer_id, bid)
            if key in self._staged_buffer:
                self._staged_buffer.move_to_end(key)
            else:
                if len(self._staged_buffer) >= self.buffer_capacity:
                    # Evict oldest entry (LRU)
                    self._staged_buffer.popitem(last=False)
                self._staged_buffer[key] = True

            # Also maintain legacy set for backwards compatibility
            if len(self._legacy_staged_blocks) < self.buffer_capacity:
                self._legacy_staged_blocks.add(bid)

            self.total_prefetched_bytes += self.bytes_per_block

    def prefetch_next_layer(
        self,
        current_layer_id: int,
        active_block_ids: List[int],
        storage_backend: Optional[Any] = None,
    ) -> List[int]:
        """
        Asynchronously predicts and stages blocks for Layer L+1.
        
        Returns:
            List of predicted and staged block IDs for Layer L+1.
        """
        next_layer, predicted_bids = self.predictor.predict_next_layer_blocks(
            current_layer_id=current_layer_id,
            current_block_ids=active_block_ids,
        )

        self.stage_blocks(predicted_bids, layer_id=next_layer)

        # Also trigger SSD Controller DRAM prefetch if supported
        if storage_backend and hasattr(storage_backend, "prefetch_kv"):
            try:
                storage_backend.prefetch_kv(predicted_bids, layer_id=next_layer)
            except Exception:
                pass

        return predicted_bids

    def check_staging(
        self,
        block_ids: List[int],
        layer_id: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """Splits candidate blocks into (hit_blocks, missing_blocks)."""
        hits = []
        missing = []
        for bid in block_ids:
            if layer_id is not None:
                is_in = (layer_id, bid) in self._staged_buffer
            else:
                is_in = bid in self._legacy_staged_blocks or any(k[1] == bid for k in self._staged_buffer)
            if is_in:
                hits.append(bid)
            else:
                missing.append(bid)
        return hits, missing

    def is_staged(
        self,
        block_ids: List[int],
        layer_id: Optional[int] = None,
        estimated_flash_latency_us: float = 70.0,
    ) -> bool:
        """
        Checks if required blocks are present in host DRAM staging buffer.
        Evaluates block-level hit rate and pipeline bubble penalties.
        """
        if not block_ids:
            return True

        hits, missing = self.check_staging(block_ids, layer_id)
        self.blocks_hit += len(hits)
        self.blocks_requested += len(block_ids)

        hit_ratio = len(hits) / float(len(block_ids))
        is_hit = (hit_ratio >= 0.80)

        if is_hit:
            self.hits += 1
            if missing:
                # Partial hit: penalty only for missing blocks
                per_block_us = estimated_flash_latency_us / len(block_ids)
                stall_us = max(0.0, (len(missing) * per_block_us) - self.gpu_compute_time_per_layer_us)
                self.total_stall_penalty_us += stall_us
        else:
            self.misses += 1
            stall_us = max(0.0, estimated_flash_latency_us - self.gpu_compute_time_per_layer_us)
            self.pipeline_stalls += 1
            self.total_stall_penalty_us += stall_us

        return is_hit

    @property
    def hit_rate(self) -> float:
        """Block-level cache hit rate."""
        if self.blocks_requested > 0:
            return self.blocks_hit / float(self.blocks_requested)
        total = self.hits + self.misses
        return (self.hits / float(total)) if total > 0 else 0.0

    @property
    def request_hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / float(total)) if total > 0 else 0.0

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive prefetch performance metrics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "blocks_hit": self.blocks_hit,
            "blocks_requested": self.blocks_requested,
            "hit_rate": self.hit_rate,
            "hit_rate_percent": round(self.hit_rate * 100.0, 1),
            "request_hit_rate": self.request_hit_rate,
            "pipeline_stalls": self.pipeline_stalls,
            "total_stall_penalty_us": self.total_stall_penalty_us,
            "buffer_capacity_blocks": self.buffer_capacity,
            "used_capacity_blocks": self.used_capacity_blocks,
            "used_capacity_bytes": self.used_capacity_bytes,
        }

    def clear(self) -> None:
        """Resets the staging buffer and statistics."""
        self._staged_buffer.clear()
        self._legacy_staged_blocks.clear()
        self.hits = 0
        self.misses = 0
        self.blocks_hit = 0
        self.blocks_requested = 0
        self.pipeline_stalls = 0
        self.total_stall_penalty_us = 0.0
        self.total_prefetched_bytes = 0
