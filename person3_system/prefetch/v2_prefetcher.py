"""
AI-SSD V2 Speculative Prefetcher with Rigorous Storage and Hit Accounting.

Features:
- Dispatches actual storage requests to StorageBackend (no free reads).
- Tracks useful, useless (pollution), and late prefetches.
- Accurately accounts for memory consumption and pipeline bubble penalties.
"""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Set, List, Dict, Any, Optional, Tuple
from person3_system.storage.backend import StorageBackend, StorageRequest, StorageResult
from person3_system.prefetch.predictor import NextLayerPredictor


@dataclass
class StagedBlockEntry:
    """Metadata for a block staged in host DRAM."""
    block_id: int
    layer_id: int
    size_bytes: int
    staged_at_us: float
    ready_at_us: float
    is_useful: bool = False
    accessed_at_us: Optional[float] = None


class V2Prefetcher:
    """
    Host DRAM Staging Buffer and Realistic Speculative Prefetch Engine.
    Ensures every prefetch read is accounted for in storage metrics.
    """

    def __init__(
        self,
        storage_backend: StorageBackend,
        buffer_capacity_blocks: int = 512,
        bytes_per_block: int = 4096,
        gpu_compute_time_per_layer_us: float = 65.0,
    ):
        self.backend = storage_backend
        self.buffer_capacity = buffer_capacity_blocks
        self.bytes_per_block = bytes_per_block
        self.gpu_compute_time_per_layer_us = gpu_compute_time_per_layer_us

        # LRU Staging Buffer: (layer_id, block_id) -> StagedBlockEntry
        self._staged_buffer: OrderedDict[Tuple[int, int], StagedBlockEntry] = OrderedDict()

        # Telemetry & Accounting Counters
        self.prefetch_requests: int = 0
        self.useful_prefetches: int = 0
        self.useless_prefetches: int = 0
        self.late_prefetches: int = 0
        self.demand_hits: int = 0
        self.demand_misses: int = 0
        self.extra_bytes_read: int = 0
        self.pipeline_stalls: int = 0
        self.total_stall_penalty_us: float = 0.0
        self.peak_memory_bytes: int = 0

        self.predictor = NextLayerPredictor()

    @property
    def current_memory_bytes(self) -> int:
        return len(self._staged_buffer) * self.bytes_per_block

    def prefetch_blocks(
        self,
        block_ids: List[int],
        layer_id: int,
        current_time_us: float,
    ) -> List[int]:
        """
        Dispatches speculative prefetch requests to the StorageBackend.
        """
        prefetched = []
        requests = []
        for bid in block_ids:
            key = (layer_id, bid)
            if key in self._staged_buffer:
                continue

            # Evict if full
            if len(self._staged_buffer) >= self.buffer_capacity:
                oldest_key, oldest_entry = self._staged_buffer.popitem(last=False)
                if not oldest_entry.is_useful:
                    self.useless_prefetches += 1
                    self.extra_bytes_read += oldest_entry.size_bytes

            requests.append(
                StorageRequest(
                    block_id=bid,
                    layer_id=layer_id,
                    length=self.bytes_per_block,
                    metadata={"prefetch": True},
                )
            )
            prefetched.append(bid)

        if requests:
            # Issue batch to storage backend
            results = self.backend.submit_batch(requests)
            for res in results:
                self.prefetch_requests += 1
                completion_time_us = current_time_us + res.latency_us
                entry = StagedBlockEntry(
                    block_id=res.block_id,
                    layer_id=layer_id,
                    size_bytes=res.length,
                    staged_at_us=current_time_us,
                    ready_at_us=completion_time_us,
                    is_useful=False,
                )
                self._staged_buffer[(layer_id, res.block_id)] = entry

            self.peak_memory_bytes = max(self.peak_memory_bytes, self.current_memory_bytes)

        return prefetched

    def access_blocks(
        self,
        block_ids: List[int],
        layer_id: int,
        demand_time_us: float,
    ) -> Tuple[List[int], List[int], float]:
        """
        Processes demand read requests for a layer.
        Returns: (hit_block_ids, miss_block_ids, latency_us)
        """
        hits = []
        misses = []
        stalls_for_layer = 0.0

        for bid in block_ids:
            key = (layer_id, bid)
            if key in self._staged_buffer:
                entry = self._staged_buffer[key]
                self._staged_buffer.move_to_end(key)
                if not entry.is_useful:
                    entry.is_useful = True
                    self.useful_prefetches += 1

                entry.accessed_at_us = demand_time_us
                self.demand_hits += 1
                hits.append(bid)

                # Check for late prefetch (completion is after demand time)
                if entry.ready_at_us > demand_time_us:
                    self.late_prefetches += 1
                    wait_us = entry.ready_at_us - demand_time_us
                    stalls_for_layer = max(stalls_for_layer, wait_us)
            else:
                self.demand_misses += 1
                misses.append(bid)

        # Handle demand misses: must read from storage synchronously
        miss_latency_us = 0.0
        if misses:
            miss_reqs = [
                StorageRequest(
                    block_id=bid,
                    layer_id=layer_id,
                    length=self.bytes_per_block,
                    metadata={"demand_miss": True},
                )
                for bid in misses
            ]
            results = self.backend.submit_batch(miss_reqs)
            miss_latency_us = sum(r.latency_us for r in results)
            stalls_for_layer += miss_latency_us

        # Net stall penalty considering GPU execution overlap
        net_stall_us = max(0.0, stalls_for_layer - self.gpu_compute_time_per_layer_us)
        if net_stall_us > 0.0:
            self.pipeline_stalls += 1
            self.total_stall_penalty_us += net_stall_us

        return hits, misses, net_stall_us

    def get_telemetry(self) -> Dict[str, Any]:
        """Calculates precise empirical prefetch metrics."""
        # Remaining unread entries at end of run count as useless prefetches
        unaccessed = sum(1 for e in self._staged_buffer.values() if not e.is_useful)
        total_useless = self.useless_prefetches + unaccessed
        total_demand = self.demand_hits + self.demand_misses
        demand_hit_rate = (self.demand_hits / total_demand) if total_demand > 0 else 0.0
        prefetch_accuracy = (self.useful_prefetches / self.prefetch_requests) if self.prefetch_requests > 0 else 0.0

        return {
            "prefetch_requests": self.prefetch_requests,
            "useful_prefetches": self.useful_prefetches,
            "useless_prefetches": total_useless,
            "late_prefetches": self.late_prefetches,
            "demand_hits": self.demand_hits,
            "demand_misses": self.demand_misses,
            "demand_hit_rate": round(demand_hit_rate, 4),
            "demand_hit_rate_pct": round(demand_hit_rate * 100.0, 2),
            "prefetch_accuracy": round(prefetch_accuracy, 4),
            "prefetch_accuracy_pct": round(prefetch_accuracy * 100.0, 2),
            "extra_bytes_read": self.extra_bytes_read + (unaccessed * self.bytes_per_block),
            "pipeline_stalls": self.pipeline_stalls,
            "total_stall_penalty_us": round(self.total_stall_penalty_us, 2),
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_mb": round(self.peak_memory_bytes / (1024 * 1024), 2),
        }

    def clear(self) -> None:
        """Reset state and counters."""
        self._staged_buffer.clear()
        self.prefetch_requests = 0
        self.useful_prefetches = 0
        self.useless_prefetches = 0
        self.late_prefetches = 0
        self.demand_hits = 0
        self.demand_misses = 0
        self.extra_bytes_read = 0
        self.pipeline_stalls = 0
        self.total_stall_penalty_us = 0.0
        self.peak_memory_bytes = 0
