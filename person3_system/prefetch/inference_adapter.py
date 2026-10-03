"""
Real Inference Prefetch Adapter for AI-SSD V2 (P3).

Provides a high-performance, non-blocking prefetch adapter that connects
real LLM inference loops (P1) to AI-SSD storage backends (P2/P3).

Key Capabilities:
- Non-blocking Speculative Prefetch: Dispatches storage read requests to host DRAM staging.
- Actual Block Data Retrieval: Returns real tensor slices (NumPy) or raw bytes for K and V pages.
- Rigorous Metric Accounting: Tracks demand reads, prefetch requests, useful prefetches,
  late prefetches, useless prefetches, bytes, and latencies.
- Zero Artificial Latency: Operates at native hardware/in-memory speed with no artificial delays.
- Event-Driven Progression: Completely driven by the inference loop's step/layer calls,
  with zero reliance on analytical tokens/second models.
"""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Set, List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import Future
import time
import numpy as np

from person3_system.storage.backend import StorageBackend, StorageRequest, StorageResult
from person3_system.storage.mock_backend import MockStorageBackend
from person3_system.prefetch.v2_prefetcher import V2Prefetcher
from person3_system.prefetch.predictor import NextLayerPredictor
from common.schemas.kv_block import KEY_PAGE_BYTES, VALUE_PAGE_BYTES, LOGICAL_BLOCK_BYTES, KVBlock


@dataclass
class StagedInferenceBlock:
    """Descriptor for an actual KV block staged in host DRAM."""
    block_id: int
    layer_id: int
    head_id: int = 0
    size_bytes: int = LOGICAL_BLOCK_BYTES
    data: Optional[bytes] = None
    future: Optional[Future[StorageResult]] = None
    staged_time_ns: int = 0
    ready_time_ns: int = 0
    is_useful: bool = False
    is_late: bool = False
    accessed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealInferencePrefetchAdapter:
    """
    Adapter exposing V2Prefetcher functionality and storage backends to real LLM inference.
    """

    def __init__(
        self,
        storage_backend: Optional[StorageBackend] = None,
        buffer_capacity_blocks: int = 512,
        bytes_per_block: int = LOGICAL_BLOCK_BYTES,
        tokens_per_block: int = 16,
        kv_heads_per_block: int = 1,
        head_dim: int = 64,
        dtype: str = "FP32",
    ):
        self.storage_backend = storage_backend if storage_backend is not None else MockStorageBackend()
        self.buffer_capacity_blocks = buffer_capacity_blocks
        self.bytes_per_block = bytes_per_block
        self.tokens_per_block = tokens_per_block
        self.kv_heads_per_block = kv_heads_per_block
        self.head_dim = head_dim
        self.dtype_str = dtype.upper()

        self.bytes_per_elem = 4 if self.dtype_str in ("FP32", "FLOAT32") else (2 if self.dtype_str in ("FP16", "FLOAT16", "BF16") else 1)
        self.np_dtype = np.float32 if self.bytes_per_elem == 4 else (np.float16 if self.bytes_per_elem == 2 else np.int8)

        # LRU DRAM Staging Buffer: (layer_id, block_id) -> StagedInferenceBlock
        self._staging_buffer: OrderedDict[Tuple[int, int], StagedInferenceBlock] = OrderedDict()

        # Geometry metadata cache: (layer_id, block_id) -> metadata dict
        self._block_meta: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Payload backing store for backends that only model latency
        self._block_payloads: Dict[int, bytes] = {}

        # Predictive prefetch model
        self.predictor = NextLayerPredictor()

        # Underlying V2Prefetcher simulation reference if needed
        self.v2_prefetcher = V2Prefetcher(
            storage_backend=self.storage_backend,
            buffer_capacity_blocks=self.buffer_capacity_blocks,
            bytes_per_block=self.bytes_per_block,
        )

        # Telemetry & Metrics Counters
        self.demand_reads: int = 0
        self.demand_hits: int = 0
        self.demand_misses: int = 0

        self.prefetch_requests: int = 0
        self.useful_prefetches: int = 0
        self.late_prefetches: int = 0
        self.useless_prefetches: int = 0

        self.prefetched_bytes: int = 0
        self.useful_bytes: int = 0
        self.wasted_bytes: int = 0
        self.demand_bytes: int = 0

        self.total_latency_us: float = 0.0
        self.demand_hit_latency_us: float = 0.0
        self.demand_miss_latency_us: float = 0.0

        self.peak_memory_bytes: int = 0

    @property
    def current_memory_bytes(self) -> int:
        """Current DRAM bytes occupied by staged KV blocks."""
        return len(self._staging_buffer) * self.bytes_per_block

    def _serialize_payload(
        self,
        payload: Optional[Union[bytes, Dict[str, np.ndarray], np.ndarray]],
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Serializes tensor payload into byte buffers preserving tensor geometry metadata."""
        if payload is None:
            return b"\x00" * self.bytes_per_block, {}

        if isinstance(payload, dict) and "k" in payload and "v" in payload:
            k_arr = np.ascontiguousarray(payload["k"])
            v_arr = np.ascontiguousarray(payload["v"])
            k_bytes = k_arr.tobytes()
            v_bytes = v_arr.tobytes()
            combined = k_bytes + v_bytes
            meta = {
                "k_shape": k_arr.shape,
                "v_shape": v_arr.shape,
                "dtype": str(k_arr.dtype),
                "k_bytes": len(k_bytes),
                "v_bytes": len(v_bytes),
            }
            return combined, meta

        if isinstance(payload, np.ndarray):
            arr = np.ascontiguousarray(payload)
            return arr.tobytes(), {"shape": arr.shape, "dtype": str(arr.dtype), "bytes": arr.nbytes}

        if isinstance(payload, (bytes, bytearray)):
            raw = bytes(payload)
            if len(raw) < self.bytes_per_block:
                raw = raw + b"\x00" * (self.bytes_per_block - len(raw))
            return raw, {"bytes": len(raw)}

        raise TypeError(f"Unsupported payload type: {type(payload)}")

    def _ensure_buffer_capacity(self) -> None:
        """Evicts LRU entries if DRAM staging capacity is exceeded."""
        while len(self._staging_buffer) >= self.buffer_capacity_blocks:
            oldest_key, oldest_entry = self._staging_buffer.popitem(last=False)
            if not oldest_entry.is_useful:
                self.useless_prefetches += 1
                self.wasted_bytes += oldest_entry.size_bytes

    def register_block(
        self,
        block_id: int,
        layer_id: int,
        head_id: int = 0,
        payload: Optional[Union[bytes, Dict[str, np.ndarray], np.ndarray]] = None,
        token_start: int = 0,
        stage_in_dram: bool = False,
        **kwargs,
    ) -> StorageResult:
        """
        Stores a KV block into the storage backend.
        
        Args:
            block_id: Logical block index.
            layer_id: Transformer layer index.
            head_id: KV attention head index.
            payload: Real block data (dict of k,v numpy arrays, raw bytes, or array).
            token_start: Sequence position offset.
            stage_in_dram: If True, immediately stages the block in host DRAM.
        """
        raw_bytes, meta = self._serialize_payload(payload)
        meta.update({
            "token_start": token_start,
            "layer_id": layer_id,
            "head_id": head_id,
            **kwargs,
        })
        key = (layer_id, block_id)
        self._block_meta[key] = meta
        self._block_payloads[block_id] = raw_bytes

        # Persist to storage backend
        res = self.storage_backend.write(
            block_id=block_id,
            offset=0,
            data=raw_bytes,
            layer_id=layer_id,
            head_id=head_id,
            length=len(raw_bytes),
            operation="PREFILL_WRITE",
            **kwargs,
        )

        if stage_in_dram:
            self._ensure_buffer_capacity()
            entry = StagedInferenceBlock(
                block_id=block_id,
                layer_id=layer_id,
                head_id=head_id,
                size_bytes=len(raw_bytes),
                data=raw_bytes,
                staged_time_ns=time.perf_counter_ns(),
                ready_time_ns=time.perf_counter_ns(),
                is_useful=True,
                accessed=False,
                metadata=meta,
            )
            self._staging_buffer[key] = entry
            self.peak_memory_bytes = max(self.peak_memory_bytes, self.current_memory_bytes)

        return res

    def register_blocks_from_adapter(
        self,
        layer_blocks: List[Tuple[KVBlock, Dict[str, np.ndarray]]],
        stage_in_dram_if_tier: bool = False,
    ) -> int:
        """
        Convenience ingestion method for output from P1's KVBlockAdapter.blockize_layer().
        """
        count = 0
        for block_desc, payload in layer_blocks:
            stage_dram = stage_in_dram_if_tier and (block_desc.storage_tier == "DRAM")
            self.register_block(
                block_id=block_desc.block_id,
                layer_id=block_desc.layer_id,
                head_id=block_desc.kv_head_start,
                payload=payload,
                token_start=block_desc.token_start,
                stage_in_dram=stage_dram,
            )
            count += 1
        return count

    def prefetch(
        self,
        block_ids: List[int],
        layer_id: int,
        head_id: int = 0,
        **kwargs,
    ) -> List[int]:
        """
        Dispatches asynchronous speculative prefetch requests for upcoming blocks.
        """
        dispatched = []
        for bid in block_ids:
            key = (layer_id, bid)
            if key in self._staging_buffer:
                continue

            self._ensure_buffer_capacity()
            staged_ns = time.perf_counter_ns()

            # Issue non-blocking async read to storage backend
            fut = self.storage_backend.async_read(
                block_id=bid,
                offset=0,
                length=self.bytes_per_block,
                layer_id=layer_id,
                head_id=head_id,
                operation="KV_PREFETCH",
                **kwargs,
            )

            entry = StagedInferenceBlock(
                block_id=bid,
                layer_id=layer_id,
                head_id=head_id,
                size_bytes=self.bytes_per_block,
                future=fut,
                staged_time_ns=staged_ns,
                ready_time_ns=0,
                is_useful=False,
                is_late=False,
            )
            self._staging_buffer[key] = entry
            self.prefetch_requests += 1
            self.prefetched_bytes += self.bytes_per_block
            dispatched.append(bid)

        self.peak_memory_bytes = max(self.peak_memory_bytes, self.current_memory_bytes)
        return dispatched

    def prefetch_blocks(
        self,
        block_ids: List[int],
        layer_id: int,
        head_id: int = 0,
        **kwargs,
    ) -> List[int]:
        """Alias for prefetch() matching V2Prefetcher interface."""
        return self.prefetch(block_ids=block_ids, layer_id=layer_id, head_id=head_id, **kwargs)

    def predict_and_prefetch(
        self,
        current_layer_id: int,
        current_block_ids: List[int],
        stride: int = 0,
    ) -> Tuple[int, List[int]]:
        """
        Predicts candidate blocks for next layer (L+1) and issues speculative prefetches.
        """
        next_layer, predicted_bids = self.predictor.predict_next_layer_blocks(
            current_layer_id=current_layer_id,
            current_block_ids=current_block_ids,
            stride=stride,
        )
        dispatched = self.prefetch(block_ids=predicted_bids, layer_id=next_layer)
        return next_layer, dispatched

    def _format_output(
        self,
        raw_bytes: bytes,
        key: Tuple[int, int],
        sub_page: str,
        return_tensors: bool,
    ) -> Union[bytes, Dict[str, np.ndarray], np.ndarray]:
        """Slices and formats raw block bytes according to sub_page and tensor requirements."""
        sub_page_upper = sub_page.upper()
        meta = self._block_meta.get(key, {})

        k_page_size = meta.get("k_bytes", KEY_PAGE_BYTES)
        v_page_size = meta.get("v_bytes", VALUE_PAGE_BYTES)

        if sub_page_upper == "KEY":
            k_bytes = raw_bytes[:k_page_size]
            if not return_tensors:
                return k_bytes
            shape = meta.get("k_shape", (self.kv_heads_per_block, self.tokens_per_block, self.head_dim))
            return np.frombuffer(k_bytes, dtype=self.np_dtype).reshape(shape)

        elif sub_page_upper == "VALUE":
            v_bytes = raw_bytes[k_page_size : k_page_size + v_page_size]
            if not return_tensors:
                return v_bytes
            shape = meta.get("v_shape", (self.kv_heads_per_block, self.tokens_per_block, self.head_dim))
            return np.frombuffer(v_bytes, dtype=self.np_dtype).reshape(shape)

        elif sub_page_upper == "BOTH":
            if not return_tensors:
                return raw_bytes
            k_bytes = raw_bytes[:k_page_size]
            v_bytes = raw_bytes[k_page_size : k_page_size + v_page_size]
            k_shape = meta.get("k_shape", (self.kv_heads_per_block, self.tokens_per_block, self.head_dim))
            v_shape = meta.get("v_shape", (self.kv_heads_per_block, self.tokens_per_block, self.head_dim))
            return {
                "k": np.frombuffer(k_bytes, dtype=self.np_dtype).reshape(k_shape),
                "v": np.frombuffer(v_bytes, dtype=self.np_dtype).reshape(v_shape),
            }

        raise ValueError(f"Unknown sub_page: {sub_page}. Must be 'KEY', 'VALUE', or 'BOTH'.")

    def get_block(
        self,
        block_id: int,
        layer_id: int,
        head_id: int = 0,
        sub_page: str = "BOTH",
        return_tensors: bool = False,
        **kwargs,
    ) -> Union[bytes, Dict[str, np.ndarray], np.ndarray]:
        """
        Demands an actual KV block for attention computation.
        
        Hit semantics:
          - If prefetched and I/O completed: returns immediately (useful prefetch hit).
          - If prefetched but I/O still in flight: awaits I/O completion (late prefetch).
          - If not prefetched: synchronously fetches from storage (demand miss).
        """
        key = (layer_id, block_id)
        t_start_ns = time.perf_counter_ns()
        self.demand_reads += 1

        if key in self._staging_buffer:
            self.demand_hits += 1
            entry = self._staging_buffer[key]
            self._staging_buffer.move_to_end(key)

            # Resolve async future if still pending
            if entry.future is not None:
                if not entry.future.done():
                    # Storage I/O caught in flight: Late prefetch
                    entry.is_late = True
                    self.late_prefetches += 1
                    res = entry.future.result()
                    entry.ready_time_ns = time.perf_counter_ns()
                else:
                    res = entry.future.result()
                    entry.ready_time_ns = entry.staged_time_ns

                entry.data = res.data if res.data is not None else self._block_payloads.get(block_id, b"\x00" * self.bytes_per_block)
                entry.future = None

            if not entry.is_useful:
                entry.is_useful = True
                self.useful_prefetches += 1
                self.useful_bytes += entry.size_bytes

            elapsed_us = (time.perf_counter_ns() - t_start_ns) / 1000.0
            self.demand_hit_latency_us += elapsed_us
            self.total_latency_us += elapsed_us
            self.demand_bytes += entry.size_bytes

            raw_bytes = entry.data if entry.data is not None else self._block_payloads.get(block_id, b"\x00" * self.bytes_per_block)
            return self._format_output(raw_bytes, key, sub_page, return_tensors)

        else:
            # Demand Miss: Synchronous storage read
            self.demand_misses += 1
            res = self.storage_backend.read(
                block_id=block_id,
                offset=0,
                length=self.bytes_per_block,
                layer_id=layer_id,
                head_id=head_id,
                operation="DECODE_READ",
                sub_page=sub_page,
                **kwargs,
            )
            elapsed_us = (time.perf_counter_ns() - t_start_ns) / 1000.0
            self.demand_miss_latency_us += elapsed_us
            self.total_latency_us += elapsed_us
            self.demand_bytes += (res.length or self.bytes_per_block)

            raw_bytes = res.data if res.data is not None else self._block_payloads.get(block_id, b"\x00" * self.bytes_per_block)

            # Cache the demanded block in host DRAM
            self._ensure_buffer_capacity()
            miss_entry = StagedInferenceBlock(
                block_id=block_id,
                layer_id=layer_id,
                head_id=head_id,
                size_bytes=len(raw_bytes),
                data=raw_bytes,
                staged_time_ns=t_start_ns,
                ready_time_ns=time.perf_counter_ns(),
                is_useful=True,
                accessed=True,
            )
            self._staging_buffer[key] = miss_entry
            self.peak_memory_bytes = max(self.peak_memory_bytes, self.current_memory_bytes)

            return self._format_output(raw_bytes, key, sub_page, return_tensors)

    def get_blocks(
        self,
        block_ids: List[int],
        layer_id: int,
        head_id: int = 0,
        sub_page: str = "BOTH",
        return_tensors: bool = False,
    ) -> Dict[int, Union[bytes, Dict[str, np.ndarray], np.ndarray]]:
        """Batch demand retrieval helper."""
        return {
            bid: self.get_block(
                block_id=bid,
                layer_id=layer_id,
                head_id=head_id,
                sub_page=sub_page,
                return_tensors=return_tensors,
            )
            for bid in block_ids
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Returns comprehensive performance metrics and accounting.
        """
        unaccessed_count = sum(1 for e in self._staging_buffer.values() if not e.is_useful)
        total_useless = self.useless_prefetches + unaccessed_count
        total_wasted_bytes = self.wasted_bytes + (unaccessed_count * self.bytes_per_block)

        hit_rate = (self.demand_hits / self.demand_reads) if self.demand_reads > 0 else 0.0
        prefetch_accuracy = (self.useful_prefetches / self.prefetch_requests) if self.prefetch_requests > 0 else 0.0
        avg_latency_us = (self.total_latency_us / self.demand_reads) if self.demand_reads > 0 else 0.0
        hit_avg_lat = (self.demand_hit_latency_us / self.demand_hits) if self.demand_hits > 0 else 0.0
        miss_avg_lat = (self.demand_miss_latency_us / self.demand_misses) if self.demand_misses > 0 else 0.0

        return {
            # Core Accounting Counters (Phase 5C Requirement)
            "demand_reads": self.demand_reads,
            "demand_hits": self.demand_hits,
            "demand_misses": self.demand_misses,
            "demand_hit_rate": round(hit_rate, 4),
            "demand_hit_rate_pct": round(hit_rate * 100.0, 2),

            "prefetch_requests": self.prefetch_requests,
            "useful_prefetches": self.useful_prefetches,
            "late_prefetches": self.late_prefetches,
            "useless_prefetches": total_useless,
            "prefetch_accuracy": round(prefetch_accuracy, 4),
            "prefetch_accuracy_pct": round(prefetch_accuracy * 100.0, 2),

            # Byte Metrics
            "demand_bytes": self.demand_bytes,
            "prefetched_bytes": self.prefetched_bytes,
            "useful_bytes": self.useful_bytes,
            "wasted_bytes": total_wasted_bytes,
            "total_bytes": self.demand_bytes + total_wasted_bytes,

            # Latency Metrics (microseconds)
            "total_latency_us": round(self.total_latency_us, 2),
            "avg_latency_us": round(avg_latency_us, 2),
            "demand_hit_avg_latency_us": round(hit_avg_lat, 2),
            "demand_miss_avg_latency_us": round(miss_avg_lat, 2),

            # Buffer & Host DRAM Metrics
            "buffer_capacity_blocks": self.buffer_capacity_blocks,
            "current_staged_blocks": len(self._staging_buffer),
            "current_memory_bytes": self.current_memory_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,

            # Backend Stats
            "storage_backend": self.storage_backend.get_telemetry(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Alias for get_telemetry()."""
        return self.get_telemetry()

    def clear(self) -> None:
        """Resets DRAM staging buffer and telemetry counters."""
        self._staging_buffer.clear()
        self._block_meta.clear()
        self._block_payloads.clear()
        self.demand_reads = 0
        self.demand_hits = 0
        self.demand_misses = 0
        self.prefetch_requests = 0
        self.useful_prefetches = 0
        self.late_prefetches = 0
        self.useless_prefetches = 0
        self.prefetched_bytes = 0
        self.useful_bytes = 0
        self.wasted_bytes = 0
        self.demand_bytes = 0
        self.total_latency_us = 0.0
        self.demand_hit_latency_us = 0.0
        self.demand_miss_latency_us = 0.0
        self.peak_memory_bytes = 0

    def close(self) -> None:
        """Closes storage backend and releases resources."""
        if hasattr(self.storage_backend, "close"):
            self.storage_backend.close()
