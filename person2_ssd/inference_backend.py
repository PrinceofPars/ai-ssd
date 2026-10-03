"""
Real Inference Storage Backend Adapter for AI-SSD V2.

Exposes Person 2's DeterministicTensorMapper and multi-channel FTL subsystem
through a direct, high-performance, Python-callable storage engine designed
for consumption by Person 1 during real LLM inference.

Classification: ANALYTICAL
  - Real payload retention (stores and returns actual KV tensors/bytes)
  - Analytical FTL multi-channel mapping and telemetry
  - Zero synthetic sleep latency injection
"""

from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np
import time

from common.schemas.kv_block import KVBlock
from person2_ssd.kv_allocator.tensor_mapping import (
    DeterministicTensorMapper,
    TensorCoordinate,
    NANDPhysicalCoordinate,
    LBAAddress,
)

# Physical Flash Page & Logical KV Block Sizes (Codified Contract)
KEY_PAGE_BYTES: int = 4096        # 4 KiB Key Page
VALUE_PAGE_BYTES: int = 4096      # 4 KiB Value Page
LOGICAL_BLOCK_BYTES: int = 8192   # 8 KiB Combined K+V Logical Block


class RealInferenceStorageBackend:
    """
    Python-callable storage backend for real LLM inference callers.
    
    Adheres strictly to the AI-SSD V2 Contract:
    - K page size: exactly 4,096 bytes (4 KiB)
    - V page size: exactly 4,096 bytes (4 KiB)
    - Combined block size: exactly 8,192 bytes (8 KiB)
    - Mapping: DeterministicTensorMapper (channels, dies, planes, blocks, pages)
    - Telemetry: Real-time request and channel distribution tracking
    - Latency: Pure analytical calculation — zero simulated sleep injection
    - Classification: ANALYTICAL
    """

    CLASSIFICATION: str = "ANALYTICAL"

    def __init__(
        self,
        channels: int = 8,
        dies_per_channel: int = 4,
        planes_per_die: int = 2,
        pages_per_block: int = 256,
        blocks_per_plane: int = 1024,
        num_layers: int = 24,
        num_heads: int = 2,
        tokens_per_block: int = 16,
        head_dim: int = 64,
        dtype: str = "float32",
        mapping_mode: str = "tensor_aware",
        # Analytical NAND timing parameters (MLC baseline):
        t_r_us: float = 35.0,
        t_prog_us: float = 350.0,
        t_xfer_us: float = 25.0,
    ):
        self.channels = channels
        self.dies_per_channel = dies_per_channel
        self.planes_per_die = planes_per_die
        self.pages_per_block = pages_per_block
        self.blocks_per_plane = blocks_per_plane
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.tokens_per_block = tokens_per_block
        self.head_dim = head_dim
        self.dtype = dtype
        self.mapping_mode = mapping_mode

        # Timing parameters for analytical metrics
        self.t_r_us = t_r_us
        self.t_prog_us = t_prog_us
        self.t_xfer_us = t_xfer_us

        # Mapper instance
        self.mapper = DeterministicTensorMapper(
            channels=self.channels,
            dies_per_channel=self.dies_per_channel,
            planes_per_die=self.planes_per_die,
            pages_per_block=self.pages_per_block,
            blocks_per_plane=self.blocks_per_plane,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            sector_size_bytes=4096,
        )

        # In-memory storage table: indexed by (layer_id, block_id) -> payload
        self._storage: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Telemetry counters
        self._channel_counters: Dict[int, int] = {c: 0 for c in range(self.channels)}
        self._per_channel_reads: Dict[int, int] = {c: 0 for c in range(self.channels)}
        self._per_channel_writes: Dict[int, int] = {c: 0 for c in range(self.channels)}
        self._per_channel_read_bytes: Dict[int, int] = {c: 0 for c in range(self.channels)}
        self._per_channel_write_bytes: Dict[int, int] = {c: 0 for c in range(self.channels)}

        self.total_read_requests: int = 0
        self.total_write_requests: int = 0
        self.total_read_bytes: int = 0
        self.total_write_bytes: int = 0

        self.key_page_read_requests: int = 0
        self.value_page_read_requests: int = 0
        self.combined_block_read_requests: int = 0

        self._access_log: List[Dict[str, Any]] = []

    def _validate_payload_size(self, data: Union[np.ndarray, bytes], expected_bytes: int, name: str) -> bytes:
        """Validates that tensor payload matches exactly the expected page size."""
        if isinstance(data, np.ndarray):
            raw = data.tobytes()
        elif isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        else:
            raise TypeError(f"{name} must be numpy.ndarray or bytes, got {type(data)}")

        if len(raw) != expected_bytes:
            raise ValueError(
                f"{name} payload byte mismatch: expected {expected_bytes} bytes, got {len(raw)} bytes."
            )
        return raw

    def store_kv(
        self,
        block_id: int,
        layer_id: int,
        key_data: Union[np.ndarray, bytes],
        value_data: Union[np.ndarray, bytes],
        metadata: Optional[Any] = None,
        head_id: int = 0,
        token_start: int = 0,
    ) -> bool:
        """
        Stores a KV block from inference host into the storage subsystem.
        
        Preserves:
            Key Page = exactly 4,096 B
            Value Page = exactly 4,096 B
            Combined = 8,192 B
        """
        k_bytes = self._validate_payload_size(key_data, KEY_PAGE_BYTES, "Key")
        v_bytes = self._validate_payload_size(value_data, VALUE_PAGE_BYTES, "Value")

        coord = TensorCoordinate(
            layer_id=layer_id,
            head_id=head_id,
            token_idx=token_start,
            tokens_per_block=self.tokens_per_block,
            block_id=block_id,
        )

        lba_addr = self.mapper.tensor_to_lba(coord, mode=self.mapping_mode)
        nand_coord = self.mapper.tensor_to_nand_physical(
            coord,
            mode=self.mapping_mode,
            channel_counters=self._channel_counters,
        )

        ch = nand_coord.channel
        self._channel_counters[ch] += 1
        self._per_channel_writes[ch] += 1
        self._per_channel_write_bytes[ch] += LOGICAL_BLOCK_BYTES

        self.total_write_requests += 1
        self.total_write_bytes += LOGICAL_BLOCK_BYTES

        stored_k = key_data if isinstance(key_data, np.ndarray) else np.frombuffer(k_bytes, dtype=np.uint8)
        stored_v = value_data if isinstance(value_data, np.ndarray) else np.frombuffer(v_bytes, dtype=np.uint8)

        self._storage[(layer_id, block_id)] = {
            "k": stored_k,
            "v": stored_v,
            "k_bytes": k_bytes,
            "v_bytes": v_bytes,
            "coord": coord,
            "lba": lba_addr.lba,
            "nand": nand_coord,
            "metadata": metadata,
            "timestamp": time.time(),
        }

        self._access_log.append({
            "op": "WRITE",
            "layer_id": layer_id,
            "head_id": head_id,
            "block_id": block_id,
            "lba": lba_addr.lba,
            "channel": ch,
            "die": nand_coord.die,
            "bytes": LOGICAL_BLOCK_BYTES,
        })

        return True

    def load_kv(
        self,
        block_id: int,
        layer_id: int,
        head_id: int = 0,
        token_start: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads both Key and Value pages (8,192 bytes total) for the specified block.
        """
        coord = TensorCoordinate(
            layer_id=layer_id,
            head_id=head_id,
            token_idx=token_start,
            tokens_per_block=self.tokens_per_block,
            block_id=block_id,
        )

        nand_coord = self.mapper.tensor_to_nand_physical(coord, mode=self.mapping_mode)
        ch = nand_coord.channel

        self._per_channel_reads[ch] += 1
        self._per_channel_read_bytes[ch] += LOGICAL_BLOCK_BYTES
        self.total_read_requests += 1
        self.total_read_bytes += LOGICAL_BLOCK_BYTES
        self.combined_block_read_requests += 1

        entry = self._storage.get((layer_id, block_id))
        if entry is None:
            k_ret = np.zeros(KEY_PAGE_BYTES // 4, dtype=np.float32)
            v_ret = np.zeros(VALUE_PAGE_BYTES // 4, dtype=np.float32)
        else:
            k_ret = entry["k"].copy()
            v_ret = entry["v"].copy()

        self._access_log.append({
            "op": "READ_KV",
            "layer_id": layer_id,
            "head_id": head_id,
            "block_id": block_id,
            "channel": ch,
            "bytes": LOGICAL_BLOCK_BYTES,
        })

        return k_ret, v_ret

    def load_key_page(
        self,
        block_id: int,
        layer_id: int,
        head_id: int = 0,
        token_start: int = 0,
    ) -> np.ndarray:
        """
        Reads ONLY the Key page (4,096 bytes).
        Models TOPK_FILTER streaming scans.
        """
        coord = TensorCoordinate(
            layer_id=layer_id,
            head_id=head_id,
            token_idx=token_start,
            tokens_per_block=self.tokens_per_block,
            block_id=block_id,
        )
        nand_coord = self.mapper.tensor_to_nand_physical(coord, mode=self.mapping_mode)
        ch = nand_coord.channel

        self._per_channel_reads[ch] += 1
        self._per_channel_read_bytes[ch] += KEY_PAGE_BYTES
        self.total_read_requests += 1
        self.total_read_bytes += KEY_PAGE_BYTES
        self.key_page_read_requests += 1

        entry = self._storage.get((layer_id, block_id))
        if entry is None:
            k_ret = np.zeros(KEY_PAGE_BYTES // 4, dtype=np.float32)
        else:
            k_ret = entry["k"].copy()

        self._access_log.append({
            "op": "TOPK_FILTER",
            "layer_id": layer_id,
            "head_id": head_id,
            "block_id": block_id,
            "channel": ch,
            "bytes": KEY_PAGE_BYTES,
        })

        return k_ret

    def load_value_page(
        self,
        block_id: int,
        layer_id: int,
        head_id: int = 0,
        token_start: int = 0,
    ) -> np.ndarray:
        """
        Reads ONLY the Value page (4,096 bytes).
        Models TOPK_FETCH selective gather.
        """
        coord = TensorCoordinate(
            layer_id=layer_id,
            head_id=head_id,
            token_idx=token_start,
            tokens_per_block=self.tokens_per_block,
            block_id=block_id,
        )
        nand_coord = self.mapper.tensor_to_nand_physical(coord, mode=self.mapping_mode)
        ch = nand_coord.channel

        self._per_channel_reads[ch] += 1
        self._per_channel_read_bytes[ch] += VALUE_PAGE_BYTES
        self.total_read_requests += 1
        self.total_read_bytes += VALUE_PAGE_BYTES
        self.value_page_read_requests += 1

        entry = self._storage.get((layer_id, block_id))
        if entry is None:
            v_ret = np.zeros(VALUE_PAGE_BYTES // 4, dtype=np.float32)
        else:
            v_ret = entry["v"].copy()

        self._access_log.append({
            "op": "TOPK_FETCH",
            "layer_id": layer_id,
            "head_id": head_id,
            "block_id": block_id,
            "channel": ch,
            "bytes": VALUE_PAGE_BYTES,
        })

        return v_ret

    def evict_kv(self, block_id: int, layer_id: int) -> bool:
        """Erases/invalidates a stored KV block."""
        if (layer_id, block_id) in self._storage:
            del self._storage[(layer_id, block_id)]
            return True
        return False

    def contains_block(self, block_id: int, layer_id: int) -> bool:
        """Returns True if block is resident in storage."""
        return (layer_id, block_id) in self._storage

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Returns full hardware, channel, and interface performance counters.
        Reports backend classification as ANALYTICAL.
        """
        total_requests = self.total_read_requests + self.total_write_requests
        total_bytes = self.total_read_bytes + self.total_write_bytes

        per_channel_total_reqs = {
            c: self._per_channel_reads[c] + self._per_channel_writes[c]
            for c in range(self.channels)
        }
        per_channel_total_bytes = {
            c: self._per_channel_read_bytes[c] + self._per_channel_write_bytes[c]
            for c in range(self.channels)
        }

        channel_loads = list(per_channel_total_reqs.values())
        max_load = max(channel_loads) if channel_loads else 0
        min_load = min(channel_loads) if channel_loads else 0
        mean_load = (sum(channel_loads) / self.channels) if self.channels > 0 else 0.0

        imbalance_pct = ((max_load - mean_load) / mean_load * 100.0) if mean_load > 0 else 0.0
        contention_ratio = (max_load / (total_requests / self.channels)) if total_requests > 0 else 1.0

        max_ch_time_us = 0.0
        for c in range(self.channels):
            r_count = self._per_channel_reads[c]
            w_count = self._per_channel_writes[c]
            ch_latency = (r_count * (self.t_r_us + self.t_xfer_us)) + (w_count * (self.t_prog_us + self.t_xfer_us))
            if ch_latency > max_ch_time_us:
                max_ch_time_us = ch_latency

        analytical_service_time_ms = max_ch_time_us / 1000.0

        return {
            "backend_classification": self.CLASSIFICATION,
            "architecture": {
                "channels": self.channels,
                "dies_per_channel": self.dies_per_channel,
                "planes_per_die": self.planes_per_die,
                "key_page_bytes": KEY_PAGE_BYTES,
                "value_page_bytes": VALUE_PAGE_BYTES,
                "logical_block_bytes": LOGICAL_BLOCK_BYTES,
                "mapping_mode": self.mapping_mode,
            },
            "requests": {
                "total": total_requests,
                "reads": self.total_read_requests,
                "writes": self.total_write_requests,
                "read_key_pages": self.key_page_read_requests,
                "read_value_pages": self.value_page_read_requests,
                "read_combined_blocks": self.combined_block_read_requests,
            },
            "bytes": {
                "total": total_bytes,
                "reads": self.total_read_bytes,
                "writes": self.total_write_bytes,
                "key_page_bytes": KEY_PAGE_BYTES,
                "value_page_bytes": VALUE_PAGE_BYTES,
            },
            "channel_distribution": {
                "per_channel_reads": self._per_channel_reads,
                "per_channel_writes": self._per_channel_writes,
                "per_channel_total_requests": per_channel_total_reqs,
                "per_channel_total_bytes": per_channel_total_bytes,
                "max_channel_load": max_load,
                "min_channel_load": min_load,
                "mean_channel_load": round(mean_load, 2),
                "load_imbalance_percent": round(imbalance_pct, 2),
                "contention_ratio": round(contention_ratio, 2),
            },
            "simulated_metrics": {
                "analytical_service_time_ms": round(analytical_service_time_ms, 2),
                "sleep_latency_injected": False,
            },
            "stored_blocks_count": len(self._storage),
        }

    def reset_telemetry(self) -> None:
        """Resets all metrics counters to zero."""
        self._channel_counters = {c: 0 for c in range(self.channels)}
        self._per_channel_reads = {c: 0 for c in range(self.channels)}
        self._per_channel_writes = {c: 0 for c in range(self.channels)}
        self._per_channel_read_bytes = {c: 0 for c in range(self.channels)}
        self._per_channel_write_bytes = {c: 0 for c in range(self.channels)}
        self.total_read_requests = 0
        self.total_write_requests = 0
        self.total_read_bytes = 0
        self.total_write_bytes = 0
        self.key_page_read_requests = 0
        self.value_page_read_requests = 0
        self.combined_block_read_requests = 0
        self._access_log.clear()

    def get_trace_log(self) -> List[Dict[str, Any]]:
        """Returns chronological access log for workload and FTL analysis."""
        return list(self._access_log)
