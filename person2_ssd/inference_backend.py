"""
Real Inference Storage Backend Adapter for AI-SSD V2.

Exposes Person 2's DeterministicTensorMapper and multi-channel FTL subsystem
through a high-performance, Python-callable storage engine designed for direct
consumption by Person 1 during live LLM inference.

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

# Physical Flash Page & Canonical KV Block Geometry (Codified Contract)
# 16 tokens x 1 KV head x 64 dimensions x 4 bytes (FP32)
KEY_PAGE_BYTES: int = 4096        # 4 KiB Key Page
VALUE_PAGE_BYTES: int = 4096      # 4 KiB Value Page
LOGICAL_BLOCK_BYTES: int = 8192   # 8 KiB Combined K+V Logical Block


class RealInferenceStorageBackend:
    """
    Python-callable storage backend for real LLM inference callers.
    
    Adheres strictly to the AI-SSD V2 Contract:
    - Canonical Geometry:
        * 16 tokens x 1 head x 64 dim x FP32 = 4,096 bytes per page
        * K page size: 4,096 bytes (4 KiB)
        * V page size: 4,096 bytes (4 KiB)
        * Combined logical block size: 8,192 bytes (8 KiB)
      Also transparently supports grouped 2-head blocks (16, 2, 64) for Qwen2.5-0.5B (8,192 B per tensor).
    - Mapping: DeterministicTensorMapper (channels, dies, planes, blocks, pages)
    - Telemetry: Real-time request and per-channel distribution tracking
    - Latency: Pure analytical calculation — ZERO simulated sleep latency
    - Classification: ANALYTICAL
    
    Drop-in compatible with P1's AISSDBlockStorageBackend interface.
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

        # In-memory storage table: indexed by (layer_id, block_id) -> payload dict
        self._storage: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Telemetry counters
        self._channel_counters: Dict[int, int] = {c: 0 for c in range(self.channels)}
        self._per_channel_reads: Dict[int, int] = {c: 0 for c in range(self.channels)}
        self._per_channel_writes: Dict[int, int] = {c: 0 for c in range(self.channels)}
        self._per_channel_read_bytes: Dict[int, int] = {c: 0 for c in range(self.channels)}
        self._per_channel_write_bytes: Dict[int, int] = {c: 0 for c in range(self.channels)}

        # Cumulative counters (P1 compatibility properties)
        self.bytes_read: int = 0
        self.bytes_written: int = 0
        self.blocks_read: int = 0
        self.blocks_written: int = 0
        self.requests: int = 0

        # Sub-page tracking
        self.key_page_read_requests: int = 0
        self.value_page_read_requests: int = 0
        self.combined_block_read_requests: int = 0

        self._access_log: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Core Write Interface (Supports both P1 write_block and P2 store_kv)
    # -------------------------------------------------------------------------

    def write_block(
        self,
        layer_idx: int,
        block_id: int,
        k_block: Union[np.ndarray, bytes],
        v_block: Union[np.ndarray, bytes],
        head_id: int = 0,
        token_start: int = 0,
    ) -> None:
        """
        Writes a KV block into storage.
        
        Args:
            layer_idx: Transformer layer index (0..num_layers-1)
            block_id: Unique block index within layer
            k_block: Key tensor array or byte buffer
            v_block: Value tensor array or byte buffer
            head_id: Head index (default 0)
            token_start: Token offset in sequence (default 0)
        """
        # Convert / validate arrays
        if isinstance(k_block, np.ndarray):
            k_arr = np.ascontiguousarray(k_block)
            k_bytes = k_arr.tobytes()
        else:
            k_bytes = bytes(k_block)
            k_arr = np.frombuffer(k_bytes, dtype=np.float32)

        if isinstance(v_block, np.ndarray):
            v_arr = np.ascontiguousarray(v_block)
            v_bytes = v_arr.tobytes()
        else:
            v_bytes = bytes(v_block)
            v_arr = np.frombuffer(v_bytes, dtype=np.float32)

        k_size = len(k_bytes)
        v_size = len(v_bytes)
        total_block_bytes = k_size + v_size

        # Determine number of heads represented in this block
        # For canonical (16, 1, 64) -> 1 head. For Qwen grouped (16, 2, 64) -> 2 heads.
        heads_in_block = 1
        if isinstance(k_block, np.ndarray) and k_block.ndim == 3:
            heads_in_block = k_block.shape[1]

        # Map through DeterministicTensorMapper across channels
        assigned_channels = []
        for h_off in range(heads_in_block):
            h_eff = head_id + h_off
            coord = TensorCoordinate(
                layer_id=layer_idx,
                head_id=h_eff,
                token_idx=token_start,
                tokens_per_block=self.tokens_per_block,
                block_id=block_id,
            )
            nand_coord = self.mapper.tensor_to_nand_physical(
                coord,
                mode=self.mapping_mode,
                channel_counters=self._channel_counters,
            )
            ch = nand_coord.channel
            assigned_channels.append(ch)
            self._channel_counters[ch] += 1
            self._per_channel_writes[ch] += 1
            self._per_channel_write_bytes[ch] += total_block_bytes // heads_in_block

        primary_ch = assigned_channels[0] if assigned_channels else 0
        lba_addr = self.mapper.tensor_to_lba(
            TensorCoordinate(layer_id=layer_idx, head_id=head_id, token_idx=token_start, block_id=block_id),
            mode=self.mapping_mode,
        )

        # Store in table
        self._storage[(layer_idx, block_id)] = {
            "k": k_arr.copy(),
            "v": v_arr.copy(),
            "k_shape": k_arr.shape if isinstance(k_arr, np.ndarray) else None,
            "v_shape": v_arr.shape if isinstance(v_arr, np.ndarray) else None,
            "k_bytes": k_bytes,
            "v_bytes": v_bytes,
            "head_id": head_id,
            "token_start": token_start,
            "channel": primary_ch,
            "channels": assigned_channels,
            "lba": lba_addr.lba,
            "timestamp": time.time(),
        }

        # Update telemetry
        self.blocks_written += 1
        self.bytes_written += total_block_bytes
        self.requests += 1

        self._access_log.append({
            "op": "WRITE",
            "layer_id": layer_idx,
            "head_id": head_id,
            "block_id": block_id,
            "channel": primary_ch,
            "bytes": total_block_bytes,
        })

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
        """Alias for write_block to satisfy KVStorageInterface specification."""
        self.write_block(
            layer_idx=layer_id,
            block_id=block_id,
            k_block=key_data,
            v_block=value_data,
            head_id=head_id,
            token_start=token_start,
        )
        return True

    # -------------------------------------------------------------------------
    # Core Read Interface (Supports P1 read_key_page, read_value_page, read_block)
    # -------------------------------------------------------------------------

    def read_key_page(
        self,
        layer_idx: int,
        block_id: int,
        head_id: int = 0,
        token_start: int = 0,
    ) -> np.ndarray:
        """
        Reads the Key page for in-storage filtering (TOPK_FILTER).
        Returns actual stored numpy array.
        """
        coord = TensorCoordinate(
            layer_id=layer_idx,
            head_id=head_id,
            token_idx=token_start,
            tokens_per_block=self.tokens_per_block,
            block_id=block_id,
        )
        nand_coord = self.mapper.tensor_to_nand_physical(coord, mode=self.mapping_mode)
        ch = nand_coord.channel

        entry = self._storage.get((layer_idx, block_id))
        if entry is None:
            # Clean zero fallback matching canonical geometry
            k_ret = np.zeros((self.tokens_per_block, self.num_heads, self.head_dim), dtype=np.float32)
            k_bytes_count = k_ret.nbytes
        else:
            k_ret = entry["k"].copy()
            k_bytes_count = len(entry["k_bytes"])

        # Update telemetry
        self._per_channel_reads[ch] += 1
        self._per_channel_read_bytes[ch] += k_bytes_count
        self.requests += 1
        self.bytes_read += k_bytes_count
        self.key_page_read_requests += 1

        self._access_log.append({
            "op": "READ_KEY",
            "layer_id": layer_idx,
            "head_id": head_id,
            "block_id": block_id,
            "channel": ch,
            "bytes": k_bytes_count,
        })

        return k_ret

    def load_key_page(self, block_id: int, layer_id: int, head_id: int = 0, token_start: int = 0) -> np.ndarray:
        """Alias with (block_id, layer_id) argument order."""
        return self.read_key_page(layer_idx=layer_id, block_id=block_id, head_id=head_id, token_start=token_start)

    def read_value_page(
        self,
        layer_idx: int,
        block_id: int,
        head_id: int = 0,
        token_start: int = 0,
    ) -> np.ndarray:
        """
        Reads the Value page over the PCIe interface (TOPK_FETCH).
        Returns actual stored numpy array.
        """
        coord = TensorCoordinate(
            layer_id=layer_idx,
            head_id=head_id,
            token_idx=token_start,
            tokens_per_block=self.tokens_per_block,
            block_id=block_id,
        )
        nand_coord = self.mapper.tensor_to_nand_physical(coord, mode=self.mapping_mode)
        ch = nand_coord.channel

        entry = self._storage.get((layer_idx, block_id))
        if entry is None:
            v_ret = np.zeros((self.tokens_per_block, self.num_heads, self.head_dim), dtype=np.float32)
            v_bytes_count = v_ret.nbytes
        else:
            v_ret = entry["v"].copy()
            v_bytes_count = len(entry["v_bytes"])

        # Update telemetry
        self._per_channel_reads[ch] += 1
        self._per_channel_read_bytes[ch] += v_bytes_count
        self.requests += 1
        self.blocks_read += 1
        self.bytes_read += v_bytes_count
        self.value_page_read_requests += 1

        self._access_log.append({
            "op": "READ_VALUE",
            "layer_id": layer_idx,
            "head_id": head_id,
            "block_id": block_id,
            "channel": ch,
            "bytes": v_bytes_count,
        })

        return v_ret

    def load_value_page(self, block_id: int, layer_id: int, head_id: int = 0, token_start: int = 0) -> np.ndarray:
        """Alias with (block_id, layer_id) argument order."""
        return self.read_value_page(layer_idx=layer_id, block_id=block_id, head_id=head_id, token_start=token_start)

    def read_block(
        self,
        layer_idx: int,
        block_id: int,
        head_id: int = 0,
        token_start: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads both Key and Value pages for the block.
        Returns (key_tensor, value_tensor).
        """
        coord = TensorCoordinate(
            layer_id=layer_idx,
            head_id=head_id,
            token_idx=token_start,
            tokens_per_block=self.tokens_per_block,
            block_id=block_id,
        )
        nand_coord = self.mapper.tensor_to_nand_physical(coord, mode=self.mapping_mode)
        ch = nand_coord.channel

        entry = self._storage.get((layer_idx, block_id))
        if entry is None:
            k_ret = np.zeros((self.tokens_per_block, self.num_heads, self.head_dim), dtype=np.float32)
            v_ret = np.zeros((self.tokens_per_block, self.num_heads, self.head_dim), dtype=np.float32)
            total_bytes = k_ret.nbytes + v_ret.nbytes
        else:
            k_ret = entry["k"].copy()
            v_ret = entry["v"].copy()
            total_bytes = len(entry["k_bytes"]) + len(entry["v_bytes"])

        # Update telemetry
        self._per_channel_reads[ch] += 1
        self._per_channel_read_bytes[ch] += total_bytes
        self.requests += 1
        self.blocks_read += 1
        self.bytes_read += total_bytes
        self.combined_block_read_requests += 1

        self._access_log.append({
            "op": "READ_BLOCK",
            "layer_id": layer_idx,
            "head_id": head_id,
            "block_id": block_id,
            "channel": ch,
            "bytes": total_bytes,
        })

        return k_ret, v_ret

    def load_kv(self, block_id: int, layer_id: int, head_id: int = 0, token_start: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Alias for read_block with (block_id, layer_id) argument order."""
        return self.read_block(layer_idx=layer_id, block_id=block_id, head_id=head_id, token_start=token_start)

    def evict_block(self, layer_idx: int, block_id: int) -> bool:
        """Erases/invalidates a stored KV block."""
        if (layer_idx, block_id) in self._storage:
            del self._storage[(layer_idx, block_id)]
            return True
        return False

    def evict_kv(self, block_id: int, layer_id: int) -> bool:
        """Alias with (block_id, layer_id) argument order."""
        return self.evict_block(layer_idx=layer_id, block_id=block_id)

    def contains_block(self, layer_idx: int, block_id: int) -> bool:
        """Checks if block exists in storage."""
        return (layer_idx, block_id) in self._storage

    # -------------------------------------------------------------------------
    # Telemetry & Performance Counters
    # -------------------------------------------------------------------------

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Returns full hardware, channel, and interface performance counters.
        Reports backend classification as ANALYTICAL.
        """
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
        contention_ratio = (max_load / (self.requests / self.channels)) if self.requests > 0 else 1.0

        # Mathematical MLC NAND timing model (QD=8 concurrency)
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
                "total": self.requests,
                "reads": self.blocks_read + self.key_page_read_requests,
                "writes": self.blocks_written,
                "read_key_pages": self.key_page_read_requests,
                "read_value_pages": self.value_page_read_requests,
                "read_combined_blocks": self.combined_block_read_requests,
            },
            "bytes": {
                "total": self.bytes_read + self.bytes_written,
                "reads": self.bytes_read,
                "writes": self.bytes_written,
                "canonical_key_page_bytes": KEY_PAGE_BYTES,
                "canonical_value_page_bytes": VALUE_PAGE_BYTES,
            },
            "channel_distribution": {
                "channel_read_counts": self._per_channel_reads,
                "channel_write_counts": self._per_channel_writes,
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

    def reset_stats(self) -> None:
        """Resets all metrics counters to zero (P1 compatible)."""
        self._channel_counters = {c: 0 for c in range(self.channels)}
        self._per_channel_reads = {c: 0 for c in range(self.channels)}
        self._per_channel_writes = {c: 0 for c in range(self.channels)}
        self._per_channel_read_bytes = {c: 0 for c in range(self.channels)}
        self._per_channel_write_bytes = {c: 0 for c in range(self.channels)}
        self.bytes_read = 0
        self.bytes_written = 0
        self.blocks_read = 0
        self.blocks_written = 0
        self.requests = 0
        self.key_page_read_requests = 0
        self.value_page_read_requests = 0
        self.combined_block_read_requests = 0
        self._access_log.clear()

    def reset_telemetry(self) -> None:
        """Alias for reset_stats."""
        self.reset_stats()

    def get_trace_log(self) -> List[Dict[str, Any]]:
        """Returns chronological access log for workload and FTL analysis."""
        return list(self._access_log)
