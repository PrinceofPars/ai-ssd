"""
Physical Tensor-Aware Storage Adapter: Bridges Person 1 KV Engine with Person 2 Physical Flash Model.

Implements Person 1's KVStorageInterface contract while executing block placement,
multi-channel striping, and contention latency simulation via Person 2's StorageSimulator.
"""

from typing import Tuple, List, Dict, Any, Optional, Callable, Set
import time
import numpy as np

from person1_kv_engine.storage_backend.kv_storage_api import KVStorageInterface, KVBlockMetadata
from person1_kv_engine.storage_backend.flash_model import FlashModel, FlashTimingConfig, EnergyConfig
from person2_ssd.storage_model.io_model import StorageSimulator
from common.schemas.kv_block import KVBlock as CommonKVBlock, StorageTier
from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    SSD_PAGES_PER_BLOCK,
    BUS_TRANSFER_US_PER_PAGE,
    PCIE_OVERHEAD_US,
    T_R_US,
)


def _encode_global_block_id(layer_id: int, block_id: int) -> int:
    """Encodes (layer_id, block_id) into a unique 32-bit integer for Person 2 FTL."""
    return (layer_id << 16) | (block_id & 0xFFFF)


class PhysicalTensorAwareStorageAdapter(KVStorageInterface):
    """
    Adapter implementing KVStorageInterface backed by Person 2's physical StorageSimulator.
    Provides tensor-aware channel striping, multi-channel contention physics, and in-storage Top-k.
    """

    def __init__(
        self,
        mode: str = "tensor_aware",
        channels: int = SSD_CHANNELS,
        dies_per_channel: int = SSD_DIES_PER_CHANNEL,
        planes_per_die: int = SSD_PLANES_PER_DIE,
        blocks_per_plane: int = 64,
        controller_dram_size_mb: int = 1024,
        flash_model: Optional[FlashModel] = None,
        c_kernel_fn: Optional[Callable] = None,
    ):
        self.mode = mode
        self.channels = channels
        self.controller_dram_capacity_bytes = controller_dram_size_mb * 1024 * 1024
        self.flash_model = flash_model or FlashModel(
            timing_config=FlashTimingConfig(
                t_read_us=T_R_US,
                t_prog_us=200.0,
                internal_channels=channels,
            )
        )

        # Person 2 Physical Storage Simulator
        self.storage_simulator = StorageSimulator(
            mode=mode,
            channels=channels,
            dies_per_channel=dies_per_channel,
            planes_per_die=planes_per_die,
            blocks_per_plane=blocks_per_plane,
        )

        # C-Kernel for In-Storage Attention
        if c_kernel_fn is None:
            try:
                from person1_kv_engine.c_kernel.kernel_binding import get_native_c_kernel
                native_k = get_native_c_kernel()
                if native_k.is_available():
                    c_kernel_fn = native_k.compute_topk
            except Exception:
                c_kernel_fn = None
        self.c_kernel_fn = c_kernel_fn

        # Internal tensor payload store: (layer_id, block_id) -> {"k": np.ndarray, "v": np.ndarray, "meta": KVBlockMetadata}
        self._flash_store: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Controller DRAM cache: set of (layer_id, block_id)
        self._controller_dram_cache: Set[Tuple[int, int]] = set()
        self._controller_dram_used_bytes: int = 0

        # Physical block mapping tracking: (layer_id, block_id) -> global_id
        self._global_id_map: Dict[Tuple[int, int], int] = {}

        # Telemetry & Performance Counters
        self._host_to_device_bytes: int = 0
        self._device_to_host_bytes: int = 0
        self._flash_read_bytes: int = 0
        self._flash_write_bytes: int = 0
        self._controller_dram_read_bytes: int = 0
        self._controller_dram_write_bytes: int = 0
        self._compute_mac_count: int = 0
        self._nand_read_ops: int = 0
        self._nand_write_ops: int = 0
        self._simulated_time_us: float = 0.0

        # Trace Log & Listeners (for Person 3 Prefetcher)
        self._trace_log: List[Dict[str, Any]] = []
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []

    def set_mode(self, mode: str) -> None:
        """Switch FTL mode dynamically between 'tensor_aware' and 'conventional'."""
        self.mode = mode
        self.storage_simulator = StorageSimulator(mode=mode, channels=self.channels)
        # Re-register all stored blocks into the new FTL simulator
        for (lid, bid), entry in self._flash_store.items():
            meta = entry["meta"]
            gid = self._global_id_map[(lid, bid)]
            p2_block = CommonKVBlock(
                block_id=gid,
                layer_id=lid,
                token_start=meta.token_start,
                token_count=meta.token_count,
                kv_head_start=0,
                kv_head_count=meta.num_heads,
                head_dim=meta.head_dim,
                dtype="FP16" if meta.dtype_bytes == 2 else "FP32",
                key_size_bytes=entry["k"].nbytes,
                value_size_bytes=entry["v"].nbytes,
                storage_tier=StorageTier.SSD.value,
            )
            self.storage_simulator.store_block(p2_block)

    def store_kv(
        self,
        block_id: int,
        layer_id: int,
        key_data: np.ndarray,
        value_data: np.ndarray,
        metadata: Optional[KVBlockMetadata] = None,
    ) -> bool:
        """Writes a KV block from host to physical SSD storage."""
        k_bytes = key_data.nbytes
        v_bytes = value_data.nbytes
        payload_bytes = k_bytes + v_bytes

        # 1. PCIe Transfer Host -> Controller DRAM
        self._host_to_device_bytes += payload_bytes
        pcie_us = self.flash_model.calculate_pcie_transfer_time_us(payload_bytes)

        # 2. Map block into Person 2's Physical Storage Simulator
        gid = _encode_global_block_id(layer_id, block_id)
        self._global_id_map[(layer_id, block_id)] = gid

        token_cnt = key_data.shape[0] if key_data.ndim >= 3 else 16
        num_h = key_data.shape[1] if key_data.ndim >= 3 else 1
        h_dim = key_data.shape[2] if key_data.ndim >= 3 else 128
        tok_start = metadata.token_start if metadata else (block_id * token_cnt)

        p2_block = CommonKVBlock(
            block_id=gid,
            layer_id=layer_id,
            token_start=tok_start,
            token_count=token_cnt,
            kv_head_start=0,
            kv_head_count=num_h,
            head_dim=h_dim,
            dtype="FP16" if key_data.itemsize == 2 else "FP32",
            key_size_bytes=k_bytes,
            value_size_bytes=v_bytes,
            storage_tier=StorageTier.SSD.value,
        )
        self.storage_simulator.store_block(p2_block)

        # 3. Physical Flash Programming Latency
        self._flash_write_bytes += payload_bytes
        self._nand_write_ops += 1
        flash_write_us = self.flash_model.calculate_flash_write_time_us(payload_bytes)
        self._simulated_time_us += (pcie_us + flash_write_us)

        # 4. Save metadata and payload
        if metadata is None:
            metadata = KVBlockMetadata(
                block_id=block_id,
                layer_id=layer_id,
                token_start=tok_start,
                token_count=token_cnt,
                num_heads=num_h,
                head_dim=h_dim,
                dtype_str=str(key_data.dtype),
                dtype_bytes=key_data.itemsize,
            )

        self._flash_store[(layer_id, block_id)] = {
            "k": np.ascontiguousarray(key_data),
            "v": np.ascontiguousarray(value_data),
            "meta": metadata,
            "global_id": gid,
        }
        return True

    def load_kv(
        self,
        block_id: int,
        layer_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reads a KV block from storage into host memory using physical FTL timing."""
        key = (layer_id, block_id)
        if key not in self._flash_store:
            raise KeyError(f"Block {block_id} for layer {layer_id} not found in physical storage")

        entry = self._flash_store[key]
        payload_bytes = entry["k"].nbytes + entry["v"].nbytes

        # Physical read latency via Person 2 StorageSimulator
        gid = entry["global_id"]
        if key not in self._controller_dram_cache:
            read_lat_us = self.storage_simulator.estimate_read_latency([gid])
            self._flash_read_bytes += payload_bytes
            self._nand_read_ops += 1
            self._simulated_time_us += read_lat_us
        else:
            self._controller_dram_read_bytes += payload_bytes
            self._simulated_time_us += 1.0  # 1 us DRAM hit

        # PCIe Transfer SSD -> Host
        self._device_to_host_bytes += payload_bytes
        pcie_us = self.flash_model.calculate_pcie_transfer_time_us(payload_bytes)
        self._simulated_time_us += pcie_us

        # Record access event
        self._record_event("LOAD_KV", layer_id, [block_id], payload_bytes)
        return entry["k"], entry["v"]

    def evict_kv(self, block_id: int, layer_id: int) -> bool:
        """Erases/invalidates a KV block in storage."""
        key = (layer_id, block_id)
        if key in self._flash_store:
            del self._flash_store[key]
            self._controller_dram_cache.discard(key)
            self._global_id_map.pop(key, None)
            return True
        return False

    def prefetch_kv(self, block_ids: List[int], layer_id: int) -> int:
        """Prefetches blocks from flash into SSD controller DRAM cache."""
        staged = 0
        gids_to_read = []

        for bid in block_ids:
            key = (layer_id, bid)
            if key in self._flash_store and key not in self._controller_dram_cache:
                entry = self._flash_store[key]
                payload_bytes = entry["k"].nbytes + entry["v"].nbytes
                if self._controller_dram_used_bytes + payload_bytes <= self.controller_dram_capacity_bytes:
                    self._controller_dram_cache.add(key)
                    self._controller_dram_used_bytes += payload_bytes
                    self._flash_read_bytes += payload_bytes
                    self._nand_read_ops += 1
                    gids_to_read.append(entry["global_id"])
                    staged += 1

        if gids_to_read:
            # Physical multi-channel read latency to load into controller DRAM
            lat_us = self.storage_simulator.estimate_read_latency(gids_to_read)
            self._simulated_time_us += lat_us

        self._record_event("PREFETCH_KV", layer_id, block_ids, staged * 4096)
        return staged

    def in_storage_topk_attention(
        self,
        query: np.ndarray,
        layer_id: int,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes computational attention filtering directly inside the SSD controller.
        Host sends ONLY query Q over PCIe. Controller scores cold blocks, reads Top-k Values
        through Person 2's physical multi-channel NAND bus, and streams back only Top-k.
        """
        layer_blocks = [(bid, entry) for (lid, bid), entry in self._flash_store.items() if lid == layer_id]
        if not layer_blocks:
            empty_idx = np.empty((0,), dtype=np.int32)
            empty_val = np.empty((0, 0, 0, 0), dtype=query.dtype)
            empty_log = np.empty((query.shape[0], 0), dtype=np.float32)
            empty_sc = np.empty((0,), dtype=np.float32)
            return empty_idx, empty_val, empty_log, empty_sc

        layer_blocks.sort(key=lambda x: x[0])
        num_blocks = len(layer_blocks)
        effective_k = min(top_k, num_blocks)

        # 1. PCIe: Host transmits Query vector Q to SSD controller
        query_bytes = query.nbytes
        self._host_to_device_bytes += query_bytes
        self._simulated_time_us += self.flash_model.calculate_pcie_transfer_time_us(query_bytes)

        # 2. Key Tensors Physical Read / Cache Check
        uncached_gids = []
        for bid, entry in layer_blocks:
            key = (layer_id, bid)
            k_bytes = entry["k"].nbytes
            if key not in self._controller_dram_cache:
                self._flash_read_bytes += k_bytes
                self._nand_read_ops += 1
                uncached_gids.append(entry["global_id"])
            else:
                self._controller_dram_read_bytes += k_bytes

        if uncached_gids:
            # Person 2 Multi-Channel Physical Read Latency for Keys
            key_lat_us = self.storage_simulator.estimate_read_latency(uncached_gids)
            self._simulated_time_us += key_lat_us

        # Compute MACs inside controller
        first_entry = layer_blocks[0][1]
        tokens_per_block = first_entry["k"].shape[0]
        num_heads = query.shape[0]
        head_dim = query.shape[1]
        macs = num_blocks * tokens_per_block * num_heads * head_dim
        self._compute_mac_count += macs
        # Embedded controller DSP latency @ 1.2 GHz (16 MACs/cycle = 19.2 GMAC/s)
        self._simulated_time_us += (macs / (19.2 * 1e3))

        # 3. Dot-Product Scoring & Top-k Selection
        if self.c_kernel_fn is not None:
            topk_block_ids, topk_values, topk_scores = self.c_kernel_fn(
                query=query,
                layer_blocks=layer_blocks,
                top_k=effective_k,
            )
        else:
            block_scores = []
            block_ids = []
            scale = 1.0 / np.sqrt(head_dim, dtype=np.float32)

            for bid, entry in layer_blocks:
                k_block = entry["k"]
                dots = np.einsum("hd,thd->th", query, k_block) * scale
                max_score = float(np.max(dots))
                block_scores.append(max_score)
                block_ids.append(bid)

            block_scores = np.array(block_scores, dtype=np.float32)
            block_ids = np.array(block_ids, dtype=np.int32)

            if effective_k < len(block_scores):
                top_indices = np.argpartition(block_scores, -effective_k)[-effective_k:]
                top_indices = top_indices[np.argsort(-block_scores[top_indices])]
            else:
                top_indices = np.argsort(-block_scores)

            topk_block_ids = block_ids[top_indices]
            topk_scores = block_scores[top_indices]
            topk_values = np.stack([layer_blocks[idx][1]["v"] for idx in top_indices], axis=0)

        # 4. Physical Read of Top-k Value Tensors via Person 2 StorageSimulator
        val_gids = []
        for bid in topk_block_ids:
            key = (layer_id, int(bid))
            v_bytes = self._flash_store[key]["v"].nbytes
            if key not in self._controller_dram_cache:
                self._flash_read_bytes += v_bytes
                self._nand_read_ops += 1
                val_gids.append(self._global_id_map[key])
            else:
                self._controller_dram_read_bytes += v_bytes

        if val_gids:
            val_lat_us = self.storage_simulator.estimate_read_latency(val_gids)
            self._simulated_time_us += val_lat_us

        # 5. Compute per-token logits inside controller DRAM
        scale = 1.0 / np.sqrt(head_dim, dtype=np.float32)
        logits_list = []
        for bid in topk_block_ids:
            k_b = self._flash_store[(layer_id, int(bid))]["k"]
            dots = np.einsum("hd,thd->ht", query, k_b) * scale
            logits_list.append(dots)

        if logits_list:
            topk_logits = np.concatenate(logits_list, axis=-1)
        else:
            topk_logits = np.empty((num_heads, 0), dtype=np.float32)

        # 6. PCIe Transfer SSD -> Host (Top-k Values, Logits, Scores, IDs only)
        ret_bytes = topk_values.nbytes + topk_logits.nbytes + topk_scores.nbytes + topk_block_ids.nbytes
        self._device_to_host_bytes += ret_bytes
        self._simulated_time_us += self.flash_model.calculate_pcie_transfer_time_us(ret_bytes)

        # Record access event & trigger listeners (e.g. for Person 3 prefetcher)
        self._record_event(
            op="IN_STORAGE_TOPK",
            layer_id=layer_id,
            block_ids=[int(b) for b in topk_block_ids],
            streamed_bytes=ret_bytes,
        )

        return topk_block_ids, topk_values, topk_logits, topk_scores

    def _record_event(self, op: str, layer_id: int, block_ids: List[int], streamed_bytes: int = 0) -> None:
        event = {
            "op": op,
            "layer_id": layer_id,
            "block_ids": block_ids,
            "streamed_bytes": streamed_bytes,
            "sim_time_us": self._simulated_time_us,
        }
        self._trace_log.append(event)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive physical and interface telemetry."""
        energy = self.flash_model.calculate_energy_joules(
            pcie_bytes=self._host_to_device_bytes + self._device_to_host_bytes,
            controller_dram_bytes=self._controller_dram_read_bytes + self._controller_dram_write_bytes,
            flash_read_bytes=self._flash_read_bytes,
            flash_write_bytes=self._flash_write_bytes,
            compute_macs=self._compute_mac_count,
        )
        return {
            "host_to_device_bytes": self._host_to_device_bytes,
            "device_to_host_bytes": self._device_to_host_bytes,
            "total_pcie_bytes": self._host_to_device_bytes + self._device_to_host_bytes,
            "flash_read_bytes": self._flash_read_bytes,
            "flash_write_bytes": self._flash_write_bytes,
            "controller_dram_read_bytes": self._controller_dram_read_bytes,
            "controller_dram_write_bytes": self._controller_dram_write_bytes,
            "compute_mac_count": self._compute_mac_count,
            "nand_read_ops": self._nand_read_ops,
            "nand_write_ops": self._nand_write_ops,
            "simulated_time_us": self._simulated_time_us,
            "simulated_time_ms": self._simulated_time_us / 1000.0,
            "energy_joules": energy,
            "mode": self.mode,
        }

    def reset_telemetry(self) -> None:
        """Resets all metrics counters to zero."""
        self._host_to_device_bytes = 0
        self._device_to_host_bytes = 0
        self._flash_read_bytes = 0
        self._flash_write_bytes = 0
        self._controller_dram_read_bytes = 0
        self._controller_dram_write_bytes = 0
        self._compute_mac_count = 0
        self._nand_read_ops = 0
        self._nand_write_ops = 0
        self._simulated_time_us = 0.0

    def register_access_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._listeners.append(callback)

    def get_trace_log(self) -> List[Dict[str, Any]]:
        return list(self._trace_log)
