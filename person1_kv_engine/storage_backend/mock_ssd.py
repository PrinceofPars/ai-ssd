"""High-Fidelity Simulated SSD Controller with Physical Timing and Telemetry.

Implements KVStorageInterface, modeling controller DRAM cache, multi-channel
NAND flash storage, DMA bus transactions, and in-storage computational offload.
"""

from typing import Tuple, List, Dict, Any, Optional, Callable
import time
import numpy as np

from person1_kv_engine.storage_backend.flash_model import FlashModel, FlashTimingConfig, EnergyConfig
from person1_kv_engine.storage_backend.kv_storage_api import KVStorageInterface, KVBlockMetadata


class MockSSDController(KVStorageInterface):
    """Simulates an AI-aware SSD with an embedded computational storage engine."""

    def __init__(
        self,
        controller_dram_size_mb: int = 1024,
        flash_model: Optional[FlashModel] = None,
        c_kernel_fn: Optional[Callable] = None,
    ):
        self.dram_capacity_bytes = controller_dram_size_mb * 1024 * 1024
        self.flash_model = flash_model or FlashModel()
        if c_kernel_fn is None:
            try:
                from person1_kv_engine.c_kernel.kernel_binding import get_native_c_kernel
                native_k = get_native_c_kernel()
                if native_k.is_available():
                    c_kernel_fn = native_k.compute_topk
            except Exception:
                c_kernel_fn = None
        self.c_kernel_fn = c_kernel_fn

        # Internal storage: (layer_id, block_id) -> {"k": np.ndarray, "v": np.ndarray, "meta": KVBlockMetadata}
        self._flash_store: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Controller DRAM cache: set of (layer_id, block_id)
        self._controller_dram_cache: set = set()
        self._controller_dram_used_bytes: int = 0

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

        # Access Trace & Listeners
        self._trace_log: List[Dict[str, Any]] = []
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []

    def set_c_kernel(self, c_kernel_fn: Callable) -> None:
        """Sets or updates the native C kernel function for in-storage filtering."""
        self.c_kernel_fn = c_kernel_fn

    def store_kv(
        self,
        block_id: int,
        layer_id: int,
        key_data: np.ndarray,
        value_data: np.ndarray,
        metadata: Optional[KVBlockMetadata] = None,
    ) -> bool:
        """Writes a KV block from host to storage."""
        k_bytes = key_data.nbytes
        v_bytes = value_data.nbytes
        payload_bytes = k_bytes + v_bytes

        # 1. PCIe Transfer Host -> Controller DRAM
        self._host_to_device_bytes += payload_bytes
        pcie_transfer_us = self.flash_model.calculate_pcie_transfer_time_us(payload_bytes)

        # 2. Controller writes to NAND Flash (background or direct)
        self._flash_write_bytes += payload_bytes
        self._nand_write_ops += 1
        flash_write_us = self.flash_model.calculate_flash_write_time_us(payload_bytes)

        # Simulated physical time elapsed
        self._simulated_time_us += (pcie_transfer_us + flash_write_us)

        # Update metadata
        if metadata is None:
            metadata = KVBlockMetadata(
                block_id=block_id,
                layer_id=layer_id,
                token_start=0,
                token_count=key_data.shape[0],
                num_heads=key_data.shape[1],
                head_dim=key_data.shape[2],
                dtype_str=str(key_data.dtype),
                dtype_bytes=key_data.itemsize,
            )

        # Store contiguous copy in simulated physical flash
        self._flash_store[(layer_id, block_id)] = {
            "k": np.ascontiguousarray(key_data),
            "v": np.ascontiguousarray(value_data),
            "meta": metadata,
        }

        # Log trace event
        event = {
            "op": "STORE",
            "layer_id": layer_id,
            "block_id": block_id,
            "bytes": payload_bytes,
            "sim_time_us": self._simulated_time_us,
        }
        self._trace_log.append(event)
        for listener in self._listeners:
            listener(event)

        return True

    def load_kv(
        self,
        block_id: int,
        layer_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reads a KV block from storage into host memory (conventional full offload)."""
        key = (layer_id, block_id)
        if key not in self._flash_store:
            raise KeyError(f"KV Block (layer={layer_id}, block={block_id}) not found in storage.")

        entry = self._flash_store[key]
        k_data = entry["k"]
        v_data = entry["v"]
        payload_bytes = k_data.nbytes + v_data.nbytes

        # 1. Read from NAND into Controller DRAM (if not cached in controller DRAM)
        if key not in self._controller_dram_cache:
            self._flash_read_bytes += payload_bytes
            self._nand_read_ops += 1
            flash_read_us = self.flash_model.calculate_flash_read_time_us(payload_bytes)
            self._simulated_time_us += flash_read_us
            self._controller_dram_write_bytes += payload_bytes
        else:
            # Controller DRAM hit
            self._controller_dram_read_bytes += payload_bytes

        # 2. PCIe Transfer from Controller to Host
        self._device_to_host_bytes += payload_bytes
        pcie_transfer_us = self.flash_model.calculate_pcie_transfer_time_us(payload_bytes)
        self._simulated_time_us += pcie_transfer_us

        entry["meta"].last_accessed_timestamp = time.time()
        entry["meta"].access_count += 1

        event = {
            "op": "LOAD",
            "layer_id": layer_id,
            "block_id": block_id,
            "bytes": payload_bytes,
            "sim_time_us": self._simulated_time_us,
        }
        self._trace_log.append(event)
        for listener in self._listeners:
            listener(event)

        return k_data.copy(), v_data.copy()

    def evict_kv(self, block_id: int, layer_id: int) -> bool:
        """Erases/invalidates a KV block."""
        key = (layer_id, block_id)
        if key in self._flash_store:
            del self._flash_store[key]
            self._controller_dram_cache.discard(key)
            return True
        return False

    def prefetch_kv(self, block_ids: List[int], layer_id: int) -> int:
        """Prefetches blocks from NAND into Controller DRAM cache."""
        staged = 0
        for bid in block_ids:
            key = (layer_id, bid)
            if key in self._flash_store and key not in self._controller_dram_cache:
                entry = self._flash_store[key]
                payload_bytes = entry["k"].nbytes + entry["v"].nbytes
                if self._controller_dram_used_bytes + payload_bytes <= self.dram_capacity_bytes:
                    self._controller_dram_cache.add(key)
                    self._controller_dram_used_bytes += payload_bytes
                    self._flash_read_bytes += payload_bytes
                    self._nand_read_ops += 1
                    flash_read_us = self.flash_model.calculate_flash_read_time_us(payload_bytes)
                    self._simulated_time_us += flash_read_us
                    staged += 1
        return staged

    def in_storage_topk_attention(
        self,
        query: np.ndarray,
        layer_id: int,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Executes in-storage computational attention filtering directly inside controller DRAM.
        
        Host sends ONLY query Q to SSD controller over PCIe.
        SSD computes dot-products Q * K^T against cold Keys in controller domain.
        SSD streams ONLY top_k Value blocks and top_k attention scores over PCIe.
        """
        # Collect all blocks for this layer
        layer_blocks = [(bid, entry) for (lid, bid), entry in self._flash_store.items() if lid == layer_id]
        if not layer_blocks:
            empty_idx = np.empty((0,), dtype=np.int32)
            empty_val = np.empty((0, 0, 0, 0), dtype=query.dtype)
            empty_sc = np.empty((0,), dtype=np.float32)
            return empty_idx, empty_val, empty_sc

        # Sort by block_id to ensure deterministic order
        layer_blocks.sort(key=lambda x: x[0])
        num_blocks = len(layer_blocks)
        effective_k = min(top_k, num_blocks)

        # 1. PCIe: Host transmits Query vector Q to SSD Controller
        query_bytes = query.nbytes
        self._host_to_device_bytes += query_bytes
        self._simulated_time_us += self.flash_model.calculate_pcie_transfer_time_us(query_bytes)

        # 2. In-Storage Processing:
        # Key tensors reside in SSD controller memory / Flash buffer.
        # Check NAND reads for Key tensors
        key_bytes_total = 0
        for bid, entry in layer_blocks:
            key = (layer_id, bid)
            k_bytes = entry["k"].nbytes
            if key not in self._controller_dram_cache:
                self._flash_read_bytes += k_bytes
                self._nand_read_ops += 1
                key_bytes_total += k_bytes
            else:
                self._controller_dram_read_bytes += k_bytes

        if key_bytes_total > 0:
            flash_read_us = self.flash_model.calculate_flash_read_time_us(key_bytes_total)
            self._simulated_time_us += flash_read_us

        # Dot-Product Computation & Top-k Filtering
        # Shape: query [num_heads, head_dim]
        # Each block K: [tokens_per_block, num_heads, head_dim]
        # Calculate MAC operations: num_blocks * tokens_per_block * num_heads * head_dim
        first_entry = layer_blocks[0][1]
        tokens_per_block = first_entry["k"].shape[0]
        num_heads = query.shape[0]
        head_dim = query.shape[1]
        macs = num_blocks * tokens_per_block * num_heads * head_dim
        self._compute_mac_count += macs

        # Controller compute latency: embedded DSP @ 1.2 GHz, e.g. 16 MACs/cycle = 19.2 GMAC/s
        controller_compute_us = (macs / (19.2 * 1e3))
        self._simulated_time_us += controller_compute_us

        # Execute dot product computation
        if self.c_kernel_fn is not None:
            # Native C Kernel path
            topk_block_ids, topk_values, topk_scores = self.c_kernel_fn(
                query=query,
                layer_blocks=layer_blocks,
                top_k=effective_k,
            )
        else:
            # High-performance NumPy reference path
            block_scores = []
            block_ids = []
            scale = 1.0 / np.sqrt(head_dim, dtype=np.float32)

            for bid, entry in layer_blocks:
                k_block = entry["k"]  # [T, H, D]
                # dot product: sum over D: Q[H, D] * K[T, H, D] -> [T, H]
                # score per block = mean or max across tokens and heads
                dots = np.einsum("hd,thd->th", query, k_block) * scale
                max_score = float(np.max(dots))
                block_scores.append(max_score)
                block_ids.append(bid)

            block_scores = np.array(block_scores, dtype=np.float32)
            block_ids = np.array(block_ids, dtype=np.int32)

            # Find top-k block indices
            if effective_k < len(block_scores):
                top_indices = np.argpartition(block_scores, -effective_k)[-effective_k:]
                top_indices = top_indices[np.argsort(-block_scores[top_indices])]
            else:
                top_indices = np.argsort(-block_scores)

            topk_block_ids = block_ids[top_indices]
            topk_scores = block_scores[top_indices]
            topk_values = np.stack([layer_blocks[idx][1]["v"] for idx in top_indices], axis=0)

        # 3. Read ONLY Top-k Value tensors from Flash (if not cached)
        val_read_bytes = 0
        for bid in topk_block_ids:
            key = (layer_id, int(bid))
            v_bytes = self._flash_store[key]["v"].nbytes
            if key not in self._controller_dram_cache:
                self._flash_read_bytes += v_bytes
                self._nand_read_ops += 1
                val_read_bytes += v_bytes
            else:
                self._controller_dram_read_bytes += v_bytes

        if val_read_bytes > 0:
            self._simulated_time_us += self.flash_model.calculate_flash_read_time_us(val_read_bytes)

        # 4. Compute per-token logits for the selected top-k blocks inside controller DRAM
        scale = 1.0 / np.sqrt(head_dim, dtype=np.float32)
        logits_list = []
        for bid in topk_block_ids:
            k_b = self._flash_store[(layer_id, int(bid))]["k"]  # [T, H, D]
            dots = np.einsum("hd,thd->ht", query, k_b) * scale   # [H, T]
            logits_list.append(dots)

        if logits_list:
            topk_logits = np.concatenate(logits_list, axis=-1)
        else:
            topk_logits = np.empty((num_heads, 0), dtype=np.float32)

        # 5. PCIe: Stream ONLY Top-k Values, per-token Logits, and Block IDs from SSD Controller to Host
        returned_payload_bytes = topk_values.nbytes + topk_logits.nbytes + topk_scores.nbytes + topk_block_ids.nbytes
        self._device_to_host_bytes += returned_payload_bytes
        self._simulated_time_us += self.flash_model.calculate_pcie_transfer_time_us(returned_payload_bytes)

        # Log trace event
        event = {
            "op": "IN_STORAGE_TOPK",
            "layer_id": layer_id,
            "num_candidates": num_blocks,
            "top_k": effective_k,
            "streamed_bytes": returned_payload_bytes,
            "sim_time_us": self._simulated_time_us,
        }
        self._trace_log.append(event)
        for listener in self._listeners:
            listener(event)

        return topk_block_ids, topk_values, topk_logits, topk_scores

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive hardware performance and telemetry metrics."""
        energy_dict = self.flash_model.calculate_energy_joules(
            host_dram_bytes=self._device_to_host_bytes,  # host receives and stores into DRAM
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
            "stored_blocks_count": len(self._flash_store),
            "energy": energy_dict,
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
        self._trace_log.clear()

    def register_access_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._listeners.append(callback)

    def get_trace_log(self) -> List[Dict[str, Any]]:
        return list(self._trace_log)
