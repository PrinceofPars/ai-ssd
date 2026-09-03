"""Standardized KV Storage Interface Contract.

Defines the unified API connecting Person 1 (KV Engine / In-Storage Compute),
Person 2 (Tensor-Aware FTL), and Person 3 (Speculative Prefetcher & Integration).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional, Callable
import time
import numpy as np


@dataclass
class KVBlockMetadata:
    """Metadata describing a single tensor-aware KV block."""
    block_id: int
    layer_id: int
    token_start: int
    token_count: int
    num_heads: int
    head_dim: int
    dtype_str: str = "float32"
    dtype_bytes: int = 4
    created_timestamp: float = field(default_factory=time.time)
    last_accessed_timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    is_resident_in_controller_dram: bool = False

    @property
    def total_elements(self) -> int:
        """Total scalar elements for K or V (tokens * heads * head_dim)."""
        return self.token_count * self.num_heads * self.head_dim

    @property
    def payload_bytes_single_tensor(self) -> int:
        """Byte size of Key or Value tensor."""
        return self.total_elements * self.dtype_bytes

    @property
    def payload_bytes_total(self) -> int:
        """Total byte size of both Key and Value tensors combined."""
        return 2 * self.payload_bytes_single_tensor


class KVStorageInterface(ABC):
    """Abstract interface contract for AI-aware storage devices and simulators."""

    @abstractmethod
    def store_kv(
        self,
        block_id: int,
        layer_id: int,
        key_data: np.ndarray,
        value_data: np.ndarray,
        metadata: Optional[KVBlockMetadata] = None,
    ) -> bool:
        """Writes a KV block from host to storage.
        
        Args:
            block_id: Unique identifier for the block within the layer.
            layer_id: Transformer layer index.
            key_data: Key tensor chunk (shape: [token_count, num_heads, head_dim]).
            value_data: Value tensor chunk (shape: [token_count, num_heads, head_dim]).
            metadata: Optional block metadata.
        Returns:
            True if write succeeded.
        """
        pass

    @abstractmethod
    def load_kv(
        self,
        block_id: int,
        layer_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reads a KV block from storage into host memory.
        
        Transfers both Key and Value data across the PCIe bus.
        """
        pass

    @abstractmethod
    def evict_kv(self, block_id: int, layer_id: int) -> bool:
        """Erases/invalidates a KV block in storage when context is closed."""
        pass

    @abstractmethod
    def prefetch_kv(self, block_ids: List[int], layer_id: int) -> int:
        """Speculatively loads flash blocks into SSD controller DRAM cache ahead of host demand.
        
        Used by Person 3's prefetcher.
        Returns the number of blocks successfully staged.
        """
        pass

    @abstractmethod
    def in_storage_topk_attention(
        self,
        query: np.ndarray,
        layer_id: int,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Executes in-storage computational attention filtering directly inside controller DRAM.
        
        Host sends ONLY query vector Q to SSD, controller computes dot-products against stored Key blocks,
        and streams only the top_k Value vectors, per-token logits, and block scores over PCIe.
        
        Args:
            query: Query tensor for the token (shape: [num_heads, head_dim]).
            layer_id: Transformer layer index.
            top_k: Number of most relevant KV blocks to retrieve.
            
        Returns:
            Tuple of:
            - topk_block_ids: np.ndarray of shape [top_k]
            - topk_values: np.ndarray of shape [top_k, token_count, num_heads, head_dim]
            - topk_logits: np.ndarray of shape [num_heads, top_k * token_count]
            - topk_scores: np.ndarray of raw block scores
        """
        pass

    @abstractmethod
    def get_telemetry(self) -> Dict[str, Any]:
        """Returns hardware and interface performance counters."""
        pass

    @abstractmethod
    def reset_telemetry(self) -> None:
        """Resets all metrics counters to zero."""
        pass

    @abstractmethod
    def register_access_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Registers a listener callback triggered on every block access (for Person 3)."""
        pass

    @abstractmethod
    def get_trace_log(self) -> List[Dict[str, Any]]:
        """Returns sequential access log for workload and FTL analysis (for Person 2)."""
        pass
