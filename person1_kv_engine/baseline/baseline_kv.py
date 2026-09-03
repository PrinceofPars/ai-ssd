"""Baseline monolithic in-memory KV cache.

Acts as the ground truth reference where all KV tensors remain un-evicted in GPU/Host memory.
Maintains exact theoretical byte calculations and real contiguous tensor storage.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from common.schemas.kv_block import KVBlock, StorageTier
from common.utils import calculate_kv_cache_size_bytes


class BaselineKVCache:
    def __init__(
        self,
        layers: int = 32,
        heads: int = 32,
        head_dim: int = 128,
        dtype: str = "FP16",
    ):
        self.layers = layers
        self.heads = heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.current_tokens = 0
        self.blocks: Dict[int, KVBlock] = {}

        # Real tensor storage per layer: [total_tokens, heads, head_dim]
        self._np_dtype = np.float16 if dtype.upper() == "FP16" else np.float32
        self._keys: List[Optional[np.ndarray]] = [None] * layers
        self._values: List[Optional[np.ndarray]] = [None] * layers

    def get_memory_bytes(self, context_length: int) -> int:
        """Calculate total in-memory size without offload."""
        return calculate_kv_cache_size_bytes(
            context_length=context_length,
            layers=self.layers,
            heads=self.heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
        )

    def get_memory_mb(self, context_length: int) -> float:
        return self.get_memory_bytes(context_length) / (1024 * 1024)

    def prefill(self, layer_id: int, k_seq: np.ndarray, v_seq: np.ndarray) -> None:
        """Stores prompt prefill tensors in ground-truth memory."""
        self._keys[layer_id] = np.ascontiguousarray(k_seq, dtype=self._np_dtype)
        self._values[layer_id] = np.ascontiguousarray(v_seq, dtype=self._np_dtype)
        if layer_id == 0:
            self.current_tokens = k_seq.shape[0]

    def append_token(self, layer_id: int, k_token: np.ndarray, v_token: np.ndarray) -> None:
        """Appends a new decode token tensor to ground-truth memory."""
        k_t = k_token[np.newaxis, ...]
        v_t = v_token[np.newaxis, ...]
        if self._keys[layer_id] is None:
            self._keys[layer_id] = np.ascontiguousarray(k_t, dtype=self._np_dtype)
            self._values[layer_id] = np.ascontiguousarray(v_t, dtype=self._np_dtype)
        else:
            self._keys[layer_id] = np.concatenate([self._keys[layer_id], k_t], axis=0)
            self._values[layer_id] = np.concatenate([self._values[layer_id], v_t], axis=0)
        if layer_id == self.layers - 1:
            self.current_tokens += 1

    def get_kv(self, layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves raw Key and Value matrices for a layer."""
        if self._keys[layer_id] is None or self._values[layer_id] is None:
            empty = np.empty((0, self.heads, self.head_dim), dtype=self._np_dtype)
            return empty, empty
        return self._keys[layer_id], self._values[layer_id]
