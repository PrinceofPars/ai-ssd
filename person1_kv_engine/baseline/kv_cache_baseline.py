"""Baseline In-Memory KV Cache with exact memory usage tracking."""

from typing import List, Optional, Tuple, Any
import numpy as np
from person1_kv_engine.baseline.transformer_attention import AttentionConfig


class BaselineKVCache:
    """Ground truth in-memory KV cache for comparison with tiered storage."""

    def __init__(self, config: AttentionConfig):
        self.config = config
        self.layers = config.num_layers
        self.heads = config.num_heads
        self.head_dim = config.head_dim
        self.dtype = config.dtype
        self.current_tokens = 0

        self._keys: List[Optional[np.ndarray]] = [None] * self.layers
        self._values: List[Optional[np.ndarray]] = [None] * self.layers

    def prefill(self, layer_id: int, k_seq: np.ndarray, v_seq: np.ndarray) -> None:
        """Stores prefill KV tensors."""
        self._keys[layer_id] = np.ascontiguousarray(k_seq, dtype=self.dtype)
        self._values[layer_id] = np.ascontiguousarray(v_seq, dtype=self.dtype)
        if layer_id == 0:
            self.current_tokens = k_seq.shape[0]

    def append_token(self, layer_id: int, k_token: np.ndarray, v_token: np.ndarray) -> None:
        """Appends single-token tensors."""
        k_t = k_token[np.newaxis, ...]
        v_t = v_token[np.newaxis, ...]
        if self._keys[layer_id] is None:
            self._keys[layer_id] = np.ascontiguousarray(k_t, dtype=self.dtype)
            self._values[layer_id] = np.ascontiguousarray(v_t, dtype=self.dtype)
        else:
            self._keys[layer_id] = np.concatenate([self._keys[layer_id], k_t], axis=0)
            self._values[layer_id] = np.concatenate([self._values[layer_id], v_t], axis=0)
        if layer_id == self.layers - 1:
            self.current_tokens += 1

    def get_kv(self, layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._keys[layer_id] is None or self._values[layer_id] is None:
            empty = np.empty((0, self.heads, self.head_dim), dtype=self.dtype)
            return empty, empty
        return self._keys[layer_id], self._values[layer_id]

    def get_memory_usage_bytes(self) -> int:
        """Returns exact bytes allocated for all KV pairs in ground truth memory."""
        elem_bytes = np.dtype(self.dtype).itemsize
        # 2 (K+V) * layers * heads * head_dim * current_tokens * elem_bytes
        return self.current_tokens * (2 * self.layers * self.heads * self.head_dim * elem_bytes)

    def get_memory_mb(self) -> float:
        return self.get_memory_usage_bytes() / (1024 * 1024)

    def get_memory_usage_mb(self) -> float:
        return self.get_memory_mb()
