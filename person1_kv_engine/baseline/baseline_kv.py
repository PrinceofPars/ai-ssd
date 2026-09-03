"""
Baseline monolithic in-memory KV cache.
Acts as the ground truth reference where all KV tensors remain un-evicted in GPU memory.
"""

from typing import Dict, Any
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
