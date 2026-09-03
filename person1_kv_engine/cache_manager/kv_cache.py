"""
Paged KV Cache Manager: Coordinates allocation, hot/cold tiering, and SSD offloading.
"""

from typing import Dict, List, Tuple, Any, Optional
from common.schemas.kv_block import KVBlock, StorageTier
from person1_kv_engine.cache_manager.block_manager import BlockManager
from person1_kv_engine.cache_manager.hot_cold import HotColdClassifier
from person1_kv_engine.cache_manager.eviction import EvictionPolicy


class PagedKVCache:
    def __init__(
        self,
        layers: int = 32,
        heads: int = 32,
        head_dim: int = 128,
        block_tokens: int = 16,
        dtype: str = "FP16",
        sink_tokens: int = 64,
        recent_tokens: int = 512,
        ssd_backend: Optional[Any] = None,
    ):
        self.layers = layers
        self.heads = heads
        self.head_dim = head_dim
        self.block_tokens = block_tokens
        self.dtype = dtype

        self.block_manager = BlockManager(block_tokens=block_tokens, head_dim=head_dim, dtype=dtype)
        self.classifier = HotColdClassifier(sink_tokens=sink_tokens, recent_tokens=recent_tokens)
        self.eviction = EvictionPolicy()
        self.ssd_backend = ssd_backend

    def initialize_context(self, context_length: int) -> int:
        """
        Populates paged blocks for all layers and heads for a given context length.
        """
        num_blocks_per_head = (context_length + self.block_tokens - 1) // self.block_tokens
        for layer_id in range(self.layers):
            for head_id in range(self.heads):
                for b_idx in range(num_blocks_per_head):
                    token_start = b_idx * self.block_tokens
                    self.block_manager.allocate_block(
                        layer_id=layer_id,
                        token_start=token_start,
                        token_count=min(self.block_tokens, context_length - token_start),
                        kv_head_start=head_id,
                        kv_head_count=1,
                        tier=StorageTier.GPU.value,
                    )
        return self.block_manager.total_blocks()

    def run_offload_pass(self, context_length: int) -> Tuple[int, int]:
        """
        Partitions blocks into hot/cold and offloads cold blocks to SSD if available.
        Returns (hot_count, cold_count).
        """
        all_blocks = self.block_manager.all_blocks()
        hot, cold = self.classifier.partition_blocks(all_blocks, context_length)

        if self.ssd_backend:
            self.eviction.evict_to_ssd(cold, self.ssd_backend)

        return len(hot), len(cold)
