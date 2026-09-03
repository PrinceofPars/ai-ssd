"""
Mock KV Engine for Person 2 Standalone Development and Testing.
Allows Person 2 to feed realistic KV block allocations and access traces
directly into FTL and NAND models without waiting for Person 1.
"""

from typing import List, Dict, Any
from common.schemas.kv_block import KVBlock, StorageTier
from common.schemas.request import KVRequest, KVOperation


class MockKVEngine:
    """
    Generates synthetic KVBlock lists and access traces.
    """
    def __init__(self, layers: int = 32, heads: int = 32):
        self.layers = layers
        self.heads = heads
        self._block_id_counter = 0

    def generate_kv_blocks(
        self,
        num_blocks: int = 64,
        layer_id: int = 0,
        token_count: int = 16,
    ) -> List[KVBlock]:
        """Generate a batch of synthetic KV blocks."""
        blocks = []
        for i in range(num_blocks):
            bid = self._block_id_counter
            self._block_id_counter += 1
            blocks.append(
                KVBlock.create_default(
                    block_id=bid,
                    layer_id=layer_id,
                    token_start=i * token_count,
                    token_count=token_count,
                    kv_head_start=i % self.heads,
                    kv_head_count=1,
                    storage_tier=StorageTier.GPU.value,
                )
            )
        return blocks

    def generate_attention_trace(
        self,
        layer_id: int,
        total_blocks: int,
        k: int = 16,
    ) -> List[int]:
        """
        Simulate an attention step that accesses k blocks within a layer.
        """
        return list(range(layer_id * total_blocks, layer_id * total_blocks + min(k, total_blocks)))
