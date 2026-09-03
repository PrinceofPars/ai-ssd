"""
Mock KV Engine for Person 2 Standalone Development and Testing.
Allows Person 2 to feed realistic KV block allocations and access traces
directly into FTL and NAND models without waiting for Person 1.
"""

from typing import List, Dict, Any, Optional
from common.schemas.kv_block import KVBlock, StorageTier
from common.schemas.request import KVRequest, KVOperation


class MockKVEngine:
    """
    Generates synthetic KVBlock lists and access traces faithful to
    transformer multi-head attention cache hierarchies.
    """
    def __init__(self, layers: int = 32, heads: int = 32):
        self.layers = layers
        self.heads = heads
        self._block_id_counter = 0

    def reset(self) -> None:
        """Resets the internal block ID counter."""
        self._block_id_counter = 0

    def generate_kv_blocks(
        self,
        num_blocks: int = 64,
        layer_id: int = 0,
        token_count: int = 16,
        layout: str = "token_major",
    ) -> List[KVBlock]:
        """
        Generate a batch of synthetic KV blocks.

        Layouts:
        - "token_major": Consecutive blocks represent distinct heads for the same
          token chunk (head-interleaved). Faithfully models parallel multi-head token
          generation and enables full multi-channel striping.
        - "head_major": Consecutive blocks represent sequential token chunks for the
          same head before advancing to the next head.
        """
        blocks = []
        blocks_per_head = max(1, num_blocks // self.heads)

        for i in range(num_blocks):
            bid = self._block_id_counter
            self._block_id_counter += 1

            if layout == "head_major":
                head_id = (i // blocks_per_head) % self.heads
                token_block_idx = i % blocks_per_head
            else:  # default "token_major"
                head_id = i % self.heads
                token_block_idx = i // self.heads

            token_start = token_block_idx * token_count

            blocks.append(
                KVBlock.create_default(
                    block_id=bid,
                    layer_id=layer_id,
                    token_start=token_start,
                    token_count=token_count,
                    kv_head_start=head_id,
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
        pattern: str = "concurrent_heads",
    ) -> List[int]:
        """
        Simulate an attention step that accesses k blocks within a layer.

        Patterns:
        - "concurrent_heads": Accesses the first k blocks in the layer (co-accessed heads).
        - "strided": Accesses every s-th block to model sparse multi-head sampling.
        """
        base_id = layer_id * total_blocks
        max_k = min(k, total_blocks)
        if pattern == "strided" and total_blocks > max_k:
            stride = max(1, total_blocks // max_k)
            return [base_id + (j * stride) for j in range(max_k)]
        return list(range(base_id, base_id + max_k))

    def generate_sparse_attention_request(
        self,
        blocks: List[KVBlock],
        k: int = 16,
        sink_ratio: float = 0.25,
    ) -> List[int]:
        """
        Generates realistic top-k block ID requests by selecting attention sink blocks
        (prompt start) and recent context blocks (sequence end).
        """
        if not blocks or k <= 0:
            return []
        if len(blocks) <= k:
            return [b.block_id for b in blocks]

        k_sink = max(1, int(k * sink_ratio))
        k_recent = k - k_sink

        sink_ids = [b.block_id for b in blocks[:k_sink]]
        recent_ids = [b.block_id for b in blocks[-k_recent:]]
        return sink_ids + recent_ids
