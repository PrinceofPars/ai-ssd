"""
High-Level KV Storage Allocator.
Dispatches placement between Conventional and Tensor-Aware FTL.
"""

from typing import Dict, List, Optional
from common.schemas.kv_block import KVBlock
from person2_ssd.ftl.conventional import ConventionalFTL
from person2_ssd.ftl.tensor_aware import TensorAwareFTL


class KVStorageAllocator:
    def __init__(self, mode: str = "tensor_aware", channels: int = 8):
        self.mode = mode
        self.ftl = TensorAwareFTL(channels=channels) if mode == "tensor_aware" else ConventionalFTL(channels=channels)
        self._stored_blocks: Dict[int, KVBlock] = {}

    def store_block(self, block: KVBlock) -> str:
        loc = self.ftl.allocate(block)
        self._stored_blocks[block.block_id] = block
        return loc

    def load_block(self, block_id: int) -> Optional[KVBlock]:
        return self._stored_blocks.get(block_id)

    def get_location(self, block_id: int) -> Optional[str]:
        return self.ftl.get_location(block_id)
