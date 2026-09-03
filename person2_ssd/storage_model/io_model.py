"""
Storage Simulator: End-to-end SSD model combining FTL, NAND allocation, and latency calculation.
"""

from typing import List, Dict, Optional
from common.schemas.kv_block import KVBlock, StorageTier
from person2_ssd.kv_allocator.allocator import KVStorageAllocator
from person2_ssd.storage_model.latency import LatencyModel


class StorageSimulator:
    def __init__(self, mode: str = "tensor_aware", channels: int = 8):
        self.allocator = KVStorageAllocator(mode=mode, channels=channels)
        self.latency_model = LatencyModel(channels=channels)
        self.mode = mode

    def store_block(self, block: KVBlock) -> str:
        block.storage_tier = StorageTier.SSD.value
        return self.allocator.store_block(block)

    def load_block(self, block_id: int) -> Optional[KVBlock]:
        return self.allocator.load_block(block_id)

    def get_location(self, block_id: int) -> Optional[str]:
        return self.allocator.get_location(block_id)

    def estimate_read_latency(self, block_ids: List[int]) -> float:
        locations = []
        for bid in block_ids:
            loc = self.get_location(bid)
            if loc:
                locations.append(loc)
        return self.latency_model.calculate_batch_read_latency(locations)
