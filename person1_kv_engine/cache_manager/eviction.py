"""
Eviction Policy: Handles moving cold KV blocks from GPU -> DRAM -> SSD.
"""

from typing import List, Any
from common.schemas.kv_block import KVBlock, StorageTier


class EvictionPolicy:
    def __init__(self, policy: str = "lru"):
        self.policy = policy

    def evict_to_ssd(self, blocks: List[KVBlock], ssd_backend: Any) -> List[int]:
        """
        Evicts a list of cold blocks to the SSD backend (supports MockSSD or Real SSD).
        Returns list of evicted block IDs.
        """
        evicted_ids = []
        for b in blocks:
            if b.storage_tier != StorageTier.SSD.value:
                ssd_backend.store_block(b)
                b.storage_tier = StorageTier.SSD.value
                evicted_ids.append(b.block_id)
        return evicted_ids
