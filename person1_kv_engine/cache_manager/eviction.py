"""Eviction Policy: Handles moving cold KV blocks from GPU -> DRAM -> SSD."""

from typing import List, Any
from common.schemas.kv_block import KVBlock, StorageTier


class EvictionPolicy:
    def __init__(self, policy: str = "lru"):
        self.policy = policy

    def evict_to_ssd(self, blocks: List[KVBlock], ssd_backend: Any) -> List[int]:
        """Evicts a list of cold blocks to the SSD backend (supports MockSSD or Real SSD).
        Frees host RAM tensors if present.
        Returns list of evicted block IDs.
        """
        evicted_ids = []
        for b in blocks:
            if b.storage_tier != StorageTier.SSD.value:
                ssd_backend.store_block(b)
                b.storage_tier = StorageTier.SSD.value
                # Deallocate host RAM tensors to free memory
                if hasattr(b, "k_data"):
                    b.k_data = None
                if hasattr(b, "v_data"):
                    b.v_data = None
                evicted_ids.append(b.block_id)
        return evicted_ids
