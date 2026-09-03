"""
Mock SSD Implementation for Person 1 Standalone Development and Testing.
Allows Person 1 to develop KV cache eviction, offloading, and retrieval
without waiting for Person 2's physical SSD simulator.
"""

from typing import Dict, List, Optional
from common.schemas.kv_block import KVBlock, StorageTier
from common.constants import T_R_US, T_PROG_US, BUS_TRANSFER_US_PER_PAGE


class MockSSD:
    """
    In-memory mock of the SSD storage backend.
    """
    def __init__(self, latency_per_block_us: float = T_R_US + BUS_TRANSFER_US_PER_PAGE):
        self._storage: Dict[int, KVBlock] = {}
        self.latency_per_block_us = latency_per_block_us
        self.write_count = 0
        self.read_count = 0

    def store_block(self, block: KVBlock) -> str:
        """Store a KVBlock into the mock SSD."""
        block.storage_tier = StorageTier.SSD.value
        loc = f"mock_ch{block.block_id % 8}_die0_pg{block.block_id}"
        block.physical_location = loc
        self._storage[block.block_id] = block
        self.write_count += 1
        return loc

    def load_block(self, block_id: int) -> Optional[KVBlock]:
        """Retrieve a KVBlock by ID."""
        self.read_count += 1
        block = self._storage.get(block_id)
        if block:
            block.storage_tier = StorageTier.DRAM.value
        return block

    def get_location(self, block_id: int) -> Optional[str]:
        """Get the physical location of a block."""
        block = self._storage.get(block_id)
        return block.physical_location if block else None

    def estimate_read_latency(self, block_ids: List[int]) -> float:
        """Estimate the latency (in microseconds) to read given blocks."""
        return len(block_ids) * self.latency_per_block_us

    def prefetch(self, block_ids: List[int]) -> List[int]:
        """Prefetch blocks into simulated staging buffer."""
        return [bid for bid in block_ids if bid in self._storage]

    def has_block(self, block_id: int) -> bool:
        return block_id in self._storage

    def clear(self) -> None:
        self._storage.clear()
        self.write_count = 0
        self.read_count = 0
