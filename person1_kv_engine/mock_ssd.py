"""Mock SSD Implementation for Person 1 Standalone Development and Testing.

Allows Person 1 to develop KV cache eviction, offloading, and retrieval
without waiting for Person 2's physical SSD simulator.
Maintains full backward compatibility with team interfaces while exposing
hardware-level PCIe DMA byte counters, NAND operations, and physical timing.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from common.schemas.kv_block import KVBlock, StorageTier
from common.constants import T_R_US, T_PROG_US, BUS_TRANSFER_US_PER_PAGE


class MockSSD:
    """In-memory mock of the SSD storage backend with hardware metrics instrumentation."""

    def __init__(self, latency_per_block_us: float = T_R_US + BUS_TRANSFER_US_PER_PAGE):
        self._storage: Dict[int, KVBlock] = {}
        self.latency_per_block_us = latency_per_block_us
        self.write_count = 0
        self.read_count = 0

        # Detailed hardware counters
        self.host_to_device_bytes = 0
        self.device_to_host_bytes = 0
        self.flash_read_bytes = 0
        self.flash_write_bytes = 0
        self.simulated_time_us = 0.0

    def store_block(self, block: KVBlock) -> str:
        """Store a KVBlock into the mock SSD."""
        block.storage_tier = StorageTier.SSD.value
        loc = f"mock_ch{block.block_id % 8}_die0_pg{block.block_id}"
        block.physical_location = loc
        self._storage[block.block_id] = block
        self.write_count += 1

        payload_bytes = block.total_size_bytes
        self.host_to_device_bytes += payload_bytes
        self.flash_write_bytes += payload_bytes
        # 4 KB / 15.75 GB/s DMA ~ 1.5 us + tPROG (200 us)
        self.simulated_time_us += 1.5 + T_PROG_US
        return loc

    def load_block(self, block_id: int) -> Optional[KVBlock]:
        """Retrieve a KVBlock by ID."""
        self.read_count += 1
        block = self._storage.get(block_id)
        if block:
            block.storage_tier = StorageTier.DRAM.value
            payload_bytes = block.total_size_bytes
            self.device_to_host_bytes += payload_bytes
            self.flash_read_bytes += payload_bytes
            self.simulated_time_us += self.latency_per_block_us
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
        self.host_to_device_bytes = 0
        self.device_to_host_bytes = 0
        self.flash_read_bytes = 0
        self.flash_write_bytes = 0
        self.simulated_time_us = 0.0

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive telemetry dictionary."""
        return {
            "write_count": self.write_count,
            "read_count": self.read_count,
            "stored_blocks": len(self._storage),
            "host_to_device_bytes": self.host_to_device_bytes,
            "device_to_host_bytes": self.device_to_host_bytes,
            "total_pcie_bytes": self.host_to_device_bytes + self.device_to_host_bytes,
            "flash_read_bytes": self.flash_read_bytes,
            "flash_write_bytes": self.flash_write_bytes,
            "simulated_time_us": self.simulated_time_us,
        }
