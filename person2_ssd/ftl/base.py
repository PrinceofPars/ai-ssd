"""
Base Flash Translation Layer (FTL) Class.
Defines common interface contracts for Conventional and Tensor-Aware strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from common.schemas.kv_block import KVBlock
from person2_ssd.ftl.mapping import MappingTable
from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    SSD_PAGES_PER_BLOCK,
)


class BaseFTL(ABC):
    """
    Abstract Base Class for all Flash Translation Layer implementations.
    Guarantees consistent mapping table management and translation contracts.
    """
    def __init__(
        self,
        channels: int = SSD_CHANNELS,
        dies_per_channel: int = SSD_DIES_PER_CHANNEL,
        planes_per_die: int = SSD_PLANES_PER_DIE,
        blocks_per_plane: int = 64,
        pages_per_block: int = SSD_PAGES_PER_BLOCK,
    ):
        self.channels = channels
        self.dies_per_channel = dies_per_channel
        self.planes_per_die = planes_per_die
        self.blocks_per_plane = blocks_per_plane
        self.pages_per_block = pages_per_block
        self.total_pages = (
            channels * dies_per_channel * planes_per_die * blocks_per_plane * pages_per_block
        )
        self.mapping_table = MappingTable()
        self._reverse_map: Dict[str, int] = {}

    def _record_mapping(self, block_id: int, physical_loc: str) -> None:
        """Synchronizes forward and reverse mapping entries."""
        old_loc = self.mapping_table.get_mapping(block_id)
        if old_loc is not None:
            self._reverse_map.pop(old_loc, None)
        self.mapping_table.set_mapping(block_id, physical_loc)
        self._reverse_map[physical_loc] = block_id

    @abstractmethod
    def allocate(self, block: KVBlock) -> str:
        """
        Allocates a physical location for the given KVBlock adhering strictly to:
        'ch<C>_die<D>_pl<P>_blk<B>_pg<G>'
        Must record the mapping in self.mapping_table and self._reverse_map,
        and update block.physical_location.
        """
        pass

    def allocate_batch(self, blocks: List[KVBlock]) -> Dict[int, str]:
        """Allocates physical locations for a batch of blocks."""
        results = {}
        for b in blocks:
            results[b.block_id] = self.allocate(b)
        return results

    def translate(self, block_id: int) -> Optional[str]:
        """Translates logical block ID to physical location string."""
        return self.mapping_table.get_mapping(block_id)

    def reverse_translate(self, physical_loc: str) -> Optional[int]:
        """Translates physical location string back to logical block ID."""
        return self._reverse_map.get(physical_loc)

    def get_location(self, block_id: int) -> Optional[str]:
        """Backward-compatible alias for translate()."""
        return self.translate(block_id)

    def get_mapping_table(self) -> Dict[int, str]:
        """Returns a snapshot copy of the logical-to-physical mapping table dictionary."""
        return dict(self.mapping_table._map)

    def get_reverse_mapping_table(self) -> Dict[str, int]:
        """Returns a snapshot copy of the physical-to-logical mapping table dictionary."""
        return dict(self._reverse_map)

    def reset(self) -> None:
        """Resets the mapping table and internal state."""
        self.mapping_table._map.clear()
        self._reverse_map.clear()
