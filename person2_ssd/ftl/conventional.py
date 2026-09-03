"""
Conventional FTL: Linear sequential allocation across physical blocks.
Causes channel contention during parallel attention head reads because
contiguous KV blocks end up in the same channel or die.
"""

from typing import List, Dict, Optional
from common.schemas.kv_block import KVBlock
from person2_ssd.ftl.mapping import MappingTable
from common.constants import SSD_CHANNELS, SSD_DIES_PER_CHANNEL


class ConventionalFTL:
    def __init__(self, channels: int = SSD_CHANNELS, dies_per_channel: int = SSD_DIES_PER_CHANNEL):
        self.channels = channels
        self.dies_per_channel = dies_per_channel
        self.mapping_table = MappingTable()
        self._linear_counter = 0

    def allocate(self, block: KVBlock) -> str:
        """
        Conventional sequential placement: assigns pages sequentially to channel 0, die 0 first
        until full, creating hot-spot channel bottlenecks.
        """
        ch = (self._linear_counter // 128) % self.channels
        die = (self._linear_counter // 512) % self.dies_per_channel
        pg = self._linear_counter % 128
        self._linear_counter += 1

        loc = f"ch{ch}_die{die}_pl0_blk0_pg{pg}"
        self.mapping_table.set_mapping(block.block_id, loc)
        block.physical_location = loc
        return loc

    def get_location(self, block_id: int) -> Optional[str]:
        return self.mapping_table.get_mapping(block_id)
