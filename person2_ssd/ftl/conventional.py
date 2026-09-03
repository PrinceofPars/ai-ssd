"""
Conventional FTL: Linear sequential allocation across physical flash hierarchy.
Causes channel contention during parallel attention head reads because
contiguous KV blocks end up in the same channel.
"""

from typing import Optional
from common.schemas.kv_block import KVBlock
from person2_ssd.ftl.base import BaseFTL
from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    SSD_PAGES_PER_BLOCK,
)


class ConventionalFTL(BaseFTL):
    """
    Conventional sequential placement: assigns pages sequentially across
    Page -> Block -> Plane -> Die -> Channel hierarchy.
    Concentrates concurrent attention requests onto Channel 0, producing
    channel hot-spotting bottlenecks.
    """
    def __init__(
        self,
        channels: int = SSD_CHANNELS,
        dies_per_channel: int = SSD_DIES_PER_CHANNEL,
        planes_per_die: int = SSD_PLANES_PER_DIE,
        blocks_per_plane: int = 64,
        pages_per_block: int = SSD_PAGES_PER_BLOCK,
    ):
        super().__init__(
            channels=channels,
            dies_per_channel=dies_per_channel,
            planes_per_die=planes_per_die,
            blocks_per_plane=blocks_per_plane,
            pages_per_block=pages_per_block,
        )
        self._linear_counter = 0

    def allocate(self, block: KVBlock) -> str:
        """
        Allocates the next sequential physical page.
        Raises RuntimeError if physical capacity is exhausted.
        """
        if self._linear_counter >= self.total_pages:
            raise RuntimeError(
                f"SSD capacity exceeded: cannot allocate block {block.block_id}. "
                f"Total capacity {self.total_pages} pages exhausted."
            )

        idx = self._linear_counter
        pg = idx % self.pages_per_block
        idx //= self.pages_per_block

        blk = idx % self.blocks_per_plane
        idx //= self.blocks_per_plane

        pl = idx % self.planes_per_die
        idx //= self.planes_per_die

        die = idx % self.dies_per_channel
        ch = idx // self.dies_per_channel

        self._linear_counter += 1

        loc = f"ch{ch}_die{die}_pl{pl}_blk{blk}_pg{pg}"
        self._record_mapping(block.block_id, loc)
        block.physical_location = loc
        return loc

    def reset(self) -> None:
        """Resets the linear allocation counter and clears mapping table."""
        self._linear_counter = 0
        super().reset()
