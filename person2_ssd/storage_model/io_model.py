"""
Storage Simulator: End-to-end SSD model combining FTL, NAND allocation, and latency calculation.
"""

import re
from typing import List, Dict, Optional, Union, Tuple
from common.schemas.kv_block import KVBlock, StorageTier
from person2_ssd.kv_allocator.allocator import KVStorageAllocator
from person2_ssd.storage_model.latency import LatencyModel
from person2_ssd.channels.channel import FlashChannel, ChannelTransferRequest
from person2_ssd.nand.page import PageState
from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    BUS_TRANSFER_US_PER_PAGE,
)

ADDRESS_PATTERN = re.compile(
    r"ch(?P<ch>\d+)_die(?P<die>\d+)_pl(?:ane)?(?P<pl>\d+)_blk(?P<blk>\d+)_pg(?P<pg>\d+)",
    re.IGNORECASE,
)


def parse_physical_location(loc: str) -> Optional[Tuple[int, int, int, int, int]]:
    """Parses canonical flash location string into (ch, die, pl, blk, pg)."""
    if not loc or not isinstance(loc, str):
        return None
    m = ADDRESS_PATTERN.search(loc)
    if not m:
        return None
    return (
        int(m.group("ch")),
        int(m.group("die")),
        int(m.group("pl")),
        int(m.group("blk")),
        int(m.group("pg")),
    )


class StorageSimulator:
    def __init__(
        self,
        mode: str = "tensor_aware",
        channels: int = SSD_CHANNELS,
        dies_per_channel: int = SSD_DIES_PER_CHANNEL,
        planes_per_die: int = SSD_PLANES_PER_DIE,
        blocks_per_plane: int = 64,
    ):
        self.mode = mode
        self.num_channels = channels
        self.channels: List[FlashChannel] = [
            FlashChannel(c, dies_per_channel, planes_per_die, blocks_per_plane)
            for c in range(channels)
        ]
        self.allocator = KVStorageAllocator(mode=mode, channels=channels)
        self.latency_model = LatencyModel(channels=channels)

    def store_block(self, block: KVBlock) -> str:
        """
        Allocates physical location via FTL, marks tier, programs physical page,
        and schedules bus/die activity.
        """
        block.storage_tier = StorageTier.SSD.value
        loc = self.allocator.store_block(block)

        parsed = parse_physical_location(loc)
        if parsed:
            ch, die, pl, blk, pg = parsed
            if 0 <= ch < len(self.channels):
                channel = self.channels[ch]
                if 0 <= die < len(channel.dies):
                    die_obj = channel.dies[die]
                    if 0 <= pl < len(die_obj.planes):
                        plane = die_obj.planes[pl]
                        if 0 <= blk < len(plane.blocks):
                            block_obj = plane.blocks[blk]
                            if 0 <= pg < len(block_obj.pages):
                                page_obj = block_obj.pages[pg]
                                if page_obj.state == PageState.FREE:
                                    page_obj.program(data_block_id=block.block_id)
                                    block_obj.free_page_index = max(
                                        block_obj.free_page_index, pg + 1
                                    )
                    channel.schedule_transfer(0.0, BUS_TRANSFER_US_PER_PAGE)
                    die_obj.schedule_program(0.0)

        return loc

    def load_block(self, block_id: int) -> Optional[KVBlock]:
        """Retrieves KVBlock metadata by logical ID."""
        return self.allocator.load_block(block_id)

    def get_location(self, block_id: int) -> Optional[str]:
        """Translates logical block ID to physical location string."""
        return self.allocator.get_location(block_id)

    def read_blocks(self, blocks: List[Union[KVBlock, int]]) -> float:
        """
        Resolves physical locations and evaluates batch read latency (PROJECT.md contract).
        Supports list of KVBlock instances or list of integer block IDs.
        """
        block_ids: List[int] = []
        for b in blocks:
            if hasattr(b, "block_id"):
                block_ids.append(b.block_id)
            elif isinstance(b, int):
                block_ids.append(b)
        return self.estimate_read_latency(block_ids)

    def estimate_read_latency(self, block_ids: List[int]) -> float:
        """
        Evaluates batch read latency, checks physical page states, updates read counts,
        enqueues channel transfer requests, and calculates contention latency.
        """
        locations = []
        for bid in block_ids:
            loc = self.get_location(bid)
            if loc:
                locations.append(loc)
                parsed = parse_physical_location(loc)
                if parsed:
                    ch, die, pl, blk, pg = parsed
                    if 0 <= ch < len(self.channels):
                        channel = self.channels[ch]
                        if 0 <= die < len(channel.dies):
                            die_obj = channel.dies[die]
                            if 0 <= pl < len(die_obj.planes):
                                plane = die_obj.planes[pl]
                                if 0 <= blk < len(plane.blocks):
                                    block_obj = plane.blocks[blk]
                                    if 0 <= pg < len(block_obj.pages):
                                        page = block_obj.pages[pg]
                                        page.read()
                        channel.enqueue_request(
                            ChannelTransferRequest(
                                request_id=bid,
                                die_id=die,
                                plane_id=pl,
                                block_id=blk,
                                page_id=pg,
                                op_type="READ",
                            )
                        )

        # Process channel transfer queues
        for ch in self.channels:
            ch.process_queue()

        return self.latency_model.calculate_batch_read_latency(locations)

    def reset(self) -> None:
        """Resets physical channels and allocator state."""
        for ch in self.channels:
            ch.reset()

