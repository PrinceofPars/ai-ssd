"""
Deterministic Tensor-to-Storage Mapping Architecture (Canonical P2 Implementation).

Formulates the formal 4-level translation hierarchy:
    Tensor Coordinate (layer, head, token_start/block_id)
        ↓
    KV Block Descriptor (token_block_idx, 4096-byte page / 8192-byte block)
        ↓
    Logical Block Address (LBA, 64-bit sector offset in NVMe namespace)
        ↓
    NAND Physical Coordinate (channel, die, plane, block, page)

NOTE ON PHYSICAL PLACEMENT VS LOGICAL STRIPING:
On commodity black-box NVMe SSDs, the host controls the LBA, but the drive's internal
closed-source FTL ultimately determines physical NAND placement. An LBA mapping alone
does NOT prove physical channel parallelism on a consumer SSD without an open-channel
architecture, ZNS, or a programmable emulator like FEMU. 
On FEMU or our cycle-accurate FTL simulator, the mapping is guaranteed end-to-end.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    SSD_PAGES_PER_BLOCK,
    DEFAULT_BLOCK_SIZE_BYTES,
)


@dataclass(frozen=True)
class TensorCoordinate:
    """
    Identifies a discrete KV block in tensor space.
    Supports either token-index or explicit block_id resolution.
    """
    layer_id: int
    head_id: int
    token_idx: int = 0
    tokens_per_block: int = 16
    block_id: Optional[int] = None
    blocks_per_head: int = 44

    @property
    def token_block_idx(self) -> int:
        if self.token_idx > 0:
            return self.token_idx // max(1, self.tokens_per_block)
        if self.block_id is not None and self.blocks_per_head > 0:
            return self.block_id % self.blocks_per_head
        return self.token_idx // max(1, self.tokens_per_block)


@dataclass(frozen=True)
class LBAAddress:
    """
    Represents a Logical Block Address (LBA) on a standard NVMe block device.
    """
    lba: int
    sector_size_bytes: int = 4096
    mode: str = "tensor_aware"

    @property
    def byte_offset(self) -> int:
        return self.lba * self.sector_size_bytes


@dataclass(frozen=True)
class NANDPhysicalCoordinate:
    """
    Physical coordinates within the multi-channel flash geometry.
    """
    channel: int
    die: int
    plane: int
    block: int
    page: int

    def to_location_str(self) -> str:
        return f"ch{self.channel}_die{self.die}_pl{self.plane}_blk{self.block}_pg{self.page}"


class DeterministicTensorMapper:
    """
    Translates tensor coordinates to LBA space and physical NAND locations.
    Can be directly instantiated and used by Person 3 without duplicating logic.
    """
    def __init__(
        self,
        channels: int = SSD_CHANNELS,
        dies_per_channel: int = SSD_DIES_PER_CHANNEL,
        planes_per_die: int = SSD_PLANES_PER_DIE,
        blocks_per_plane: int = 64,
        pages_per_block: int = SSD_PAGES_PER_BLOCK,
        sector_size_bytes: int = DEFAULT_BLOCK_SIZE_BYTES,
        max_tokens_per_head: int = 32768,
        num_layers: int = 24,
        num_heads: int = 2,
    ):
        self.channels = channels
        self.dies_per_channel = dies_per_channel
        self.planes_per_die = planes_per_die
        self.blocks_per_plane = blocks_per_plane
        self.pages_per_block = pages_per_block
        self.sector_size_bytes = sector_size_bytes
        self.max_tokens_per_head = max_tokens_per_head
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.blocks_per_head = max_tokens_per_head // 16
        self.total_device_blocks = channels * dies_per_channel * planes_per_die * blocks_per_plane * pages_per_block

    def tensor_to_lba(self, coord: TensorCoordinate, mode: str = "tensor_aware") -> LBAAddress:
        """
        Maps a 3D tensor coordinate (layer, head, token) to a 1D Logical Block Address.
        """
        b_idx = coord.token_block_idx
        layer = coord.layer_id
        head = coord.head_id

        if mode == "conventional":
            # Linear sequential allocation:
            global_head_idx = (layer * self.num_heads) + head
            raw_lba = (global_head_idx * self.blocks_per_head) + b_idx
            lba = raw_lba % self.total_device_blocks
            return LBAAddress(lba=lba, sector_size_bytes=self.sector_size_bytes, mode="conventional")

        elif mode == "tensor_aware":
            # Channel-striped LBA allocation incorporating layer, head, and token block index
            target_ch = (layer + head + b_idx + (b_idx // self.channels)) % self.channels
            target_die = (layer + (head // self.channels) + (b_idx // self.channels)) % self.dies_per_channel
            stripe_unit = self.total_device_blocks // (self.channels * self.dies_per_channel)
            offset_in_stripe = ((layer * self.num_heads + head) * self.blocks_per_head + b_idx) % stripe_unit
            lba = (target_ch * self.dies_per_channel + target_die) * stripe_unit + offset_in_stripe
            return LBAAddress(lba=lba, sector_size_bytes=self.sector_size_bytes, mode="tensor_aware")

        else:
            raise ValueError(f"Unknown mapping mode: {mode}")

    def tensor_to_nand_physical(
        self,
        coord: TensorCoordinate,
        mode: str = "tensor_aware",
        channel_counters: Optional[Dict[int, int]] = None,
    ) -> NANDPhysicalCoordinate:
        """
        Calculates physical NAND coordinates (channel, die, plane, block, page).
        """
        b_idx = coord.token_block_idx
        layer = coord.layer_id
        head = coord.head_id

        if mode == "conventional":
            # Conventional FTL concentrates traffic sequentially onto channel 0
            ch = 0
            die = 0
            pl = 0
            counter = 0 if channel_counters is None else channel_counters.get(ch, 0)
            pg = counter % self.pages_per_block
            blk = (counter // self.pages_per_block) % self.blocks_per_plane
            return NANDPhysicalCoordinate(channel=ch, die=die, plane=pl, block=blk, page=pg)

        elif mode == "tensor_aware":
            # Factors in layer, head, and block index to guarantee multi-channel reachability
            # even when the model architecture has only 2 KV heads (GQA)
            ch = (layer + head + b_idx + (b_idx // self.channels)) % self.channels
            die = (layer + (head // self.channels) + (b_idx // self.channels)) % self.dies_per_channel
            pl = (b_idx // (self.channels * self.dies_per_channel)) % self.planes_per_die

            counter = 0 if channel_counters is None else channel_counters.get(ch, 0)
            pg = counter % self.pages_per_block
            blk = (counter // self.pages_per_block) % self.blocks_per_plane
            return NANDPhysicalCoordinate(channel=ch, die=die, plane=pl, block=blk, page=pg)

        else:
            raise ValueError(f"Unknown mapping mode: {mode}")