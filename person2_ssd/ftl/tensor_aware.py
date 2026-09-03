"""
Tensor-Aware FTL: Co-designed placement strategy that stripes KV blocks
across independent NAND channels and dies based on layer and attention head geometry.
Eliminates channel bottlenecks and maximizes read parallelism.
"""

from typing import List, Dict, Optional
from common.schemas.kv_block import KVBlock
from person2_ssd.ftl.base import BaseFTL
from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    SSD_PAGES_PER_BLOCK,
)


class TensorAwareFTL(BaseFTL):
    """
    Tensor-Aware striped placement:
    Distributes blocks across channels, dies, and planes round-robin
    based on layer, attention head, and token block coordinates.
    Eliminates channel contention and odd-channel parity starvation.
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
        self._channel_counters = [0] * channels

    def allocate(self, block: KVBlock) -> str:
        """
        Tensor-aware striped allocation:
        Distributes blocks across channels, dies, and planes round-robin
        based on layer, attention head, and token block coordinates.
        Eliminates channel contention and modulo parity starvation.
        """
        token_count = max(1, getattr(block, "token_count", 16) or 16)
        token_start = getattr(block, "token_start", 0) or 0
        token_block_idx = token_start // token_count
        head = getattr(block, "kv_head_start", 0) or 0
        layer = getattr(block, "layer_id", 0) or 0

        # Uniform 8-channel striping eliminating parity starvation
        ch = (head + token_block_idx + (token_block_idx // self.channels)) % self.channels

        # Multi-die striping across dies per channel
        die = (layer + (head // self.channels) + (token_block_idx // self.channels)) % self.dies_per_channel

        # Multi-plane striping
        pl = (token_block_idx // (self.channels * self.dies_per_channel)) % self.planes_per_die

        # Page and block allocation within the selected channel
        pg = self._channel_counters[ch] % self.pages_per_block
        blk = (self._channel_counters[ch] // self.pages_per_block) % self.blocks_per_plane
        self._channel_counters[ch] += 1

        loc = f"ch{ch}_die{die}_pl{pl}_blk{blk}_pg{pg}"
        self._record_mapping(block.block_id, loc)
        block.physical_location = loc
        return loc

    def reset(self) -> None:
        """Resets channel counters and clears mapping table."""
        self._channel_counters = [0] * self.channels
        super().reset()
