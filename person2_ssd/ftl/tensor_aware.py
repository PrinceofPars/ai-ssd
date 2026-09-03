"""
Tensor-Aware FTL: Co-designed placement strategy that stripes KV blocks
across independent NAND channels and dies based on layer and attention head geometry.
Eliminates channel bottlenecks and maximizes read parallelism.
"""

from typing import List, Dict, Optional
from common.schemas.kv_block import KVBlock
from person2_ssd.ftl.mapping import MappingTable
from common.constants import SSD_CHANNELS, SSD_DIES_PER_CHANNEL


class TensorAwareFTL:
    def __init__(self, channels: int = SSD_CHANNELS, dies_per_channel: int = SSD_DIES_PER_CHANNEL):
        self.channels = channels
        self.dies_per_channel = dies_per_channel
        self.mapping_table = MappingTable()
        self._channel_counters = [0] * channels

    def allocate(self, block: KVBlock) -> str:
        """
        Tensor-aware striped allocation:
        Blocks belonging to consecutive heads/tokens within the same layer are assigned
        across distinct channels and dies round-robin.
        """
        # Map (layer, head, token_idx) to channel
        token_block_idx = block.token_start // max(1, block.token_count)
        ch = (block.kv_head_start + token_block_idx) % self.channels
        die = (block.layer_id + (token_block_idx // self.channels)) % self.dies_per_channel
        pg = self._channel_counters[ch] % 128
        blk = self._channel_counters[ch] // 128
        self._channel_counters[ch] += 1

        loc = f"ch{ch}_die{die}_pl0_blk{blk}_pg{pg}"
        self.mapping_table.set_mapping(block.block_id, loc)
        block.physical_location = loc
        return loc

    def allocate_batch(self, blocks: List[KVBlock]) -> Dict[int, str]:
        """Striped allocation for a batch of blocks."""
        results = {}
        for b in blocks:
            results[b.block_id] = self.allocate(b)
        return results

    def get_location(self, block_id: int) -> Optional[str]:
        return self.mapping_table.get_mapping(block_id)
