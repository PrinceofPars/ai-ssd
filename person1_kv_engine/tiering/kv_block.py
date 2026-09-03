"""KV Block Data Structure and Block Pool for Paged KV Cache Tiering."""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class KVBlock:
    block_id: int
    layer_id: int
    token_start: int
    token_count: int
    is_pinned: bool = False
    is_in_ram: bool = True
    tier: str = "HOT_RAM"
    k_data: Optional[np.ndarray] = None
    v_data: Optional[np.ndarray] = None
    access_count: int = 0
    last_access_step: int = 0

    @property
    def total_size_bytes(self) -> int:
        if self.k_data is not None and self.v_data is not None:
            return self.k_data.nbytes + self.v_data.nbytes
        # Estimated: tokens * heads * head_dim * 4 bytes * 2
        return self.token_count * 4 * 32 * 4 * 2


class KVBlockPool:
    def __init__(self, tokens_per_block: int = 16):
        self.tokens_per_block = tokens_per_block
        self._next_id = 0
        self._blocks: Dict[int, KVBlock] = {}
        self._layer_blocks: Dict[int, List[KVBlock]] = {}

    def allocate_block(
        self,
        layer_id: int,
        token_start: int,
        token_count: int,
        k_data: Optional[np.ndarray] = None,
        v_data: Optional[np.ndarray] = None,
        is_pinned: bool = False,
    ) -> KVBlock:
        b_id = self._next_id
        self._next_id += 1

        block = KVBlock(
            block_id=b_id,
            layer_id=layer_id,
            token_start=token_start,
            token_count=token_count,
            is_pinned=is_pinned,
            is_in_ram=True,
            tier="HOT_RAM",
            k_data=k_data,
            v_data=v_data,
        )
        self._blocks[b_id] = block
        if layer_id not in self._layer_blocks:
            self._layer_blocks[layer_id] = []
        self._layer_blocks[layer_id].append(block)
        return block

    def get_layer_blocks(self, layer_id: int) -> List[KVBlock]:
        return self._layer_blocks.get(layer_id, [])

    def get_block(self, block_id: int) -> Optional[KVBlock]:
        return self._blocks.get(block_id)

    def all_blocks(self) -> List[KVBlock]:
        return list(self._blocks.values())
