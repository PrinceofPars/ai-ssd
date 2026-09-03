"""Block Manager: Responsible for allocating, indexing, and managing KVBlocks."""

from typing import Dict, List, Optional
import numpy as np
from common.schemas.kv_block import KVBlock, StorageTier
from common.constants import DEFAULT_BLOCK_TOKENS, DEFAULT_HEAD_DIM, DEFAULT_DTYPE


class BlockManager:
    def __init__(
        self,
        block_tokens: int = DEFAULT_BLOCK_TOKENS,
        head_dim: int = DEFAULT_HEAD_DIM,
        dtype: str = DEFAULT_DTYPE,
    ):
        self.block_tokens = block_tokens
        self.head_dim = head_dim
        self.dtype = dtype
        self._next_block_id = 0
        self._blocks: Dict[int, KVBlock] = {}

    def allocate_block(
        self,
        layer_id: int,
        token_start: int,
        token_count: Optional[int] = None,
        kv_head_start: int = 0,
        kv_head_count: int = 1,
        tier: str = StorageTier.GPU.value,
        attach_tensors: bool = False,
    ) -> KVBlock:
        b_id = self._next_block_id
        self._next_block_id += 1
        cnt = token_count or self.block_tokens

        block = KVBlock.create_default(
            block_id=b_id,
            layer_id=layer_id,
            token_start=token_start,
            token_count=cnt,
            kv_head_start=kv_head_start,
            kv_head_count=kv_head_count,
            head_dim=self.head_dim,
            dtype=self.dtype,
            storage_tier=tier,
        )

        if attach_tensors:
            np_dtype = np.float16 if self.dtype.upper() == "FP16" else np.float32
            block.k_data = np.zeros((cnt, kv_head_count, self.head_dim), dtype=np_dtype)
            block.v_data = np.zeros((cnt, kv_head_count, self.head_dim), dtype=np_dtype)
        else:
            block.k_data = None
            block.v_data = None

        self._blocks[b_id] = block
        return block

    def get_block(self, block_id: int) -> Optional[KVBlock]:
        return self._blocks.get(block_id)

    def total_blocks(self) -> int:
        return len(self._blocks)

    def all_blocks(self) -> List[KVBlock]:
        return list(self._blocks.values())

    def clear(self) -> None:
        self._blocks.clear()
        self._next_block_id = 0
