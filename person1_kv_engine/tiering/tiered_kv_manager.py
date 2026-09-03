"""Tiered KV Cache Manager: Manages Host RAM <-> SSD Paged KV Blocks."""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from person1_kv_engine.baseline.transformer_attention import AttentionConfig
from person1_kv_engine.tiering.kv_block import KVBlock, KVBlockPool
from person1_kv_engine.tiering.hot_cold_classifier import HotColdClassifier, TieringPolicy


class TieredKVManager:
    """Coordinates allocation, hot/cold tiering, and SSD offloading of KV blocks."""

    def __init__(
        self,
        config: AttentionConfig,
        storage: Any,
        policy: Optional[TieringPolicy] = None,
    ):
        self.config = config
        self.storage = storage
        self.policy = policy or TieringPolicy()
        self.block_pool = KVBlockPool(tokens_per_block=self.policy.tokens_per_block)
        self.classifier = HotColdClassifier(self.policy)
        self._elem_bytes = np.dtype(config.dtype).itemsize

    def prefill_sequence(self, layer_id: int, k_seq: np.ndarray, v_seq: np.ndarray) -> None:
        """Slices prefill sequence into KVBlocks and offloads cold blocks to SSD."""
        seq_len = k_seq.shape[0]
        tokens_per_block = self.policy.tokens_per_block

        num_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block
        layer_blocks = []

        for b_idx in range(num_blocks):
            t_start = b_idx * tokens_per_block
            t_end = min(t_start + tokens_per_block, seq_len)
            cnt = t_end - t_start

            k_chunk = np.ascontiguousarray(k_seq[t_start:t_end])
            v_chunk = np.ascontiguousarray(v_seq[t_start:t_end])

            is_sink = (t_start < self.policy.sink_tokens)

            block = self.block_pool.allocate_block(
                layer_id=layer_id,
                token_start=t_start,
                token_count=cnt,
                k_data=k_chunk,
                v_data=v_chunk,
                is_pinned=is_sink,
            )
            layer_blocks.append(block)

        # Evict cold blocks to SSD
        for block in layer_blocks:
            if not self.classifier.is_hot(block, seq_len):
                if hasattr(self.storage, "store_kv"):
                    self.storage.store_kv(
                        block_id=block.block_id,
                        layer_id=layer_id,
                        key_data=block.k_data,
                        value_data=block.v_data,
                    )
                elif hasattr(self.storage, "store_block"):
                    self.storage.store_block(block)
                block.is_in_ram = False
                block.tier = "COLD_SSD"
                block.k_data = None
                block.v_data = None

    def append_token(self, layer_id: int, k_token: np.ndarray, v_token: np.ndarray) -> None:
        """Appends a new decode token and maintains hot/cold tiering boundaries."""
        layer_blocks = self.block_pool.get_layer_blocks(layer_id)
        k_t = k_token[np.newaxis, ...]
        v_t = v_token[np.newaxis, ...]
        tokens_per_block = self.policy.tokens_per_block

        if layer_blocks and layer_blocks[-1].token_count < tokens_per_block:
            last_block = layer_blocks[-1]
            if last_block.k_data is not None and last_block.v_data is not None:
                last_block.k_data = np.concatenate([last_block.k_data, k_t], axis=0)
                last_block.v_data = np.concatenate([last_block.v_data, v_t], axis=0)
            else:
                last_block.k_data = k_t
                last_block.v_data = v_t
            last_block.token_count += 1
        else:
            total_tokens = sum(b.token_count for b in layer_blocks)
            new_block = self.block_pool.allocate_block(
                layer_id=layer_id,
                token_start=total_tokens,
                token_count=1,
                k_data=k_t,
                v_data=v_t,
                is_pinned=False,
            )
            layer_blocks = self.block_pool.get_layer_blocks(layer_id)

        # Evict historical cold blocks if exceeding hot limit
        total_tokens = sum(b.token_count for b in layer_blocks)
        for block in layer_blocks:
            if block.is_in_ram and not block.is_pinned and not self.classifier.is_hot(block, total_tokens):
                if hasattr(self.storage, "store_kv"):
                    self.storage.store_kv(
                        block_id=block.block_id,
                        layer_id=layer_id,
                        key_data=block.k_data,
                        value_data=block.v_data,
                    )
                elif hasattr(self.storage, "store_block"):
                    self.storage.store_block(block)
                block.is_in_ram = False
                block.tier = "COLD_SSD"
                block.k_data = None
                block.v_data = None

    def fetch_cold_block(self, block_id: int, layer_id: int) -> KVBlock:
        """Fetches a cold block from SSD storage back into RAM."""
        block = self.block_pool.get_block(block_id)
        if block is None:
            raise ValueError(f"Block {block_id} not found in pool")

        if not block.is_in_ram:
            if hasattr(self.storage, "load_kv"):
                k_block, v_block = self.storage.load_kv(block_id=block_id, layer_id=layer_id)
            elif hasattr(self.storage, "load_block"):
                ret = self.storage.load_block(block_id)
                k_block, v_block = ret.k_data, ret.v_data
            else:
                raise AttributeError("Storage backend missing load method")
            block.k_data = np.ascontiguousarray(k_block)
            block.v_data = np.ascontiguousarray(v_block)
            block.is_in_ram = True
            block.tier = "HOT_RAM"

        return block

    def get_hot_kv(self, layer_id: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Retrieves contiguous K and V tensors for all blocks currently resident in RAM."""
        layer_blocks = self.block_pool.get_layer_blocks(layer_id)
        hot_k_list = []
        hot_v_list = []
        hot_indices = []

        for b in layer_blocks:
            if b.is_in_ram and b.k_data is not None and b.v_data is not None:
                hot_k_list.append(b.k_data)
                hot_v_list.append(b.v_data)
                hot_indices.extend(range(b.token_start, b.token_start + b.token_count))

        if not hot_k_list:
            empty = np.empty((0, self.config.num_heads, self.config.head_dim), dtype=self.config.dtype)
            return empty, empty, []

        return np.concatenate(hot_k_list, axis=0), np.concatenate(hot_v_list, axis=0), hot_indices

    def get_cold_blocks(self, layer_id: int) -> List[KVBlock]:
        """Returns all cold blocks offloaded to SSD for the given layer."""
        layer_blocks = self.block_pool.get_layer_blocks(layer_id)
        return [b for b in layer_blocks if not b.is_in_ram]

    def get_host_ram_usage_bytes(self) -> int:
        """Returns total memory bytes occupied by currently resident RAM blocks."""
        ram_bytes = 0
        for b in self.block_pool.all_blocks():
            if b.is_in_ram and b.k_data is not None and b.v_data is not None:
                ram_bytes += (b.k_data.nbytes + b.v_data.nbytes)
        return ram_bytes

    def get_host_ram_usage_mb(self) -> float:
        return self.get_host_ram_usage_bytes() / (1024 * 1024)

    def get_offloaded_storage_bytes(self) -> int:
        """Returns total bytes offloaded to SSD storage."""
        ssd_bytes = 0
        for b in self.block_pool.all_blocks():
            if not b.is_in_ram:
                # 2 * token_count * heads * head_dim * elem_bytes
                ssd_bytes += 2 * b.token_count * self.config.num_heads * self.config.head_dim * self._elem_bytes
        return ssd_bytes

    def get_offloaded_storage_mb(self) -> float:
        return self.get_offloaded_storage_bytes() / (1024 * 1024)
