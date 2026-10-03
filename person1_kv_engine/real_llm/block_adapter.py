"""Adapter between Real Model KV Tensors and the V2 KVBlock Representation.

Strictly enforces physical NAND page separation:
- Key Page = 4 KiB (at standard 16 tokens x 1 head x 128 dim in FP16, or 16 tokens x 1 head x 64 dim in FP32)
- Value Page = 4 KiB
- Combined Logical KV Block = 8 KiB

Programmatically validates all byte-size calculations against underlying tensor buffer sizes.
"""

from typing import List, Dict, Tuple, Any, Optional
import math
import numpy as np

from common.schemas.kv_block import KVBlock, StorageTier


class KVBlockAdapter:
    """Slices real model KV cache tensors into structured KVBlock descriptors and payload buffers."""

    def __init__(
        self,
        tokens_per_block: int = 16,
        kv_heads_per_block: int = 1,
        head_dim: int = 64,
        dtype: str = "FP32",
        attention_sink_tokens: int = 4,
        recent_window_tokens: int = 16,
    ):
        self.tokens_per_block = tokens_per_block
        self.kv_heads_per_block = kv_heads_per_block
        self.head_dim = head_dim
        self.dtype_str = dtype.upper()
        self.attention_sink_tokens = attention_sink_tokens
        self.recent_window_tokens = recent_window_tokens

        # Derive byte metrics
        self.bytes_per_elem = self._get_bytes_per_elem(self.dtype_str)
        self.key_size_bytes = self.tokens_per_block * self.kv_heads_per_block * self.head_dim * self.bytes_per_elem
        self.value_size_bytes = self.key_size_bytes
        self.logical_block_bytes = self.key_size_bytes + self.value_size_bytes

        # Physical 4 KiB Flash Page Counts
        self.key_flash_pages = math.ceil(self.key_size_bytes / 4096)
        self.value_flash_pages = math.ceil(self.value_size_bytes / 4096)
        self.total_flash_pages = self.key_flash_pages + self.value_flash_pages

    @staticmethod
    def _get_bytes_per_elem(dtype: str) -> int:
        dt = dtype.upper()
        if dt in ("FP32", "FLOAT32"):
            return 4
        elif dt in ("FP16", "FLOAT16", "BF16"):
            return 2
        elif dt in ("FP8", "INT8"):
            return 1
        return 4

    def validate_geometry(self, k_slice: np.ndarray, v_slice: np.ndarray) -> bool:
        """Validates that real tensor slice byte size matches physical geometry."""
        expected_bytes = self.key_size_bytes
        if k_slice.nbytes != expected_bytes:
            raise ValueError(
                f"K slice byte mismatch: expected {expected_bytes} bytes, got {k_slice.nbytes} bytes "
                f"(shape: {k_slice.shape}, dtype: {k_slice.dtype})"
            )
        if v_slice.nbytes != expected_bytes:
            raise ValueError(
                f"V slice byte mismatch: expected {expected_bytes} bytes, got {v_slice.nbytes} bytes "
                f"(shape: {v_slice.shape}, dtype: {v_slice.dtype})"
            )
        return True

    def blockize_layer(
        self,
        layer_id: int,
        k_tensor: np.ndarray,
        v_tensor: np.ndarray,
        global_block_offset: int = 0,
    ) -> List[Tuple[KVBlock, Dict[str, np.ndarray]]]:
        """Converts [kv_heads, seq_len, head_dim] tensors into a list of (KVBlock, payload) tuples.
        
        Args:
            layer_id: Transformer layer index
            k_tensor: Key array [kv_heads, seq_len, head_dim]
            v_tensor: Value array [kv_heads, seq_len, head_dim]
            global_block_offset: Starting integer for global block_id
            
        Returns:
            List of (KVBlock metadata, {"k": k_block, "v": v_block})
        """
        kv_heads, seq_len, head_dim = k_tensor.shape
        assert head_dim == self.head_dim, f"head_dim mismatch: expected {self.head_dim}, got {head_dim}"

        blocks = []
        block_idx = global_block_offset

        for h in range(0, kv_heads, self.kv_heads_per_block):
            h_count = min(self.kv_heads_per_block, kv_heads - h)

            for t_start in range(0, seq_len, self.tokens_per_block):
                t_end = min(t_start + self.tokens_per_block, seq_len)
                t_count = t_end - t_start

                # Slice tensor
                k_slice = k_tensor[h:h + h_count, t_start:t_end, :]
                v_slice = v_tensor[h:h + h_count, t_start:t_end, :]

                # If partial block at end of sequence, pad to full block geometry
                if t_count < self.tokens_per_block:
                    pad_len = self.tokens_per_block - t_count
                    k_pad = np.zeros((h_count, pad_len, head_dim), dtype=k_tensor.dtype)
                    v_pad = np.zeros((h_count, pad_len, head_dim), dtype=v_tensor.dtype)
                    k_slice = np.concatenate([k_slice, k_pad], axis=1)
                    v_slice = np.concatenate([v_slice, v_pad], axis=1)

                # Classify hotness and storage tier
                is_sink = (t_start < self.attention_sink_tokens)
                is_recent = (t_end > (seq_len - self.recent_window_tokens))

                if is_sink or is_recent:
                    tier = StorageTier.DRAM.value
                    hotness = 1.0 if is_sink else 0.9
                else:
                    tier = StorageTier.SSD.value
                    # Salience decay for middle context
                    pos_norm = float(t_start) / max(1.0, float(seq_len))
                    hotness = max(0.05, 0.5 * (1.0 - pos_norm))

                # Validate byte size
                self.validate_geometry(k_slice, v_slice)

                block_desc = KVBlock(
                    block_id=block_idx,
                    layer_id=layer_id,
                    token_start=t_start,
                    token_count=self.tokens_per_block,
                    kv_head_start=h,
                    kv_head_count=h_count,
                    head_dim=self.head_dim,
                    dtype=self.dtype_str,
                    key_size_bytes=self.key_size_bytes,
                    value_size_bytes=self.value_size_bytes,
                    storage_tier=tier,
                    hotness=hotness,
                )

                payload = {
                    "k": np.ascontiguousarray(k_slice, dtype=np.float32),
                    "v": np.ascontiguousarray(v_slice, dtype=np.float32),
                }

                blocks.append((block_desc, payload))
                block_idx += 1

        return blocks

    def blockize_all_layers(
        self,
        layer_kv: Dict[int, Dict[str, np.ndarray]],
    ) -> Dict[int, List[Tuple[KVBlock, Dict[str, np.ndarray]]]]:
        """Blockizes all layers in a real LLM KV cache."""
        all_layer_blocks = {}
        curr_id = 0
        for l_idx in sorted(layer_kv.keys()):
            k_t = layer_kv[l_idx]["k"]
            v_t = layer_kv[l_idx]["v"]
            layer_blocks = self.blockize_layer(l_idx, k_t, v_t, global_block_offset=curr_id)
            all_layer_blocks[l_idx] = layer_blocks
            curr_id += len(layer_blocks)
        return all_layer_blocks
