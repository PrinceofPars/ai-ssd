from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any


# Physical Flash Page & Logical KV Block Sizes (Codified Contract)
KEY_PAGE_BYTES: int = 4096        # 4 KiB Key Page
VALUE_PAGE_BYTES: int = 4096      # 4 KiB Value Page
LOGICAL_BLOCK_BYTES: int = 8192   # 8 KiB Combined K+V Logical Block


class StorageTier(str, Enum):
    GPU = "GPU"
    DRAM = "DRAM"
    SSD = "SSD"


class DType(str, Enum):
    FP16 = "FP16"
    FP8 = "FP8"
    FP32 = "FP32"


class SubPageType(str, Enum):
    KEY = "KEY"
    VALUE = "VALUE"
    BOTH = "BOTH"


@dataclass
class KVBlock:
    """
    Fundamental unit of KV cache storage and transfer in AI-SSD.
    Supports MHA, GQA, and MQA configurations via kv_head_start/count.
    Key and Value tensors are partitioned into independent 4 KiB physical flash pages
    forming an 8 KiB combined logical block.
    """
    block_id: int
    layer_id: int

    token_start: int
    token_count: int

    kv_head_start: int
    kv_head_count: int

    head_dim: int
    dtype: str

    key_size_bytes: int = KEY_PAGE_BYTES
    value_size_bytes: int = VALUE_PAGE_BYTES

    storage_tier: str = StorageTier.SSD.value  # "GPU", "DRAM", "SSD"
    hotness: float = 1.0  # Access salience/recency score [0.0, 1.0]
    physical_location: Optional[str] = None  # e.g., "ch2_die1_plane0_blk12_pg4"

    @property
    def total_size_bytes(self) -> int:
        return self.key_size_bytes + self.value_size_bytes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KVBlock:
        return cls(**data)

    @classmethod
    def create_default(
        cls,
        block_id: int,
        layer_id: int,
        token_start: int,
        token_count: int = 16,
        kv_head_start: int = 0,
        kv_head_count: int = 1,
        head_dim: int = 128,
        dtype: str = "FP16",
        storage_tier: str = "SSD",
        hotness: float = 1.0,
    ) -> KVBlock:
        """
        Factory to create a standard KVBlock with codified physical page sizes:
        Default: 16 tokens * 1 head * 128 head_dim * 2 bytes (FP16) = 4096 bytes Key + 4096 bytes Value = 8192 bytes total.
        For FP32 with 64 dim (e.g. Qwen2.5-0.5B): 16 tokens * 1 head * 64 dim * 4 bytes = 4096 bytes Key + 4096 bytes Value = 8192 bytes total.
        """
        bytes_per_elem = 4 if dtype.upper() == "FP32" else (2 if dtype.upper() == "FP16" else 1)
        calc_page_size = token_count * kv_head_count * head_dim * bytes_per_elem
        # Standard page size is 4096 bytes
        page_size = calc_page_size if calc_page_size > 0 else KEY_PAGE_BYTES

        return cls(
            block_id=block_id,
            layer_id=layer_id,
            token_start=token_start,
            token_count=token_count,
            kv_head_start=kv_head_start,
            kv_head_count=kv_head_count,
            head_dim=head_dim,
            dtype=dtype,
            key_size_bytes=page_size,
            value_size_bytes=page_size,
            storage_tier=storage_tier,
            hotness=hotness,
        )
