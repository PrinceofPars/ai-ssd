from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any


class StorageTier(str, Enum):
    GPU = "GPU"
    DRAM = "DRAM"
    SSD = "SSD"


class DType(str, Enum):
    FP16 = "FP16"
    FP8 = "FP8"


@dataclass
class KVBlock:
    """
    Fundamental unit of KV cache storage and transfer in AI-SSD.
    Supports MHA, GQA, and MQA configurations via kv_head_start/count.
    """
    block_id: int
    layer_id: int

    token_start: int
    token_count: int

    kv_head_start: int
    kv_head_count: int

    head_dim: int
    dtype: str

    key_size_bytes: int
    value_size_bytes: int

    storage_tier: str   # "GPU", "DRAM", "SSD"
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
        storage_tier: str = "GPU",
        hotness: float = 1.0,
    ) -> KVBlock:
        """
        Factory to create a standard KVBlock with calculated byte sizes.
        Default: 16 tokens * 1 head * 128 head_dim * 2 bytes = 2048 bytes per Key/Value (4096 bytes total).
        """
        bytes_per_elem = 2 if dtype.upper() == "FP16" else 1
        total_size = token_count * kv_head_count * head_dim * bytes_per_elem
        half_size = total_size // 2
        return cls(
            block_id=block_id,
            layer_id=layer_id,
            token_start=token_start,
            token_count=token_count,
            kv_head_start=kv_head_start,
            kv_head_count=kv_head_count,
            head_dim=head_dim,
            dtype=dtype,
            key_size_bytes=half_size,
            value_size_bytes=half_size,
            storage_tier=storage_tier,
            hotness=hotness,
        )
