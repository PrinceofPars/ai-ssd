"""
Configuration dataclasses for V2 benchmark baselines and ablations.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ExperimentConfig:
    experiment_id: str
    model_name: str = "Llama-3-8B"
    context_length: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    dtype: str = "FP16"
    offload_pct: float = 80.0
    topk_pct: float = 10.0
    prefetch_enabled: bool = False
    ftl_mode: str = "tensor_aware"  # "none", "conventional", "tensor_aware"
    storage_backend_type: str = "analytical"  # "mock", "file", "analytical"
    seed: int = 42
    queue_depth: int = 1
    block_size: int = 4096
    tokens_per_block: int = 16

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        return cls(**data)
