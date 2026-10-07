"""Generic Model and State Descriptors for Architecture-Adaptive AI-SSD.

Codifies model configuration metadata, state descriptors, and compatibility levels
across Transformer architectures (MHA, GQA, MQA, sliding-window) and hybrid
Attention + SSM / recurrent models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple


class CompatibilityLevel(str, Enum):
    FULL = "FULL"          # AI-SSD state path is fully exercised (KV offloaded to storage)
    PARTIAL = "PARTIAL"    # Reusable state is split (e.g. Attention KV -> AI-SSD, SSM state -> DRAM)
    UNSUPPORTED = "UNSUPPORTED"  # Model cannot currently execute through AI-SSD state path


class StateType(str, Enum):
    ATTENTION_KV = "attention_kv"
    SLIDING_WINDOW_KV = "sliding_window_kv"
    SSM = "ssm"
    RECURRENT = "recurrent"
    UNKNOWN = "unknown"


class ResidencyTier(str, Enum):
    AI_SSD = "ai_ssd"       # Offloaded to AI-SSD flash storage
    HOST_DRAM = "host_dram" # Retained resident in host DRAM


@dataclass
class StateDescriptor:
    """Model-independent descriptor for a slice or block of reusable state."""
    state_type: StateType
    layer_id: int
    residency: ResidencyTier
    tensor_shape: Tuple[int, ...]
    dtype: str
    byte_size: int
    layout: str = "tokens_heads_dim"  # e.g. [tokens, kv_heads, head_dim]
    block_size: int = 16              # tokens per block
    sequence_range: Optional[Tuple[int, int]] = None
    retrieval_requirements: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_type": self.state_type.value,
            "layer_id": self.layer_id,
            "residency": self.residency.value,
            "tensor_shape": list(self.tensor_shape),
            "dtype": self.dtype,
            "byte_size": self.byte_size,
            "layout": self.layout,
            "block_size": self.block_size,
            "sequence_range": list(self.sequence_range) if self.sequence_range else None,
            "retrieval_requirements": self.retrieval_requirements,
        }


@dataclass
class ModelArchitectureConfig:
    """Unified architectural specification for LLM models."""
    model_id: str
    architecture: str                  # e.g., "qwen2", "mistral", "llama", "jamba", "mamba"
    model_family: str                  # e.g., "transformer", "hybrid_transformer_ssm", "ssm"
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    default_precision: str             # "fp32" or "fp16"
    supported_precisions: List[str]    # ["fp32", "float32"] etc.
    supported_contexts: List[int]
    attention_type: str                # "GQA", "MHA", "MQA"
    has_separable_kv_cache: bool = True
    sliding_window: Optional[int] = None
    layer_types: Optional[List[str]] = None  # Per-layer type for hybrid models: ["ssm", "attention", ...]
    compatibility_level: CompatibilityLevel = CompatibilityLevel.FULL
    compatibility_reason: str = "Supported"
    param_count: Optional[str] = None
    expected_tokens_seed42: Optional[List[int]] = None
    dense_peak_rss_mb: Optional[float] = None

    @property
    def gqa_ratio(self) -> int:
        if self.num_key_value_heads > 0:
            return max(1, self.num_attention_heads // self.num_key_value_heads)
        return 1

    def is_layer_attention(self, layer_idx: int) -> bool:
        if self.layer_types is None:
            return True
        if 0 <= layer_idx < len(self.layer_types):
            return self.layer_types[layer_idx] in ("attention", "self_attn", "attn", "full_attention")
        return True

    def is_layer_ssm(self, layer_idx: int) -> bool:
        if self.layer_types is None:
            return False
        if 0 <= layer_idx < len(self.layer_types):
            return self.layer_types[layer_idx] in ("ssm", "mamba", "linear_attention")
        return False
