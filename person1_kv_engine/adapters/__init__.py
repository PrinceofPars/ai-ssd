"""Model Adapters and State Providers for Architecture-Adaptive AI-SSD."""

from person1_kv_engine.adapters.descriptor import (
    ModelArchitectureConfig,
    StateDescriptor,
    StateType,
    ResidencyTier,
    CompatibilityLevel,
)
from person1_kv_engine.adapters.state_provider import StateProvider
from person1_kv_engine.adapters.transformer_kv import TransformerKVStateProvider
from person1_kv_engine.adapters.sliding_window_kv import SlidingWindowKVStateProvider
from person1_kv_engine.adapters.hybrid_state import HybridStateProvider
from person1_kv_engine.adapters.model_adapter import (
    ModelAdapter,
    QwenAdapter,
    MistralAdapter,
    HybridJambaAdapter,
)
from person1_kv_engine.adapters.registry import (
    ModelRegistry,
    KNOWN_MODELS,
)

__all__ = [
    "ModelArchitectureConfig",
    "StateDescriptor",
    "StateType",
    "ResidencyTier",
    "CompatibilityLevel",
    "StateProvider",
    "TransformerKVStateProvider",
    "SlidingWindowKVStateProvider",
    "HybridStateProvider",
    "ModelAdapter",
    "QwenAdapter",
    "MistralAdapter",
    "HybridJambaAdapter",
    "ModelRegistry",
    "KNOWN_MODELS",
]
