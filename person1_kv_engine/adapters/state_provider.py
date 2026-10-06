"""Abstract Base StateProvider for Architecture-Adaptive Model State Management.

Defines the contract for extracting, blockifying, offloading, retrieving,
and reconstructing model states (KV cache, sliding-window KV, hybrid states).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch

from person1_kv_engine.adapters.descriptor import StateDescriptor, ModelArchitectureConfig


class StateProvider(ABC):
    """Abstract provider managing the conversion between model native state and AI-SSD storage."""

    def __init__(
        self,
        config: ModelArchitectureConfig,
        backend: Any,
        tokens_per_block: int = 16,
        top_k_pct: float = 10.0,
        enable_computational_storage: bool = False,
        enable_prefetch: bool = True,
        enable_async_pipeline: bool = False,
    ):
        self.config = config
        self.backend = backend
        self.tokens_per_block = tokens_per_block
        self.top_k_pct = top_k_pct
        self.enable_computational_storage = enable_computational_storage
        self.enable_prefetch = enable_prefetch
        self.enable_async_pipeline = enable_async_pipeline
        self.is_active = False

    @abstractmethod
    def inspect_state(self, native_state: Any) -> List[StateDescriptor]:
        """Inspects native state tensors/containers and returns model-independent descriptors."""
        pass

    @abstractmethod
    def init_from_prefill(self, native_state: Any) -> None:
        """Blockizes prefill state and offloads eligible blocks to storage backend."""
        pass

    @abstractmethod
    def append_new_token(self, layer_idx: int, *state_tensors: Any) -> None:
        """Incorporates newly generated token state for the specified layer."""
        pass

    @abstractmethod
    def select_and_fetch_active_state(
        self,
        layer_idx: int,
        query: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, ...]:
        """Retrieves and reconstructs the active working set needed for layer computation."""
        pass

    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """Returns statistics on active resident DRAM bytes vs total state bytes."""
        pass

    @abstractmethod
    def reset_timings(self) -> None:
        """Resets operation timers and traffic counters."""
        pass
