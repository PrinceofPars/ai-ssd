"""Hybrid State Provider for models combining Attention and SSM / Mamba layers.

Executes selective state management:
  - Attention layers: KVStateProvider -> blockized and offloaded to AI-SSD flash
  - SSM / Recurrent layers: Retained resident in host DRAM
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import torch

from person1_kv_engine.adapters.descriptor import (
    StateDescriptor,
    StateType,
    ResidencyTier,
    ModelArchitectureConfig,
)
from person1_kv_engine.adapters.state_provider import StateProvider
from person1_kv_engine.adapters.transformer_kv import TransformerKVStateProvider


class HybridStateProvider(StateProvider):
    """Selective state provider routing Attention layers to AI-SSD and SSM layers to DRAM."""

    def __init__(
        self,
        config: ModelArchitectureConfig,
        backend: Any,
        sink_tokens: int = 4,
        recent_tokens: int = 16,
        tokens_per_block: int = 16,
        top_k_pct: float = 10.0,
        enable_computational_storage: bool = False,
        enable_prefetch: bool = True,
        enable_async_pipeline: bool = False,
    ):
        self.attention_provider = TransformerKVStateProvider(
            config=config,
            backend=backend,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            tokens_per_block=tokens_per_block,
            top_k_pct=top_k_pct,
            enable_computational_storage=enable_computational_storage,
            enable_prefetch=enable_prefetch,
            enable_async_pipeline=enable_async_pipeline,
        )
        super().__init__(
            config=config,
            backend=backend,
            tokens_per_block=tokens_per_block,
            top_k_pct=top_k_pct,
            enable_computational_storage=enable_computational_storage,
            enable_prefetch=enable_prefetch,
            enable_async_pipeline=enable_async_pipeline,
        )
        # Resident DRAM storage for SSM / recurrent layer states
        self.resident_ssm_states: Dict[int, Dict[str, Any]] = {}

    @property
    def is_active(self) -> bool:
        return self.attention_provider.is_active

    @is_active.setter
    def is_active(self, val: bool) -> None:
        self.attention_provider.is_active = val

    @property
    def timings(self) -> Dict[str, float]:
        return self.attention_provider.timings

    @property
    def candidate_k_bytes_to_host(self) -> int:
        return self.attention_provider.candidate_k_bytes_to_host

    @property
    def winning_k_bytes_to_host(self) -> int:
        return self.attention_provider.winning_k_bytes_to_host

    @property
    def winning_v_bytes_to_host(self) -> int:
        return self.attention_provider.winning_v_bytes_to_host

    @property
    def topk_metadata_bytes_to_host(self) -> int:
        return self.attention_provider.topk_metadata_bytes_to_host

    def reset_timings(self) -> None:
        self.attention_provider.reset_timings()

    def inspect_state(self, native_state: Any) -> List[StateDescriptor]:
        descriptors: List[StateDescriptor] = []
        if hasattr(native_state, "layers"):
            layers = native_state.layers
        else:
            layers = native_state if isinstance(native_state, list) else []

        for i, layer in enumerate(layers):
            if hasattr(layer, "keys") and layer.keys is not None:
                # Attention layer
                k_shape = layer.keys.shape
                dtype_str = "fp16" if layer.keys.dtype in (torch.float16, torch.bfloat16) else "fp32"
                b_bytes = self.tokens_per_block * self.config.num_key_value_heads * self.config.head_dim * (2 if dtype_str == "fp16" else 4)
                descriptors.append(
                    StateDescriptor(
                        state_type=StateType.ATTENTION_KV,
                        layer_id=i,
                        residency=ResidencyTier.AI_SSD,
                        tensor_shape=(self.tokens_per_block, self.config.num_key_value_heads, self.config.head_dim),
                        dtype=dtype_str,
                        byte_size=b_bytes * 2,
                        layout="tokens_heads_dim",
                        block_size=self.tokens_per_block,
                    )
                )
            else:
                # SSM layer (LinearAttentionLayer / conv_states / recurrent_states)
                conv_shape = tuple(layer.conv_states.shape) if hasattr(layer, "conv_states") and layer.conv_states is not None else (0,)
                descriptors.append(
                    StateDescriptor(
                        state_type=StateType.SSM,
                        layer_id=i,
                        residency=ResidencyTier.HOST_DRAM,
                        tensor_shape=conv_shape,
                        dtype="fp32",
                        byte_size=1024,
                        layout="ssm_state",
                        block_size=0,
                        retrieval_requirements={"resident": True},
                    )
                )
        return descriptors

    def init_from_prefill(self, native_state: Any) -> None:
        # Separate attention layers and SSM layers
        if hasattr(native_state, "layers"):
            attn_layers = {}
            for i, layer in enumerate(native_state.layers):
                if hasattr(layer, "keys") and layer.keys is not None:
                    attn_layers[i] = (layer.keys, layer.values)
                else:
                    self.resident_ssm_states[i] = {
                        "conv_states": getattr(layer, "conv_states", None),
                        "recurrent_states": getattr(layer, "recurrent_states", None),
                    }
        else:
            attn_layers = native_state

        # Initialize attention provider on attention layers
        self.attention_provider.init_from_prefill(attn_layers)
        self.is_active = True

    def append_new_token(self, layer_idx: int, *state_tensors: Any) -> None:
        if self.config.is_layer_attention(layer_idx):
            self.attention_provider.append_new_token(layer_idx, *state_tensors)
        else:
            # SSM state updates remain in DRAM
            if layer_idx in self.resident_ssm_states:
                self.resident_ssm_states[layer_idx]["updated"] = True

    def select_and_fetch_active_state(
        self,
        layer_idx: int,
        query: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, ...]:
        if self.config.is_layer_attention(layer_idx):
            return self.attention_provider.select_and_fetch_active_state(layer_idx, query=query, **kwargs)
        else:
            # Return resident SSM state reference
            return (self.resident_ssm_states.get(layer_idx),)

    def get_memory_stats(self) -> Dict[str, Any]:
        stats = self.attention_provider.get_memory_stats()
        # Add resident SSM memory
        ssm_mb = len(self.resident_ssm_states) * 0.1  # Approx resident SSM overhead
        stats["active_dram_mb"] += ssm_mb
        stats["ssm_resident_layers"] = len(self.resident_ssm_states)
        return stats
