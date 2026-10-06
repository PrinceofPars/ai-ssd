"""Model Adapter Interface and Concrete Implementations for AI-SSD.

Encapsulates model forward patching, positional embeddings, projection shapes,
and state provider lifecycle across Qwen, Mistral, LLaMA, and Hybrid architectures.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Callable
import time
import math
import torch
import numpy as np

from person1_kv_engine.adapters.descriptor import (
    ModelArchitectureConfig,
    CompatibilityLevel,
    StateType,
)
from person1_kv_engine.adapters.state_provider import StateProvider
from person1_kv_engine.adapters.transformer_kv import TransformerKVStateProvider
from person1_kv_engine.adapters.sliding_window_kv import SlidingWindowKVStateProvider
from person1_kv_engine.adapters.hybrid_state import HybridStateProvider


class ModelAdapter(ABC):
    """Abstract model adapter defining the interface between model architectures and AI-SSD."""

    def __init__(self, config: ModelArchitectureConfig):
        self.config = config

    @abstractmethod
    def create_state_provider(
        self,
        backend: Any,
        top_k_pct: float = 10.0,
        enable_computational_storage: bool = False,
        enable_prefetch: bool = True,
        enable_async_pipeline: bool = False,
    ) -> StateProvider:
        """Instantiates the appropriate StateProvider for this architecture."""
        pass

    @abstractmethod
    def wrap_model_for_aissd(
        self,
        model: torch.nn.Module,
        state_provider: StateProvider,
        model_timings: Dict[str, float],
        attn_forward_durations: List[float],
    ) -> Tuple[Dict[int, Callable], Callable[[], None]]:
        """Wraps the model's inner layers to route attention/state through the state provider.
        
        Returns:
            (original_forwards, restore_func)
        """
        pass


class QwenAdapter(ModelAdapter):
    """Adapter for Qwen-family models (Qwen2, Qwen2.5, Qwen3)."""

    def create_state_provider(
        self,
        backend: Any,
        top_k_pct: float = 10.0,
        enable_computational_storage: bool = False,
        enable_prefetch: bool = True,
        enable_async_pipeline: bool = False,
    ) -> StateProvider:
        return TransformerKVStateProvider(
            config=self.config,
            backend=backend,
            tokens_per_block=16,
            top_k_pct=top_k_pct,
            enable_computational_storage=enable_computational_storage,
            enable_prefetch=enable_prefetch,
            enable_async_pipeline=enable_async_pipeline,
        )

    def wrap_model_for_aissd(
        self,
        model: torch.nn.Module,
        state_provider: StateProvider,
        model_timings: Dict[str, float],
        attn_forward_durations: List[float],
    ) -> Tuple[Dict[int, Callable], Callable[[], None]]:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        orig_forwards = {}
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            orig_forwards[i] = attn.forward

            def make_aissd_forward(layer_idx: int, original_fwd: Any):
                def forward(hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor] = None, past_key_values: Optional[Any] = None, **kwargs):
                    if not state_provider.is_active or hidden_states.shape[1] > 1:
                        return original_fwd(hidden_states, position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)

                    t_attn_fwd_start = time.perf_counter()
                    attn_module = model.model.layers[layer_idx].self_attn
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, attn_module.head_dim)

                    t_qkv = time.perf_counter()
                    q_raw = attn_module.q_proj(hidden_states).view(hidden_shape)
                    k_raw = attn_module.k_proj(hidden_states).view(hidden_shape)
                    if hasattr(attn_module, "q_norm"):
                        q_raw = attn_module.q_norm(q_raw)
                    if hasattr(attn_module, "k_norm"):
                        k_raw = attn_module.k_norm(k_raw)

                    q = q_raw.transpose(1, 2)
                    k = k_raw.transpose(1, 2)
                    v = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    model_timings["qkv_proj_s"] += time.perf_counter() - t_qkv

                    t_rope = time.perf_counter()
                    cos, sin = position_embeddings
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    model_timings["rope_s"] += time.perf_counter() - t_rope

                    state_provider.append_new_token(layer_idx, k, v)
                    act_k, act_v = state_provider.select_and_fetch_active_state(layer_idx, query=q)

                    t_attn = time.perf_counter()
                    scaling = attn_module.scaling
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q, act_k, act_v, scale=scaling, enable_gqa=True
                    )
                    out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                    model_timings["attn_matmul_s"] += time.perf_counter() - t_attn

                    t_out = time.perf_counter()
                    out = attn_module.o_proj(out)
                    model_timings["out_proj_s"] += time.perf_counter() - t_out

                    attn_forward_durations.append(time.perf_counter() - t_attn_fwd_start)
                    return out, None

                return forward

            attn.forward = make_aissd_forward(i, orig_forwards[i])

        def restore():
            for i, layer in enumerate(model.model.layers):
                layer.self_attn.forward = orig_forwards[i]

        return orig_forwards, restore


class MistralAdapter(ModelAdapter):
    """Adapter for Mistral-family models with GQA and optional sliding-window attention."""

    def create_state_provider(
        self,
        backend: Any,
        top_k_pct: float = 10.0,
        enable_computational_storage: bool = False,
        enable_prefetch: bool = True,
        enable_async_pipeline: bool = False,
    ) -> StateProvider:
        if self.config.sliding_window is not None:
            return SlidingWindowKVStateProvider(
                config=self.config,
                backend=backend,
                tokens_per_block=16,
                top_k_pct=top_k_pct,
                enable_computational_storage=enable_computational_storage,
                enable_prefetch=enable_prefetch,
                enable_async_pipeline=enable_async_pipeline,
            )
        return TransformerKVStateProvider(
            config=self.config,
            backend=backend,
            tokens_per_block=16,
            top_k_pct=top_k_pct,
            enable_computational_storage=enable_computational_storage,
            enable_prefetch=enable_prefetch,
            enable_async_pipeline=enable_async_pipeline,
        )

    def wrap_model_for_aissd(
        self,
        model: torch.nn.Module,
        state_provider: StateProvider,
        model_timings: Dict[str, float],
        attn_forward_durations: List[float],
    ) -> Tuple[Dict[int, Callable], Callable[[], None]]:
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

        orig_forwards = {}
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            orig_forwards[i] = attn.forward

            def make_aissd_forward(layer_idx: int, original_fwd: Any):
                def forward(hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor] = None, past_key_values: Optional[Any] = None, **kwargs):
                    if not state_provider.is_active or hidden_states.shape[1] > 1:
                        return original_fwd(hidden_states, position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)

                    t_attn_fwd_start = time.perf_counter()
                    attn_module = model.model.layers[layer_idx].self_attn
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, attn_module.head_dim)

                    t_qkv = time.perf_counter()
                    q = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    k = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    v = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    model_timings["qkv_proj_s"] += time.perf_counter() - t_qkv

                    t_rope = time.perf_counter()
                    cos, sin = position_embeddings
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    model_timings["rope_s"] += time.perf_counter() - t_rope

                    state_provider.append_new_token(layer_idx, k, v)
                    act_k, act_v = state_provider.select_and_fetch_active_state(layer_idx, query=q)

                    t_attn = time.perf_counter()
                    scaling = attn_module.scaling
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q, act_k, act_v, scale=scaling, enable_gqa=True
                    )
                    out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                    model_timings["attn_matmul_s"] += time.perf_counter() - t_attn

                    t_out = time.perf_counter()
                    out = attn_module.o_proj(out)
                    model_timings["out_proj_s"] += time.perf_counter() - t_out

                    attn_forward_durations.append(time.perf_counter() - t_attn_fwd_start)
                    return out, None

                return forward

            attn.forward = make_aissd_forward(i, orig_forwards[i])

        def restore():
            for i, layer in enumerate(model.model.layers):
                layer.self_attn.forward = orig_forwards[i]

        return orig_forwards, restore


class HybridJambaAdapter(ModelAdapter):
    """Adapter for hybrid architectures like Jamba (Attention + Mamba/SSM)."""

    def create_state_provider(
        self,
        backend: Any,
        top_k_pct: float = 10.0,
        enable_computational_storage: bool = False,
        enable_prefetch: bool = True,
        enable_async_pipeline: bool = False,
    ) -> StateProvider:
        return HybridStateProvider(
            config=self.config,
            backend=backend,
            tokens_per_block=16,
            top_k_pct=top_k_pct,
            enable_computational_storage=enable_computational_storage,
            enable_prefetch=enable_prefetch,
            enable_async_pipeline=enable_async_pipeline,
        )

    def wrap_model_for_aissd(
        self,
        model: torch.nn.Module,
        state_provider: StateProvider,
        model_timings: Dict[str, float],
        attn_forward_durations: List[float],
    ) -> Tuple[Dict[int, Callable], Callable[[], None]]:
        orig_forwards = {}
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn") and layer.self_attn is not None:
                attn = layer.self_attn
                orig_forwards[i] = attn.forward

                def make_hybrid_attn_fwd(layer_idx: int, original_fwd: Any):
                    def forward(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, past_key_values: Optional[Any] = None, **kwargs):
                        if not state_provider.is_active or hidden_states.shape[1] > 1:
                            return original_fwd(hidden_states, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)

                        t_attn_fwd_start = time.perf_counter()
                        attn_module = model.model.layers[layer_idx].self_attn
                        input_shape = hidden_states.shape[:-1]
                        hidden_shape = (*input_shape, -1, attn_module.head_dim)

                        t_qkv = time.perf_counter()
                        q = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        k = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        v = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        model_timings["qkv_proj_s"] += time.perf_counter() - t_qkv

                        state_provider.append_new_token(layer_idx, k, v)
                        act_k, act_v = state_provider.select_and_fetch_active_state(layer_idx, query=q)

                        t_attn = time.perf_counter()
                        scaling = attn_module.scaling
                        out = torch.nn.functional.scaled_dot_product_attention(
                            q, act_k, act_v, scale=scaling, enable_gqa=True
                        )
                        out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                        model_timings["attn_matmul_s"] += time.perf_counter() - t_attn

                        t_out = time.perf_counter()
                        out = attn_module.o_proj(out)
                        model_timings["out_proj_s"] += time.perf_counter() - t_out

                        attn_forward_durations.append(time.perf_counter() - t_attn_fwd_start)
                        return out, None

                    return forward

                attn.forward = make_hybrid_attn_fwd(i, orig_forwards[i])

        def restore():
            for i, layer in enumerate(model.model.layers):
                if i in orig_forwards:
                    layer.self_attn.forward = orig_forwards[i]

        return orig_forwards, restore
