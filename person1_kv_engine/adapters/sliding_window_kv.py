"""Sliding Window KV State Provider for models with local attention windows (e.g. Mistral).

Distinguishes:
  - Active Attention Window: historical tokens inside sliding window
  - Stored KV: blocks partitioned across sliding window
  - Out-of-window evicted/skipped state
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import math
import time
import numpy as np
import torch

from person1_kv_engine.adapters.descriptor import (
    StateDescriptor,
    StateType,
    ResidencyTier,
    ModelArchitectureConfig,
)
from person1_kv_engine.adapters.transformer_kv import TransformerKVStateProvider


class SlidingWindowKVStateProvider(TransformerKVStateProvider):
    """Window-aware StateProvider that enforces attention window boundaries."""

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
        super().__init__(
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
        self.sliding_window = config.sliding_window or 4096

    def inspect_state(self, past_key_values: Any) -> List[StateDescriptor]:
        descriptors = super().inspect_state(past_key_values)
        for d in descriptors:
            d.state_type = StateType.SLIDING_WINDOW_KV
            d.retrieval_requirements["sliding_window"] = self.sliding_window
        return descriptors

    def select_and_fetch_active_state(
        self,
        layer_idx: int,
        query: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filters candidate blocks to only those falling within the effective sliding window."""
        ld = self.layer_data[layer_idx]
        total_tokens = ld["total_tokens"]

        # If sequence length exceeds sliding window, filter candidate blocks outside the window
        orig_cand_blocks = ld["candidate_blocks"]
        if total_tokens > self.sliding_window:
            window_start = total_tokens - self.sliding_window
            # Candidate block b covers tokens [sink_tokens + b*tok_per_block, sink_tokens + (b+1)*tok_per_block]
            valid_cands = []
            for bid, act_tok in orig_cand_blocks:
                b_tok_start = self.sink_tokens + bid * self.tokens_per_block
                b_tok_end = b_tok_start + act_tok
                if b_tok_end > window_start:
                    valid_cands.append((bid, act_tok))
            ld["candidate_blocks"] = valid_cands

        try:
            return super().select_and_fetch_active_state(layer_idx=layer_idx, query=query, **kwargs)
        finally:
            ld["candidate_blocks"] = orig_cand_blocks
