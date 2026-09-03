"""Transformer Multi-Head Attention, Rotary Position Embedding, and Config."""

from dataclasses import dataclass
from typing import Tuple, Optional, Any
import numpy as np
from person1_kv_engine.attention.attention import (
    precompute_rope_frequencies,
    apply_rotary_pos_emb,
    AttentionEngine,
)


@dataclass
class AttentionConfig:
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    head_dim: int = 64
    max_seq_len: int = 2048
    dtype: Any = np.float32

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

    @property
    def d_model(self) -> int:
        return self.num_heads * self.head_dim


class ScaledDotProductAttention:
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.engine = AttentionEngine(head_dim=config.head_dim, max_seq_len=config.max_seq_len)

    def compute_full_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        causal: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes multi-head attention.
        Args:
            q, k, v: [seq_len, num_heads, head_dim]
        Returns:
            out: [seq_len, num_heads, head_dim]
            weights: [num_heads, seq_len, seq_len]
        """
        return self.engine.compute_full_attention(q, k, v, causal=causal)

    def compute_decode_step(
        self,
        q_token: np.ndarray,
        k_history: np.ndarray,
        v_history: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes single-token decode attention against history.
        Args:
            q_token: [num_heads, head_dim]
            k_history, v_history: [seq_len, num_heads, head_dim]
        Returns:
            out: [num_heads, head_dim]
            weights: [num_heads, seq_len]
        """
        return self.engine.compute_decode_step(q_token, k_history, v_history)
