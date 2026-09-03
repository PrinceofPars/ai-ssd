"""Attention computation layer abstraction.

Implements multi-head scaled dot-product attention, causal masking,
and Rotary Position Embedding (RoPE).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from common.schemas.kv_block import KVBlock
from person1_kv_engine.attention.scoring import AttentionScorer


def precompute_rope_frequencies(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Precomputes cosine and sine tables for Rotary Position Embedding."""
    dim_half = head_dim // 2
    freq_indices = np.arange(0, dim_half, dtype=np.float32)
    inv_freq = 1.0 / (theta ** (freq_indices / dim_half))
    positions = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(positions, inv_freq)
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return cos, sin


def apply_rotary_pos_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, pos_offset: int = 0) -> np.ndarray:
    """Applies RoPE to query or key tensor."""
    is_single_token = (x.ndim == 2)
    if is_single_token:
        x = x[np.newaxis, ...]

    seq_len, num_heads, head_dim = x.shape
    dim_half = head_dim // 2

    x1 = x[..., :dim_half]
    x2 = x[..., dim_half:]

    cos_slice = cos[pos_offset : pos_offset + seq_len, np.newaxis, :]
    sin_slice = sin[pos_offset : pos_offset + seq_len, np.newaxis, :]

    rot_x1 = x1 * cos_slice - x2 * sin_slice
    rot_x2 = x1 * sin_slice + x2 * cos_slice

    out = np.concatenate([rot_x1, rot_x2], axis=-1)
    if is_single_token:
        return out[0]
    return out


class AttentionEngine:
    def __init__(self, head_dim: int = 128, max_seq_len: int = 65536):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.scale = 1.0 / np.sqrt(head_dim, dtype=np.float32)
        self.scorer = AttentionScorer(head_dim=head_dim)
        self.cos, self.sin = precompute_rope_frequencies(head_dim, max_seq_len)

    def compute_attention(
        self,
        query: Any,
        blocks: List[KVBlock],
    ) -> Dict[int, float]:
        return self.scorer.score_blocks(blocks, query_vector=query)

    def compute_full_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        causal: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes batched multi-head attention for prompt prefill."""
        seq_len, num_heads, head_dim = q.shape

        q_t = np.transpose(q, (1, 0, 2))
        k_t = np.transpose(k, (1, 0, 2))
        v_t = np.transpose(v, (1, 0, 2))

        scores = np.matmul(q_t, np.transpose(k_t, (0, 2, 1))) * self.scale

        if causal and seq_len > 1:
            mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
            scores = scores + mask[np.newaxis, :, :]

        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
        attn_weights = exp_scores / (sum_exp + 1e-12)

        out_t = np.matmul(attn_weights, v_t)
        out = np.transpose(out_t, (1, 0, 2))
        return out, attn_weights

    def compute_decode_step(
        self,
        q_token: np.ndarray,
        k_history: np.ndarray,
        v_history: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes single-token decode attention against cached history."""
        q_expanded = q_token[:, np.newaxis, :]
        k_t = np.transpose(k_history, (1, 2, 0))
        v_t = np.transpose(v_history, (1, 0, 2))

        logits = np.matmul(q_expanded, k_t) * self.scale
        logits = logits.squeeze(axis=1)

        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        attn_weights = exp_logits / (sum_exp + 1e-12)

        attn_expanded = attn_weights[:, np.newaxis, :]
        out = np.matmul(attn_expanded, v_t).squeeze(axis=1)
        return out, attn_weights
