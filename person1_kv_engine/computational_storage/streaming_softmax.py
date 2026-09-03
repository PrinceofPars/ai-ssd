"""Numerically Stable Online Streaming Softmax Merger.

Based on the FlashAttention online softmax formulation, this module merges
partial attention results from Host RAM (Hot tokens) and SSD Controller (Cold Top-k tokens)
with zero numerical overflow or precision loss.
"""

from typing import Tuple
import numpy as np


class OnlineSoftmaxAccumulator:
    """Maintains running FlashAttention-style online softmax state across token partitions."""

    def __init__(self, num_heads: int, head_dim: int, dtype: np.dtype = np.float32):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Running max logit per head: [num_heads]
        self.m = np.full((num_heads,), -1e30, dtype=dtype)
        # Running sum of exponentials per head: [num_heads]
        self.l = np.zeros((num_heads,), dtype=dtype)
        # Running accumulated context vector: [num_heads, head_dim]
        self.acc = np.zeros((num_heads, head_dim), dtype=dtype)

    def update_with_partition(
        self,
        logits: np.ndarray,  # [num_heads, num_tokens]
        values: np.ndarray,  # [num_tokens, num_heads, head_dim]
    ) -> None:
        """Merges a new partition of tokens into the running accumulator."""
        num_tokens = logits.shape[1]
        if num_tokens == 0:
            return

        # Max logit for this partition: [num_heads]
        m_part = np.max(logits, axis=-1)
        # New combined maximum
        m_new = np.maximum(self.m, m_part)

        # Scaling factors to rescale previous running sums
        alpha_prev = np.exp(self.m - m_new)
        # Exponentials for this partition rescaled by m_new
        exp_part = np.exp(logits - m_new[:, np.newaxis])
        # Sum of exponentials for this partition: [num_heads]
        l_part = np.sum(exp_part, axis=-1)

        # Rescaled previous unnormalized context
        acc_prev_rescaled = self.acc * alpha_prev[:, np.newaxis]

        # Context vector from this partition: [num_heads, head_dim]
        # values: [num_tokens, num_heads, head_dim] -> transpose to [num_heads, num_tokens, head_dim]
        v_t = np.transpose(values, (1, 0, 2))
        acc_part = np.matmul(exp_part[:, np.newaxis, :], v_t).squeeze(axis=1)

        # Update running state
        self.m = m_new
        self.l = self.l * alpha_prev + l_part
        self.acc = acc_prev_rescaled + acc_part

    def finalize(self) -> np.ndarray:
        """Returns the final normalized attention context vector [num_heads, head_dim]."""
        safe_l = np.where(self.l > 0, self.l, 1.0)[:, np.newaxis]
        return (self.acc / safe_l).astype(self.dtype)


def merge_online_attention(
    m_a: np.ndarray,
    l_a: np.ndarray,
    out_a: np.ndarray,
    m_b: np.ndarray,
    l_b: np.ndarray,
    out_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combines two independently normalized attention results into a unified output.
    
    Args:
        m_a, m_b: Max logits per head [num_heads]
        l_a, l_b: Sum of exponentials per head [num_heads]
        out_a, out_b: Normalized context vectors [num_heads, head_dim]
        
    Returns:
        Tuple of (m_merged, l_merged, out_merged)
    """
    m_merged = np.maximum(m_a, m_b)

    d_a = np.exp(m_a - m_merged)
    d_b = np.exp(m_b - m_merged)

    l_merged = (d_a * l_a) + (d_b * l_b)
    safe_l = np.where(l_merged > 0, l_merged, 1.0)[:, np.newaxis]

    # Weighted combination of context vectors
    weighted_a = (d_a * l_a)[:, np.newaxis] * out_a
    weighted_b = (d_b * l_b)[:, np.newaxis] * out_b
    out_merged = (weighted_a + weighted_b) / safe_l

    return m_merged, l_merged, out_merged
