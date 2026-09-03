"""Workload Generator: Generates synthetic prompt tokens and autoregressive decode traces."""

from typing import List, Dict, Any, Tuple
import numpy as np


class WorkloadGenerator:
    def __init__(self, context_length: int = 32768, layers: int = 32, heads: int = 32, head_dim: int = 128):
        self.context_length = context_length
        self.layers = layers
        self.heads = heads
        self.head_dim = head_dim

    def generate_decode_step(self, step: int, current_length: int) -> Dict[str, Any]:
        """Generates simulated attention access request for one decoding step."""
        return {
            "step": step,
            "context_length": current_length,
            "query_token_id": current_length,
            "layers": self.layers,
            "heads": self.heads,
        }

    def generate_synthetic_tensors(
        self,
        seq_len: int,
        num_topics: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Generates prompt Key and Value tensors with realistic semantic topic clusters."""
        scale_factor = np.sqrt(float(self.head_dim))
        topics = [
            (np.random.randn(self.heads, self.head_dim).astype(np.float32) / np.sqrt(self.head_dim)) * scale_factor
            for _ in range(num_topics)
        ]

        k_seq = np.random.randn(seq_len, self.heads, self.head_dim).astype(np.float32) * 0.3
        v_seq = np.random.randn(seq_len, self.heads, self.head_dim).astype(np.float32)

        # Plant topic clusters
        for i in range(seq_len // 16):
            t_idx = i % num_topics
            k_seq[i * 16 : (i + 1) * 16] += topics[t_idx] * 0.8

        query = topics[0] + np.random.randn(self.heads, self.head_dim).astype(np.float32) * 0.2
        return k_seq, v_seq, query, topics
