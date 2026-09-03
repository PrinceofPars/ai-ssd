"""Unit tests for Phase 1: Baseline Attention & Full In-Memory KV Cache."""

import unittest
import numpy as np

from person1_kv_engine.baseline.transformer_attention import (
    AttentionConfig,
    ScaledDotProductAttention,
    apply_rotary_pos_emb,
    precompute_rope_frequencies,
)
from person1_kv_engine.baseline.kv_cache_baseline import BaselineKVCache
from person1_kv_engine.baseline.telemetry import InferenceTelemetry


class TestBaselineAttention(unittest.TestCase):

    def setUp(self):
        self.config = AttentionConfig(
            num_layers=4,
            num_heads=8,
            num_kv_heads=8,
            head_dim=64,
            max_seq_len=2048,
            dtype=np.float32,
        )
        self.attn_engine = ScaledDotProductAttention(self.config)

    def test_rope_norm_preservation(self):
        """RoPE is an orthogonal 2D rotation: it must preserve the Euclidean norm of vectors."""
        cos, sin = precompute_rope_frequencies(head_dim=64, max_seq_len=100)
        vec = np.random.randn(8, 64).astype(np.float32)
        norm_before = np.linalg.norm(vec, axis=-1)

        vec_rotated = apply_rotary_pos_emb(vec, cos, sin, pos_offset=15)
        norm_after = np.linalg.norm(vec_rotated, axis=-1)

        np.testing.assert_allclose(norm_before, norm_after, rtol=1e-5)

    def test_causal_masking_structure(self):
        """Attention weights with causal mask must be strictly zero in upper triangular entries."""
        seq_len = 16
        q = np.random.randn(seq_len, 8, 64).astype(np.float32)
        k = np.random.randn(seq_len, 8, 64).astype(np.float32)
        v = np.random.randn(seq_len, 8, 64).astype(np.float32)

        out, weights = self.attn_engine.compute_full_attention(q, k, v, causal=True)
        self.assertEqual(out.shape, (seq_len, 8, 64))
        self.assertEqual(weights.shape, (8, seq_len, seq_len))

        # Check upper triangular is strictly 0
        for head in range(8):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    self.assertEqual(weights[head, i, j], 0.0)

    def test_decode_step_matches_full_causal_attention(self):
        """Single decode step attention must numerically match the last row of full causal attention."""
        seq_len = 32
        q = np.random.randn(seq_len, 8, 64).astype(np.float32)
        k = np.random.randn(seq_len, 8, 64).astype(np.float32)
        v = np.random.randn(seq_len, 8, 64).astype(np.float32)

        out_full, weights_full = self.attn_engine.compute_full_attention(q, k, v, causal=True)
        last_out_full = out_full[-1]  # [num_heads, head_dim]

        # Now compute single decode step for token at pos = seq_len - 1
        q_token = q[-1]
        out_decode, weights_decode = self.attn_engine.compute_decode_step(q_token, k, v)

        np.testing.assert_allclose(last_out_full, out_decode, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(weights_full[:, -1, :], weights_decode, rtol=1e-5, atol=1e-5)

    def test_kv_cache_baseline_memory_scaling(self):
        """KV cache memory tracking must scale strictly linearly O(N) with context length."""
        cache = BaselineKVCache(self.config)

        # Prefill 64 tokens
        seq_len = 64
        for l in range(self.config.num_layers):
            k = np.random.randn(seq_len, 8, 64).astype(np.float32)
            v = np.random.randn(seq_len, 8, 64).astype(np.float32)
            cache.prefill(l, k, v)

        expected_bytes_64 = 64 * (2 * 4 * 8 * 64 * 4)  # 64 tokens * 2(K+V) * 4 layers * 8 heads * 64 dim * 4 bytes
        self.assertEqual(cache.get_memory_usage_bytes(), expected_bytes_64)

        # Append 16 tokens
        for _ in range(16):
            for l in range(self.config.num_layers):
                k_tok = np.random.randn(8, 64).astype(np.float32)
                v_tok = np.random.randn(8, 64).astype(np.float32)
                cache.append_token(l, k_tok, v_tok)

        expected_bytes_80 = 80 * (2 * 4 * 8 * 64 * 4)
        self.assertEqual(cache.get_memory_usage_bytes(), expected_bytes_80)
        self.assertEqual(cache.current_tokens, 80)


if __name__ == "__main__":
    unittest.main()
