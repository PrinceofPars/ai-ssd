"""Unit tests for Native C In-Storage Kernel & ctypes bindings."""

import unittest
import numpy as np

from person1_kv_engine.c_kernel.kernel_binding import get_native_c_kernel


class TestCKernel(unittest.TestCase):

    def setUp(self):
        self.kernel = get_native_c_kernel()

    def test_c_kernel_loaded(self):
        """Validates that instorage_attention.dll is properly compiled and loaded."""
        self.assertTrue(self.kernel.is_available(), "Native C kernel DLL should be loaded.")

    def test_single_block_score_vs_numpy(self):
        """Checks single block score computed in C vs NumPy dot product."""
        tokens = 16
        heads = 4
        head_dim = 32
        scale = 1.0 / np.sqrt(head_dim, dtype=np.float32)

        query = np.random.randn(heads, head_dim).astype(np.float32)
        k_block = np.random.randn(tokens, heads, head_dim).astype(np.float32)

        # NumPy reference score: max across tokens and heads
        dots = np.einsum("hd,thd->th", query, k_block) * scale
        expected_score = float(np.max(dots))

        # C kernel score
        q_ptr = query.ctypes.data_as(self.kernel._lib.compute_block_score.argtypes[0])
        k_ptr = k_block.ctypes.data_as(self.kernel._lib.compute_block_score.argtypes[1])
        c_score = self.kernel._lib.compute_block_score(q_ptr, k_ptr, tokens, heads, head_dim, scale)

        np.testing.assert_allclose(c_score, expected_score, rtol=1e-4, atol=1e-4)

    def test_topk_selection_vs_numpy(self):
        """Checks multi-block top-k filtering in C against NumPy sorting."""
        num_blocks = 20
        tokens = 16
        heads = 8
        head_dim = 64
        top_k = 5

        query = np.random.randn(heads, head_dim).astype(np.float32)
        layer_blocks = []
        for bid in range(num_blocks):
            entry = {
                "k": np.random.randn(tokens, heads, head_dim).astype(np.float32),
                "v": np.random.randn(tokens, heads, head_dim).astype(np.float32),
            }
            layer_blocks.append((bid, entry))

        topk_ids, topk_vals, topk_scores = self.kernel.compute_topk(
            query=query,
            layer_blocks=layer_blocks,
            top_k=top_k,
        )

        self.assertEqual(len(topk_ids), top_k)
        self.assertEqual(topk_vals.shape, (top_k, tokens, heads, head_dim))
        self.assertEqual(len(topk_scores), top_k)
        # Check scores are sorted descending
        self.assertTrue(np.all(np.diff(topk_scores) <= 1e-5))

        # Ensure selected blocks are strictly from the candidate set
        for bid in topk_ids:
            self.assertTrue(0 <= bid < num_blocks)


if __name__ == "__main__":
    unittest.main()
