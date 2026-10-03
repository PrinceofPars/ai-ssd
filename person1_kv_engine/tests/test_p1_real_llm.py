"""Focused Unit Tests for P1 Real LLM Engine, Block Adapter, Trace Generator, and Top-K Evaluator."""

import unittest
import os
import json
import numpy as np

from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.block_adapter import KVBlockAdapter
from person1_kv_engine.real_llm.trace_generator import RealKVTraceGenerator
from person1_kv_engine.real_llm.topk_evaluator import TopKEvaluator
from person1_kv_engine.c_kernel.kernel_binding import get_native_c_kernel


class TestP1RealLLM(unittest.TestCase):
    """Validates real LLM components, physical blockization, and evaluation logic."""

    def setUp(self):
        self.engine = RealLLMEngine(use_mock=True)
        self.adapter = KVBlockAdapter(
            tokens_per_block=16,
            kv_heads_per_block=1,
            head_dim=64,
            dtype="FP32",
            attention_sink_tokens=4,
            recent_window_tokens=16,
        )
        self.evaluator = TopKEvaluator(
            head_dim=64,
            tokens_per_block=16,
            attention_sink_tokens=4,
            recent_window_tokens=16,
            bytes_per_elem=4,
        )
        self.kernel = get_native_c_kernel()

    def test_byte_size_calculation(self):
        """Validates that 16 tokens x 1 head x 64 dim x 4 bytes = 4096 bytes per page."""
        self.assertEqual(self.adapter.key_size_bytes, 4096)
        self.assertEqual(self.adapter.value_size_bytes, 4096)
        self.assertEqual(self.adapter.logical_block_bytes, 8192)
        self.assertEqual(self.adapter.key_flash_pages, 1)
        self.assertEqual(self.adapter.value_flash_pages, 1)
        self.assertEqual(self.adapter.total_flash_pages, 2)

    def test_canonical_fp16_geometry(self):
        """Validates canonical 16 tokens x 1 head x 128 dim in FP16 = 4096 bytes K, 4096 bytes V."""
        fp16_adapter = KVBlockAdapter(
            tokens_per_block=16,
            kv_heads_per_block=1,
            head_dim=128,
            dtype="FP16",
        )
        self.assertEqual(fp16_adapter.key_size_bytes, 4096)
        self.assertEqual(fp16_adapter.value_size_bytes, 4096)
        self.assertEqual(fp16_adapter.logical_block_bytes, 8192)

    def test_blockization_and_kv_separation(self):
        """Validates that blockize_layer accurately segments tensors into 4096-byte buffers."""
        seq_len = 64
        kv_heads = 2
        head_dim = 64
        k_tensor = np.random.randn(kv_heads, seq_len, head_dim).astype(np.float32)
        v_tensor = np.random.randn(kv_heads, seq_len, head_dim).astype(np.float32)

        blocks = self.adapter.blockize_layer(layer_id=0, k_tensor=k_tensor, v_tensor=v_tensor)
        # 64 tokens / 16 = 4 blocks per head * 2 heads = 8 blocks
        self.assertEqual(len(blocks), 8)

        for block_desc, payload in blocks:
            self.assertEqual(block_desc.key_size_bytes, 4096)
            self.assertEqual(block_desc.value_size_bytes, 4096)
            self.assertEqual(block_desc.total_size_bytes, 8192)
            self.assertEqual(payload["k"].nbytes, 4096)
            self.assertEqual(payload["v"].nbytes, 4096)
            self.assertEqual(payload["k"].shape, (1, 16, 64))

    def test_deterministic_inference_and_extraction(self):
        """Validates deterministic output and KV shape extraction."""
        res1 = self.engine.run_inference("Test prompt for determinism", max_new_tokens=8, seed=123)
        res2 = self.engine.run_inference("Test prompt for determinism", max_new_tokens=8, seed=123)

        self.assertEqual(res1["total_tokens"], res2["total_tokens"])
        np.testing.assert_array_equal(res1["layer_kv"][0]["k"], res2["layer_kv"][0]["k"])
        np.testing.assert_array_equal(res1["layer_kv"][0]["v"], res2["layer_kv"][0]["v"])

    def test_trace_generation_and_deserialization(self):
        """Validates JSONL trace serialization and field integrity."""
        inf_res = self.engine.run_inference("Prompt for trace test", max_new_tokens=4, seed=42)
        layer_blocks = self.adapter.blockize_all_layers(inf_res["layer_kv"])

        tmp_dir = "/tmp/test_trace_dir"
        os.makedirs(tmp_dir, exist_ok=True)
        trace_gen = RealKVTraceGenerator(output_dir=tmp_dir)

        trace_path, manifest_path, sha_digest = trace_gen.generate_trace_from_run(
            inference_result=inf_res,
            layer_blocks=layer_blocks,
            workload_name="unit_test_trace",
            top_k_percent=10.0,
        )

        self.assertTrue(os.path.exists(trace_path))
        self.assertTrue(os.path.exists(manifest_path))
        self.assertTrue(len(sha_digest) == 64)

        # Deserialization test
        with open(trace_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 0)

        first_event = json.loads(lines[0])
        self.assertIn("query_id", first_event)
        self.assertIn("layer_id", first_event)
        self.assertIn("head_id", first_event)
        self.assertIn("block_id", first_event)
        self.assertIn("operation", first_event)
        self.assertIn("byte_size", first_event)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["sha256_checksum"], sha_digest)

    def test_topk_sparse_vs_dense_comparison(self):
        """Validates Top-k evaluator metrics against dense reference."""
        rng = np.random.RandomState(42)
        seq_len = 128
        q_heads = 8
        kv_heads = 2
        head_dim = 64

        query = rng.randn(q_heads, head_dim).astype(np.float32)
        k_seq = rng.randn(kv_heads, seq_len, head_dim).astype(np.float32)
        v_seq = rng.randn(kv_heads, seq_len, head_dim).astype(np.float32)

        results = self.evaluator.evaluate_query(
            query=query,
            k_seq=k_seq,
            v_seq=v_seq,
            sparsity_levels=[5.0, 10.0, 20.0, 50.0],
        )

        evals = results["sparsity_evaluations"]
        # Higher sparsity budget should yield higher or equal cosine similarity
        cos_5 = evals["5.0%"]["mean_cosine_similarity"]
        cos_50 = evals["50.0%"]["mean_cosine_similarity"]
        self.assertGreater(cos_50, 0.70)
        self.assertLessEqual(cos_5, cos_50 + 0.15)

        # Traffic reduction should decrease as sparsity budget increases
        self.assertGreater(evals["5.0%"]["pcie_traffic_reduction_percent"], evals["50.0%"]["pcie_traffic_reduction_percent"])

    def test_native_c_kernel_correctness(self):
        """Validates native C kernel numerical accuracy vs NumPy reference."""
        if not self.kernel.is_available():
            self.skipTest("Native C kernel library not loaded.")

        tokens = 16
        heads = 4
        head_dim = 64
        scale = float(1.0 / np.sqrt(head_dim))
        rng = np.random.RandomState(99)

        query = rng.randn(heads, head_dim).astype(np.float32)
        k_block = rng.randn(tokens, heads, head_dim).astype(np.float32)

        ref_dots = np.einsum("hd,thd->th", query, k_block) * scale
        ref_score = float(np.max(ref_dots))

        q_ptr = np.ascontiguousarray(query).ctypes.data_as(self.kernel._lib.compute_block_score.argtypes[0])
        k_ptr = np.ascontiguousarray(k_block).ctypes.data_as(self.kernel._lib.compute_block_score.argtypes[1])
        c_score = float(self.kernel._lib.compute_block_score(q_ptr, k_ptr, tokens, heads, head_dim, scale))

        self.assertAlmostEqual(c_score, ref_score, places=4)


if __name__ == "__main__":
    unittest.main()
