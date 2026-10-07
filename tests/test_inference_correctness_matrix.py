"""
AI-SSD End-to-End Correctness & Storage Telemetry Regression Test Suite.

Validates:
1. KV block round trip across precisions (FP32, FP16) and architectures (Qwen, Mistral, Qwen3.5 sparse).
2. Multi-inference storage state isolation (no cross-contamination between consecutive runs).
3. Exact token match / semantic parity between Baseline and AI-SSD modes across supported models.
4. Candidate K bytes to host == 0 under computational storage Top-K filtering.
5. Inference-level telemetry reporting (Total Read MB, Total Write MB) with before/after delta snapshots.
"""

import sys
import os
import math
import hashlib
from pathlib import Path
import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.adapters.registry import ModelRegistry, KNOWN_MODELS, QwenAdapter
from person1_kv_engine.adapters.descriptor import ModelArchitectureConfig, CompatibilityLevel
from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.aissd_inference import (
    create_default_storage_backend,
    run_baseline_decode,
    run_aissd_decode,
)


class TestKVBlockRoundTrip:
    """Phase 4: Tensor -> Blockize -> Serialize -> Write -> Read -> Deserialize -> Reconstruct."""

    @pytest.mark.parametrize("dtype_str,np_dtype", [
        ("float32", np.float32),
        ("float16", np.float16),
    ])
    def test_roundtrip_standard_geometries(self, dtype_str, np_dtype):
        """Tests FP32 and FP16 round-trip with Qwen/Mistral tensor shapes."""
        tokens_per_block = 16
        num_kv_heads = 8
        head_dim = 128
        layer_idx = 5
        block_id = 12

        backend = create_default_storage_backend(
            channels=4,
            num_layers=16,
            num_heads=num_kv_heads,
            tokens_per_block=tokens_per_block,
            head_dim=head_dim,
            dtype=dtype_str,
            mapping_mode="tensor_aware",
            storage_mode="file",
            enable_prefetch=False,
            enable_async_pipeline=False,
        )

        rng = np.random.RandomState(42)
        orig_k = rng.randn(tokens_per_block, num_kv_heads, head_dim).astype(np_dtype)
        orig_v = rng.randn(tokens_per_block, num_kv_heads, head_dim).astype(np_dtype)

        # Write
        backend.write_block(layer_idx, block_id, orig_k, orig_v)

        # Read back
        k_back = backend.read_key_page(layer_idx, block_id)
        v_back = backend.read_value_page(layer_idx, block_id)

        assert k_back.shape == orig_k.shape, f"Key shape mismatch: {k_back.shape} vs {orig_k.shape}"
        assert v_back.shape == orig_v.shape, f"Value shape mismatch: {v_back.shape} vs {orig_v.shape}"
        assert k_back.dtype == orig_k.dtype, f"Key dtype mismatch: {k_back.dtype} vs {orig_k.dtype}"
        assert v_back.dtype == orig_v.dtype, f"Value dtype mismatch: {v_back.dtype} vs {orig_v.dtype}"
        np.testing.assert_array_equal(k_back, orig_k, err_msg="Key block numerical mismatch")
        np.testing.assert_array_equal(v_back, orig_v, err_msg="Value block numerical mismatch")

    def test_roundtrip_qwen35_sparse_layers(self):
        """Tests Qwen3.5 sparse full-attention geometry (heads=4, dim=256) at sparse layer IDs."""
        sparse_layers = [3, 7, 11, 15, 19, 23, 27, 31]
        tokens_per_block = 16
        num_kv_heads = 4
        head_dim = 256
        dtype_str = "float16"
        np_dtype = np.float16

        backend = create_default_storage_backend(
            channels=8,
            num_layers=32,
            num_heads=num_kv_heads,
            tokens_per_block=tokens_per_block,
            head_dim=head_dim,
            dtype=dtype_str,
            mapping_mode="tensor_aware",
            storage_mode="file",
            enable_prefetch=False,
            enable_async_pipeline=False,
        )

        rng = np.random.RandomState(1337)
        for l_idx in sparse_layers:
            for bid in range(4):
                k_block = rng.randn(tokens_per_block, num_kv_heads, head_dim).astype(np_dtype)
                v_block = rng.randn(tokens_per_block, num_kv_heads, head_dim).astype(np_dtype)

                backend.write_block(l_idx, bid, k_block, v_block)

                k_back = backend.read_key_page(l_idx, bid)
                v_back = backend.read_value_page(l_idx, bid)

                np.testing.assert_array_equal(k_back, k_block, err_msg=f"Layer {l_idx} block {bid} K mismatch")
                np.testing.assert_array_equal(v_back, v_block, err_msg=f"Layer {l_idx} block {bid} V mismatch")


class TestStorageStateIsolation:
    """Phase 7: Two consecutive inference runs must not contaminate each other."""

    def test_consecutive_runs_isolation(self):
        """Verifies that two consecutive inferences use clean state and produce identical results."""
        engine = RealLLMEngine(model_name="Qwen/Qwen2.5-0.5B", device="cpu", dtype="float32", num_threads=4)
        prompt = "Artificial intelligence hardware acceleration enables efficient large language model serving."
        input_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"]

        backend1 = create_default_storage_backend(
            channels=4,
            num_layers=engine.num_layers,
            num_heads=engine.num_kv_heads,
            head_dim=engine.head_dim,
            dtype="float32",
            storage_mode="file",
        )
        res1 = run_aissd_decode(
            engine.model, engine.tokenizer, input_ids,
            decode_tokens=8, top_k_pct=10.0, storage_backend=backend1,
            enable_prefetch=True, enable_computational_storage=True, seed=42, show_progress=False
        )

        backend2 = create_default_storage_backend(
            channels=4,
            num_layers=engine.num_layers,
            num_heads=engine.num_kv_heads,
            head_dim=engine.head_dim,
            dtype="float32",
            storage_mode="file",
        )
        res2 = run_aissd_decode(
            engine.model, engine.tokenizer, input_ids,
            decode_tokens=8, top_k_pct=10.0, storage_backend=backend2,
            enable_prefetch=True, enable_computational_storage=True, seed=42, show_progress=False
        )

        assert res1["token_ids"] == res2["token_ids"], f"Consecutive run token mismatch: {res1['token_ids']} vs {res2['token_ids']}"


class TestModelParityAndTelemetry:
    """Phase 1, 8, 9, 10: Model Correctness Matrix and Storage Telemetry."""

    def test_qwen25_05b_parity_and_telemetry(self):
        engine = RealLLMEngine(model_name="Qwen/Qwen2.5-0.5B", device="cpu", dtype="float32", num_threads=4)
        # Use a prompt exceeding the recent window (e.g. 150+ tokens) to trigger KV offloading
        sentence = "The PCI Express (PCIe) bus standard defines high-speed serial computer expansion bus communications for solid state drive NVMe storage controllers. "
        prompt = sentence * 10
        input_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"]
        assert input_ids.shape[1] > 100

        b_res = run_baseline_decode(engine.model, engine.tokenizer, input_ids, decode_tokens=16, seed=42, show_progress=False)
        assert b_res["total_read_mb"] == 0.0
        assert b_res["total_write_mb"] == 0.0

        backend = create_default_storage_backend(
            channels=8,
            num_layers=engine.num_layers,
            num_heads=engine.num_kv_heads,
            head_dim=engine.head_dim,
            dtype="float32",
            storage_mode="file",
        )
        a_res = run_aissd_decode(
            engine.model, engine.tokenizer, input_ids,
            decode_tokens=16, top_k_pct=10.0, storage_backend=backend,
            enable_prefetch=True, enable_computational_storage=True, seed=42, show_progress=False
        )

        # Correctness check: 100% token match
        assert b_res["token_ids"] == a_res["token_ids"], f"Qwen2.5-0.5B mismatch: {b_res['token_ids']} vs {a_res['token_ids']}"

        # Zero-bus candidate check
        assert a_res["candidate_k_bytes_to_host"] == 0, f"Candidate K to host must be 0, got {a_res['candidate_k_bytes_to_host']}"

        # Telemetry check: Storage bytes transferred > 0 for AI-SSD when blocks offloaded
        assert a_res["total_read_mb"] > 0.0, "AI-SSD Total Read MB must be positive"
        assert a_res["total_write_mb"] > 0.0, "AI-SSD Total Write MB must be positive"

    def test_qwen3_4b_parity(self):
        engine = RealLLMEngine(model_name="Qwen/Qwen3-4B-Instruct-2507", device="cpu", dtype="float32", num_threads=4)
        prompt = "The PCI Express standard defines high-speed serial computer expansion bus technology."
        input_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"]

        b_res = run_baseline_decode(engine.model, engine.tokenizer, input_ids, decode_tokens=16, seed=42, show_progress=False)

        backend = create_default_storage_backend(
            channels=8,
            num_layers=engine.num_layers,
            num_heads=engine.num_kv_heads,
            head_dim=engine.head_dim,
            dtype="float32",
            storage_mode="file",
        )
        a_res = run_aissd_decode(
            engine.model, engine.tokenizer, input_ids,
            decode_tokens=16, top_k_pct=10.0, storage_backend=backend,
            enable_prefetch=True, enable_computational_storage=True, seed=42, show_progress=False
        )

        assert b_res["token_ids"] == a_res["token_ids"], f"Qwen3-4B mismatch: {b_res['token_ids']} vs {a_res['token_ids']}"
        assert a_res["candidate_k_bytes_to_host"] == 0

    def test_tiny_mistral_parity(self):
        engine = RealLLMEngine(model_name="openaccess-ai-collective/tiny-mistral", device="cpu", dtype="float32", num_threads=4)
        prompt = "The PCI Express standard defines high-speed serial computer expansion bus technology."
        input_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"]

        b_res = run_baseline_decode(engine.model, engine.tokenizer, input_ids, decode_tokens=16, seed=42, show_progress=False)

        backend = create_default_storage_backend(
            channels=8,
            num_layers=engine.num_layers,
            num_heads=engine.num_kv_heads,
            head_dim=engine.head_dim,
            dtype="float32",
            storage_mode="file",
        )
        a_res = run_aissd_decode(
            engine.model, engine.tokenizer, input_ids,
            decode_tokens=16, top_k_pct=10.0, storage_backend=backend,
            enable_prefetch=True, enable_computational_storage=True, seed=42, show_progress=False
        )

        assert b_res["token_ids"] == a_res["token_ids"], f"Tiny-Mistral mismatch: {b_res['token_ids']} vs {a_res['token_ids']}"
        assert a_res["candidate_k_bytes_to_host"] == 0
