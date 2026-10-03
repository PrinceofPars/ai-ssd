"""Unit tests for Phase 5A Real Qwen Inference + AI-SSD KV Integration."""

import os
import json
import pytest
import numpy as np
import torch

from person1_kv_engine.real_llm.aissd_inference import (
    AISSDBlockStorageBackend,
    AISSDKVManager,
)

RESULTS_DIR = "/opt/ai-ssd-v2/results/p1"
BASELINE_JSON = os.path.join(RESULTS_DIR, "real_inference_baseline.json")
AISSD_JSON = os.path.join(RESULTS_DIR, "real_inference_ai_ssd.json")


def test_aissd_storage_backend():
    """Validates block storage read/write and accounting semantics."""
    backend = AISSDBlockStorageBackend(num_layers=2, tokens_per_block=16, head_dim=64)

    k_blk = np.ones((16, 2, 64), dtype=np.float32)
    v_blk = np.full((16, 2, 64), 2.0, dtype=np.float32)

    backend.write_block(layer_idx=0, block_id=0, k_block=k_blk, v_block=v_blk)
    assert backend.blocks_written == 1
    assert backend.bytes_written == 8192

    # Read Key page (controller internal scan)
    k_read = backend.read_key_page(layer_idx=0, block_id=0)
    assert k_read.shape == (16, 2, 64)
    assert np.allclose(k_read, 1.0)
    assert backend.bytes_read == 4096
    assert backend.requests == 1

    # Read Value page (host PCIe retrieval)
    v_read = backend.read_value_page(layer_idx=0, block_id=0)
    assert v_read.shape == (16, 2, 64)
    assert np.allclose(v_read, 2.0)
    assert backend.blocks_read == 1
    assert backend.bytes_read == 8192
    assert backend.requests == 2


def test_aissd_kv_manager_mock():
    """Validates KV manager partitioning and block selection."""
    backend = AISSDBlockStorageBackend(num_layers=2, tokens_per_block=16, head_dim=64)
    kv_mgr = AISSDKVManager(backend, num_layers=2, sink_tokens=4, recent_tokens=16, top_k_pct=20.0)

    # Mock past_key_values with 64 tokens
    class MockLayer:
        def __init__(self):
            self.keys = torch.randn(1, 2, 64, 64)
            self.values = torch.randn(1, 2, 64, 64)

    class MockPKV:
        def __init__(self):
            self.layers = [MockLayer(), MockLayer()]

    pkv = MockPKV()
    kv_mgr.init_from_prefill(pkv)

    assert kv_mgr.is_active
    assert len(kv_mgr.layer_data) == 2
    ld = kv_mgr.layer_data[0]
    assert ld["sink_k"].shape == (1, 2, 4, 64)
    assert ld["recent_k"].shape == (1, 2, 16, 64)
    # Historical tokens: 64 - 4 - 16 = 44 tokens -> ceil(44/16) = 3 blocks
    assert len(ld["candidate_blocks"]) == 3

    # Test query selection
    query = torch.randn(1, 14, 1, 64)
    act_k, act_v = kv_mgr.select_and_fetch_active_kv(0, query)
    assert act_k.shape[0] == 1
    assert act_k.shape[1] == 2
    assert act_k.shape[3] == 64
    # Sinks (4) + Top-k (1 block of 16) + Recent (16) = 36 tokens
    assert act_k.shape[2] >= 20

    mem_stats = kv_mgr.get_memory_stats()
    assert mem_stats["offload_pct"] > 0.0


def test_benchmark_result_artifacts():
    """Validates existence and contents of benchmark JSON results."""
    assert os.path.exists(BASELINE_JSON), f"Missing {BASELINE_JSON}"
    assert os.path.exists(AISSD_JSON), f"Missing {AISSD_JSON}"

    with open(BASELINE_JSON, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    assert base_data["mode"] == "BASELINE"
    assert base_data["tokens_per_second"]["mean"] > 10.0
    assert base_data["kv_offloaded_pct"] == 0.0
    assert "REAL INFERENCE" in base_data["classification"]

    with open(AISSD_JSON, "r", encoding="utf-8") as f:
        aissd_data = json.load(f)

    assert aissd_data["mode"] == "AI-SSD"
    assert aissd_data["tokens_per_second"]["mean"] > 10.0
    assert aissd_data["kv_offloaded_pct"] > 70.0
    assert aissd_data["storage"]["bytes_read"] > 0
    assert "correctness_vs_baseline" in aissd_data
    cmp = aissd_data["correctness_vs_baseline"]
    assert cmp["token_match_percent"] > 0.0
    assert cmp["logits_cosine_similarity"] > 0.0
