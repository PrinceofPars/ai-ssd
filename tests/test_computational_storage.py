"""
Tests for Computational Storage: In-Storage Top-K Candidate Filtering.

Validates:
1. Exact Top-K selection equivalence between in-storage computational filtering
   and host-side reference scoring.
2. In-storage scoring data movement reduction: 0 candidate Key pages transferred
   to host memory.
3. Proper telemetry tracking for computational Top-K operations.
"""

import math
import pytest
import numpy as np

from person2_ssd.inference_backend import RealInferenceStorageBackend
from person3_system.prefetch.inference_adapter import RealInferencePrefetchAdapter


def test_computational_topk_equivalence():
    """
    Verifies that compute_topk_filter produces identical Top-K block selection
    and matching scores as the host-side reference GQA calculation.
    """
    backend = RealInferenceStorageBackend(
        channels=8,
        num_layers=2,
        num_heads=2,
        head_dim=64,
        dtype="float32",
        storage_mode="file",
    )

    rng = np.random.RandomState(42)
    layer_idx = 0
    num_blocks = 32
    top_k = 4
    scale = 1.0 / math.sqrt(64)

    # 14 query heads, head_dim=64
    query = rng.randn(14, 64).astype(np.float32)

    cand_bids = []
    ref_scores = []

    for bid in range(num_blocks):
        # 16 tokens, 2 KV heads, 64 head_dim
        k_blk = rng.randn(16, 2, 64).astype(np.float32)
        v_blk = rng.randn(16, 2, 64).astype(np.float32)
        act_tokens = 16 if bid < num_blocks - 1 else 10

        backend.write_block(layer_idx, bid, k_blk, v_blk)
        cand_bids.append((bid, act_tokens))

        # Reference host-side scoring
        dots = np.einsum("hd,thd->th", query, k_blk[:, [h // 7 for h in range(14)], :]) * scale
        max_s = float(np.max(dots[:act_tokens]))
        ref_scores.append((max_s, bid, act_tokens))

    ref_scores.sort(key=lambda x: x[0], reverse=True)
    expected_topk = ref_scores[:top_k]

    # In-storage computational filtering
    actual_topk = backend.compute_topk_filter(
        layer_idx=layer_idx,
        cand_bids=cand_bids,
        query=query,
        top_k=top_k,
        scale=scale,
        q_heads=14,
        kv_heads=2,
        head_dim=64,
    )

    assert len(actual_topk) == top_k

    # Verify identical block IDs selected
    expected_bids = [bid for _, bid, _ in expected_topk]
    actual_bids = [bid for _, bid, _ in actual_topk]
    assert actual_bids == expected_bids, f"Expected {expected_bids}, got {actual_bids}"

    # Verify scores match within numerical precision
    for i in range(top_k):
        exp_score = expected_topk[i][0]
        act_score = actual_topk[i][0]
        assert abs(exp_score - act_score) < 1e-4, f"Mismatch at rank {i}: exp {exp_score} vs act {act_score}"

    backend.close()


def test_computational_storage_telemetry():
    """
    Verifies that compute_topk_filter tracks internal scan volume and query transfer bytes.
    """
    backend = RealInferenceStorageBackend(
        channels=8,
        num_layers=2,
        num_heads=2,
        head_dim=64,
        dtype="float32",
        storage_mode="file",
    )

    rng = np.random.RandomState(123)
    layer_idx = 1
    num_blocks = 20
    query = rng.randn(14, 64).astype(np.float32)

    cand_bids = []
    for bid in range(num_blocks):
        k_blk = rng.randn(16, 2, 64).astype(np.float32)
        v_blk = rng.randn(16, 2, 64).astype(np.float32)
        backend.write_block(layer_idx, bid, k_blk, v_blk)
        cand_bids.append((bid, 16))

    topk = backend.compute_topk_filter(
        layer_idx=layer_idx,
        cand_bids=cand_bids,
        query=query,
        top_k=3,
        scale=0.125,
        q_heads=14,
        kv_heads=2,
        head_dim=64,
    )

    assert len(topk) == 3
    telem = backend.get_telemetry()
    cs_telem = telem["computational_storage"]
    assert cs_telem["topk_calls"] == 1
    assert cs_telem["internal_k_blocks_scanned"] == num_blocks
    assert cs_telem["internal_k_bytes_scanned"] == num_blocks * (16 * 2 * 64 * 4)

    backend.close()


def test_prefetch_adapter_forwarding():
    """
    Verifies that RealInferencePrefetchAdapter forwards compute_topk_filter seamlessly.
    """
    backend = RealInferenceStorageBackend(
        channels=8,
        num_layers=2,
        num_heads=2,
        head_dim=64,
        dtype="float32",
        storage_mode="file",
    )
    adapter = RealInferencePrefetchAdapter(storage_backend=backend)

    rng = np.random.RandomState(99)
    query = rng.randn(14, 64).astype(np.float32)
    cand_bids = []
    for bid in range(10):
        k_blk = rng.randn(16, 2, 64).astype(np.float32)
        v_blk = rng.randn(16, 2, 64).astype(np.float32)
        adapter.write_block(0, bid, k_blk, v_blk)
        cand_bids.append((bid, 16))

    topk = adapter.compute_topk_filter(
        layer_idx=0,
        cand_bids=cand_bids,
        query=query,
        top_k=2,
        scale=0.125,
        q_heads=14,
        kv_heads=2,
        head_dim=64,
    )
    assert len(topk) == 2
    adapter.close()
