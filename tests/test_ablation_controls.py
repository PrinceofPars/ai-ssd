#!/usr/bin/env python3
"""
Tests for Phase 6 Controlled Ablation features and telemetry breakdown.
Verifies batching toggle, telemetry reset, timing breakdown structure, and prefetch toggles.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person2_ssd.inference_backend import RealInferenceStorageBackend
from person1_kv_engine.real_llm.aissd_inference import (
    AISSDKVManager,
    create_default_storage_backend,
)


def test_batching_ablation_toggle():
    """Verifies that enable_batching=False bypasses batched dispatch."""
    # Batched backend
    backend_batched = RealInferenceStorageBackend(
        channels=8,
        num_layers=2,
        tokens_per_block=16,
        head_dim=64,
        dtype="float32",
        enable_batching=True,
    )
    # Unbatched backend
    backend_unbatched = RealInferenceStorageBackend(
        channels=8,
        num_layers=2,
        tokens_per_block=16,
        head_dim=64,
        dtype="float32",
        enable_batching=False,
    )

    k = np.ones((16, 2, 64), dtype=np.float32)
    v = np.ones((16, 2, 64), dtype=np.float32)

    for bid in range(4):
        backend_batched.write_block(0, bid, k, v)
        backend_unbatched.write_block(0, bid, k, v)

    # Batched read
    res_b = backend_batched.read_key_page_batch(0, [0, 1, 2, 3])
    assert len(res_b) == 4
    assert backend_batched.storage_batches == 1
    assert backend_batched.batched_requests == 4

    # Unbatched read
    res_u = backend_unbatched.read_key_page_batch(0, [0, 1, 2, 3])
    assert len(res_u) == 4
    assert backend_unbatched.storage_batches == 0
    assert backend_unbatched.batched_requests == 0
    assert backend_unbatched.key_page_read_requests == 4

    backend_batched.close()
    backend_unbatched.close()


def test_telemetry_reset():
    """Verifies that reset_stats properly clears all metrics and batch counters."""
    backend = RealInferenceStorageBackend(
        channels=8,
        num_layers=2,
        tokens_per_block=16,
        head_dim=64,
        dtype="float32",
    )
    k = np.ones((16, 2, 64), dtype=np.float32)
    v = np.ones((16, 2, 64), dtype=np.float32)
    backend.write_block(0, 0, k, v)
    backend.read_key_page_batch(0, [0])

    assert backend.requests > 0
    assert backend.storage_batches > 0

    backend.reset_stats()
    assert backend.requests == 0
    assert backend.storage_batches == 0
    assert backend.batched_requests == 0
    assert backend.bytes_read == 0
    backend.close()


def test_kv_manager_timing_breakdown():
    """Verifies that AISSDKVManager records all required Phase 6 sub-operation timers."""
    backend = create_default_storage_backend(
        channels=8,
        num_layers=2,
        num_heads=2,
        head_dim=64,
        enable_prefetch=False,
    )
    mgr = AISSDKVManager(backend=backend, num_layers=2, top_k_pct=10.0)
    assert "candidate_k_reads_s" in mgr.timings
    assert "topk_scoring_s" in mgr.timings
    assert "candidate_selection_s" in mgr.timings
    assert "prefetch_s" in mgr.timings
    assert "winning_v_reads_s" in mgr.timings
    assert "tensor_recon_s" in mgr.timings
    assert "active_concat_s" in mgr.timings

    mgr.timings["topk_scoring_s"] = 1.23
    mgr.reset_timings()
    assert mgr.timings["topk_scoring_s"] == 0.0

    if hasattr(backend, "close"):
        backend.close()
