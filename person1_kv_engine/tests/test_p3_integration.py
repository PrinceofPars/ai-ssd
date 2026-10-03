"""Unit and Integration Tests for Person 3 Prefetch Adapter Wrapping Person 2 Backend in Person 1.

Verifies:
1. Pipeline initialization: P1 KV Engine -> P3 Prefetch Adapter -> P2 Storage Backend.
2. Bit-for-bit exact numerical roundtrip data retention through P3 staging and P2 FTL.
3. Speculative prefetch hit accounting and useful byte tracking.
4. Unified telemetry reporting: demand hits/misses, prefetch hits/misses, staging memory, and P2 8-channel distribution.
5. AISSDKVManager inter-layer speculative prefetching across transformer layers.
6. Zero artificial latency injection (no simulated sleep delays, genuine native execution).
"""

import os
import sys
import time
import pytest
import numpy as np
import torch

from person1_kv_engine.real_llm.aissd_inference import (
    AISSDKVManager,
    create_default_storage_backend,
    RealInferenceStorageBackend,
    RealInferencePrefetchAdapter,
    _P2_AVAILABLE,
    _P3_AVAILABLE,
)


@pytest.mark.skipif(not (_P2_AVAILABLE and _P3_AVAILABLE), reason="P2 or P3 integration not available")
class TestPerson3PrefetchAdapterIntegration:
    """Test suite validating Person 3's RealInferencePrefetchAdapter wrapping P2 in the P1 engine."""

    def test_pipeline_stack_initialization(self):
        """Verifies that create_default_storage_backend returns P3 wrapping P2."""
        pipeline = create_default_storage_backend(enable_prefetch=True)
        assert isinstance(pipeline, RealInferencePrefetchAdapter)
        assert isinstance(pipeline.storage_backend, RealInferenceStorageBackend)
        assert pipeline.predictor.total_layers == 24
        assert pipeline.buffer_capacity_blocks == 512

        # Direct P2 backend when enable_prefetch=False
        direct_p2 = create_default_storage_backend(enable_prefetch=False)
        assert isinstance(direct_p2, RealInferenceStorageBackend)

    def test_exact_numerical_roundtrip_through_pipeline(self):
        """Validates that stored Key and Value tensors pass bit-for-bit through P3 and P2."""
        pipeline = create_default_storage_backend(enable_prefetch=True)
        rng = np.random.RandomState(42)

        k_orig = rng.randn(16, 2, 64).astype(np.float32)
        v_orig = rng.randn(16, 2, 64).astype(np.float32)

        # Write through P3 adapter to P2 storage
        pipeline.write_block(layer_idx=3, block_id=10, k_block=k_orig, v_block=v_orig)

        # Read back via individual page reads and combined read
        k_read = pipeline.read_key_page(layer_idx=3, block_id=10)
        v_read = pipeline.read_value_page(layer_idx=3, block_id=10)
        k_both, v_both = pipeline.read_block(layer_idx=3, block_id=10)

        assert np.array_equal(k_read, k_orig), "Key tensor corrupted during P3->P2 roundtrip"
        assert np.array_equal(v_read, v_orig), "Value tensor corrupted during P3->P2 roundtrip"
        assert np.array_equal(k_both, k_orig), "Combined Key tensor corrupted"
        assert np.array_equal(v_both, v_orig), "Combined Value tensor corrupted"

    def test_speculative_prefetch_hits_and_telemetry(self):
        """Verifies that speculative prefetching stages blocks and generates cache hits."""
        pipeline = create_default_storage_backend(enable_prefetch=True)
        rng = np.random.RandomState(123)

        # Store 4 blocks at layer 1
        for b in range(4):
            k_data = rng.randn(16, 2, 64).astype(np.float32)
            v_data = rng.randn(16, 2, 64).astype(np.float32)
            pipeline.write_block(layer_idx=1, block_id=b, k_block=k_data, v_block=v_data)

        # Speculatively predict and prefetch layer 1 blocks from layer 0
        pipeline.predict_and_prefetch(current_layer_id=0, current_block_ids=[0, 1, 2, 3])

        # Verify staged in DRAM
        assert pipeline.staging_memory_bytes == 4 * pipeline.bytes_per_block
        assert len(pipeline._staging_buffer) == 4

        # Read the prefetched blocks - all should be hits
        for b in range(4):
            pipeline.read_key_page(layer_idx=1, block_id=b)
            pipeline.read_value_page(layer_idx=1, block_id=b)

        telem = pipeline.get_telemetry()
        assert telem["demand_reads"] == 8
        assert telem["demand_hits"] == 8
        assert telem["demand_misses"] == 0
        assert telem["demand_hit_rate_pct"] == 100.0
        assert telem["useful_prefetches"] == 4
        assert telem["prefetch_accuracy_pct"] == 100.0
        assert telem["useful_bytes"] > 0
        assert telem["wasted_bytes"] == 0

    def test_aissd_kv_manager_interlayer_prefetching(self):
        """Validates that AISSDKVManager with P3 adapter executes multi-layer inference and prefetching."""
        pipeline = create_default_storage_backend(enable_prefetch=True)
        mgr = AISSDKVManager(backend=pipeline, num_layers=24, sink_tokens=4, recent_tokens=16, top_k_pct=10.0)

        # Mock prefill cache with 512 tokens
        class MockLayer:
            def __init__(self):
                self.keys = torch.randn(1, 2, 512, 64, dtype=torch.float32)
                self.values = torch.randn(1, 2, 512, 64, dtype=torch.float32)

        class MockPKV:
            def __init__(self):
                self.layers = [MockLayer() for _ in range(24)]

        pkv = MockPKV()
        mgr.init_from_prefill(pkv)

        # Reset decode stats
        pipeline.reset_stats()

        # Simulate 2 decode steps across all 24 layers
        for _ in range(2):
            for l in range(24):
                q = torch.randn(1, 14, 1, 64, dtype=torch.float32)
                act_k, act_v = mgr.select_and_fetch_active_kv(layer_idx=l, query_states=q)
                assert act_k.shape[0] == 1
                assert act_k.shape[1] == 2
                assert act_k.shape[3] == 64
                assert act_v.shape == act_k.shape

        telem = pipeline.get_telemetry()
        assert telem["demand_reads"] > 0
        assert telem["demand_hits"] > 0
        assert telem["prefetch_requests"] > 0
        assert telem["useful_prefetches"] > 0

        # Check underlying P2 telemetry
        p2_telem = telem["storage_backend"]
        assert p2_telem["architecture"]["channels"] == 8
        ch_counts = p2_telem["channel_distribution"]["per_channel_total_requests"]
        assert len(ch_counts) == 8
        for ch, count in ch_counts.items():
            assert count > 0, f"P2 Channel {ch} starved!"

    def test_zero_artificial_sleep_latency(self):
        """Guarantees zero sleep latency injection through P3->P2 pipeline."""
        pipeline = create_default_storage_backend(enable_prefetch=True)
        k_buf = np.zeros((16, 2, 64), dtype=np.float32)
        v_buf = np.zeros((16, 2, 64), dtype=np.float32)

        start = time.perf_counter()
        for i in range(100):
            pipeline.write_block(0, i, k_buf, v_buf)
            pipeline.read_key_page(0, i)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.25, f"Execution took {elapsed:.4f}s, potential sleep detected!"
        telem = pipeline.get_telemetry()
        assert telem["storage_backend"]["simulated_metrics"]["sleep_latency_injected"] is False
