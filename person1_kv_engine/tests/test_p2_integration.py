"""Unit and Integration Tests for Person 2 Storage Backend in Person 1 Inference Path.

Verifies:
1. Canonical Flash page & block constants (4096 B Key, 4096 B Value, 8192 B Logical Block).
2. Exact numerical round-trip data retention (no data corruption through FTL).
3. Interoperability of both parameter orders: (layer_idx, block_id) and (block_id, layer_id).
4. Multi-channel distribution across all 8 channels without channel starvation.
5. Integration with AISSDKVManager (prefill offloading and decode in-storage Top-k selection).
6. Zero artificial latency injection (no simulated sleep delays).
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
    _P2_AVAILABLE,
)


@pytest.mark.skipif(not _P2_AVAILABLE, reason="Person 2 backend not available")
class TestPerson2StorageBackendIntegration:
    """Test suite validating Person 2's RealInferenceStorageBackend in the P1 engine."""

    def test_canonical_geometry_constants(self):
        """Verifies canonical 4 KiB page and 8 KiB block size constants."""
        backend = create_default_storage_backend()
        telemetry = backend.get_telemetry()
        arch = telemetry["architecture"]
        assert arch["channels"] == 8
        assert arch["key_page_bytes"] == 4096
        assert arch["value_page_bytes"] == 4096
        assert arch["logical_block_bytes"] == 8192
        assert arch["mapping_mode"] == "tensor_aware"

    def test_exact_numerical_roundtrip(self):
        """Validates that stored Key and Value tensors are retrieved bit-for-bit identical."""
        backend = create_default_storage_backend()
        rng = np.random.RandomState(42)

        # Qwen grouped KV block: 16 tokens x 2 heads x 64 dim FP32
        k_orig = rng.randn(16, 2, 64).astype(np.float32)
        v_orig = rng.randn(16, 2, 64).astype(np.float32)

        # Store block
        backend.write_block(layer_idx=5, block_id=12, k_block=k_orig, v_block=v_orig)

        # Retrieve and verify
        k_read = backend.read_key_page(layer_idx=5, block_id=12)
        v_read = backend.read_value_page(layer_idx=5, block_id=12)

        assert np.array_equal(k_read, k_orig), "Key tensor corrupted during storage roundtrip"
        assert np.array_equal(v_read, v_orig), "Value tensor corrupted during storage roundtrip"

    def test_parameter_order_interoperability(self):
        """Validates that both P1 and P2 argument conventions work identically."""
        backend = create_default_storage_backend()
        rng = np.random.RandomState(99)

        k_data = rng.randn(16, 2, 64).astype(np.float32)
        v_data = rng.randn(16, 2, 64).astype(np.float32)

        # Write using P2 convention: store_kv(block_id, layer_id, ...)
        backend.store_kv(block_id=7, layer_id=3, key_data=k_data, value_data=v_data)

        # Read using P1 convention: read_key_page(layer_idx, block_id)
        k_p1 = backend.read_key_page(layer_idx=3, block_id=7)
        v_p1 = backend.read_value_page(layer_idx=3, block_id=7)

        # Read using P2 convention: load_key_page(block_id, layer_id)
        k_p2 = backend.load_key_page(block_id=7, layer_id=3)
        v_p2 = backend.load_value_page(block_id=7, layer_id=3)

        assert np.array_equal(k_p1, k_data)
        assert np.array_equal(v_p1, v_data)
        assert np.array_equal(k_p2, k_data)
        assert np.array_equal(v_p2, v_data)

    def test_multi_channel_striping_all_8_channels(self):
        """Verifies deterministic multi-channel distribution across all 8 channels."""
        backend = create_default_storage_backend(channels=8)
        k_buf = np.zeros((16, 2, 64), dtype=np.float32)
        v_buf = np.zeros((16, 2, 64), dtype=np.float32)

        # Write 96 blocks across 24 layers
        for i in range(96):
            l_idx = i % 24
            b_id = i // 24
            backend.write_block(l_idx, b_id, k_buf, v_buf)

        # Read all blocks back
        for i in range(96):
            l_idx = i % 24
            b_id = i // 24
            backend.read_key_page(l_idx, b_id)
            backend.read_value_page(l_idx, b_id)

        telemetry = backend.get_telemetry()
        ch_dist = telemetry["channel_distribution"]
        per_ch_reqs = ch_dist["per_channel_total_requests"]

        # Ensure all 8 channels received traffic (zero starvation)
        assert len(per_ch_reqs) == 8
        for ch, count in per_ch_reqs.items():
            assert count > 0, f"Channel {ch} received 0 requests!"

        # Contention ratio must be <= 1.50
        assert ch_dist["contention_ratio"] <= 1.50

    def test_aissd_kv_manager_with_p2_backend(self):
        """Validates that AISSDKVManager offloads and retrieves active KV tensors via P2 backend."""
        backend = create_default_storage_backend()
        mgr = AISSDKVManager(backend=backend, num_layers=24, sink_tokens=4, recent_tokens=16, top_k_pct=10.0)

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

        # Verify historical blocks offloaded to backend
        assert len(backend._storage) > 0
        telemetry_prefill = backend.get_telemetry()
        assert telemetry_prefill["requests"]["writes"] == 744  # 31 blocks * 24 layers

        # Reset stats to measure only decode traffic
        backend.reset_stats()
        assert backend.requests == 0

        # Simulate decode step query
        q = torch.randn(1, 14, 1, 64, dtype=torch.float32)
        act_k, act_v = mgr.select_and_fetch_active_kv(layer_idx=0, query_states=q)

        # Active tokens = 4 (sinks) + 16 (recent) + ceil(31 * 0.1) * 16 = 4 + 16 + 64 = 84 tokens
        assert act_k.shape == (1, 2, 84, 64)
        assert act_v.shape == (1, 2, 84, 64)

        # Verify backend read requests and channel distribution
        telemetry_decode = backend.get_telemetry()
        # 31 candidate filter scans + 4 winning key fetches = 35 key page reads
        assert telemetry_decode["requests"]["read_key_pages"] == 35
        # 4 winning value fetches
        assert telemetry_decode["requests"]["read_value_pages"] == 4
        assert telemetry_decode["requests"]["total"] == 39
        assert telemetry_decode["channel_distribution"]["contention_ratio"] < 2.0

    def test_zero_artificial_sleep_latency(self):
        """Guarantees zero sleep latency injection and high-throughput execution."""
        backend = create_default_storage_backend()
        k_buf = np.zeros((16, 2, 64), dtype=np.float32)
        v_buf = np.zeros((16, 2, 64), dtype=np.float32)

        start = time.perf_counter()
        for i in range(100):
            backend.write_block(0, i, k_buf, v_buf)
            backend.read_key_page(0, i)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.20, f"Execution took {elapsed:.4f}s, potential sleep detected!"
        telemetry = backend.get_telemetry()
        assert telemetry["simulated_metrics"]["sleep_latency_injected"] is False
