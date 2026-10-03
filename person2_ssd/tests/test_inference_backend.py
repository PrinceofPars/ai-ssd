"""
Tests for Person 2 Real Inference Storage Backend Adapter.

Validates:
1. Writing known KV blocks and retrieving them via DeterministicTensorMapper.
2. Data fidelity (exact float / byte equality between written and retrieved payload).
3. Exact page geometry preservation: K = 4096 B, V = 4096 B, K+V = 8192 B.
4. Selective sub-page retrieval (Key-only for TOPK_FILTER, Value-only for TOPK_FETCH).
5. Multi-channel distribution across all 8 channels using DeterministicTensorMapper.
6. Telemetry accuracy (reads, writes, per-channel loads, contention ratio).
7. Zero simulated sleep latency (line-rate execution for real LLM inference).
8. Analytical classification reporting.
"""

import time
import pytest
import numpy as np

from person2_ssd.inference_backend import (
    RealInferenceStorageBackend,
    KEY_PAGE_BYTES,
    VALUE_PAGE_BYTES,
    LOGICAL_BLOCK_BYTES,
)


class TestRealInferenceStorageBackend:

    def test_write_and_read_known_kv_blocks(self):
        """Verifies that known numerical KV blocks are accurately persisted and retrieved."""
        backend = RealInferenceStorageBackend(channels=8, num_layers=24, num_heads=2)

        # 1024 float32 numbers = 4,096 bytes
        known_k = np.linspace(-10.0, 10.0, 1024, dtype=np.float32)
        known_v = np.sin(np.linspace(0, 2 * np.pi, 1024)).astype(np.float32)

        assert known_k.nbytes == KEY_PAGE_BYTES
        assert known_v.nbytes == VALUE_PAGE_BYTES

        # Write to Layer 3, Block 7, Head 1
        success = backend.store_kv(
            block_id=7,
            layer_id=3,
            key_data=known_k,
            value_data=known_v,
            head_id=1,
            token_start=112,
        )
        assert success is True
        assert backend.contains_block(block_id=7, layer_id=3) is True

        # Retrieve both Key and Value
        ret_k, ret_v = backend.load_kv(
            block_id=7,
            layer_id=3,
            head_id=1,
            token_start=112,
        )

        assert np.array_equal(ret_k, known_k)
        assert np.array_equal(ret_v, known_v)

    def test_preserve_exact_geometry_bytes(self):
        """Verifies enforcement of 4096 B Key, 4096 B Value, 8192 B combined."""
        backend = RealInferenceStorageBackend()

        # Under-sized tensor (512 floats = 2048 bytes) must fail loudly
        bad_k = np.zeros(512, dtype=np.float32)
        valid_v = np.zeros(1024, dtype=np.float32)

        with pytest.raises(ValueError, match="payload byte mismatch: expected 4096 bytes"):
            backend.store_kv(block_id=0, layer_id=0, key_data=bad_k, value_data=valid_v)

        # Over-sized tensor (2048 floats = 8192 bytes) must fail loudly
        bad_v = np.zeros(2048, dtype=np.float32)
        valid_k = np.zeros(1024, dtype=np.float32)

        with pytest.raises(ValueError, match="payload byte mismatch: expected 4096 bytes"):
            backend.store_kv(block_id=0, layer_id=0, key_data=valid_k, value_data=bad_v)

    def test_subpage_reads_filter_and_fetch(self):
        """Verifies separate Key and Value subpage retrieval for TOPK operations."""
        backend = RealInferenceStorageBackend()

        k_data = np.arange(1024, dtype=np.float32) + 100.0
        v_data = np.arange(1024, dtype=np.float32) + 200.0

        backend.store_kv(block_id=12, layer_id=5, key_data=k_data, value_data=v_data)

        # Read only Key (TOPK_FILTER simulation)
        k_read = backend.load_key_page(block_id=12, layer_id=5)
        assert np.array_equal(k_read, k_data)

        # Read only Value (TOPK_FETCH simulation)
        v_read = backend.load_value_page(block_id=12, layer_id=5)
        assert np.array_equal(v_read, v_data)

        # Verify telemetry recorded individual 4096 B reads
        telemetry = backend.get_telemetry()
        assert telemetry["requests"]["read_key_pages"] == 1
        assert telemetry["requests"]["read_value_pages"] == 1
        assert telemetry["bytes"]["reads"] == (KEY_PAGE_BYTES + VALUE_PAGE_BYTES)

    def test_multi_channel_striping_and_telemetry(self):
        """Verifies that DeterministicTensorMapper distributes requests across all 8 channels."""
        backend = RealInferenceStorageBackend(channels=8, num_layers=24, num_heads=2)

        dummy_k = np.zeros(1024, dtype=np.float32)
        dummy_v = np.ones(1024, dtype=np.float32)

        # Write 64 blocks spanning diverse layers and heads
        num_blocks = 64
        for i in range(num_blocks):
            layer = i % 24
            head = (i // 24) % 2
            backend.store_kv(block_id=i, layer_id=layer, key_data=dummy_k, value_data=dummy_v, head_id=head)

        # Read all blocks back
        for i in range(num_blocks):
            layer = i % 24
            head = (i // 24) % 2
            backend.load_kv(block_id=i, layer_id=layer, head_id=head)

        telemetry = backend.get_telemetry()

        # Telemetry assertions
        assert telemetry["backend_classification"] == "ANALYTICAL"
        assert telemetry["requests"]["writes"] == 64
        assert telemetry["requests"]["reads"] == 64
        assert telemetry["bytes"]["writes"] == 64 * LOGICAL_BLOCK_BYTES
        assert telemetry["bytes"]["reads"] == 64 * LOGICAL_BLOCK_BYTES

        # Channel distribution: all 8 channels must be utilized
        ch_dist = telemetry["channel_distribution"]
        total_per_channel = ch_dist["per_channel_total_requests"]
        assert len(total_per_channel) == 8
        for ch, count in total_per_channel.items():
            assert count > 0, f"Channel {ch} was starved!"

        # Contention ratio should be close to 1.0 (optimal balance)
        assert ch_dist["contention_ratio"] < 1.30

    def test_zero_simulated_sleep_latency(self):
        """Verifies that no time.sleep is injected, guaranteeing real inference throughput."""
        backend = RealInferenceStorageBackend()
        dummy_k = np.zeros(1024, dtype=np.float32)
        dummy_v = np.zeros(1024, dtype=np.float32)

        # Perform 200 operations
        start_time = time.perf_counter()
        for i in range(100):
            backend.store_kv(block_id=i, layer_id=0, key_data=dummy_k, value_data=dummy_v)
            backend.load_kv(block_id=i, layer_id=0)
        elapsed = time.perf_counter() - start_time

        # 200 operations without sleep should take < 150 milliseconds
        assert elapsed < 0.15, f"Operations took too long ({elapsed:.4f}s), possible sleep injection!"

        telemetry = backend.get_telemetry()
        assert telemetry["simulated_metrics"]["sleep_latency_injected"] is False
        assert telemetry["simulated_metrics"]["analytical_service_time_ms"] > 0.0

    def test_eviction_and_state_management(self):
        """Verifies block eviction and residency queries."""
        backend = RealInferenceStorageBackend()
        k_buf = np.zeros(1024, dtype=np.float32)
        v_buf = np.zeros(1024, dtype=np.float32)

        backend.store_kv(block_id=42, layer_id=1, key_data=k_buf, value_data=v_buf)
        assert backend.contains_block(42, 1) is True

        evicted = backend.evict_kv(42, 1)
        assert evicted is True
        assert backend.contains_block(42, 1) is False

        # Evicting non-existent block returns False
        assert backend.evict_kv(42, 1) is False

    def test_telemetry_reset(self):
        """Verifies reset_telemetry clears all counters."""
        backend = RealInferenceStorageBackend()
        k_buf = np.zeros(1024, dtype=np.float32)
        v_buf = np.zeros(1024, dtype=np.float32)

        backend.store_kv(block_id=1, layer_id=0, key_data=k_buf, value_data=v_buf)
        backend.load_kv(block_id=1, layer_id=0)

        telemetry = backend.get_telemetry()
        assert telemetry["requests"]["total"] == 2

        backend.reset_telemetry()
        cleared_telemetry = backend.get_telemetry()
        assert cleared_telemetry["requests"]["total"] == 0
        assert cleared_telemetry["bytes"]["total"] == 0
