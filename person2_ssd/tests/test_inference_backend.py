"""
Comprehensive Unit & Integration Tests for RealInferenceStorageBackend.

Validates:
1. Exact numerical round-trip data retention:
   original K/V -> write_block() -> mapper -> read_block() -> equality.
2. Direct compatibility with P1's live Qwen inference signatures:
   - write_block(layer_idx, block_id, k_block, v_block)
   - read_key_page(layer_idx, block_id) -> np.ndarray
   - read_value_page(layer_idx, block_id) -> np.ndarray
   - read_block(layer_idx, block_id) -> Tuple[np.ndarray, np.ndarray]
3. Canonical Geometry:
   - 16 tokens x 1 KV head x 64 dim x 4 bytes = 4,096 bytes per page
   - K = 4096 bytes, V = 4096 bytes, K+V = 8192 bytes
4. Live Qwen2.5-0.5B Grouped Geometry:
   - (16 tokens, 2 KV heads, 64 dim) float32
5. Deterministic multi-channel distribution across all 8 channels.
6. Complete telemetry tracking (reads, writes, per-channel counts, bytes).
7. Zero artificial latency injection (no time.sleep).
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

    def test_canonical_geometry_constants(self):
        """Verifies canonical 4 KiB page and 8 KiB block size constants."""
        assert KEY_PAGE_BYTES == 4096
        assert VALUE_PAGE_BYTES == 4096
        assert LOGICAL_BLOCK_BYTES == 8192

    def test_canonical_single_head_roundtrip(self):
        """
        Validates round-trip for canonical 1-head geometry:
        16 tokens x 1 head x 64 dim x FP32 = 4,096 bytes per tensor.
        """
        backend = RealInferenceStorageBackend(channels=8, num_layers=24, num_heads=2)

        # Shape: (16, 1, 64), 1024 float32 elements = 4096 bytes
        rng = np.random.RandomState(42)
        k_orig = rng.randn(16, 1, 64).astype(np.float32)
        v_orig = rng.randn(16, 1, 64).astype(np.float32)

        assert k_orig.nbytes == KEY_PAGE_BYTES
        assert v_orig.nbytes == VALUE_PAGE_BYTES

        # Write block to Layer 4, Block 9
        backend.write_block(layer_idx=4, block_id=9, k_block=k_orig, v_block=v_orig)
        assert backend.contains_block(layer_idx=4, block_id=9) is True

        # Read Key page (TOPK_FILTER)
        k_read = backend.read_key_page(layer_idx=4, block_id=9)
        assert np.array_equal(k_read, k_orig)
        assert k_read.shape == k_orig.shape

        # Read Value page (TOPK_FETCH)
        v_read = backend.read_value_page(layer_idx=4, block_id=9)
        assert np.array_equal(v_read, v_orig)
        assert v_read.shape == v_orig.shape

        # Read Full Block
        k_full, v_full = backend.read_block(layer_idx=4, block_id=9)
        assert np.array_equal(k_full, k_orig)
        assert np.array_equal(v_full, v_orig)

    def test_p1_live_qwen_tensor_shape_roundtrip(self):
        """
        Validates round-trip for P1's live Qwen tensor shape:
        (16 tokens, 2 KV heads, 64 dim) float32 = 8,192 bytes per tensor.
        """
        backend = RealInferenceStorageBackend(channels=8, num_layers=24, num_heads=2)

        rng = np.random.RandomState(123)
        k_qwen = rng.randn(16, 2, 64).astype(np.float32)
        v_qwen = rng.randn(16, 2, 64).astype(np.float32)

        # Write using P1's exact signature: write_block(l_idx, bid, k_blk, v_blk)
        backend.write_block(0, 3, k_qwen, v_qwen)

        # Read using P1's exact signature: read_key_page(l_idx, bid)
        k_read = backend.read_key_page(0, 3)
        assert np.array_equal(k_read, k_qwen)
        assert k_read.shape == (16, 2, 64)

        # Read using P1's exact signature: read_value_page(l_idx, bid)
        v_read = backend.read_value_page(0, 3)
        assert np.array_equal(v_read, v_qwen)
        assert v_read.shape == (16, 2, 64)

        # Verify P1 compatibility properties
        assert backend.blocks_written == 1
        assert backend.bytes_written == (k_qwen.nbytes + v_qwen.nbytes)
        assert backend.requests == 3  # 1 write + 1 key read + 1 value read

    def test_multi_channel_striping_verification(self):
        """
        Verifies that DeterministicTensorMapper distributes live inference blocks
        across all 8 channels without serialization.
        """
        backend = RealInferenceStorageBackend(channels=8, num_layers=24, num_heads=2)

        k_dummy = np.zeros((16, 2, 64), dtype=np.float32)
        v_dummy = np.zeros((16, 2, 64), dtype=np.float32)

        # Write 96 blocks across 24 layers (4 blocks per layer)
        num_blocks = 96
        for i in range(num_blocks):
            l_idx = i % 24
            b_id = i // 24
            backend.write_block(l_idx, b_id, k_dummy, v_dummy)

        # Read all blocks back
        for i in range(num_blocks):
            l_idx = i % 24
            b_id = i // 24
            backend.read_block(l_idx, b_id)

        telemetry = backend.get_telemetry()
        ch_dist = telemetry["channel_distribution"]
        total_reqs_per_ch = ch_dist["per_channel_total_requests"]

        # All 8 channels must receive requests (zero channel starvation)
        assert len(total_reqs_per_ch) == 8
        for ch, count in total_reqs_per_ch.items():
            assert count > 0, f"Channel {ch} received zero requests!"

        # Contention ratio must remain balanced (< 1.5x, vs 8.0x on conventional)
        assert ch_dist["contention_ratio"] <= 1.50
        assert telemetry["backend_classification"] == "ANALYTICAL"

    def test_zero_artificial_latency(self):
        """
        Guarantees that no time.sleep() or artificial delays exist.
        200 I/O operations must complete in under 150 ms.
        """
        backend = RealInferenceStorageBackend()
        k_buf = np.zeros((16, 1, 64), dtype=np.float32)
        v_buf = np.zeros((16, 1, 64), dtype=np.float32)

        start = time.perf_counter()
        for i in range(100):
            backend.write_block(0, i, k_buf, v_buf)
            backend.read_key_page(0, i)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.15, f"Execution took {elapsed:.4f}s, potential sleep injection!"
        telemetry = backend.get_telemetry()
        assert telemetry["simulated_metrics"]["sleep_latency_injected"] is False

    def test_eviction_and_reset(self):
        """Validates cache eviction and stats reset."""
        backend = RealInferenceStorageBackend()
        k_buf = np.zeros((16, 1, 64), dtype=np.float32)
        v_buf = np.zeros((16, 1, 64), dtype=np.float32)

        backend.write_block(2, 5, k_buf, v_buf)
        assert backend.contains_block(2, 5) is True

        # Evict
        assert backend.evict_block(2, 5) is True
        assert backend.contains_block(2, 5) is False
        assert backend.evict_block(2, 5) is False

        # Reset stats
        backend.reset_stats()
        assert backend.requests == 0
        assert backend.bytes_read == 0
        assert backend.bytes_written == 0
        assert backend.blocks_read == 0
        assert backend.blocks_written == 0
        assert backend.storage_batches == 0
        assert backend.batched_requests == 0

    def test_batch_storage_operations(self):
        """Optimization C: Validates batched storage reads and exact tensor reconstruction."""
        backend = RealInferenceStorageBackend(num_layers=4, num_heads=2, tokens_per_block=16, head_dim=64)
        layer_idx = 1
        num_blocks = 8

        # Write test blocks
        k_blocks = {}
        v_blocks = {}
        for b in range(num_blocks):
            k_data = np.random.randn(16, 2, 64).astype(np.float32)
            v_data = np.random.randn(16, 2, 64).astype(np.float32)
            backend.write_block(layer_idx, b, k_data, v_data)
            k_blocks[b] = k_data
            v_blocks[b] = v_data

        backend.reset_stats()

        # 1. Batch Key Page Reads
        bids = [0, 2, 4, 6]
        k_batch = backend.read_key_page_batch(layer_idx, bids)
        assert len(k_batch) == 4
        assert backend.storage_batches == 1
        assert backend.batched_requests == 4
        assert backend.requests == 4
        assert backend.bytes_read == 4 * 16 * 2 * 64 * 4  # 4 * 8192 bytes
        for b in bids:
            assert np.array_equal(k_batch[b], k_blocks[b])

        # 2. Batch Value Page Reads
        v_bids = [1, 3, 5, 7]
        v_batch = backend.read_value_page_batch(layer_idx, v_bids)
        assert len(v_batch) == 4
        assert backend.storage_batches == 2
        assert backend.batched_requests == 8
        assert backend.requests == 8
        for b in v_bids:
            assert np.array_equal(v_batch[b], v_blocks[b])

        # 3. Batch Combined Block Reads
        all_bids = list(range(num_blocks))
        both_batch = backend.read_block_batch(layer_idx, all_bids)
        assert len(both_batch) == 8
        assert backend.storage_batches == 3
        for b in all_bids:
            kb, vb = both_batch[b]
            assert np.array_equal(kb, k_blocks[b])
            assert np.array_equal(vb, v_blocks[b])

        telemetry = backend.get_telemetry()
        assert telemetry["requests"]["storage_batches"] == 3
        assert telemetry["requests"]["avg_batch_size"] > 0

