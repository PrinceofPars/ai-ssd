"""
Unit and Integration Tests for RealInferencePrefetchAdapter (Phase 5C / Live Inference Integration).

Validates:
- Direct wrapping of Person 2's RealInferenceStorageBackend.
- Actual KV block data retrieval (NumPy tensors and raw bytes).
- Sub-page slicing (KEY 4KB, VALUE 4KB, BOTH 8KB).
- Direct ingestion from P1 KVBlockAdapter.
- Non-blocking speculative prefetching (useful, late, and useless prefetches).
- Staged data is NOT metadata-only (actual NumPy tensors and bytes in DRAM staging).
- Accurate metric accounting (demand requests, prefetch requests, hits, misses, bytes, staging memory).
- Zero artificial latency injection.
- Live-data round-trip verification:
    prefetch(block) -> backend read -> staging -> later demand read -> exact same K/V data.
"""

import pytest
import os
import sys
from pathlib import Path
import tempfile
import time
import numpy as np

from common.schemas.kv_block import KEY_PAGE_BYTES, VALUE_PAGE_BYTES, LOGICAL_BLOCK_BYTES
from person3_system.storage.mock_backend import MockStorageBackend
from person3_system.storage.file_backend import FileStorageBackend
from person3_system.storage.analytical_backend import AnalyticalFTLBackend
from person3_system.prefetch.inference_adapter import RealInferencePrefetchAdapter

# Resilient import of Person 2's RealInferenceStorageBackend
HAS_P2_BACKEND = False
RealInferenceStorageBackend = None
_p2_paths = [
    str(Path(__file__).resolve().parent.parent),
    "/home/ubuntu/ai-ssd",
    "/home/ubuntu/ai-ssd-p2",
    "/opt/ai-ssd-v2/p2",
]
for _p in _p2_paths:
    if os.path.isdir(_p):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        try:
            from person2_ssd.inference_backend import RealInferenceStorageBackend as _P2Cls
            RealInferenceStorageBackend = _P2Cls
            HAS_P2_BACKEND = True
            break
        except Exception:
            pass

# Import P1 KVBlockAdapter if available
HAS_P1_BLOCK_ADAPTER = False
KVBlockAdapter = None
_possible_adapter_paths = [
    str(Path(__file__).resolve().parent.parent / "person1_kv_engine" / "real_llm" / "block_adapter.py"),
    "/home/ubuntu/ai-ssd/person1_kv_engine/real_llm/block_adapter.py",
    "/home/ubuntu/ai-ssd-p1/person1_kv_engine/real_llm/block_adapter.py",
]
for p1_adapter_path in _possible_adapter_paths:
    if os.path.exists(p1_adapter_path):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("p1_real_block_adapter", p1_adapter_path)
            p1_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(p1_mod)
            KVBlockAdapter = getattr(p1_mod, "KVBlockAdapter", None)
            if KVBlockAdapter is not None:
                HAS_P1_BLOCK_ADAPTER = True
                break
        except Exception:
            pass


# =============================================================================
# 1. Initialization & Wrapper Interface Tests
# =============================================================================

def test_adapter_initialization():
    backend = MockStorageBackend()
    adapter = RealInferencePrefetchAdapter(
        storage_backend=backend,
        buffer_capacity_blocks=64,
        bytes_per_block=LOGICAL_BLOCK_BYTES,
        tokens_per_block=16,
        kv_heads_per_block=2,
        head_dim=64,
        dtype="float32",
    )
    assert adapter.buffer_capacity_blocks == 64
    assert adapter.bytes_per_block == 8192
    assert adapter.staging_memory_bytes == 0
    assert adapter.demand_reads == 0
    assert adapter.requests == 0
    assert adapter.predictor is not None


def test_p1_storage_backend_wrapper_methods():
    """Verifies that RealInferencePrefetchAdapter implements P1's exact storage interface."""
    adapter = RealInferencePrefetchAdapter()
    
    k_orig = np.random.randn(16, 2, 64).astype(np.float32)
    v_orig = np.random.randn(16, 2, 64).astype(np.float32)

    # 1. P1 write_block
    adapter.write_block(layer_idx=0, block_id=1, k_block=k_orig, v_block=v_orig)
    assert adapter.blocks_written == 1
    assert adapter.bytes_written == (k_orig.nbytes + v_orig.nbytes)

    # 2. P1 read_key_page (demand miss path)
    k_read = adapter.read_key_page(layer_idx=0, block_id=1)
    assert isinstance(k_read, np.ndarray)
    assert np.allclose(k_read, k_orig)
    assert adapter.demand_misses == 1

    # 3. P1 read_value_page (demand miss path)
    v_read = adapter.read_value_page(layer_idx=0, block_id=1)
    assert isinstance(v_read, np.ndarray)
    assert np.allclose(v_read, v_orig)
    assert adapter.demand_misses == 2

    # 4. P1 read_block
    k_both, v_both = adapter.read_block(layer_idx=0, block_id=1)
    assert np.allclose(k_both, k_orig)
    assert np.allclose(v_both, v_orig)


def test_unified_read_and_record_hit_miss():
    """Verifies read(...), record_hit(...), and record_miss(...) according to Requirement 4."""
    adapter = RealInferencePrefetchAdapter()
    k_data = np.ones((16, 2, 64), dtype=np.float32) * 3.14
    v_data = np.ones((16, 2, 64), dtype=np.float32) * 2.71
    adapter.write_block(layer_idx=1, block_id=10, k_block=k_data, v_block=v_data)

    # Unified read for KEY
    k_res = adapter.read(layer_idx=1, block_id=10, sub_page="KEY", return_tensors=True)
    assert isinstance(k_res, np.ndarray)
    assert np.allclose(k_res, k_data)

    # Unified read for VALUE
    v_res = adapter.read(layer_idx=1, block_id=10, sub_page="VALUE", return_tensors=True)
    assert isinstance(v_res, np.ndarray)
    assert np.allclose(v_res, v_data)

    # Unified read for BOTH
    kb, vb = adapter.read(layer_idx=1, block_id=10, sub_page="BOTH", return_tensors=True)
    assert np.allclose(kb, k_data)
    assert np.allclose(vb, v_data)

    # Unified read returning raw bytes
    raw_b = adapter.read(layer_idx=1, block_id=10, sub_page="BOTH", return_tensors=False)
    assert isinstance(raw_b, bytes)
    assert len(raw_b) == (k_data.nbytes + v_data.nbytes)


# =============================================================================
# 2. Live-Data Round-Trip & Actual Staging Verification (Requirements 5 & 6)
# =============================================================================

def test_live_data_round_trip_prefetch_staging():
    """
    CRITICAL REQUIREMENT 5 & 6:
    Confirm that:
        prefetch(block) -> backend read -> staging -> later demand read -> exact same K/V data.
    And verify that prefetched data in staging is NOT metadata-only (contains actual NumPy arrays!).
    """
    adapter = RealInferencePrefetchAdapter()

    rng = np.random.RandomState(42)
    k_source = rng.randn(16, 2, 64).astype(np.float32)
    v_source = rng.randn(16, 2, 64).astype(np.float32)

    adapter.write_block(layer_idx=3, block_id=7, k_block=k_source, v_block=v_source)

    # 1. Speculatively prefetch block 7
    dispatched = adapter.prefetch(block_ids=[7], layer_id=3)
    assert dispatched == [7]
    assert adapter.prefetch_requests == 1

    # 2. Inspect DRAM staging entry: MUST NOT be metadata-only!
    key = (3, 7)
    assert key in adapter._staging_buffer
    staged_entry = adapter._staging_buffer[key]
    assert staged_entry.data_k is not None, "Staged Key data must not be None!"
    assert staged_entry.data_v is not None, "Staged Value data must not be None!"
    assert isinstance(staged_entry.data_k, np.ndarray), "Staged Key must be a real NumPy tensor!"
    assert isinstance(staged_entry.data_v, np.ndarray), "Staged Value must be a real NumPy tensor!"
    assert staged_entry.data_k.shape == (16, 2, 64)
    assert np.allclose(staged_entry.data_k, k_source)
    assert np.allclose(staged_entry.data_v, v_source)

    # 3. Later demand read: MUST be a verified DRAM hit with identical data
    hits_before = adapter.demand_hits
    k_fetched = adapter.read_key_page(layer_idx=3, block_id=7)
    assert adapter.demand_hits == hits_before + 1, "Demand read must hit DRAM staging!"
    assert np.allclose(k_fetched, k_source), "Fetched Key data must match original source exactly!"

    v_fetched = adapter.read_value_page(layer_idx=3, block_id=7)
    assert np.allclose(v_fetched, v_source), "Fetched Value data must match original source exactly!"


def test_prefetch_hit_vs_demand_miss_accounting():
    """Validates real runtime metric accounting for hits vs misses (Requirement 8)."""
    adapter = RealInferencePrefetchAdapter()

    # Write blocks 10, 11, 12
    for b in (10, 11, 12):
        k_b = np.full((16, 2, 64), fill_value=float(b), dtype=np.float32)
        v_b = np.full((16, 2, 64), fill_value=float(b * 10), dtype=np.float32)
        adapter.write_block(layer_idx=0, block_id=b, k_block=k_b, v_block=v_b)

    # Prefetch blocks 10 and 11
    adapter.prefetch(block_ids=[10, 11], layer_id=0)

    # Read block 10 (Hit!)
    adapter.read_block(layer_idx=0, block_id=10)
    # Read block 11 (Hit!)
    adapter.read_block(layer_idx=0, block_id=11)
    # Read block 12 (Miss! Not prefetched)
    adapter.read_block(layer_idx=0, block_id=12)

    telem = adapter.get_telemetry()
    assert telem["demand_requests"] == 3
    assert telem["prefetch_requests"] == 2
    assert telem["prefetch_hits"] == 2
    assert telem["prefetch_misses"] == 1
    assert telem["demand_hit_rate"] == round(2 / 3, 4)
    assert telem["useful_prefetches"] == 2
    assert telem["useful_bytes"] == 2 * 8192


def test_useless_prefetch_eviction():
    """Verifies that evicted unread blocks count as useless prefetches and wasted bytes."""
    adapter = RealInferencePrefetchAdapter(buffer_capacity_blocks=2)

    for b in (1, 2, 3):
        adapter.write_block(layer_idx=0, block_id=b, k_block=np.zeros((16, 2, 64), dtype=np.float32), v_block=np.zeros((16, 2, 64), dtype=np.float32))

    # Prefetch blocks 1 and 2
    adapter.prefetch([1, 2], layer_id=0)
    assert len(adapter._staging_buffer) == 2

    # Prefetch block 3 -> evicts block 1 without it being read
    adapter.prefetch([3], layer_id=0)
    assert len(adapter._staging_buffer) == 2

    telem = adapter.get_telemetry()
    assert telem["prefetch_requests"] == 3
    assert telem["useless_prefetches"] == 3  # 1 evicted unused + 2 unaccessed remaining
    assert telem["wasted_bytes"] == 3 * 8192


def test_staging_memory_tracking():
    """Verifies staging memory bytes and MB accounting."""
    adapter = RealInferencePrefetchAdapter(buffer_capacity_blocks=10, bytes_per_block=8192)

    for b in range(5):
        adapter.write_block(layer_idx=0, block_id=b, k_block=np.zeros((16, 2, 64), dtype=np.float32), v_block=np.zeros((16, 2, 64), dtype=np.float32))

    adapter.prefetch(block_ids=list(range(5)), layer_id=0)

    assert adapter.staging_memory_bytes == 5 * 8192
    assert adapter.peak_memory_bytes == 5 * 8192

    telem = adapter.get_telemetry()
    assert telem["staging_memory_bytes"] == 5 * 8192
    assert telem["staging_memory_mb"] == round((5 * 8192) / (1024.0 * 1024.0), 4)


def test_no_artificial_latency():
    """Verifies zero synthetic sleep latency injection (Requirement 9)."""
    adapter = RealInferencePrefetchAdapter()
    for bid in range(50):
        adapter.write_block(layer_idx=0, block_id=bid, k_block=np.zeros((16, 2, 64), dtype=np.float32), v_block=np.zeros((16, 2, 64), dtype=np.float32))

    adapter.prefetch(block_ids=list(range(50)), layer_id=0)

    t0 = time.perf_counter()
    for bid in range(50):
        adapter.read_block(layer_idx=0, block_id=bid)
    elapsed_s = time.perf_counter() - t0

    assert elapsed_s < 0.1, f"Execution was too slow ({elapsed_s:.4f}s), possible artificial delay!"


# =============================================================================
# 3. P2 RealInferenceStorageBackend Wrapper Integration Test
# =============================================================================

@pytest.mark.skipif(not HAS_P2_BACKEND, reason="Person 2 RealInferenceStorageBackend not found")
def test_wrap_p2_real_inference_storage_backend():
    """
    CRITICAL INTEGRATION TEST:
    Verifies that RealInferencePrefetchAdapter directly wraps Person 2's
    RealInferenceStorageBackend, executing tensor-aware channel mapping and real KV retention.
    """
    p2_backend = RealInferenceStorageBackend(
        channels=8,
        num_layers=24,
        num_heads=2,
        tokens_per_block=16,
        head_dim=64,
        dtype="float32",
        mapping_mode="tensor_aware",
    )

    adapter = RealInferencePrefetchAdapter(
        storage_backend=p2_backend,
        buffer_capacity_blocks=256,
        bytes_per_block=LOGICAL_BLOCK_BYTES,
        tokens_per_block=16,
        kv_heads_per_block=2,
        head_dim=64,
        dtype="float32",
    )

    rng = np.random.RandomState(999)
    k_tensor = rng.randn(16, 2, 64).astype(np.float32)
    v_tensor = rng.randn(16, 2, 64).astype(np.float32)

    # 1. Write block into P2 backend via adapter
    adapter.write_block(layer_idx=5, block_id=42, k_block=k_tensor, v_block=v_tensor)
    assert p2_backend.contains_block(5, 42), "Block must be stored inside P2 backend!"

    # 2. Speculatively prefetch block 42 for layer 5
    adapter.prefetch(block_ids=[42], layer_id=5)
    assert (5, 42) in adapter._staging_buffer, "Block must be staged in adapter DRAM!"

    # 3. Demand read Key and Value pages through P1 interface
    k_fetched = adapter.read_key_page(layer_idx=5, block_id=42)
    v_fetched = adapter.read_value_page(layer_idx=5, block_id=42)

    assert np.allclose(k_fetched, k_tensor)
    assert np.allclose(v_fetched, v_tensor)

    # 4. Check telemetry from both adapter and underlying P2 backend
    telem = adapter.get_telemetry()
    assert telem["prefetch_hits"] == 2
    assert "storage_backend" in telem
    assert telem["storage_backend"]["backend_classification"] == "ANALYTICAL"
    assert "channel_distribution" in telem["storage_backend"]


# =============================================================================
# 4. Backend Interoperability & P1 Ingestion
# =============================================================================

def test_file_storage_backend_interoperability():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp_path = f.name

    try:
        block_bytes = 16384
        backend = FileStorageBackend(filepath=tmp_path, block_size=block_bytes)
        adapter = RealInferencePrefetchAdapter(storage_backend=backend, bytes_per_block=block_bytes)

        k_arr = np.ones((16, 2, 64), dtype=np.float32) * 5.5
        v_arr = np.ones((16, 2, 64), dtype=np.float32) * 6.6

        adapter.write_block(layer_idx=0, block_id=0, k_block=k_arr, v_block=v_arr)
        adapter.prefetch(block_ids=[0], layer_id=0)

        kb, vb = adapter.read_block(layer_idx=0, block_id=0)
        assert np.allclose(kb, k_arr)
        assert np.allclose(vb, v_arr)

        adapter.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.skipif(not HAS_P1_BLOCK_ADAPTER, reason="P1 KVBlockAdapter not found")
def test_p1_kv_block_adapter_direct_ingestion():
    p1_adapter = KVBlockAdapter(
        tokens_per_block=16,
        kv_heads_per_block=1,
        head_dim=64,
        dtype="FP32",
    )

    rng = np.random.RandomState(123)
    k_cache = rng.randn(1, 32, 64).astype(np.float32)
    v_cache = rng.randn(1, 32, 64).astype(np.float32)

    blocks = p1_adapter.blockize_layer(layer_id=3, k_tensor=k_cache, v_tensor=v_cache, global_block_offset=10)
    assert len(blocks) == 2

    p3_adapter = RealInferencePrefetchAdapter()
    ingested = p3_adapter.register_blocks_from_adapter(blocks)
    assert ingested == 2

    for desc, payload in blocks:
        ret = p3_adapter.get_block(block_id=desc.block_id, layer_id=desc.layer_id, sub_page="BOTH", return_tensors=True)
        assert np.allclose(ret["k"], payload["k"])
        assert np.allclose(ret["v"], payload["v"])
