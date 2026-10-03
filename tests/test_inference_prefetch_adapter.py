"""
Unit and Integration Tests for RealInferencePrefetchAdapter (Phase 5C).

Validates:
- Actual KV block data retrieval (bytes and NumPy tensors).
- Sub-page slicing (KEY 4KB, VALUE 4KB, BOTH 8KB).
- Direct ingestion from P1 KVBlockAdapter.
- Non-blocking speculative prefetching (useful, late, and useless prefetches).
- Accurate metric accounting (demand reads, prefetch requests, hit rates, bytes, latency).
- Zero artificial latency injection.
- Backend interoperability (MockStorageBackend, FileStorageBackend, AnalyticalFTLBackend).
"""

import pytest
import os
import sys
import tempfile
import time
import numpy as np

from common.schemas.kv_block import KEY_PAGE_BYTES, VALUE_PAGE_BYTES, LOGICAL_BLOCK_BYTES
from person3_system.storage.mock_backend import MockStorageBackend
from person3_system.storage.file_backend import FileStorageBackend
from person3_system.storage.analytical_backend import AnalyticalFTLBackend
from person3_system.prefetch.inference_adapter import RealInferencePrefetchAdapter

import importlib.util

p1_adapter_path = "/home/ubuntu/ai-ssd-p1/person1_kv_engine/real_llm/block_adapter.py"
HAS_P1_BLOCK_ADAPTER = False
KVBlockAdapter = None

if os.path.exists(p1_adapter_path):
    try:
        spec = importlib.util.spec_from_file_location("p1_real_block_adapter", p1_adapter_path)
        p1_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p1_mod)
        KVBlockAdapter = getattr(p1_mod, "KVBlockAdapter", None)
        HAS_P1_BLOCK_ADAPTER = KVBlockAdapter is not None
    except Exception:
        HAS_P1_BLOCK_ADAPTER = False


def test_adapter_initialization():
    backend = MockStorageBackend()
    adapter = RealInferencePrefetchAdapter(
        storage_backend=backend,
        buffer_capacity_blocks=64,
        bytes_per_block=LOGICAL_BLOCK_BYTES,
        tokens_per_block=16,
        kv_heads_per_block=1,
        head_dim=64,
        dtype="FP32",
    )
    assert adapter.buffer_capacity_blocks == 64
    assert adapter.bytes_per_block == 8192
    assert adapter.current_memory_bytes == 0
    assert adapter.demand_reads == 0
    assert adapter.predictor is not None


def test_register_and_get_raw_bytes():
    adapter = RealInferencePrefetchAdapter()
    
    # 8192 bytes payload: 4096 'K' bytes and 4096 'V' bytes
    k_raw = b"K" * 4096
    v_raw = b"V" * 4096
    payload = k_raw + v_raw

    res = adapter.register_block(block_id=1, layer_id=0, payload=payload)
    assert res.success is True

    # 1. Retrieve full combined block
    data_both = adapter.get_block(block_id=1, layer_id=0, sub_page="BOTH")
    assert len(data_both) == 8192
    assert data_both == payload

    # 2. Retrieve KEY page only
    data_k = adapter.get_block(block_id=1, layer_id=0, sub_page="KEY")
    assert len(data_k) == 4096
    assert data_k == k_raw

    # 3. Retrieve VALUE page only
    data_v = adapter.get_block(block_id=1, layer_id=0, sub_page="VALUE")
    assert len(data_v) == 4096
    assert data_v == v_raw


def test_register_and_get_real_tensors():
    adapter = RealInferencePrefetchAdapter(head_dim=64, dtype="FP32")

    # Real Qwen2.5-0.5B block geometry: [1, 16, 64] float32 = 4096 bytes each
    rng = np.random.RandomState(42)
    k_tensor = rng.randn(1, 16, 64).astype(np.float32)
    v_tensor = rng.randn(1, 16, 64).astype(np.float32)

    assert k_tensor.nbytes == 4096
    assert v_tensor.nbytes == 4096

    adapter.register_block(
        block_id=5,
        layer_id=2,
        head_id=0,
        payload={"k": k_tensor, "v": v_tensor},
    )

    # 1. Retrieve BOTH as tensors
    res_both = adapter.get_block(block_id=5, layer_id=2, sub_page="BOTH", return_tensors=True)
    assert isinstance(res_both, dict)
    assert "k" in res_both and "v" in res_both
    assert np.allclose(res_both["k"], k_tensor)
    assert np.allclose(res_both["v"], v_tensor)

    # 2. Retrieve KEY only as tensor
    res_k = adapter.get_block(block_id=5, layer_id=2, sub_page="KEY", return_tensors=True)
    assert isinstance(res_k, np.ndarray)
    assert res_k.shape == (1, 16, 64)
    assert np.allclose(res_k, k_tensor)

    # 3. Retrieve VALUE only as tensor
    res_v = adapter.get_block(block_id=5, layer_id=2, sub_page="VALUE", return_tensors=True)
    assert isinstance(res_v, np.ndarray)
    assert res_v.shape == (1, 16, 64)
    assert np.allclose(res_v, v_tensor)


def test_p1_kv_block_adapter_direct_ingestion():
    if not HAS_P1_BLOCK_ADAPTER:
        pytest.skip("P1 KVBlockAdapter not available in current environment")

    p1_adapter = KVBlockAdapter(
        tokens_per_block=16,
        kv_heads_per_block=1,
        head_dim=64,
        dtype="FP32",
    )

    # Simulate layer 3 KV cache: 1 head, 32 tokens, 64 dim -> 2 blocks
    rng = np.random.RandomState(123)
    k_cache = rng.randn(1, 32, 64).astype(np.float32)
    v_cache = rng.randn(1, 32, 64).astype(np.float32)

    blocks = p1_adapter.blockize_layer(layer_id=3, k_tensor=k_cache, v_tensor=v_cache, global_block_offset=10)
    assert len(blocks) == 2

    # Ingest into P3 RealInferencePrefetchAdapter
    p3_adapter = RealInferencePrefetchAdapter()
    ingested = p3_adapter.register_blocks_from_adapter(blocks)
    assert ingested == 2

    # Verify retrieval of both blocks
    for desc, payload in blocks:
        ret = p3_adapter.get_block(block_id=desc.block_id, layer_id=desc.layer_id, sub_page="BOTH", return_tensors=True)
        assert np.allclose(ret["k"], payload["k"])
        assert np.allclose(ret["v"], payload["v"])


def test_speculative_prefetch_hit_and_accounting():
    adapter = RealInferencePrefetchAdapter()

    # Pre-register blocks 10, 11, 12
    for bid in (10, 11, 12):
        adapter.register_block(block_id=bid, layer_id=1, payload=f"block_{bid}_payload".encode("utf-8") * 100)

    # Speculatively prefetch blocks 10 and 11
    dispatched = adapter.prefetch(block_ids=[10, 11], layer_id=1)
    assert dispatched == [10, 11]
    assert adapter.prefetch_requests == 2

    # Demand read blocks 10 and 11 (Hits!)
    b10 = adapter.get_block(block_id=10, layer_id=1)
    b11 = adapter.get_block(block_id=11, layer_id=1)
    assert len(b10) == 8192
    assert len(b11) == 8192

    # Demand read block 12 (Miss!)
    b12 = adapter.get_block(block_id=12, layer_id=1)
    assert len(b12) == 8192

    telem = adapter.get_telemetry()
    assert telem["demand_reads"] == 3
    assert telem["demand_hits"] == 2
    assert telem["demand_misses"] == 1
    assert telem["demand_hit_rate"] == round(2 / 3, 4)
    assert telem["prefetch_requests"] == 2
    assert telem["useful_prefetches"] == 2
    assert telem["prefetch_accuracy"] == 1.0


def test_useless_prefetch_buffer_eviction():
    # Set capacity to 2 blocks
    adapter = RealInferencePrefetchAdapter(buffer_capacity_blocks=2)

    adapter.register_block(block_id=1, layer_id=0, payload=b"one" * 100)
    adapter.register_block(block_id=2, layer_id=0, payload=b"two" * 100)
    adapter.register_block(block_id=3, layer_id=0, payload=b"three" * 100)

    # Prefetch blocks 1 and 2 (fills buffer)
    adapter.prefetch(block_ids=[1, 2], layer_id=0)
    assert len(adapter._staging_buffer) == 2

    # Prefetch block 3 -> evicts block 1 without it ever being accessed!
    adapter.prefetch(block_ids=[3], layer_id=0)
    assert len(adapter._staging_buffer) == 2

    telem = adapter.get_telemetry()
    assert telem["prefetch_requests"] == 3
    # Block 1 was evicted without access -> useless!
    # Blocks 2 and 3 remain unaccessed at evaluation -> also useless!
    assert telem["useless_prefetches"] == 3
    assert telem["wasted_bytes"] == 3 * 8192


def test_predict_and_prefetch():
    adapter = RealInferencePrefetchAdapter()
    
    # Layer 0 accesses blocks [20, 21]
    # Predictor predicts Layer 1 needs blocks [20, 21]
    next_layer, prefetched = adapter.predict_and_prefetch(current_layer_id=0, current_block_ids=[20, 21])
    assert next_layer == 1
    assert prefetched == [20, 21]
    assert adapter.prefetch_requests == 2


def test_no_artificial_latency():
    adapter = RealInferencePrefetchAdapter()
    for bid in range(50):
        adapter.register_block(block_id=bid, layer_id=0, payload=b"quick_data" * 50)

    adapter.prefetch(block_ids=list(range(50)), layer_id=0)

    t0 = time.perf_counter()
    for bid in range(50):
        adapter.get_block(block_id=bid, layer_id=0)
    total_time_s = time.perf_counter() - t0

    # 50 gets should complete in well under 100 milliseconds (< 2ms per block)
    assert total_time_s < 0.1, f"Execution was too slow ({total_time_s:.4f}s), possible artificial delay!"


def test_file_storage_backend_integration():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp_file = f.name

    try:
        backend = FileStorageBackend(filepath=tmp_file, block_size=LOGICAL_BLOCK_BYTES)
        adapter = RealInferencePrefetchAdapter(storage_backend=backend)

        payload_0 = b"persistent_block_0_data" + (b"0" * (8192 - 23))
        payload_1 = b"persistent_block_1_data" + (b"1" * (8192 - 23))

        adapter.register_block(block_id=0, layer_id=0, payload=payload_0)
        adapter.register_block(block_id=1, layer_id=0, payload=payload_1)

        # Prefetch block 0
        adapter.prefetch(block_ids=[0], layer_id=0)

        # Retrieve both blocks
        b0 = adapter.get_block(block_id=0, layer_id=0)
        b1 = adapter.get_block(block_id=1, layer_id=0)

        assert b0 == payload_0
        assert b1 == payload_1

        telem = adapter.get_telemetry()
        assert telem["demand_hits"] == 1
        assert telem["demand_misses"] == 1

        adapter.close()
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def test_analytical_ftl_backend_integration():
    backend = AnalyticalFTLBackend(mode="tensor_aware", channels=8)
    adapter = RealInferencePrefetchAdapter(storage_backend=backend)

    payload = b"analytical_block_test" + (b"X" * (8192 - 21))
    adapter.register_block(block_id=7, layer_id=1, payload=payload)

    adapter.prefetch(block_ids=[7], layer_id=1)
    ret = adapter.get_block(block_id=7, layer_id=1)
    assert ret == payload

    telem = adapter.get_telemetry()
    assert telem["demand_hits"] == 1
    assert telem["useful_prefetches"] == 1
