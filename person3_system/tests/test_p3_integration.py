"""
Unit and Integration Tests for Person 3 System Subsystem:
- PhysicalTensorAwareStorageAdapter (Person 1 <-> Person 2 bridge)
- SpeculativePrefetcher (LRU buffer, next-layer pipeline, bubble penalties)
- SystemOrchestrator and InferencePipeline (Full End-to-End Simulation)
"""

import numpy as np
import pytest

from person3_system.adapters.physical_adapter import PhysicalTensorAwareStorageAdapter
from person3_system.prefetch.prefetcher import SpeculativePrefetcher
from person3_system.prefetch.predictor import NextLayerPredictor
from person3_system.integration.orchestrator import SystemOrchestrator
from person3_system.integration.pipeline import InferencePipeline
from common.schemas.request import KVRequest, KVOperation
from common.schemas.result import OperationStatus


def test_physical_adapter_store_and_load():
    adapter = PhysicalTensorAwareStorageAdapter(mode="tensor_aware", channels=8)
    k = np.random.randn(16, 8, 64).astype(np.float32)
    v = np.random.randn(16, 8, 64).astype(np.float32)

    # Store block in layer 0
    ok = adapter.store_kv(block_id=0, layer_id=0, key_data=k, value_data=v)
    assert ok is True

    # Load block back
    k_ret, v_ret = adapter.load_kv(block_id=0, layer_id=0)
    assert np.allclose(k, k_ret)
    assert np.allclose(v, v_ret)

    # Telemetry check
    telem = adapter.get_telemetry()
    assert telem["flash_write_bytes"] > 0
    assert telem["flash_read_bytes"] > 0
    assert "total_joules" in telem["energy_joules"]


def test_physical_adapter_in_storage_topk():
    adapter = PhysicalTensorAwareStorageAdapter(mode="tensor_aware", channels=8)
    num_blocks = 8

    for bid in range(num_blocks):
        k = np.random.randn(16, 4, 32).astype(np.float32)
        v = np.random.randn(16, 4, 32).astype(np.float32)
        adapter.store_kv(block_id=bid, layer_id=1, key_data=k, value_data=v)

    query = np.random.randn(4, 32).astype(np.float32)
    top_k = 3

    topk_ids, topk_vals, topk_logits, topk_scores = adapter.in_storage_topk_attention(
        query=query,
        layer_id=1,
        top_k=top_k,
    )

    assert len(topk_ids) == top_k
    assert topk_vals.shape == (top_k, 16, 4, 32)
    assert topk_logits.shape == (4, top_k * 16)
    assert len(topk_scores) == top_k
    # Top-k scores should be monotonically non-increasing
    for i in range(len(topk_scores) - 1):
        assert topk_scores[i] >= topk_scores[i + 1]


def test_speculative_prefetcher_lru_and_overlap():
    prefetcher = SpeculativePrefetcher(
        buffer_capacity_blocks=4,
        bytes_per_block=4096,
        gpu_compute_time_per_layer_us=65.0,
    )

    # Stage 4 blocks for Layer 1
    prefetcher.stage_blocks([10, 11, 12, 13], layer_id=1)
    assert prefetcher.used_capacity_blocks == 4

    # Adding a 5th block should trigger LRU eviction of block 10
    prefetcher.stage_blocks([14], layer_id=1)
    assert prefetcher.used_capacity_blocks == 4
    assert (1, 10) not in prefetcher._staged_buffer
    assert (1, 14) in prefetcher._staged_buffer

    # Test hit
    hit = prefetcher.is_staged([11, 12, 13, 14], layer_id=1, estimated_flash_latency_us=50.0)
    assert hit is True
    assert prefetcher.hits == 1
    assert prefetcher.pipeline_stalls == 0


def test_next_layer_predictor():
    predictor = NextLayerPredictor(total_layers=32)
    next_layer, predicted = predictor.predict_next_layer_blocks(
        current_layer_id=5,
        current_block_ids=[10, 20, 30],
    )
    assert next_layer == 6
    assert predicted == [10, 20, 30]


def test_system_orchestrator_wiring():
    orch = SystemOrchestrator(
        mode="tensor_aware",
        num_layers=4,
        num_heads=8,
        head_dim=64,
        dtype="FP16",
        enable_prefetch=True,
    )
    assert isinstance(orch.ssd_engine, PhysicalTensorAwareStorageAdapter)
    assert orch.prefetcher is not None

    telem = orch.get_telemetry()
    assert "storage" in telem
    assert "prefetch" in telem


def test_full_pipeline_simulation():
    orch = SystemOrchestrator(
        mode="tensor_aware",
        num_layers=8,
        num_heads=8,
        head_dim=64,
        dtype="FP16",
        enable_prefetch=True,
    )
    pipeline = InferencePipeline(orchestrator=orch)

    res = pipeline.run_simulation(
        context_length=4096,
        num_layers=8,
        num_heads=8,
        head_dim=64,
        offload_pct=80.0,
        topk_pct=10.0,
        dtype="FP16",
    )

    assert res["memory"]["reduction_percent"] == 80.0
    assert res["storage"]["traffic_reduction_percent"] >= 80.0
    assert res["ftl"]["speedup_x"] >= 5.0
    assert res["prefetch"]["cache_hit_rate"] >= 0.80
    assert res["latency"]["overhead_percent"] <= 15.0
