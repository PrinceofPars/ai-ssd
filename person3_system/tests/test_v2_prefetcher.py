import pytest
from person3_system.storage.mock_backend import MockStorageBackend
from person3_system.prefetch.v2_prefetcher import V2Prefetcher


def test_v2_prefetcher_accounting():
    backend = MockStorageBackend(read_latency_us=25.0)
    prefetcher = V2Prefetcher(
        storage_backend=backend,
        buffer_capacity_blocks=16,
        bytes_per_block=4096,
        gpu_compute_time_per_layer_us=50.0,
    )

    # 1. Speculatively prefetch blocks [1, 2, 3] for Layer 1 at t=0
    prefetched = prefetcher.prefetch_blocks(block_ids=[1, 2, 3], layer_id=1, current_time_us=0.0)
    assert prefetched == [1, 2, 3]
    assert prefetcher.prefetch_requests == 3
    assert backend.total_reads == 3  # Explicit storage read charged!

    # 2. Demand read for Layer 1 requests blocks [1, 2, 4] at t=100.0 (blocks 1, 2 hit; block 4 misses)
    hits, misses, stall_us = prefetcher.access_blocks(
        block_ids=[1, 2, 4],
        layer_id=1,
        demand_time_us=100.0,
    )
    assert hits == [1, 2]
    assert misses == [4]
    # Missing block 4 causes an on-demand read:
    assert backend.total_reads == 4

    telem = prefetcher.get_telemetry()
    assert telem["useful_prefetches"] == 2
    assert telem["useless_prefetches"] == 1  # block 3 was never accessed!
    assert telem["demand_hits"] == 2
    assert telem["demand_misses"] == 1
    assert telem["demand_hit_rate"] == round(2 / 3, 4)
    assert telem["extra_bytes_read"] == 4096  # 1 useless block * 4096


def test_no_prefetch_vs_prefetch_comparison():
    # Workload: Layer 0 accesses [10, 11], Layer 1 accesses [20, 21]
    # Run 1: No prefetch (baseline)
    backend_noprefetch = MockStorageBackend(read_latency_us=30.0)
    pref_disabled = V2Prefetcher(storage_backend=backend_noprefetch, gpu_compute_time_per_layer_us=40.0)

    # Layer 0 demand
    pref_disabled.access_blocks([10, 11], layer_id=0, demand_time_us=0.0)
    # Layer 1 demand
    pref_disabled.access_blocks([20, 21], layer_id=1, demand_time_us=100.0)

    telem_nopref = pref_disabled.get_telemetry()
    assert telem_nopref["demand_hits"] == 0
    assert telem_nopref["demand_misses"] == 4
    assert telem_nopref["prefetch_requests"] == 0
    assert backend_noprefetch.total_reads == 4

    # Run 2: Prefetch enabled
    backend_pref = MockStorageBackend(read_latency_us=30.0)
    pref_enabled = V2Prefetcher(storage_backend=backend_pref, gpu_compute_time_per_layer_us=40.0)

    # Layer 0 prefetch for Layer 1
    pref_enabled.prefetch_blocks([20, 21], layer_id=1, current_time_us=0.0)
    # Layer 0 demand
    pref_enabled.access_blocks([10, 11], layer_id=0, demand_time_us=0.0)
    # Layer 1 demand
    pref_enabled.access_blocks([20, 21], layer_id=1, demand_time_us=100.0)

    telem_pref = pref_enabled.get_telemetry()
    assert telem_pref["demand_hits"] == 2
    assert telem_pref["demand_misses"] == 2  # layer 0 missed
    assert telem_pref["useful_prefetches"] == 2
    assert telem_pref["useless_prefetches"] == 0
    # Prefetch saved latency on layer 1 because I/O overlapped with compute!
    assert telem_pref["total_stall_penalty_us"] < telem_nopref["total_stall_penalty_us"]
