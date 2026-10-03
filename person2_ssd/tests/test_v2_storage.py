"""
Unit and Integration Tests for AI-SSD V2 Storage Subsystem (Person 2).

Tests:
    1. NAND Geometry & Constants
    2. Tensor -> LBA Address Mapping
    3. Tensor -> Physical NAND Coordinates
    4. Conventional vs. Tensor-Aware FTL Channel Balancing
    5. Deterministic Trace Replay
    6. Queue Depth & Request Batching
    7. Boundary & Edge Conditions
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    SSD_PAGES_PER_BLOCK,
    SSD_PAGE_SIZE_BYTES,
    DEFAULT_BLOCK_SIZE_BYTES,
)
from person2_ssd.kv_allocator.tensor_mapping import (
    DeterministicTensorMapper,
    TensorCoordinate,
    LBAAddress,
    NANDPhysicalCoordinate,
)
from person2_ssd.ftl.conventional import ConventionalFTL
from person2_ssd.ftl.tensor_aware import TensorAwareFTL
from person2_ssd.trace_replay.replayer import StorageTraceReplayer
from common.schemas.kv_block import KVBlock


def test_nand_geometry():
    """Validates baseline physical geometry constants."""
    print("Testing NAND geometry...", end=" ")
    assert SSD_CHANNELS == 8
    assert SSD_DIES_PER_CHANNEL == 4
    assert SSD_PLANES_PER_DIE == 2
    assert SSD_PAGES_PER_BLOCK == 128
    assert SSD_PAGE_SIZE_BYTES == 4096
    assert DEFAULT_BLOCK_SIZE_BYTES == 4096
    print("[PASSED]")


def test_tensor_lba_mapping():
    """Validates deterministic mapping from TensorCoordinate to LBA."""
    print("Testing Tensor -> LBA mapping...", end=" ")
    mapper = DeterministicTensorMapper(channels=8, dies_per_channel=4)
    coord1 = TensorCoordinate(layer_id=0, head_id=0, token_idx=0)
    coord2 = TensorCoordinate(layer_id=0, head_id=1, token_idx=0)

    lba1 = mapper.tensor_to_lba(coord1, mode="tensor_aware")
    lba2 = mapper.tensor_to_lba(coord2, mode="tensor_aware")

    assert isinstance(lba1, LBAAddress)
    assert isinstance(lba2, LBAAddress)
    assert lba1.lba != lba2.lba
    assert lba1.byte_offset == lba1.lba * 4096
    print("[PASSED]")


def test_tensor_to_physical_nand():
    """Validates mapping from TensorCoordinate to physical NAND coordinates."""
    print("Testing Tensor -> Physical NAND coordinates...", end=" ")
    mapper = DeterministicTensorMapper(channels=8, dies_per_channel=4)
    counters = {c: 0 for c in range(8)}

    coord = TensorCoordinate(layer_id=1, head_id=3, token_idx=32)
    phys = mapper.tensor_to_nand_physical(coord, mode="tensor_aware", channel_counters=counters)

    assert isinstance(phys, NANDPhysicalCoordinate)
    assert 0 <= phys.channel < 8
    assert 0 <= phys.die < 4
    assert 0 <= phys.plane < 2
    assert "ch" in phys.to_location_str()
    print("[PASSED]")


def test_conventional_vs_tensor_aware_ftl():
    """Verifies that Tensor-Aware FTL balances channel load across 8 channels."""
    print("Testing Conventional vs Tensor-Aware FTL...", end=" ")
    conv_ftl = ConventionalFTL(channels=8)
    ta_ftl = TensorAwareFTL(channels=8)

    blocks = [
        KVBlock.create_default(block_id=i, layer_id=0, token_start=0, kv_head_start=i, kv_head_count=1)
        for i in range(16)
    ]

    conv_locs = [conv_ftl.allocate(b) for b in blocks]
    ta_locs = [ta_ftl.allocate(b) for b in blocks]

    # Conventional concentrates everything on channel 0
    conv_ch0_count = sum(1 for loc in conv_locs if "ch0" in loc)
    assert conv_ch0_count == 16, f"Expected 16 on ch0, got {conv_ch0_count}"

    # Tensor-aware distributes evenly (2 per channel for 16 blocks across 8 channels)
    ta_channels = [int(loc.split("ch")[1].split("_")[0]) for loc in ta_locs]
    channel_counts = {c: ta_channels.count(c) for c in range(8)}
    for c in range(8):
        assert channel_counts[c] == 2, f"Channel {c} expected 2 blocks, got {channel_counts[c]}"
    print("[PASSED]")


def test_trace_replayer_deterministic_replay():
    """Verifies that replaying the same trace yields identical results."""
    print("Testing Deterministic Trace Replay...", end=" ")
    replayer = StorageTraceReplayer(channels=8, queue_depth=8)
    trace = replayer.discover_or_synthesize_trace(num_layers=2, num_heads=4, seq_length=64)

    res1 = replayer.replay(trace, mode="tensor_aware")
    res2 = replayer.replay(trace, mode="tensor_aware")

    assert res1["total_requests"] == res2["total_requests"]
    assert res1["bytes_read"] == res2["bytes_read"]
    assert res1["total_simulated_latency_us"] == res2["total_simulated_latency_us"]
    assert res1["contention_ratio"] == res2["contention_ratio"]
    assert res1["channel_access_counts"] == res2["channel_access_counts"]
    print("[PASSED]")


def test_queue_depth_batching():
    """Verifies that queue depth scales batch processing correctly."""
    print("Testing Queue Depth Batching...", end=" ")
    replayer_qd4 = StorageTraceReplayer(channels=8, queue_depth=4)
    replayer_qd16 = StorageTraceReplayer(channels=8, queue_depth=16)

    trace = replayer_qd4.discover_or_synthesize_trace(num_layers=2, num_heads=4, seq_length=128)

    res_qd4 = replayer_qd4.replay(trace, mode="tensor_aware")
    res_qd16 = replayer_qd16.replay(trace, mode="tensor_aware")

    assert res_qd4["total_requests"] == res_qd16["total_requests"]
    assert res_qd16["total_simulated_latency_us"] <= res_qd4["total_simulated_latency_us"]
    print("[PASSED]")


def test_boundary_and_edge_conditions():
    """Tests empty trace, single item, and boundary coordinates."""
    print("Testing Boundary & Edge Conditions...", end=" ")
    replayer = StorageTraceReplayer(channels=8)
    empty_res = replayer.replay([], mode="tensor_aware")
    assert empty_res["total_requests"] == 0
    assert empty_res["total_simulated_latency_us"] == 0.0

    single_trace = [{"op": "READ", "layer_id": 0, "head_id": 0, "token_idx": 0, "bytes": 4096}]
    single_res = replayer.replay(single_trace, mode="tensor_aware")
    assert single_res["total_requests"] == 1
    assert single_res["bytes_read"] == 4096
    assert single_res["total_simulated_latency_us"] > 0.0
    print("[PASSED]")


if __name__ == "__main__":
    print("==================================================")
    print("       AI-SSD V2 Storage Unit Tests (P2)         ")
    print("==================================================")
    test_nand_geometry()
    test_tensor_lba_mapping()
    test_tensor_to_physical_nand()
    test_conventional_vs_tensor_aware_ftl()
    test_trace_replayer_deterministic_replay()
    test_queue_depth_batching()
    test_boundary_and_edge_conditions()
    print("==================================================")
    print("  [SUCCESS] All 7 V2 storage test suites passed!  ")
    print("==================================================")