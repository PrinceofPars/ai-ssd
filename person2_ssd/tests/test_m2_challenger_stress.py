"""
Challenger M2.1 Empirical Stress Test Suite:
FTL Speedup Acceptance, Address Format Conformance, and Bidirectional Mapping Integrity.

Validates:
1. Benchmark speedup >= 2.5x for all batch sizes >= 64 (64, 128, 256).
2. Strict canonical address format: 'ch<C>_die<D>_pl<P>_blk<B>_pg<G>' with correct physical bounds.
3. Bidirectional translation integrity (translate <-> reverse_translate) across 5,000+ allocations.
4. Channel distribution uniformity & absence of odd-channel starvation.
5. Edge cases: unmapped lookups, re-allocations, resets, and StorageSimulator query polymorphism.
"""

import sys
import re
import time
from pathlib import Path
from typing import List, Dict, Set

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.schemas.kv_block import KVBlock, StorageTier
from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    SSD_PAGES_PER_BLOCK,
    T_R_US,
    BUS_TRANSFER_US_PER_PAGE,
    PCIE_OVERHEAD_US,
)
from person2_ssd.ftl.base import BaseFTL
from person2_ssd.ftl.conventional import ConventionalFTL
from person2_ssd.ftl.tensor_aware import TensorAwareFTL
from person2_ssd.mock_kv_engine import MockKVEngine
from person2_ssd.storage_model.io_model import StorageSimulator, parse_physical_location
from person2_ssd.storage_model.latency import LatencyModel


CANONICAL_REGEX = re.compile(
    r"^ch(?P<ch>[0-7])_die(?P<die>[0-3])_pl(?P<pl>[0-1])_blk(?P<blk>\d+)_pg(?P<pg>\d+)$"
)


def suite_1_benchmark_speedup_acceptance():
    print("\n--- Suite 1: Benchmark Speedup Acceptance (16, 32, 64, 128, 256) ---")
    mock = MockKVEngine(layers=32, heads=32)
    batch_sizes = [16, 32, 64, 128, 256]

    for bsz in batch_sizes:
        conv_sim = StorageSimulator(mode="conventional", channels=8)
        ta_sim = StorageSimulator(mode="tensor_aware", channels=8)

        mock.reset()
        blocks = mock.generate_kv_blocks(num_blocks=bsz, layer_id=0)

        for b in blocks:
            conv_sim.store_block(b)
            ta_sim.store_block(b)

        b_ids = [b.block_id for b in blocks]
        conv_lat = conv_sim.estimate_read_latency(b_ids)
        ta_lat = ta_sim.estimate_read_latency(b_ids)
        assert ta_lat > 0, f"Tensor-aware latency must be > 0, got {ta_lat}"
        speedup = conv_lat / ta_lat

        print(f"  Batch {bsz:3d}: Conv = {conv_lat:6.1f} us | TA = {ta_lat:6.1f} us | Speedup = {speedup:.2f}x")

        # R3 / Acceptance Criteria: Speedup >= 2.5x for batch size >= 64
        if bsz >= 64:
            assert speedup >= 2.5, (
                f"FAILED: Speedup {speedup:.2f}x at batch size {bsz} is below the required 2.5x threshold!"
            )
            print(f"    [PASS] Batch {bsz} speedup {speedup:.2f}x >= 2.5x threshold satisfied.")
        else:
            assert speedup >= 1.0, f"Speedup at batch {bsz} must be >= 1.0x, got {speedup:.2f}x"
            print(f"    [PASS] Batch {bsz} speedup {speedup:.2f}x >= 1.0x baseline satisfied.")


def suite_2_speedup_across_extended_batch_sizes():
    print("\n--- Suite 2: Extended Batch Sizes & Attention Layouts ---")
    mock = MockKVEngine(layers=32, heads=32)
    extended_batches = [48, 64, 80, 96, 128, 160, 192, 256, 384, 512]

    for bsz in extended_batches:
        conv_sim = StorageSimulator(mode="conventional", channels=8)
        ta_sim = StorageSimulator(mode="tensor_aware", channels=8)

        mock.reset()
        blocks = mock.generate_kv_blocks(num_blocks=bsz, layer_id=0, layout="token_major")
        for b in blocks:
            conv_sim.store_block(b)
            ta_sim.store_block(b)

        b_ids = [b.block_id for b in blocks]
        conv_lat = conv_sim.estimate_read_latency(b_ids)
        ta_lat = ta_sim.estimate_read_latency(b_ids)
        speedup = conv_lat / ta_lat

        assert speedup >= 2.5, f"Extended batch {bsz} failed >= 2.5x requirement: got {speedup:.2f}x"
        print(f"  [PASS] Extended Batch {bsz:4d}: Conv={conv_lat:7.1f}us, TA={ta_lat:7.1f}us -> Speedup={speedup:.2f}x")

    # Test Head-Major layout
    mock.reset()
    hm_blocks = mock.generate_kv_blocks(num_blocks=128, layer_id=0, layout="head_major")
    conv_sim = StorageSimulator(mode="conventional", channels=8)
    ta_sim = StorageSimulator(mode="tensor_aware", channels=8)
    for b in hm_blocks:
        conv_sim.store_block(b)
        ta_sim.store_block(b)
    b_ids = [b.block_id for b in hm_blocks]
    conv_lat = conv_sim.estimate_read_latency(b_ids)
    ta_lat = ta_sim.estimate_read_latency(b_ids)
    speedup = conv_lat / ta_lat
    print(f"  [PASS] Head-Major Batch 128: Conv={conv_lat:7.1f}us, TA={ta_lat:7.1f}us -> Speedup={speedup:.2f}x")
    assert speedup >= 2.5, f"Head-Major layout failed >= 2.5x requirement: got {speedup:.2f}x"

    # Test Sparse Attention Request (Sink + Recent tokens)
    mock.reset()
    pool_blocks = mock.generate_kv_blocks(num_blocks=256, layer_id=0)
    for b in pool_blocks:
        conv_sim.store_block(b)
        ta_sim.store_block(b)
    sparse_ids = mock.generate_sparse_attention_request(pool_blocks, k=64, sink_ratio=0.25)
    assert len(sparse_ids) == 64
    conv_lat = conv_sim.estimate_read_latency(sparse_ids)
    ta_lat = ta_sim.estimate_read_latency(sparse_ids)
    speedup = conv_lat / ta_lat
    print(f"  [PASS] Sparse Attention k=64: Conv={conv_lat:7.1f}us, TA={ta_lat:7.1f}us -> Speedup={speedup:.2f}x")
    assert speedup >= 2.5, f"Sparse attention request failed >= 2.5x requirement: got {speedup:.2f}x"


def suite_3_canonical_address_format_thousands_of_blocks():
    print("\n--- Suite 3: Canonical Address Format Strict Conformance (5,000+ Allocations) ---")
    N = 6000
    mock = MockKVEngine(layers=32, heads=32)

    for mode, ftl_cls in [("Conventional", ConventionalFTL), ("TensorAware", TensorAwareFTL)]:
        ftl = ftl_cls(channels=8, dies_per_channel=4, planes_per_die=2, blocks_per_plane=64, pages_per_block=128)
        mock.reset()
        blocks = mock.generate_kv_blocks(num_blocks=N)

        for i, b in enumerate(blocks):
            loc = ftl.allocate(b)
            # 1. Exact string regex match
            m = CANONICAL_REGEX.match(loc)
            assert m is not None, f"{mode} allocation #{i} format invalid: '{loc}'"

            ch = int(m.group("ch"))
            die = int(m.group("die"))
            pl = int(m.group("pl"))
            blk = int(m.group("blk"))
            pg = int(m.group("pg"))

            # 2. Strict physical bounds
            assert 0 <= ch < 8, f"{mode} channel out of bounds: {ch}"
            assert 0 <= die < 4, f"{mode} die out of bounds: {die}"
            assert 0 <= pl < 2, f"{mode} plane out of bounds: {pl}"
            assert 0 <= blk < 64, f"{mode} block out of bounds: {blk}"
            assert 0 <= pg < 128, f"{mode} page out of bounds: {pg}"

            # 3. parse_physical_location helper consistency
            parsed = parse_physical_location(loc)
            assert parsed == (ch, die, pl, blk, pg), f"parse_physical_location mismatch for '{loc}': {parsed}"

            # 4. LatencyModel.extract_channel consistency
            extracted_ch = LatencyModel.extract_channel(loc)
            assert extracted_ch == ch, f"extract_channel mismatch for '{loc}': {extracted_ch} != {ch}"

        print(f"  [PASS] Verified {N} consecutive {mode} allocations strictly adhere to canonical format.")


def suite_4_bidirectional_translation_integrity():
    print("\n--- Suite 4: Bidirectional Translation Integrity (5,000+ Blocks) ---")
    N = 5000
    mock = MockKVEngine(layers=16, heads=16)

    for mode, ftl_cls in [("Conventional", ConventionalFTL), ("TensorAware", TensorAwareFTL)]:
        ftl = ftl_cls(channels=8)
        mock.reset()
        blocks = mock.generate_kv_blocks(num_blocks=N)

        # Track allocated pairs
        expected_mappings: Dict[int, str] = {}
        for b in blocks:
            loc = ftl.allocate(b)
            expected_mappings[b.block_id] = loc

        # Verify forward translation contracts
        for bid, exp_loc in expected_mappings.items():
            trans_loc = ftl.translate(bid)
            get_loc = ftl.get_location(bid)
            assert trans_loc == exp_loc, f"translate({bid}) expected {exp_loc}, got {trans_loc}"
            assert get_loc == exp_loc, f"get_location({bid}) expected {exp_loc}, got {get_loc}"

        # Verify reverse translation contracts
        for bid, exp_loc in expected_mappings.items():
            rev_bid = ftl.reverse_translate(exp_loc)
            assert rev_bid == bid, f"reverse_translate('{exp_loc}') expected {bid}, got {rev_bid}"

        # Verify table snapshot integrity
        fwd_table = ftl.get_mapping_table()
        rev_table = ftl.get_reverse_mapping_table()
        assert len(fwd_table) == N, f"Forward mapping table size mismatch: {len(fwd_table)} vs {N}"
        assert len(rev_table) == N, f"Reverse mapping table size mismatch: {len(rev_table)} vs {N}"

        for bid, loc in fwd_table.items():
            assert expected_mappings[bid] == loc
            assert rev_table[loc] == bid

        print(f"  [PASS] {mode}: 100% bidirectional translation parity across all {N} allocations.")


def suite_5_unmapped_and_edge_case_lookups():
    print("\n--- Suite 5: Unmapped & Edge Case Lookups ---")
    ftl = TensorAwareFTL()
    conv = ConventionalFTL()

    for name, f in [("TensorAware", ftl), ("Conventional", conv)]:
        # Lookups on empty FTL
        assert f.translate(9999) is None
        assert f.get_location(9999) is None
        assert f.reverse_translate("ch0_die0_pl0_blk0_pg0") is None
        assert f.reverse_translate("") is None
        assert f.reverse_translate("invalid_address_string") is None

        # Negative and zero IDs
        b0 = KVBlock.create_default(block_id=0, layer_id=0, token_start=0)
        loc0 = f.allocate(b0)
        assert f.translate(0) == loc0
        assert f.reverse_translate(loc0) == 0

        # Non-existent negative ID
        assert f.translate(-1) is None
        assert f.translate(-999) is None

        print(f"  [PASS] {name}: Edge cases (empty, non-existent, 0-id, negative, malformed) handled cleanly.")


def suite_6_reallocation_and_reverse_map_consistency():
    print("\n--- Suite 6: Re-allocation & Overwrite Synchronization ---")
    ftl = TensorAwareFTL()

    b1 = KVBlock.create_default(block_id=42, layer_id=0, token_start=0)
    loc_v1 = ftl.allocate(b1)
    assert ftl.translate(42) == loc_v1
    assert ftl.reverse_translate(loc_v1) == 42

    # Re-allocate block 42 with different parameters (simulating update/migration)
    b1_updated = KVBlock.create_default(block_id=42, layer_id=1, token_start=16)
    loc_v2 = ftl.allocate(b1_updated)
    assert loc_v2 != loc_v1, f"Re-allocation expected different physical address, got {loc_v2}"

    # Forward lookup must return new location
    assert ftl.translate(42) == loc_v2
    # Reverse lookup of new location must return 42
    assert ftl.reverse_translate(loc_v2) == 42
    # Reverse lookup of old location must be evicted (return None)
    assert ftl.reverse_translate(loc_v1) is None, (
        f"Stale mapping detected! reverse_translate('{loc_v1}') should be None, got {ftl.reverse_translate(loc_v1)}"
    )
    print("  [PASS] Re-allocation evicted stale reverse mapping and bound new physical address.")


def suite_7_channel_distribution_and_parity_starvation():
    print("\n--- Suite 7: Channel Distribution Uniformity (No Odd-Channel Starvation) ---")
    mock = MockKVEngine(layers=32, heads=32)

    for bsz in [32, 64, 128, 256, 512]:
        ta = TensorAwareFTL(channels=8)
        mock.reset()
        blocks = mock.generate_kv_blocks(num_blocks=bsz, layer_id=0)

        channel_counts = {c: 0 for c in range(8)}
        for b in blocks:
            loc = ta.allocate(b)
            ch = LatencyModel.extract_channel(loc)
            channel_counts[ch] += 1

        print(f"  Batch {bsz:3d} channel distribution: {channel_counts}")

        # Check: All 8 channels MUST have non-zero allocation
        for c in range(8):
            assert channel_counts[c] > 0, f"Channel {c} starved at batch size {bsz}!"

        # Check: Distribution must be balanced
        expected_per_channel = bsz // 8
        for c in range(8):
            assert channel_counts[c] == expected_per_channel, (
                f"Channel {c} count {channel_counts[c]} != expected {expected_per_channel} at batch {bsz}"
            )
        print(f"  [PASS] Batch {bsz:3d}: 100% uniform (exactly {expected_per_channel} blocks/channel).")


def suite_8_reset_lifecycle():
    print("\n--- Suite 8: FTL Reset Lifecycle & Idempotence ---")
    conv = ConventionalFTL()
    ta = TensorAwareFTL()
    mock = MockKVEngine()

    for name, ftl in [("Conventional", conv), ("TensorAware", ta)]:
        mock.reset()
        blocks = mock.generate_kv_blocks(num_blocks=64)
        for b in blocks:
            ftl.allocate(b)

        assert len(ftl.get_mapping_table()) == 64
        assert len(ftl.get_reverse_mapping_table()) == 64

        # Reset
        ftl.reset()
        assert len(ftl.get_mapping_table()) == 0
        assert len(ftl.get_reverse_mapping_table()) == 0
        assert ftl.translate(0) is None

        # Re-allocate after reset
        mock.reset()
        new_blocks = mock.generate_kv_blocks(num_blocks=16)
        for b in new_blocks:
            ftl.allocate(b)
        assert len(ftl.get_mapping_table()) == 16
        assert len(ftl.get_reverse_mapping_table()) == 16
        print(f"  [PASS] {name}: reset() cleanly cleared tables and permitted fresh re-allocations.")


def suite_9_capacity_exhaustion_in_conventional():
    print("\n--- Suite 9: Capacity Exhaustion Bounds in Conventional FTL ---")
    # Tiny SSD: 1 channel, 1 die, 1 plane, 2 blocks, 2 pages = 4 pages total
    tiny_conv = ConventionalFTL(channels=1, dies_per_channel=1, planes_per_die=1, blocks_per_plane=2, pages_per_block=2)
    assert tiny_conv.total_pages == 4

    for i in range(4):
        b = KVBlock.create_default(block_id=i, layer_id=0, token_start=0)
        loc = tiny_conv.allocate(b)
        assert loc is not None

    # 5th allocation must raise RuntimeError
    overflow_b = KVBlock.create_default(block_id=99, layer_id=0, token_start=0)
    try:
        tiny_conv.allocate(overflow_b)
        assert False, "Expected RuntimeError on capacity exhaustion"
    except RuntimeError as e:
        assert "capacity exceeded" in str(e).lower()
        print(f"  [PASS] Capacity exhaustion correctly raised RuntimeError: {e}")


def suite_10_storage_simulator_polymorphism_and_diagnostics():
    print("\n--- Suite 10: StorageSimulator Polymorphism & Latency Model Breakdown ---")
    sim = StorageSimulator(mode="tensor_aware")
    mock = MockKVEngine()
    blocks = mock.generate_kv_blocks(num_blocks=32)

    for b in blocks:
        sim.store_block(b)

    # 1. Query by KVBlock objects
    lat_objs = sim.read_blocks(blocks)
    # 2. Query by integer IDs
    lat_ints = sim.read_blocks([b.block_id for b in blocks])
    # 3. Query by mixed list
    mixed_list = [blocks[0], blocks[1].block_id, blocks[2]]
    lat_mix = sim.read_blocks(mixed_list)

    assert lat_objs == lat_ints
    assert lat_objs > 0.0
    assert lat_mix > 0.0
    print(f"  [PASS] StorageSimulator polymorphic queries consistent: lat_objs={lat_objs}us == lat_ints={lat_ints}us")

    # 4. Latency breakdown diagnostics
    locs = [sim.get_location(b.block_id) for b in blocks]
    breakdown = sim.latency_model.get_latency_breakdown(locs)
    assert breakdown["total_requests"] == 32
    assert breakdown["max_channel_load"] == 4  # 32 / 8 channels = 4
    expected_read_time = PCIE_OVERHEAD_US + 4 * (T_R_US + BUS_TRANSFER_US_PER_PAGE)
    assert abs(breakdown["total_read_latency_us"] - expected_read_time) < 1e-6
    print(f"  [PASS] Latency diagnostic breakdown verified: max_channel_load=4, read_latency={expected_read_time}us.")


def main():
    print("===================================================================")
    print("  CHALLENGER M2.1: EMPIRICAL FTL STRESS TEST & ACCEPTANCE HARNESS  ")
    print("===================================================================")

    t0 = time.perf_counter()
    try:
        suite_1_benchmark_speedup_acceptance()
        suite_2_speedup_across_extended_batch_sizes()
        suite_3_canonical_address_format_thousands_of_blocks()
        suite_4_bidirectional_translation_integrity()
        suite_5_unmapped_and_edge_case_lookups()
        suite_6_reallocation_and_reverse_map_consistency()
        suite_7_channel_distribution_and_parity_starvation()
        suite_8_reset_lifecycle()
        suite_9_capacity_exhaustion_in_conventional()
        suite_10_storage_simulator_polymorphism_and_diagnostics()

        elapsed = time.perf_counter() - t0
        print("\n===================================================================")
        print(f"  ALL 10 EMPIRICAL STRESS SUITES PASSED CLEANLY in {elapsed:.3f}s!  ")
        print("===================================================================")
        return 0
    except Exception as e:
        print(f"\n[FATAL CHALLENGER STRESS FAILURE]: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
