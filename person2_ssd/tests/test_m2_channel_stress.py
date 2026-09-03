"""
Empirical Stress Test Suite: Channel Load Balance, Zero Odd-Channel Starvation,
and Edge Cases for Milestone 2 (Challenger M2.2).
"""

import math
import re
import sys
from pathlib import Path
from typing import List, Dict, Set

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.schemas.kv_block import KVBlock, StorageTier
from person2_ssd.ftl.base import BaseFTL
from person2_ssd.ftl.conventional import ConventionalFTL
from person2_ssd.ftl.tensor_aware import TensorAwareFTL
from person2_ssd.mock_kv_engine import MockKVEngine
from person2_ssd.storage_model.io_model import StorageSimulator
from person2_ssd.storage_model.latency import LatencyModel

CANONICAL_ADDR_REGEX = re.compile(
    r"^ch(?P<ch>[0-7])_die(?P<die>[0-3])_pl(?P<pl>[0-1])_blk(?P<blk>\d+)_pg(?P<pg>\d+)$"
)


def _get_channel_distribution(ftl: BaseFTL, blocks: List[KVBlock], channels: int = 8) -> List[int]:
    loads = [0] * channels
    for b in blocks:
        loc = ftl.translate(b.block_id)
        assert loc is not None, f"Block {b.block_id} has no mapping!"
        m = CANONICAL_ADDR_REGEX.match(loc)
        assert m is not None, f"Location {loc} does not match canonical format!"
        ch = int(m.group("ch"))
        loads[ch] += 1
    return loads


def test_suite_1_channel_load_uniformity_standard_batches():
    """
    Suite 1: 100% Balanced Channel Load Verification across N in (8, 16, 64, 128, 256, 512).
    Verifies:
      1. Every channel load == N / 8 exactly.
      2. Min load == Max load == N / 8 (stdev == 0.0).
      3. Zero odd-channel starvation (odd_sum == even_sum == N / 2).
    """
    print("\n--- Suite 1: Channel Load Uniformity Across Standard Batches (N=8..512) ---")
    mock = MockKVEngine(layers=32, heads=32)
    batch_sizes = [8, 16, 64, 128, 256, 512]

    for n in batch_sizes:
        ta = TensorAwareFTL(channels=8)
        mock.reset()
        blocks = mock.generate_kv_blocks(num_blocks=n, layer_id=0, layout="token_major")
        ta.allocate_batch(blocks)

        loads = _get_channel_distribution(ta, blocks, channels=8)
        expected_per_ch = n // 8
        min_load = min(loads)
        max_load = max(loads)
        odd_sum = sum(loads[1::2])
        even_sum = sum(loads[0::2])

        # Invariants
        assert min_load == expected_per_ch, f"N={n}: min_load {min_load} != expected {expected_per_ch}"
        assert max_load == expected_per_ch, f"N={n}: max_load {max_load} != expected {expected_per_ch}"
        assert min_load == max_load, f"N={n}: Imbalanced channels! loads={loads}"
        assert odd_sum == even_sum == (n // 2), f"N={n}: Parity imbalance! odd={odd_sum}, even={even_sum}"
        assert all(loads[ch] == expected_per_ch for ch in range(8)), f"N={n}: Unequal loads: {loads}"
        assert min(loads[1::2]) > 0, f"N={n}: Odd channel starvation detected! {loads}"

        print(f"  [PASS] N={n:3d}: loads={loads}, min={min_load}, max={max_load}, odd_sum={odd_sum}, even_sum={even_sum} (100% uniform)")


def test_suite_2_channel_load_arbitrary_batch_sizes():
    """
    Suite 2: Channel distribution for arbitrary non-powers-of-2 and non-multiples of 8.
    Verifies:
      1. Optimal round-robin: max_load - min_load <= 1 for all N.
      2. No channel starvation for all N >= 8.
    """
    print("\n--- Suite 2: Channel Load Balance for Arbitrary / Non-Power-of-2 Batches ---")
    mock = MockKVEngine(layers=32, heads=32)
    test_sizes = [1, 2, 3, 5, 7, 9, 13, 17, 23, 31, 33, 65, 99, 127, 255, 333, 500]

    for n in test_sizes:
        ta = TensorAwareFTL(channels=8)
        mock.reset()
        blocks = mock.generate_kv_blocks(num_blocks=n, layer_id=0, layout="token_major")
        ta.allocate_batch(blocks)

        loads = _get_channel_distribution(ta, blocks, channels=8)
        diff = max(loads) - min(loads)
        assert diff <= 1, f"N={n}: Load spread > 1 (max={max(loads)}, min={min(loads)}, loads={loads})"
        assert sum(loads) == n, f"N={n}: Total load {sum(loads)} != {n}"
        if n >= 8:
            assert all(ch_load > 0 for ch_load in loads), f"N={n}: Starvation observed: {loads}"
            assert all(ch_load >= (n // 8) for ch_load in loads)

        print(f"  [PASS] N={n:3d}: loads={loads}, max-min={diff} (balanced)")


def test_suite_3_head_major_and_variable_head_geometries():
    """
    Suite 3: Head-Major Layout & Variable Head Counts (heads=8, 16, 24, 32, 40, 64, 128).
    Verifies:
      1. Standard power-of-two/multiple-of-8 architectures achieve 100% uniform balance
         in BOTH token_major and head_major layouts across all batch sizes.
      2. Non-power-of-two architectures (heads=24, 40) achieve 100% uniform balance in token_major,
         and bounded variation with zero odd-channel starvation in head_major.
    """
    print("\n--- Suite 3: Head-Major Layout & Variable Attention Head Geometries ---")
    power_of_two_heads = [8, 16, 32, 64, 128]
    batches = [8, 16, 64, 128, 256, 512]

    # Standard power-of-two / multiple-of-8 heads
    for heads in power_of_two_heads:
        mock = MockKVEngine(layers=32, heads=heads)
        for n in batches:
            for layout in ["token_major", "head_major"]:
                ta = TensorAwareFTL(channels=8)
                mock.reset()
                blocks = mock.generate_kv_blocks(num_blocks=n, layer_id=0, layout=layout)
                ta.allocate_batch(blocks)

                loads = _get_channel_distribution(ta, blocks, channels=8)
                assert min(loads) == max(loads) == (n // 8), (
                    f"Imbalance with heads={heads}, layout={layout}, N={n}: loads={loads}"
                )
                assert all(loads[ch] > 0 for ch in range(8))
        print(f"  [PASS] heads={heads:3d}: 100% exact uniform balance across all batch sizes for both layouts")

    # Non-power-of-two heads (heads=24, 40)
    for heads in [24, 40]:
        mock = MockKVEngine(layers=32, heads=heads)
        for n in batches:
            # Token major is 100% balanced
            ta = TensorAwareFTL(channels=8)
            mock.reset()
            blocks_tm = mock.generate_kv_blocks(num_blocks=n, layer_id=0, layout="token_major")
            ta.allocate_batch(blocks_tm)
            loads_tm = _get_channel_distribution(ta, blocks_tm, channels=8)
            assert min(loads_tm) == max(loads_tm) == (n // 8)

            # Head major has 0 odd-channel starvation and spread <= 3
            ta.reset()
            mock.reset()
            blocks_hm = mock.generate_kv_blocks(num_blocks=n, layer_id=0, layout="head_major")
            ta.allocate_batch(blocks_hm)
            loads_hm = _get_channel_distribution(ta, blocks_hm, channels=8)
            assert all(ch_load > 0 for ch_load in loads_hm)
            assert max(loads_hm) - min(loads_hm) <= 3
        print(f"  [PASS] heads={heads:3d}: token_major 100% balanced; head_major zero starvation (spread <= 3)")


def test_suite_4_boundary_edge_cases():
    """
    Suite 4: Boundary Edge Cases (N=0, N=1, Missing/Malformed Block Attributes).
    Verifies:
      1. N=0: returns empty list/dict without error, latency evaluates to 0.0.
      2. N=1: single block allocation, valid canonical address format, bidirectional lookup.
      3. Missing/None attributes: token_count=None, token_start=None, layer_id=None, kv_head_start=None safely default.
      4. Zero token count: token_count=0 does not cause ZeroDivisionError.
    """
    print("\n--- Suite 4: Boundary Edge Cases (N=0, N=1, Zero/None Attributes) ---")
    ta = TensorAwareFTL(channels=8)
    sim = StorageSimulator(mode="tensor_aware", channels=8)
    mock = MockKVEngine()

    # N=0
    blocks_0 = mock.generate_kv_blocks(num_blocks=0)
    assert blocks_0 == []
    alloc_0 = ta.allocate_batch(blocks_0)
    assert alloc_0 == {}
    lat_0 = sim.read_blocks([])
    assert lat_0 == 0.0
    print("  [PASS] N=0: Empty batch correctly handled (0.0 us latency, empty dict)")

    # N=1
    ta.reset()
    blocks_1 = mock.generate_kv_blocks(num_blocks=1)
    loc_1 = ta.allocate(blocks_1[0])
    m = CANONICAL_ADDR_REGEX.match(loc_1)
    assert m is not None, f"Malformed address for N=1: {loc_1}"
    assert ta.translate(blocks_1[0].block_id) == loc_1
    assert ta.reverse_translate(loc_1) == blocks_1[0].block_id
    sim.store_block(blocks_1[0])
    lat_1 = sim.read_blocks(blocks_1)
    assert lat_1 == 40.0  # 10us PCIe + 1 * (25us t_R + 5us t_bus)
    print(f"  [PASS] N=1: Single block correctly allocated ({loc_1}), latency={lat_1} us")

    # Missing / None attributes
    b_none = KVBlock.create_default(block_id=100, layer_id=0, token_start=0, token_count=16, kv_head_start=0)
    b_none.token_count = None
    b_none.token_start = None
    b_none.layer_id = None
    b_none.kv_head_start = None
    loc_none = ta.allocate(b_none)
    assert CANONICAL_ADDR_REGEX.match(loc_none) is not None
    print(f"  [PASS] None attributes safely handled, allocated: {loc_none}")

    # Zero token count
    b_zero = KVBlock.create_default(block_id=101, layer_id=0, token_start=0, token_count=16, kv_head_start=0)
    b_zero.token_count = 0
    loc_zero = ta.allocate(b_zero)
    assert CANONICAL_ADDR_REGEX.match(loc_zero) is not None
    print(f"  [PASS] token_count=0 safely handled without ZeroDivisionError: {loc_zero}")


def test_suite_5_large_sequence_lengths():
    """
    Suite 5: Large Sequence Length Context Window Stress (up to 131,072 tokens).
    Verifies:
      1. All coordinates strictly satisfy physical limits:
         0 <= ch < 8, 0 <= die < 4, 0 <= pl < 2, 0 <= blk < 64, 0 <= pg < 128.
      2. Bidirectional translation is 100% bijective with zero mapping collisions.
    """
    print("\n--- Suite 5: Large Sequence Length Context Window Stress (up to 131,072 tokens) ---")
    ta = TensorAwareFTL(channels=8)
    seq_lengths = [4096, 16384, 32768, 65536, 131072]

    for seq_len in seq_lengths:
        token_count = 16
        num_chunks = seq_len // token_count
        ta.reset()

        for chunk_idx in range(num_chunks):
            b = KVBlock.create_default(
                block_id=chunk_idx,
                layer_id=0,
                token_start=chunk_idx * token_count,
                token_count=token_count,
                kv_head_start=chunk_idx % 32,
                kv_head_count=1,
                storage_tier=StorageTier.GPU.value,
            )
            loc = ta.allocate(b)
            m = CANONICAL_ADDR_REGEX.match(loc)
            assert m is not None, f"Invalid address format {loc} at chunk {chunk_idx}"
            ch, die, pl, blk, pg = [int(x) for x in m.groups()]
            assert 0 <= ch < 8
            assert 0 <= die < 4
            assert 0 <= pl < 2
            assert 0 <= blk < 64
            assert 0 <= pg < 128
            assert ta.translate(chunk_idx) == loc
            assert ta.reverse_translate(loc) == chunk_idx

        print(f"  [PASS] Sequence length {seq_len:6d} tokens ({num_chunks:5d} blocks): All address bounds and translations valid")


def test_suite_6_multi_layer_allocations_and_traces():
    """
    Suite 6: Multi-Layer Transformer Allocations & Attention Traces.
    Verifies:
      1. Allocations across 32 layers distribute across dies and channels.
      2. Realistic attention retrieval traces (concurrent_heads, strided, sparse)
         achieve speedup under Tensor-Aware FTL compared to Conventional FTL.
    """
    print("\n--- Suite 6: Multi-Layer Allocations & Attention Traces ---")
    mock = MockKVEngine(layers=32, heads=32)
    conv_sim = StorageSimulator(mode="conventional", channels=8)
    ta_sim = StorageSimulator(mode="tensor_aware", channels=8)

    all_blocks = []
    for l in range(32):
        layer_blocks = mock.generate_kv_blocks(num_blocks=32, layer_id=l)
        for b in layer_blocks:
            conv_sim.store_block(b)
            ta_sim.store_block(b)
        all_blocks.extend(layer_blocks)

    assert len(all_blocks) == 1024

    # Check die spread across layers in TensorAware
    dies_used = set()
    for b in all_blocks:
        loc = ta_sim.get_location(b.block_id)
        m = CANONICAL_ADDR_REGEX.match(loc)
        dies_used.add(int(m.group("die")))
    assert dies_used == {0, 1, 2, 3}, f"Dies used {dies_used} did not cover all 4 dies!"
    print(f"  [PASS] 1024 multi-layer blocks stored: utilized all 4 dies per channel ({dies_used})")

    # Trace 1: Concurrent heads for Layer 0 (k=16)
    t1 = mock.generate_attention_trace(layer_id=0, total_blocks=32, k=16, pattern="concurrent_heads")
    c_lat1 = conv_sim.estimate_read_latency(t1)
    t_lat1 = ta_sim.estimate_read_latency(t1)
    assert t_lat1 < c_lat1
    print(f"  [PASS] Concurrent heads trace (k=16): Conv={c_lat1:.1f} us, TA={t_lat1:.1f} us, Speedup={c_lat1/t_lat1:.2f}x")

    # Trace 2: Strided heads for Layer 3 (k=16)
    t2 = mock.generate_attention_trace(layer_id=3, total_blocks=32, k=16, pattern="strided")
    c_lat2 = conv_sim.estimate_read_latency(t2)
    t_lat2 = ta_sim.estimate_read_latency(t2)
    assert t_lat2 < c_lat2
    print(f"  [PASS] Strided heads trace (k=16): Conv={c_lat2:.1f} us, TA={t_lat2:.1f} us, Speedup={c_lat2/t_lat2:.2f}x")

    # Trace 3: Sparse attention request (sink + recent, k=32)
    t3 = mock.generate_sparse_attention_request(all_blocks[:64], k=32, sink_ratio=0.25)
    c_lat3 = conv_sim.estimate_read_latency(t3)
    t_lat3 = ta_sim.estimate_read_latency(t3)
    assert t_lat3 < c_lat3
    print(f"  [PASS] Sparse request trace (k=32): Conv={c_lat3:.1f} us, TA={t_lat3:.1f} us, Speedup={c_lat3/t_lat3:.2f}x")


def test_suite_7_conventional_capacity_exhaustion_and_rollover():
    """
    Suite 7: Conventional FTL Capacity Exhaustion, Boundary & Rollover.
    Verifies:
      1. Exact capacity exhaustion at total_pages limit.
      2. Sequential allocation order adheres strictly to Page -> Block -> Plane -> Die -> Channel.
      3. Over-capacity allocation raises RuntimeError with descriptive message.
      4. Reset cleanly clears counter and tables, allowing fresh allocations from ch0_die0_pl0_blk0_pg0.
    """
    print("\n--- Suite 7: Conventional FTL Capacity Exhaustion & Boundary Rollover ---")
    # Small drive: 2 channels, 2 dies/ch, 1 plane/die, 2 blocks/plane, 4 pages/block = 32 pages
    conv = ConventionalFTL(
        channels=2,
        dies_per_channel=2,
        planes_per_die=1,
        blocks_per_plane=2,
        pages_per_block=4,
    )
    assert conv.total_pages == 32

    # Allocate all 32 pages
    for i in range(32):
        b = KVBlock.create_default(block_id=i, layer_id=0, token_start=i*16, token_count=16, kv_head_start=0)
        loc = conv.allocate(b)
        assert conv.translate(i) == loc
        assert conv.reverse_translate(loc) == i

    loc_first = conv.translate(0)
    loc_last = conv.translate(31)
    assert loc_first == "ch0_die0_pl0_blk0_pg0", f"Expected ch0_die0_pl0_blk0_pg0, got {loc_first}"
    assert loc_last == "ch1_die1_pl0_blk1_pg3", f"Expected ch1_die1_pl0_blk1_pg3, got {loc_last}"
    print(f"  [PASS] 32/32 pages allocated sequentially: first={loc_first}, last={loc_last}")

    # 33rd allocation must raise RuntimeError
    overflow_raised = False
    try:
        b_overflow = KVBlock.create_default(block_id=32, layer_id=0, token_start=32*16, token_count=16, kv_head_start=0)
        conv.allocate(b_overflow)
    except RuntimeError as e:
        overflow_raised = True
        assert "capacity exceeded" in str(e).lower()
        print(f"  [PASS] Over-capacity allocation properly rejected with RuntimeError: {e}")
    assert overflow_raised, "ConventionalFTL failed to raise RuntimeError on capacity overflow!"

    # Reset behavior
    conv.reset()
    assert conv._linear_counter == 0
    assert len(conv.get_mapping_table()) == 0
    assert len(conv.get_reverse_mapping_table()) == 0

    # Re-allocation after reset
    b_fresh = KVBlock.create_default(block_id=999, layer_id=0, token_start=0, token_count=16, kv_head_start=0)
    loc_fresh = conv.allocate(b_fresh)
    assert loc_fresh == "ch0_die0_pl0_blk0_pg0"
    print(f"  [PASS] Reset restored initial state: block 999 allocated at {loc_fresh}")


def test_suite_8_mapping_table_integrity_and_snapshot_immutability():
    """
    Suite 8: Mapping Table Integrity & Snapshot Immutability.
    Verifies:
      1. Bijective mapping: translate(reverse_translate(loc)) == loc.
      2. Re-allocation updates forward map and removes stale reverse map entry.
      3. get_mapping_table() and get_reverse_mapping_table() return isolated copies.
      4. Querying non-existent block_id or location returns None.
    """
    print("\n--- Suite 8: Mapping Table Integrity & Snapshot Immutability ---")
    ta = TensorAwareFTL(channels=8)
    b = KVBlock.create_default(block_id=50, layer_id=0, token_start=0, token_count=16, kv_head_start=0)

    # Allocation
    loc1 = ta.allocate(b)
    assert ta.translate(50) == loc1
    assert ta.reverse_translate(loc1) == 50

    # Query unknown
    assert ta.translate(99999) is None
    assert ta.reverse_translate("ch7_die3_pl1_blk63_pg127") is None
    print("  [PASS] Unknown block_id and location return None")

    # Re-allocation of same block_id
    b.token_start = 32
    loc2 = ta.allocate(b)
    assert ta.translate(50) == loc2
    assert ta.reverse_translate(loc2) == 50
    assert ta.reverse_translate(loc1) is None, "Stale reverse mapping was not pruned on re-allocation!"
    print("  [PASS] Re-allocation updated forward map and pruned stale reverse entry")

    # Snapshot immutability
    fwd_table = ta.get_mapping_table()
    rev_table = ta.get_reverse_mapping_table()
    fwd_table[8888] = "corrupted_entry"
    rev_table["corrupted_loc"] = 7777
    assert ta.translate(8888) is None, "Internal forward mapping corrupted by modifying snapshot!"
    assert ta.reverse_translate("corrupted_loc") is None, "Internal reverse mapping corrupted by modifying snapshot!"
    print("  [PASS] Snapshot dictionary copies are safely decoupled from internal state")


def test_suite_9_speedup_acceptance_and_latency_monotonicity():
    """
    Suite 9: Speedup Acceptance Verification (>= 2.5x for N >= 64) and Latency Monotonicity.
    Verifies:
      1. Speedup exceeds 2.5x for N in (64, 128, 256, 512).
      2. Reading N blocks on same channel takes strictly longer than across 8 channels.
      3. Read latency is monotonically non-decreasing with batch size.
    """
    print("\n--- Suite 9: Speedup Acceptance (>= 2.5x) & Latency Monotonicity ---")
    mock = MockKVEngine(layers=32, heads=32)
    conv_sim = StorageSimulator(mode="conventional", channels=8)
    ta_sim = StorageSimulator(mode="tensor_aware", channels=8)

    test_batches = [8, 16, 32, 64, 128, 256, 512]
    prev_conv_lat = 0.0
    prev_ta_lat = 0.0

    for n in test_batches:
        conv_sim.reset()
        ta_sim.reset()
        mock.reset()

        blocks = mock.generate_kv_blocks(num_blocks=n, layer_id=0)
        for b in blocks:
            conv_sim.store_block(b)
            ta_sim.store_block(b)

        b_ids = [b.block_id for b in blocks]
        c_lat = conv_sim.estimate_read_latency(b_ids)
        t_lat = ta_sim.estimate_read_latency(b_ids)
        speedup = c_lat / t_lat

        # Monotonicity
        assert c_lat >= prev_conv_lat, f"Non-monotonic Conventional latency at N={n}: {c_lat} < {prev_conv_lat}"
        assert t_lat >= prev_ta_lat, f"Non-monotonic TensorAware latency at N={n}: {t_lat} < {prev_ta_lat}"
        prev_conv_lat = c_lat
        prev_ta_lat = t_lat

        # Contention inequality
        assert c_lat > t_lat, f"Contention violated at N={n}: Conv ({c_lat}) <= TA ({t_lat})"

        # Speedup threshold check (>= 2.5x for N >= 64)
        if n >= 64:
            assert speedup >= 2.5, f"Speedup requirement failed at N={n}: {speedup:.2f}x < 2.5x"

        print(f"  [PASS] Batch {n:3d} | Conv: {c_lat:7.1f} us | TA: {t_lat:7.1f} us | Speedup: {speedup:5.2f}x (>=2.5x check: {'PASS' if n>=64 else 'N/A'})")


def run_all():
    print("=" * 72)
    print("   CHALLENGER M2.2: COMPREHENSIVE CHANNEL LOAD & STRESS HARNESS   ")
    print("=" * 72)

    test_suite_1_channel_load_uniformity_standard_batches()
    test_suite_2_channel_load_arbitrary_batch_sizes()
    test_suite_3_head_major_and_variable_head_geometries()
    test_suite_4_boundary_edge_cases()
    test_suite_5_large_sequence_lengths()
    test_suite_6_multi_layer_allocations_and_traces()
    test_suite_7_conventional_capacity_exhaustion_and_rollover()
    test_suite_8_mapping_table_integrity_and_snapshot_immutability()
    test_suite_9_speedup_acceptance_and_latency_monotonicity()

    print("\n" + "=" * 72)
    print("   ALL 9 CHALLENGER M2.2 STRESS SUITES PASSED CLEANLY!            ")
    print("=" * 72)


if __name__ == "__main__":
    run_all()
