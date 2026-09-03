"""
Opaque-Box End-to-End (E2E) Test Suite for AI-SSD & FTL Subsystem.
Implements Tiers 1-4 requirement-driven acceptance tests:
- Tier 1: Category-Partition Feature Coverage (R1, R2, R3)
- Tier 2: Boundary Value Analysis (BVA) & Corner Cases
- Tier 3: Pairwise Combinatorial Cross-Feature Interactions
- Tier 4: Real-World Application Workloads (Sparse Attention, LLM Serving)

Zero external dependencies: executes on pure Python 3 standard library.
"""

import csv
import math
import os
import re
import sys
import unittest
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.schemas.kv_block import KVBlock, StorageTier
from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    SSD_PAGES_PER_BLOCK,
    SSD_PAGE_SIZE_BYTES,
    T_R_US,
    T_PROG_US,
    T_BERS_US,
    BUS_TRANSFER_US_PER_PAGE,
    PCIE_OVERHEAD_US,
)
from person2_ssd.nand.page import FlashPage, PageState
from person2_ssd.nand.block import FlashBlock
from person2_ssd.storage_model.latency import LatencyModel
from person2_ssd.storage_model.io_model import StorageSimulator, parse_physical_location
from person2_ssd.kv_allocator.allocator import KVStorageAllocator
from person2_ssd.ftl.conventional import ConventionalFTL
from person2_ssd.ftl.tensor_aware import TensorAwareFTL
from person2_ssd.mock_kv_engine import MockKVEngine


CANONICAL_ADDR_REGEX = re.compile(
    r"^ch(?P<ch>[0-7])_die(?P<die>[0-3])_pl(?P<pl>[0-1])_blk(?P<blk>\d+)_pg(?P<pg>\d+)$"
)


# ============================================================================
# Tier 1: Category-Partition Feature Coverage (R1, R2, R3)
# ============================================================================

class Tier1FeatureCoverageTests(unittest.TestCase):
    """
    Tier 1 tests exhaustively verify core feature requirements:
    - Feature 1: R1 NAND Flash Physics & Timing Model (5 tests)
    - Feature 2: R2 Conventional FTL Hot-Spotting (5 tests)
    - Feature 3: R2 Tensor-Aware Striping (5 tests)
    - Feature 4: R2 Address Formatting & Translation (5 tests)
    - Feature 5: R3 Benchmark Execution & Speedup Acceptance (5 tests)
    """

    # --- Feature 1: R1 NAND Flash Physics & Timing ---

    def test_t1_nand_read_bus_timing_exact_derivation(self):
        """Verify read latency formula: T = t_pcie + max_c (N_c * (t_R + t_bus))."""
        lm = LatencyModel()
        # Single channel serialization with 4 requests
        locs = [f"ch0_die0_pl0_blk0_pg{i}" for i in range(4)]
        lat = lm.calculate_batch_read_latency(locs)
        expected = PCIE_OVERHEAD_US + 4 * (T_R_US + BUS_TRANSFER_US_PER_PAGE)  # 10 + 4 * 30 = 130 us
        self.assertAlmostEqual(lat, expected, delta=1e-6)
        self.assertEqual(expected, 130.0)

    def test_t1_nand_program_timing_derivation(self):
        """Verify write latency formula: T = t_pcie + max_c (N_c * (t_bus + t_PROG))."""
        lm = LatencyModel()
        locs = [f"ch1_die0_pl0_blk0_pg{i}" for i in range(3)]
        lat = lm.calculate_batch_write_latency(locs)
        expected = PCIE_OVERHEAD_US + 3 * (BUS_TRANSFER_US_PER_PAGE + T_PROG_US)  # 10 + 3 * 205 = 625 us
        self.assertAlmostEqual(lat, expected, delta=1e-6)
        self.assertEqual(expected, 625.0)

    def test_t1_nand_erase_timing_derivation(self):
        """Verify erase latency formula: T = t_pcie + max_c (N_c * t_BERS)."""
        lm = LatencyModel()
        locs = [f"ch2_die0_pl0_blk{i}_pg0" for i in range(2)]
        lat = lm.calculate_batch_erase_latency(locs)
        expected = PCIE_OVERHEAD_US + 2 * T_BERS_US  # 10 + 2 * 2000 = 4010 us
        self.assertAlmostEqual(lat, expected, delta=1e-6)
        self.assertEqual(expected, 4010.0)

    def test_t1_nand_page_state_transitions(self):
        """Verify FlashPage state transitions: FREE -> VALID -> INVALID -> FREE."""
        page = FlashPage(page_id=0, size_bytes=SSD_PAGE_SIZE_BYTES)
        self.assertEqual(page.state, PageState.FREE)
        self.assertIsNone(page.data_block_id)
        self.assertEqual(page.read_count, 0)
        self.assertEqual(page.program_count, 0)

        # Program page
        page.program(data_block_id=101)
        self.assertEqual(page.state, PageState.VALID)
        self.assertEqual(page.data_block_id, 101)
        self.assertEqual(page.program_count, 1)

        # Read page
        bid = page.read(current_time_us=12.5)
        self.assertEqual(bid, 101)
        self.assertEqual(page.read_count, 1)
        self.assertEqual(page.last_accessed_us, 12.5)

        # Invalidate page
        page.invalidate()
        self.assertEqual(page.state, PageState.INVALID)
        self.assertIsNone(page.data_block_id)
        self.assertIsNone(page.read())

        # Erase page
        page.erase()
        self.assertEqual(page.state, PageState.FREE)
        self.assertIsNone(page.data_block_id)
        self.assertEqual(page.read_count, 0)

    def test_t1_nand_channel_contention_serialization(self):
        """Verify reading N blocks on same channel takes strictly longer than across N channels."""
        lm = LatencyModel()
        for n in [2, 4, 8]:
            same_channel = [f"ch0_die0_pl0_blk0_pg{i}" for i in range(n)]
            distinct_channels = [f"ch{i}_die0_pl0_blk0_pg0" for i in range(n)]

            t_same = lm.calculate_batch_read_latency(same_channel)
            t_distinct = lm.calculate_batch_read_latency(distinct_channels)

            exp_same = PCIE_OVERHEAD_US + n * (T_R_US + BUS_TRANSFER_US_PER_PAGE)
            exp_distinct = PCIE_OVERHEAD_US + 1 * (T_R_US + BUS_TRANSFER_US_PER_PAGE)

            self.assertAlmostEqual(t_same, exp_same)
            self.assertAlmostEqual(t_distinct, exp_distinct)
            self.assertGreater(t_same, t_distinct, f"Contention violated at N={n}")

    # --- Feature 2: R2 Conventional FTL Hot-Spotting ---

    def test_t1_conv_ftl_channel0_clustering(self):
        """Verify Conventional FTL sequentially fills channel 0 before touching other channels."""
        conv = ConventionalFTL(channels=8, dies_per_channel=4, planes_per_die=2, blocks_per_plane=64, pages_per_block=128)
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=128)

        for b in blocks:
            loc = conv.allocate(b)
            ch = LatencyModel.extract_channel(loc)
            self.assertEqual(ch, 0, f"Conventional FTL assigned block to non-zero channel {ch} prematurely")

    def test_t1_conv_ftl_hotspot_bottleneck_scaling(self):
        """Verify Conventional FTL latency scales linearly as 10 + 30N for parallel requests."""
        sim = StorageSimulator(mode="conventional", channels=8)
        mock = MockKVEngine()
        for n in [8, 16, 32, 64]:
            sim.reset()
            mock.reset()
            blocks = mock.generate_kv_blocks(num_blocks=n)
            for b in blocks:
                sim.store_block(b)
            lat = sim.read_blocks(blocks)
            expected = PCIE_OVERHEAD_US + n * (T_R_US + BUS_TRANSFER_US_PER_PAGE)
            self.assertAlmostEqual(lat, expected, delta=1e-6)

    def test_t1_conv_ftl_sequential_hierarchy_order(self):
        """Verify Conventional FTL progresses strictly Page -> Block -> Plane -> Die -> Channel."""
        # Drive geometry with 2 pages/block, 2 blocks/plane, 2 planes/die, 2 dies/ch, 2 channels
        conv = ConventionalFTL(channels=2, dies_per_channel=2, planes_per_die=2, blocks_per_plane=2, pages_per_block=2)
        locations = []
        for i in range(8):
            b = KVBlock.create_default(block_id=i, layer_id=0, token_start=0)
            locations.append(conv.allocate(b))

        expected = [
            "ch0_die0_pl0_blk0_pg0",
            "ch0_die0_pl0_blk0_pg1",
            "ch0_die0_pl0_blk1_pg0",
            "ch0_die0_pl0_blk1_pg1",
            "ch0_die0_pl1_blk0_pg0",
            "ch0_die0_pl1_blk0_pg1",
            "ch0_die0_pl1_blk1_pg0",
            "ch0_die0_pl1_blk1_pg1",
        ]
        self.assertEqual(locations, expected)

    def test_t1_conv_ftl_channel_load_concentration(self):
        """Verify diagnostic breakdown shows 100% channel 0 load and 0% on other channels."""
        conv = ConventionalFTL(channels=8)
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=64)
        locs = [conv.allocate(b) for b in blocks]

        lm = LatencyModel(channels=8)
        bd = lm.get_latency_breakdown(locs)
        self.assertEqual(bd["bottleneck_channel"], 0)
        self.assertEqual(bd["max_channel_load"], 64)
        self.assertEqual(bd["channel_loads"][0], 64)
        for c in range(1, 8):
            self.assertEqual(bd["channel_loads"].get(c, 0), 0)

    def test_t1_conv_ftl_inter_head_contention(self):
        """Verify parallel attention heads in the same token chunk collide on Channel 0 in Conventional FTL."""
        sim = StorageSimulator(mode="conventional", channels=8)
        mock = MockKVEngine(layers=1, heads=8)
        blocks = mock.generate_kv_blocks(num_blocks=8, layout="token_major")
        for b in blocks:
            sim.store_block(b)

        # All 8 heads should be on channel 0
        for b in blocks:
            loc = sim.get_location(b.block_id)
            self.assertEqual(LatencyModel.extract_channel(loc), 0)

        lat = sim.read_blocks(blocks)
        self.assertEqual(lat, 10.0 + 8 * 30.0)  # 250 us

    # --- Feature 3: R2 Tensor-Aware Striping ---

    def test_t1_ta_ftl_round_robin_channel_distribution(self):
        """Verify Tensor-Aware FTL distributes blocks uniformly across all 8 channels."""
        ta = TensorAwareFTL(channels=8)
        mock = MockKVEngine(layers=32, heads=32)
        blocks = mock.generate_kv_blocks(num_blocks=64, layer_id=0, layout="token_major")
        ta.allocate_batch(blocks)

        loads = {c: 0 for c in range(8)}
        for b in blocks:
            loc = ta.translate(b.block_id)
            ch = LatencyModel.extract_channel(loc)
            loads[ch] += 1

        self.assertEqual(loads, {c: 8 for c in range(8)})

    def test_t1_ta_ftl_zero_odd_channel_starvation(self):
        """Verify Tensor-Aware FTL has zero odd-channel starvation (even sum == odd sum)."""
        ta = TensorAwareFTL(channels=8)
        mock = MockKVEngine(layers=32, heads=32)
        for n in [16, 32, 64, 128]:
            ta.reset()
            mock.reset()
            blocks = mock.generate_kv_blocks(num_blocks=n, layer_id=0)
            ta.allocate_batch(blocks)

            loads = [0] * 8
            for b in blocks:
                ch = LatencyModel.extract_channel(ta.translate(b.block_id))
                loads[ch] += 1

            even_sum = sum(loads[0::2])
            odd_sum = sum(loads[1::2])
            self.assertEqual(even_sum, odd_sum)
            self.assertEqual(even_sum, n // 2)
            for c in range(8):
                self.assertGreater(loads[c], 0)

    def test_t1_ta_ftl_multi_die_distribution(self):
        """Verify Tensor-Aware FTL stripes across multiple dies per channel."""
        ta = TensorAwareFTL(channels=8, dies_per_channel=4)
        mock = MockKVEngine(layers=32, heads=32)
        dies_seen = set()
        for layer in range(4):
            blocks = mock.generate_kv_blocks(num_blocks=32, layer_id=layer)
            for b in blocks:
                loc = ta.allocate(b)
                parsed = parse_physical_location(loc)
                dies_seen.add(parsed[1])

        self.assertEqual(dies_seen, {0, 1, 2, 3})

    def test_t1_ta_ftl_multi_plane_striping(self):
        """Verify Tensor-Aware FTL stripes across planes as token block index advances."""
        ta = TensorAwareFTL(channels=8, dies_per_channel=4, planes_per_die=2)
        planes_seen = set()
        # Allocate blocks across high token chunk indices
        for chunk in range(64):
            b = KVBlock.create_default(
                block_id=chunk,
                layer_id=0,
                token_start=chunk * 16,
                token_count=16,
                kv_head_start=0,
            )
            loc = ta.allocate(b)
            parsed = parse_physical_location(loc)
            planes_seen.add(parsed[2])

        self.assertEqual(planes_seen, {0, 1})

    def test_t1_ta_ftl_parallelism_speedup(self):
        """Verify Tensor-Aware FTL max channel load is 1/8th of Conventional FTL for N=64."""
        sim_conv = StorageSimulator(mode="conventional", channels=8)
        sim_ta = StorageSimulator(mode="tensor_aware", channels=8)
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=64)

        for b in blocks:
            sim_conv.store_block(b)
            sim_ta.store_block(b)

        b_ids = [b.block_id for b in blocks]
        lat_conv = sim_conv.estimate_read_latency(b_ids)
        lat_ta = sim_ta.estimate_read_latency(b_ids)

        self.assertEqual(lat_conv, 10.0 + 64 * 30.0)  # 1930 us
        self.assertEqual(lat_ta, 10.0 + 8 * 30.0)    # 250 us
        speedup = lat_conv / lat_ta
        self.assertAlmostEqual(speedup, 1930.0 / 250.0, delta=1e-2)
        self.assertGreater(speedup, 7.7)

    # --- Feature 4: R2 Address Formatting & Translation ---

    def test_t1_address_canonical_regex_conformance(self):
        """Verify allocated physical addresses adhere strictly to canonical regex."""
        for mode, ftl_cls in [("conv", ConventionalFTL), ("ta", TensorAwareFTL)]:
            ftl = ftl_cls(channels=8)
            mock = MockKVEngine()
            blocks = mock.generate_kv_blocks(num_blocks=100)
            for b in blocks:
                loc = ftl.allocate(b)
                m = CANONICAL_ADDR_REGEX.match(loc)
                self.assertIsNotNone(m, f"{mode} produced non-canonical address '{loc}'")

    def test_t1_address_coordinate_bounds(self):
        """Verify physical coordinates respect physical hardware limits."""
        ta = TensorAwareFTL(channels=8, dies_per_channel=4, planes_per_die=2, blocks_per_plane=64, pages_per_block=128)
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=500)
        for b in blocks:
            loc = ta.allocate(b)
            ch, die, pl, blk, pg = parse_physical_location(loc)
            self.assertTrue(0 <= ch < 8)
            self.assertTrue(0 <= die < 4)
            self.assertTrue(0 <= pl < 2)
            self.assertTrue(0 <= blk < 64)
            self.assertTrue(0 <= pg < 128)

    def test_t1_address_bidirectional_translation_parity(self):
        """Verify bijection: translate(reverse_translate(loc)) == loc and vice versa."""
        ta = TensorAwareFTL(channels=8)
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=200)
        for b in blocks:
            loc = ta.allocate(b)
            bid = b.block_id
            self.assertEqual(ta.translate(bid), loc)
            self.assertEqual(ta.reverse_translate(loc), bid)
            self.assertEqual(ta.get_location(bid), loc)

    def test_t1_address_reallocation_eviction(self):
        """Verify re-allocating a block updates the forward mapping and prunes stale reverse entry."""
        ta = TensorAwareFTL(channels=8)
        b = KVBlock.create_default(block_id=77, layer_id=0, token_start=0)
        loc_v1 = ta.allocate(b)
        self.assertEqual(ta.translate(77), loc_v1)
        self.assertEqual(ta.reverse_translate(loc_v1), 77)

        # Modify coordinate and reallocate
        b.token_start = 64
        loc_v2 = ta.allocate(b)
        self.assertNotEqual(loc_v1, loc_v2)
        self.assertEqual(ta.translate(77), loc_v2)
        self.assertEqual(ta.reverse_translate(loc_v2), 77)
        self.assertIsNone(ta.reverse_translate(loc_v1), "Stale reverse mapping was not evicted!")

    def test_t1_address_table_snapshots(self):
        """Verify get_mapping_table() and get_reverse_mapping_table() return isolated copies."""
        ta = TensorAwareFTL(channels=8)
        b = KVBlock.create_default(block_id=99, layer_id=0, token_start=0)
        loc = ta.allocate(b)

        fwd = ta.get_mapping_table()
        rev = ta.get_reverse_mapping_table()
        self.assertEqual(fwd[99], loc)
        self.assertEqual(rev[loc], 99)

        # Mutate snapshot dicts
        fwd[999] = "ch0_die0_pl0_blk0_pg0"
        rev["fake_loc"] = 123
        self.assertIsNone(ta.translate(999))
        self.assertIsNone(ta.reverse_translate("fake_loc"))

    # --- Feature 5: R3 Benchmark Execution & Speedup Acceptance ---

    def test_t1_benchmark_batch_sizes_scaling(self):
        """Verify benchmark batch sizes [16, 32, 64, 128, 256] scaling behavior."""
        batch_sizes = [16, 32, 64, 128, 256]
        mock = MockKVEngine()
        sim_conv = StorageSimulator(mode="conventional", channels=8)
        sim_ta = StorageSimulator(mode="tensor_aware", channels=8)

        for bsz in batch_sizes:
            sim_conv.reset()
            sim_ta.reset()
            mock.reset()
            blocks = mock.generate_kv_blocks(num_blocks=bsz)
            for b in blocks:
                sim_conv.store_block(b)
                sim_ta.store_block(b)

            b_ids = [b.block_id for b in blocks]
            c_lat = sim_conv.estimate_read_latency(b_ids)
            t_lat = sim_ta.estimate_read_latency(b_ids)
            self.assertGreater(c_lat, t_lat)

    def test_t1_benchmark_speedup_acceptance_threshold(self):
        """Verify acceptance criterion: speedup >= 2.5x for batch sizes >= 64 across 8 channels."""
        mock = MockKVEngine()
        for bsz in [64, 128, 256]:
            sim_conv = StorageSimulator(mode="conventional", channels=8)
            sim_ta = StorageSimulator(mode="tensor_aware", channels=8)
            mock.reset()
            blocks = mock.generate_kv_blocks(num_blocks=bsz)
            for b in blocks:
                sim_conv.store_block(b)
                sim_ta.store_block(b)

            b_ids = [b.block_id for b in blocks]
            c_lat = sim_conv.estimate_read_latency(b_ids)
            t_lat = sim_ta.estimate_read_latency(b_ids)
            speedup = c_lat / t_lat
            self.assertGreaterEqual(speedup, 2.5, f"Batch {bsz} speedup {speedup:.2f}x below 2.5x threshold")

    def test_t1_benchmark_csv_schema_and_integrity(self):
        """Verify results/raw/ftl_results.csv schema and contents from benchmarks/run_ftl.py."""
        csv_path = PROJECT_ROOT / "results" / "raw" / "ftl_results.csv"
        self.assertTrue(csv_path.exists(), f"Benchmark CSV not found at {csv_path}")

        expected_fields = ["experiment", "batch_size", "conventional_latency_us", "tensor_aware_latency_us", "speedup_x"]
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, expected_fields)
            rows = list(reader)

        self.assertEqual(len(rows), 5)
        for row in rows:
            bsz = int(row["batch_size"])
            self.assertIn(bsz, [16, 32, 64, 128, 256])
            conv_lat = float(row["conventional_latency_us"])
            ta_lat = float(row["tensor_aware_latency_us"])
            speedup = float(row["speedup_x"])
            self.assertAlmostEqual(conv_lat / ta_lat, speedup, delta=0.05)
            if bsz >= 64:
                self.assertGreaterEqual(speedup, 2.5)

    def test_t1_benchmark_monotonic_latency_increase(self):
        """Verify read latency monotonically increases with batch size for both Conventional and Tensor-Aware."""
        sim_c = StorageSimulator(mode="conventional")
        sim_t = StorageSimulator(mode="tensor_aware")
        mock = MockKVEngine()

        c_prev = 0.0
        t_prev = 0.0
        for bsz in [16, 32, 64, 128, 256]:
            sim_c.reset()
            sim_t.reset()
            mock.reset()
            blocks = mock.generate_kv_blocks(num_blocks=bsz)
            for b in blocks:
                sim_c.store_block(b)
                sim_t.store_block(b)

            b_ids = [b.block_id for b in blocks]
            c_lat = sim_c.estimate_read_latency(b_ids)
            t_lat = sim_t.estimate_read_latency(b_ids)

            self.assertGreater(c_lat, c_prev)
            self.assertGreater(t_lat, t_prev)
            c_prev = c_lat
            t_prev = t_lat

    def test_t1_benchmark_speedup_monotonicity(self):
        """Verify speedup monotonically increases towards theoretical maximum (~8.0x) as batch size grows."""
        sim_c = StorageSimulator(mode="conventional")
        sim_t = StorageSimulator(mode="tensor_aware")
        mock = MockKVEngine()

        prev_speedup = 0.0
        for bsz in [16, 32, 64, 128, 256]:
            sim_c.reset()
            sim_t.reset()
            mock.reset()
            blocks = mock.generate_kv_blocks(num_blocks=bsz)
            for b in blocks:
                sim_c.store_block(b)
                sim_t.store_block(b)

            b_ids = [b.block_id for b in blocks]
            speedup = sim_c.estimate_read_latency(b_ids) / sim_t.estimate_read_latency(b_ids)
            self.assertGreater(speedup, prev_speedup)
            prev_speedup = speedup

        # At 256, speedup should be near 7.9x
        self.assertGreater(prev_speedup, 7.8)


# ============================================================================
# Tier 2: Boundary Value Analysis (BVA) & Corner Cases
# ============================================================================

class Tier2BoundaryCornerTests(unittest.TestCase):
    """
    Tier 2 tests probe extreme boundaries and corner cases:
    - Feature 1: Batch Size Boundaries (5 tests)
    - Feature 2: Limits & Capacity Exhaustion (5 tests)
    - Feature 3: Empty & Unmapped Requests (5 tests)
    - Feature 4: Irregular Head Counts & Layout Geometries (5 tests)
    - Feature 5: Large Sequence Lengths & Extended Context (5 tests)
    """

    # --- Feature 1: Batch Size Boundaries ---

    def test_t2_batch_size_zero_returns_zero_latency(self):
        """Verify batch size 0 returns 0.0 latency and empty allocation."""
        sim = StorageSimulator(mode="tensor_aware")
        self.assertEqual(sim.read_blocks([]), 0.0)
        self.assertEqual(sim.estimate_read_latency([]), 0.0)

        ta = TensorAwareFTL()
        self.assertEqual(ta.allocate_batch([]), {})

    def test_t2_batch_size_one_base_latency(self):
        """Verify batch size 1 latency equals exactly t_pcie + t_R + t_bus = 40.0 us."""
        sim = StorageSimulator(mode="tensor_aware")
        b = KVBlock.create_default(block_id=1, layer_id=0, token_start=0)
        sim.store_block(b)
        lat = sim.read_blocks([b])
        expected = PCIE_OVERHEAD_US + 1 * (T_R_US + BUS_TRANSFER_US_PER_PAGE)
        self.assertEqual(lat, expected)
        self.assertEqual(expected, 40.0)

    def test_t2_batch_size_primes_and_non_multiples(self):
        """Verify prime and non-multiple batch sizes have optimal channel spread: max - min <= 1."""
        ta = TensorAwareFTL(channels=8)
        mock = MockKVEngine()
        for prime in [3, 7, 11, 13, 17, 31, 67]:
            ta.reset()
            mock.reset()
            blocks = mock.generate_kv_blocks(num_blocks=prime)
            ta.allocate_batch(blocks)

            loads = [0] * 8
            for b in blocks:
                ch = LatencyModel.extract_channel(ta.translate(b.block_id))
                loads[ch] += 1

            self.assertLessEqual(max(loads) - min(loads), 1, f"Imbalance for prime N={prime}: {loads}")
            self.assertEqual(sum(loads), prime)

    def test_t2_batch_size_sub_channel_count(self):
        """Verify sub-channel batch sizes (N in [2..7]) result in max channel load == 1."""
        ta = TensorAwareFTL(channels=8)
        mock = MockKVEngine()
        for n in range(2, 8):
            ta.reset()
            mock.reset()
            blocks = mock.generate_kv_blocks(num_blocks=n)
            ta.allocate_batch(blocks)

            loads = [0] * 8
            for b in blocks:
                ch = LatencyModel.extract_channel(ta.translate(b.block_id))
                loads[ch] += 1

            self.assertEqual(max(loads), 1)
            self.assertEqual(sum(loads), n)

    def test_t2_batch_size_large_scale(self):
        """Verify massive batch size (N=1024) allocates and reads without memory or integer overflow."""
        sim = StorageSimulator(mode="tensor_aware", channels=8)
        mock = MockKVEngine(layers=32, heads=32)
        blocks = mock.generate_kv_blocks(num_blocks=1024)
        for b in blocks:
            sim.store_block(b)

        lat = sim.read_blocks(blocks)
        # 1024 / 8 = 128 requests per channel -> 10 + 128 * 30 = 3850 us
        expected = PCIE_OVERHEAD_US + 128 * (T_R_US + BUS_TRANSFER_US_PER_PAGE)
        self.assertAlmostEqual(lat, expected, delta=1e-6)

    # --- Feature 2: Limits & Capacity Exhaustion ---

    def test_t2_conv_ftl_exact_capacity_exhaustion(self):
        """Verify Conventional FTL raises RuntimeError on exact capacity exhaustion limit."""
        # Drive with 1 ch, 1 die, 1 pl, 2 blk, 2 pg = 4 pages total
        conv = ConventionalFTL(channels=1, dies_per_channel=1, planes_per_die=1, blocks_per_plane=2, pages_per_block=2)
        self.assertEqual(conv.total_pages, 4)

        for i in range(4):
            b = KVBlock.create_default(block_id=i, layer_id=0, token_start=0)
            conv.allocate(b)

        # 5th allocation must raise RuntimeError
        overflow = KVBlock.create_default(block_id=4, layer_id=0, token_start=0)
        with self.assertRaises(RuntimeError) as ctx:
            conv.allocate(overflow)
        self.assertIn("capacity exceeded", str(ctx.exception).lower())

    def test_t2_nand_block_full_rejection(self):
        """Verify FlashBlock returns None when allocating page beyond block capacity."""
        blk = FlashBlock(block_id=0, pages_count=4)
        for i in range(4):
            p = blk.allocate_page(data_block_id=i)
            self.assertIsNotNone(p)
            self.assertEqual(p.page_id, i)

        self.assertTrue(blk.is_full)
        self.assertEqual(blk.free_page_count, 0)
        self.assertIsNone(blk.allocate_page(data_block_id=999))

    def test_t2_nand_page_double_program_error(self):
        """Verify FlashPage raises ValueError when programmed twice without erasing."""
        p = FlashPage(page_id=0)
        p.program(data_block_id=10)
        self.assertEqual(p.state, PageState.VALID)

        with self.assertRaises(ValueError):
            p.program(data_block_id=20)

    def test_t2_nand_block_erase_endurance_limit(self):
        """Verify FlashBlock tracks erase cycles and marks is_bad_block when max cycles reached."""
        blk = FlashBlock(block_id=0, pages_count=2, max_erase_cycles=5)
        for _ in range(4):
            blk.allocate_page(1)
            blk.erase()
            self.assertFalse(blk.is_bad_block)

        # 5th erase reaches limit
        blk.allocate_page(1)
        blk.erase()
        self.assertTrue(blk.is_bad_block)
        self.assertEqual(blk.erase_count, 5)

    def test_t2_ftl_reset_capacity_recovery(self):
        """Verify reset() restores full allocation capacity and resets counters."""
        conv = ConventionalFTL(channels=1, dies_per_channel=1, planes_per_die=1, blocks_per_plane=1, pages_per_block=2)
        conv.allocate(KVBlock.create_default(block_id=0, layer_id=0, token_start=0))
        conv.allocate(KVBlock.create_default(block_id=1, layer_id=0, token_start=0))

        with self.assertRaises(RuntimeError):
            conv.allocate(KVBlock.create_default(block_id=2, layer_id=0, token_start=0))

        # Reset
        conv.reset()
        fresh = conv.allocate(KVBlock.create_default(block_id=99, layer_id=0, token_start=0))
        self.assertEqual(fresh, "ch0_die0_pl0_blk0_pg0")

    # --- Feature 3: Empty & Unmapped Requests ---

    def test_t2_empty_read_request_handling(self):
        """Verify read_blocks with empty list returns 0.0 without errors."""
        sim = StorageSimulator()
        self.assertEqual(sim.read_blocks([]), 0.0)

    def test_t2_unmapped_block_id_lookup(self):
        """Verify translating unmapped or negative logical block IDs returns None."""
        ta = TensorAwareFTL()
        self.assertIsNone(ta.translate(999999))
        self.assertIsNone(ta.translate(-1))
        self.assertIsNone(ta.get_location(5555))

    def test_t2_unmapped_physical_address_lookup(self):
        """Verify reverse translating unmapped, empty, or garbage locations returns None."""
        ta = TensorAwareFTL()
        self.assertIsNone(ta.reverse_translate("ch0_die0_pl0_blk0_pg0"))
        self.assertIsNone(ta.reverse_translate(""))
        self.assertIsNone(ta.reverse_translate("garbage_location"))

    def test_t2_read_blocks_with_nonexistent_ids(self):
        """Verify read_blocks with unmapped block IDs gracefully evaluates to 0.0."""
        sim = StorageSimulator()
        lat = sim.read_blocks([99999, 88888, 77777])
        self.assertEqual(lat, 0.0)

    def test_t2_malformed_location_channel_extraction(self):
        """Verify LatencyModel.extract_channel handles None, empty string, and malformed inputs gracefully."""
        self.assertEqual(LatencyModel.extract_channel(None), 0)
        self.assertEqual(LatencyModel.extract_channel(""), 0)
        self.assertEqual(LatencyModel.extract_channel("no_channel_here"), 0)
        self.assertEqual(LatencyModel.extract_channel(12345), 0)
        self.assertEqual(LatencyModel.extract_channel("SSD_CH5_DIE0"), 5)

    # --- Feature 4: Irregular Head Counts & Layout Geometries ---

    def test_t2_head_count_single_head_mqa(self):
        """Verify single-head architecture (MQA, heads=1) stripes consecutive token chunks across channels."""
        mock = MockKVEngine(layers=1, heads=1)
        ta = TensorAwareFTL(channels=8)
        blocks = mock.generate_kv_blocks(num_blocks=16, layout="token_major")
        ta.allocate_batch(blocks)

        loads = [0] * 8
        for b in blocks:
            ch = LatencyModel.extract_channel(ta.translate(b.block_id))
            loads[ch] += 1

        self.assertEqual(loads, [2] * 8)

    def test_t2_head_count_non_power_of_two(self):
        """Verify non-power-of-two head counts (heads=24, heads=40) achieve zero odd-channel starvation."""
        for heads in [24, 40]:
            mock = MockKVEngine(layers=4, heads=heads)
            ta = TensorAwareFTL(channels=8)
            blocks = mock.generate_kv_blocks(num_blocks=64, layout="token_major")
            ta.allocate_batch(blocks)

            loads = [0] * 8
            for b in blocks:
                ch = LatencyModel.extract_channel(ta.translate(b.block_id))
                loads[ch] += 1

            self.assertEqual(loads, [8] * 8)

    def test_t2_head_major_layout_balance(self):
        """Verify head_major layout distributes blocks evenly without channel starvation."""
        mock = MockKVEngine(layers=4, heads=16)
        ta = TensorAwareFTL(channels=8)
        blocks = mock.generate_kv_blocks(num_blocks=64, layout="head_major")
        ta.allocate_batch(blocks)

        loads = [0] * 8
        for b in blocks:
            ch = LatencyModel.extract_channel(ta.translate(b.block_id))
            loads[ch] += 1

        self.assertEqual(loads, [8] * 8)

    def test_t2_block_attribute_defaults_resilience(self):
        """Verify TensorAwareFTL handles KVBlock with None or missing attributes safely."""
        ta = TensorAwareFTL(channels=8)
        b = KVBlock.create_default(block_id=301, layer_id=0, token_start=0)
        b.token_count = None
        b.token_start = None
        b.layer_id = None
        b.kv_head_start = None

        loc = ta.allocate(b)
        self.assertIsNotNone(CANONICAL_ADDR_REGEX.match(loc))
        self.assertEqual(ta.translate(301), loc)
        self.assertEqual(ta.reverse_translate(loc), 301)

    def test_t2_zero_token_count_handling(self):
        """Verify KVBlock with token_count=0 does not cause ZeroDivisionError in FTL."""
        ta = TensorAwareFTL(channels=8)
        b = KVBlock.create_default(block_id=302, layer_id=0, token_start=0)
        b.token_count = 0

        loc = ta.allocate(b)
        self.assertIsNotNone(CANONICAL_ADDR_REGEX.match(loc))

    # --- Feature 5: Large Sequence Lengths & Extended Context ---

    def test_t2_sequence_length_4k_context(self):
        """Verify 4,096 token context (256 blocks) allocates with bounded physical coordinates."""
        ta = TensorAwareFTL(channels=8)
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=256)
        ta.allocate_batch(blocks)

        for b in blocks:
            loc = ta.translate(b.block_id)
            self.assertIsNotNone(CANONICAL_ADDR_REGEX.match(loc))

    def test_t2_sequence_length_16k_context(self):
        """Verify 16,384 token context (1,024 blocks) maintains 100% bijective translation."""
        ta = TensorAwareFTL(channels=8)
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=1024)
        ta.allocate_batch(blocks)

        self.assertEqual(len(ta.get_mapping_table()), 1024)
        self.assertEqual(len(ta.get_reverse_mapping_table()), 1024)

    def test_t2_sequence_length_64k_context(self):
        """Verify 65,536 token context (4,096 blocks) maintains continuous round-robin channel allocation."""
        ta = TensorAwareFTL(channels=8)
        for chunk in range(4096):
            b = KVBlock.create_default(block_id=chunk, layer_id=0, token_start=chunk * 16)
            ta.allocate(b)

        loads = [0] * 8
        for chunk in range(4096):
            ch = LatencyModel.extract_channel(ta.translate(chunk))
            loads[ch] += 1

        self.assertEqual(loads, [512] * 8)

    def test_t2_sequence_length_131k_context(self):
        """Verify 131,072 token context (8,192 blocks) satisfies all hardware bounds."""
        ta = TensorAwareFTL(channels=8, dies_per_channel=4, planes_per_die=2, blocks_per_plane=64, pages_per_block=128)
        for chunk in range(8192):
            b = KVBlock.create_default(block_id=chunk, layer_id=0, token_start=chunk * 16)
            loc = ta.allocate(b)
            ch, die, pl, blk, pg = parse_physical_location(loc)
            self.assertTrue(0 <= ch < 8)
            self.assertTrue(0 <= die < 4)
            self.assertTrue(0 <= pl < 2)
            self.assertTrue(0 <= blk < 64)
            self.assertTrue(0 <= pg < 128)

    def test_t2_long_sequence_subsegment_sparse_reads(self):
        """Verify sparse sampling of a 64k sequence preserves >=2.5x speedup under Tensor-Aware FTL."""
        sim_c = StorageSimulator(mode="conventional", channels=8)
        sim_t = StorageSimulator(mode="tensor_aware", channels=8)

        for i in range(1024):
            b = KVBlock.create_default(block_id=i, layer_id=0, token_start=i * 16)
            sim_c.store_block(b)
            sim_t.store_block(b)

        # Sample every 16th block (64 blocks total)
        sparse_sample = [i * 16 for i in range(64)]
        c_lat = sim_c.estimate_read_latency(sparse_sample)
        t_lat = sim_t.estimate_read_latency(sparse_sample)

        speedup = c_lat / t_lat
        self.assertGreaterEqual(speedup, 2.5)


# ============================================================================
# Tier 3: Pairwise Combinatorial Cross-Feature Interactions
# ============================================================================

class Tier3CrossFeatureCombinationTests(unittest.TestCase):
    """
    Tier 3 validates non-trivial cross-feature interactions:
    - Test 1: Tensor-Aware allocation combined with latency model contention
    - Test 2: Multi-die interleaving with sequential page programming and channel bus queueing
    - Test 3: Reverse translation across striped block re-allocations
    - Test 4: Multi-layer sparse attention retrieval across channels
    - Test 5: Dynamic StorageSimulator reset and mode switching
    - Test 6: Polymorphic storage reads with detailed latency breakdown
    """

    def test_t3_ta_allocation_combined_with_latency_contention(self):
        """Pairwise: TensorAware allocation combined with LatencyModel diagnostic breakdown."""
        ta = TensorAwareFTL(channels=8)
        lm = LatencyModel(channels=8)
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=64)
        locs = [ta.allocate(b) for b in blocks]

        bd = lm.get_latency_breakdown(locs)
        self.assertEqual(bd["total_requests"], 64)
        self.assertEqual(bd["max_channel_load"], 8)
        # Latency breakdown timing equations
        self.assertEqual(bd["read_channel_time_us"], 8 * 30.0)
        self.assertEqual(bd["total_read_latency_us"], 10.0 + 8 * 30.0)
        self.assertEqual(bd["write_channel_time_us"], 8 * 205.0)
        self.assertEqual(bd["total_write_latency_us"], 10.0 + 8 * 205.0)

    def test_t3_multi_die_interleaving_with_sequential_programming(self):
        """Pairwise: StorageSimulator storing multi-layer blocks updates physical page states and channel transfer queues."""
        sim = StorageSimulator(mode="tensor_aware", channels=8)
        mock = MockKVEngine(layers=4, heads=8)
        blocks = mock.generate_kv_blocks(num_blocks=32)

        for b in blocks:
            loc = sim.store_block(b)
            self.assertEqual(b.storage_tier, StorageTier.SSD.value)
            ch, die, pl, blk, pg = parse_physical_location(loc)
            page = sim.channels[ch].dies[die].planes[pl].blocks[blk].pages[pg]
            self.assertEqual(page.state, PageState.VALID)
            self.assertEqual(page.data_block_id, b.block_id)

        # Execute batch read to process channel transfer queues and verify completion
        lat = sim.read_blocks(blocks)
        self.assertGreater(lat, 0.0)

        # Verify all 32 read transfers completed across channels
        total_transfers = sum(len(ch.completed_transfers) for ch in sim.channels)
        self.assertEqual(total_transfers, 32)

    def test_t3_reverse_translation_across_striped_reallocations(self):
        """Pairwise: Live block migration / re-allocation across striped channels retains bijection."""
        ta = TensorAwareFTL(channels=8)
        mock = MockKVEngine(layers=2, heads=8)
        blocks = mock.generate_kv_blocks(num_blocks=16)
        locs_v1 = [ta.allocate(b) for b in blocks]

        # Re-allocate with different layer/head coordinates
        locs_v2 = []
        for b in blocks:
            b.layer_id = 1
            b.token_start += 32
            locs_v2.append(ta.allocate(b))

        # Check all new locations are valid and bound
        for i, b in enumerate(blocks):
            self.assertEqual(ta.translate(b.block_id), locs_v2[i])
            self.assertEqual(ta.reverse_translate(locs_v2[i]), b.block_id)
            # Old locations must be cleansed
            self.assertIsNone(ta.reverse_translate(locs_v1[i]))

    def test_t3_multi_layer_sparse_attention_channel_spread(self):
        """Pairwise: 32-layer allocation followed by top-k attention retrieval maintains channel balance."""
        sim = StorageSimulator(mode="tensor_aware", channels=8)
        mock = MockKVEngine(layers=32, heads=32)

        all_blocks = []
        for layer in range(32):
            layer_blocks = mock.generate_kv_blocks(num_blocks=32, layer_id=layer)
            for b in layer_blocks:
                sim.store_block(b)
            all_blocks.extend(layer_blocks)

        # Retrieve top 8 heads across 8 selected layers = 64 blocks
        query_ids = []
        for layer in range(8):
            trace = mock.generate_attention_trace(layer_id=layer, total_blocks=32, k=8)
            query_ids.extend(trace)

        self.assertEqual(len(query_ids), 64)
        lat = sim.estimate_read_latency(query_ids)
        # Verify latency matches 8 requests/channel
        expected = PCIE_OVERHEAD_US + 8 * (T_R_US + BUS_TRANSFER_US_PER_PAGE)
        self.assertEqual(lat, expected)

    def test_t3_dynamic_simulator_reset_and_mixed_ftl_workloads(self):
        """Pairwise: StorageSimulator reset() followed by switching modes isolates state."""
        sim = StorageSimulator(mode="conventional", channels=8)
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=64)
        for b in blocks:
            sim.store_block(b)
        conv_lat = sim.read_blocks(blocks)
        self.assertEqual(conv_lat, 1930.0)

        # Reset and switch to TensorAware
        sim.reset()
        sim.allocator = KVStorageAllocator(mode="tensor_aware", channels=8)
        mock.reset()
        new_blocks = mock.generate_kv_blocks(num_blocks=64)
        for b in new_blocks:
            sim.store_block(b)
        ta_lat = sim.read_blocks(new_blocks)
        self.assertEqual(ta_lat, 250.0)
        self.assertAlmostEqual(conv_lat / ta_lat, 7.72, delta=0.01)

    def test_t3_polymorphic_storage_reads_with_latency_diagnostics(self):
        """Pairwise: Verify StorageSimulator polymorphic queries (KVBlock vs int vs mixed) match LatencyModel."""
        sim = StorageSimulator(mode="tensor_aware", channels=8)
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=32)
        for b in blocks:
            sim.store_block(b)

        lat_objects = sim.read_blocks(blocks)
        lat_ints = sim.read_blocks([b.block_id for b in blocks])
        lat_mixed = sim.read_blocks([blocks[0], blocks[1].block_id, blocks[2]])

        self.assertEqual(lat_objects, lat_ints)
        self.assertEqual(lat_objects, 10.0 + 4 * 30.0)  # 130 us
        self.assertEqual(lat_mixed, 10.0 + 1 * 30.0)    # 40 us


# ============================================================================
# Tier 4: Real-World Application Workloads
# ============================================================================

class Tier4RealWorldWorkloadTests(unittest.TestCase):
    """
    Tier 4 tests model realistic end-to-end production LLM inference workloads:
    - Test 1: 32-Layer Sparse Decode Phase across 32 heads
    - Test 2: Attention sink + recent context retrieval pattern
    - Test 3: End-to-end prefill then multi-turn decode lifecycle
    - Test 4: Grouped-Query Attention (GQA-8) layout verification
    - Test 5: Throughput (GB/s) and IOPS scaling comparison
    """

    def test_t4_32_layer_sparse_decode_phase(self):
        """Simulate LLM generation decode step across 32 transformer layers, reading 4 heads/layer (128 blocks)."""
        sim_c = StorageSimulator(mode="conventional", channels=8)
        sim_t = StorageSimulator(mode="tensor_aware", channels=8)
        mock = MockKVEngine(layers=32, heads=32)

        # Store cache across 32 layers
        all_blocks = []
        for l in range(32):
            layer_blocks = mock.generate_kv_blocks(num_blocks=32, layer_id=l)
            for b in layer_blocks:
                sim_c.store_block(b)
                sim_t.store_block(b)
            all_blocks.extend(layer_blocks)

        # Decode step accesses 4 concurrent heads per layer (128 blocks total)
        decode_block_ids = []
        for l in range(32):
            trace = mock.generate_attention_trace(layer_id=l, total_blocks=32, k=4, pattern="concurrent_heads")
            decode_block_ids.extend(trace)

        self.assertEqual(len(decode_block_ids), 128)
        conv_lat = sim_c.estimate_read_latency(decode_block_ids)
        ta_lat = sim_t.estimate_read_latency(decode_block_ids)
        speedup = conv_lat / ta_lat

        self.assertGreaterEqual(speedup, 2.5)
        self.assertGreater(speedup, 3.5)

    def test_t4_attention_sink_and_recent_context_retrieval(self):
        """Simulate StreamingLLM attention sink (4 sink blocks) + recent context (28 recent blocks) per layer."""
        sim_c = StorageSimulator(mode="conventional", channels=8)
        sim_t = StorageSimulator(mode="tensor_aware", channels=8)
        mock = MockKVEngine(layers=16, heads=32)

        retrieval_ids = []
        for l in range(16):
            layer_blocks = mock.generate_kv_blocks(num_blocks=64, layer_id=l)
            for b in layer_blocks:
                sim_c.store_block(b)
                sim_t.store_block(b)
            req = mock.generate_sparse_attention_request(layer_blocks, k=16, sink_ratio=0.25)
            retrieval_ids.extend(req)

        self.assertEqual(len(retrieval_ids), 16 * 16)  # 256 blocks
        conv_lat = sim_c.estimate_read_latency(retrieval_ids)
        ta_lat = sim_t.estimate_read_latency(retrieval_ids)
        speedup = conv_lat / ta_lat

        self.assertGreaterEqual(speedup, 2.5)
        self.assertGreater(speedup, 5.0)

    def test_t4_end_to_end_prefill_then_multi_turn_decode(self):
        """Simulate prefill phase (1,024 blocks written) followed by 10 decode turns tracking read disturbs."""
        sim = StorageSimulator(mode="tensor_aware", channels=8)
        mock = MockKVEngine(layers=32, heads=32)

        # Prefill: store 1024 blocks
        blocks = []
        for l in range(32):
            b_list = mock.generate_kv_blocks(num_blocks=32, layer_id=l)
            for b in b_list:
                sim.store_block(b)
            blocks.extend(b_list)

        # 10 Decode turns: each requests 16 blocks across layer 0
        for turn in range(10):
            turn_ids = mock.generate_attention_trace(layer_id=0, total_blocks=32, k=16)
            lat = sim.estimate_read_latency(turn_ids)
            self.assertEqual(lat, 10.0 + 2 * 30.0)  # 70 us per turn

        # Check physical page read disturb metrics for layer 0 blocks
        for b in blocks[:16]:
            loc = sim.get_location(b.block_id)
            ch, die, pl, blk, pg = parse_physical_location(loc)
            page = sim.channels[ch].dies[die].planes[pl].blocks[blk].pages[pg]
            self.assertEqual(page.read_count, 10)

    def test_t4_gqa_grouped_query_attention_workload(self):
        """Simulate Grouped-Query Attention (GQA) with 8 KV heads striped across 8 SSD channels."""
        sim = StorageSimulator(mode="tensor_aware", channels=8)
        mock = MockKVEngine(layers=1, heads=8)
        gqa_blocks = mock.generate_kv_blocks(num_blocks=8, layout="token_major")
        for b in gqa_blocks:
            sim.store_block(b)

        # All 8 KV heads must be mapped to distinct channels [0..7]
        channels_assigned = set()
        for b in gqa_blocks:
            loc = sim.get_location(b.block_id)
            ch = LatencyModel.extract_channel(loc)
            channels_assigned.add(ch)

        self.assertEqual(channels_assigned, set(range(8)))
        # Parallel read of all 8 KV heads incurs zero channel queueing
        lat = sim.read_blocks(gqa_blocks)
        self.assertEqual(lat, 10.0 + 1 * 30.0)  # exactly 40 us

    def test_t4_throughput_and_iops_comparison(self):
        """Quantify throughput (GB/s) and IOPS for batch 256 blocks (1 MB data) under Conv vs TensorAware."""
        sim_c = StorageSimulator(mode="conventional", channels=8)
        sim_t = StorageSimulator(mode="tensor_aware", channels=8)
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=256)
        for b in blocks:
            sim_c.store_block(b)
            sim_t.store_block(b)

        b_ids = [b.block_id for b in blocks]
        c_lat_us = sim_c.estimate_read_latency(b_ids)  # 7690 us
        t_lat_us = sim_t.estimate_read_latency(b_ids)  # 970 us

        total_bytes = 256 * SSD_PAGE_SIZE_BYTES  # 256 * 4096 = 1,048,576 bytes (1 MB)
        c_throughput_gbps = (total_bytes / (1024 ** 3)) / (c_lat_us * 1e-6)
        t_throughput_gbps = (total_bytes / (1024 ** 3)) / (t_lat_us * 1e-6)

        c_iops = 256 / (c_lat_us * 1e-6)
        t_iops = 256 / (t_lat_us * 1e-6)

        speedup = t_throughput_gbps / c_throughput_gbps
        self.assertGreater(speedup, 7.8)
        self.assertGreater(t_throughput_gbps, 1.0)  # > 1.0 GB/s
        self.assertGreater(t_iops, 250000.0)        # > 250k IOPS


if __name__ == "__main__":
    unittest.main(verbosity=2)
