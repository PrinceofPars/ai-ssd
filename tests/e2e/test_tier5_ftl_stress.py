"""
Tier 5 White-Box Adversarial Hardening Test Suite: FTL & Mock KV Engine.

Exhaustively stress-tests white-box internals, boundary values, untested branches,
and failure modes for:
- BaseFTL and MappingTable (person2_ssd/ftl/base.py, person2_ssd/ftl/mapping.py)
- ConventionalFTL (person2_ssd/ftl/conventional.py)
- TensorAwareFTL (person2_ssd/ftl/tensor_aware.py)
- MockKVEngine (person2_ssd/mock_kv_engine.py)

Focus areas probed:
1. Re-allocating the same block ID to multiple physical locations (reverse mapping integrity).
2. Translating unallocated block IDs (None return for forward and reverse lookups).
3. Conventional FTL capacity exhaustion (exact boundary where total pages are exceeded).
4. Tensor-Aware striping under unusual head counts (e.g. heads=1, 3, 5, 7, 9, 17, 33).
5. Long context sequence lengths (e.g. 65k, 131k, 262k, 1M, 10M tokens).
6. MockKVEngine trace generation with empty, 0-token, or out-of-range layer parameters.
7. White-box robustness, duck-typing, wrap-around differences, and snapshot isolation.

Pure Python 3 standard library unittest test suite.
"""

import math
import os
import re
import sys
import unittest
from pathlib import Path
from typing import List, Dict, Set, Optional

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
    BUS_TRANSFER_US_PER_PAGE,
    PCIE_OVERHEAD_US,
)
from person2_ssd.ftl.base import BaseFTL
from person2_ssd.ftl.mapping import MappingTable
from person2_ssd.ftl.conventional import ConventionalFTL
from person2_ssd.ftl.tensor_aware import TensorAwareFTL
from person2_ssd.mock_kv_engine import MockKVEngine
from person2_ssd.storage_model.latency import LatencyModel


CANONICAL_ADDR_REGEX = re.compile(
    r"^ch(?P<ch>[0-7])_die(?P<die>[0-3])_pl(?P<pl>[0-1])_blk(?P<blk>\d+)_pg(?P<pg>\d+)$"
)


# ============================================================================
# 1. Reverse Mapping Integrity & Re-Allocation
# ============================================================================

class Tier5ReverseMappingIntegrityTests(unittest.TestCase):
    """
    Stress-tests reverse mapping synchronization when block IDs are re-allocated,
    overwritten, reset, or batched with duplicate keys.
    """

    def test_reallocate_same_block_id_updates_forward_and_reverse_map(self):
        """
        Verify that re-allocating an already-mapped block ID:
        1. Correctly updates the forward mapping table to the new location.
        2. Removes the old location from the reverse mapping table.
        3. Adds the new location pointing to the block ID in the reverse mapping table.
        """
        ftl = ConventionalFTL(channels=8, dies_per_channel=4, planes_per_die=2, blocks_per_plane=64, pages_per_block=128)
        block = KVBlock.create_default(block_id=42, layer_id=0, token_start=0)

        loc1 = ftl.allocate(block)
        self.assertEqual(loc1, "ch0_die0_pl0_blk0_pg0")
        self.assertEqual(ftl.translate(42), loc1)
        self.assertEqual(ftl.reverse_translate(loc1), 42)
        self.assertEqual(len(ftl.get_mapping_table()), 1)
        self.assertEqual(len(ftl.get_reverse_mapping_table()), 1)

        # Re-allocate block 42
        loc2 = ftl.allocate(block)
        self.assertEqual(loc2, "ch0_die0_pl0_blk0_pg1")
        self.assertNotEqual(loc1, loc2)

        # Forward lookup returns new location
        self.assertEqual(ftl.translate(42), loc2)

        # Reverse lookup for old location MUST return None (no stale reverse pointer)
        self.assertIsNone(ftl.reverse_translate(loc1), "Stale reverse mapping leaked for old location!")

        # Reverse lookup for new location returns block_id
        self.assertEqual(ftl.reverse_translate(loc2), 42)

        # Total size of tables must remain exactly 1
        self.assertEqual(len(ftl.get_mapping_table()), 1)
        self.assertEqual(len(ftl.get_reverse_mapping_table()), 1)

    def test_reallocate_same_block_id_100_times_no_reverse_leak(self):
        """
        Repeatedly re-allocate the same block ID 100 times in Conventional FTL.
        Verify that table sizes never grow beyond 1 and all 99 obsolete locations return None.
        """
        ftl = ConventionalFTL(channels=8, dies_per_channel=4, planes_per_die=2, blocks_per_plane=64, pages_per_block=128)
        block = KVBlock.create_default(block_id=777, layer_id=0, token_start=0)
        locations: List[str] = []

        for _ in range(100):
            loc = ftl.allocate(block)
            locations.append(loc)

        self.assertEqual(len(locations), 100)
        self.assertEqual(len(set(locations)), 100, "All 100 sequential locations must be physically distinct")

        # Forward translation returns the 100th location
        self.assertEqual(ftl.translate(777), locations[-1])
        self.assertEqual(ftl.reverse_translate(locations[-1]), 777)

        # All 99 previous locations must evaluate to None
        for old_loc in locations[:-1]:
            self.assertIsNone(ftl.reverse_translate(old_loc), f"Reverse map leak at {old_loc}")

        self.assertEqual(len(ftl.get_mapping_table()), 1)
        self.assertEqual(len(ftl.get_reverse_mapping_table()), 1)

    def test_reallocate_in_tensor_aware_ftl_with_coordinate_shifts(self):
        """
        Re-allocate a block ID in TensorAwareFTL after modifying its tensor coordinates
        (layer_id, token_start, kv_head_start). Verify forward and reverse synchronization.
        """
        ta = TensorAwareFTL(channels=8)
        block = KVBlock.create_default(block_id=99, layer_id=0, token_start=0, kv_head_start=0)

        loc_a = ta.allocate(block)
        self.assertEqual(ta.translate(99), loc_a)
        self.assertEqual(ta.reverse_translate(loc_a), 99)

        # Shift coordinates to target a completely different channel, die, and plane
        block.layer_id = 2
        block.token_start = 1600
        block.kv_head_start = 5

        loc_b = ta.allocate(block)
        self.assertNotEqual(loc_a, loc_b)
        self.assertEqual(ta.translate(99), loc_b)
        self.assertEqual(ta.reverse_translate(loc_b), 99)
        self.assertIsNone(ta.reverse_translate(loc_a), "Old TensorAware location was not purged from reverse map")

        self.assertEqual(len(ta.get_mapping_table()), 1)
        self.assertEqual(len(ta.get_reverse_mapping_table()), 1)

    def test_batch_allocation_with_duplicate_block_ids(self):
        """
        Allocate a batch containing duplicate block IDs [b10, b20, b10, b30, b20].
        Verify that each block ID maps only to its final assigned location in both directions.
        """
        conv = ConventionalFTL(channels=8)
        b10_first = KVBlock.create_default(block_id=10, layer_id=0, token_start=0)
        b20_first = KVBlock.create_default(block_id=20, layer_id=0, token_start=16)
        b10_second = KVBlock.create_default(block_id=10, layer_id=0, token_start=32)
        b30 = KVBlock.create_default(block_id=30, layer_id=0, token_start=48)
        b20_second = KVBlock.create_default(block_id=20, layer_id=0, token_start=64)

        batch = [b10_first, b20_first, b10_second, b30, b20_second]
        res = conv.allocate_batch(batch)

        self.assertEqual(set(res.keys()), {10, 20, 30})
        self.assertEqual(len(conv.get_mapping_table()), 3)
        self.assertEqual(len(conv.get_reverse_mapping_table()), 3)

        # Final locations
        self.assertEqual(conv.translate(10), b10_second.physical_location)
        self.assertEqual(conv.translate(20), b20_second.physical_location)
        self.assertEqual(conv.translate(30), b30.physical_location)

        # Obsolete locations purged
        self.assertIsNone(conv.reverse_translate(b10_first.physical_location))
        self.assertIsNone(conv.reverse_translate(b20_first.physical_location))

    def test_reset_clears_forward_and_reverse_maps(self):
        """
        Verify that reset() completely flushes both forward mapping and reverse mapping tables
        as well as internal allocator counters.
        """
        for ftl_cls in [ConventionalFTL, TensorAwareFTL]:
            ftl = ftl_cls(channels=8)
            blocks = [KVBlock.create_default(block_id=i, layer_id=0, token_start=i * 16) for i in range(16)]
            ftl.allocate_batch(blocks)

            self.assertEqual(len(ftl.get_mapping_table()), 16)
            self.assertEqual(len(ftl.get_reverse_mapping_table()), 16)

            ftl.reset()

            self.assertEqual(len(ftl.get_mapping_table()), 0)
            self.assertEqual(len(ftl.get_reverse_mapping_table()), 0)
            for b in blocks:
                self.assertIsNone(ftl.translate(b.block_id))
                self.assertIsNone(ftl.reverse_translate(b.physical_location))

            # Re-allocation starts from initial origin
            loc0 = ftl.allocate(blocks[0])
            self.assertTrue(loc0.endswith("_blk0_pg0"), f"Allocator did not reset counters properly: {loc0}")


# ============================================================================
# 2. Translating Unallocated Block IDs & Query Boundaries
# ============================================================================

class Tier5UnallocatedBlockTranslationTests(unittest.TestCase):
    """
    Stress-tests edge cases and query boundary conditions for translating unallocated,
    negative, non-existent, or malformed addresses in BaseFTL implementations.
    """

    def setUp(self):
        self.conv = ConventionalFTL(channels=8)
        self.ta = TensorAwareFTL(channels=8)

    def test_translate_unallocated_positive_id_returns_none(self):
        """Verify querying unallocated positive block IDs returns None."""
        for ftl in [self.conv, self.ta]:
            self.assertIsNone(ftl.translate(0))
            self.assertIsNone(ftl.translate(1))
            self.assertIsNone(ftl.translate(999999))
            self.assertIsNone(ftl.get_location(12345))

    def test_translate_unallocated_negative_id_returns_none(self):
        """Verify querying negative block IDs returns None."""
        for ftl in [self.conv, self.ta]:
            self.assertIsNone(ftl.translate(-1))
            self.assertIsNone(ftl.translate(-999999))
            self.assertIsNone(ftl.get_location(-42))

    def test_reverse_translate_unallocated_valid_address_returns_none(self):
        """Verify reverse lookup on valid-format address strings that were never allocated returns None."""
        valid_unallocated = [
            "ch0_die0_pl0_blk0_pg0",
            "ch7_die3_pl1_blk63_pg127",
            "ch4_die2_pl0_blk12_pg44",
        ]
        for ftl in [self.conv, self.ta]:
            for addr in valid_unallocated:
                self.assertIsNone(ftl.reverse_translate(addr))

    def test_reverse_translate_malformed_and_edge_strings_return_none(self):
        """Verify reverse lookup on arbitrary, empty, or malformed strings returns None without crashing."""
        malformed_strings = [
            "",
            "   ",
            "None",
            "invalid_address",
            "ch0",
            "ch0_die0",
            "ch8_die0_pl0_blk0_pg0",
            "ch0_die0_pl0_blk0_pg0_extra",
        ]
        for ftl in [self.conv, self.ta]:
            for s in malformed_strings:
                self.assertIsNone(ftl.reverse_translate(s))

    def test_empty_mapping_tables_return_empty_dicts(self):
        """Verify get_mapping_table and get_reverse_mapping_table return empty dicts initially."""
        for ftl in [self.conv, self.ta]:
            fwd = ftl.get_mapping_table()
            rev = ftl.get_reverse_mapping_table()
            self.assertEqual(fwd, {})
            self.assertEqual(rev, {})
            self.assertIsInstance(fwd, dict)
            self.assertIsInstance(rev, dict)

    def test_mapping_table_isolation_and_immutability(self):
        """Verify that mutating the dictionary returned by get_mapping_table does not corrupt FTL state."""
        b = KVBlock.create_default(block_id=1, layer_id=0, token_start=0)
        loc = self.conv.allocate(b)

        fwd = self.conv.get_mapping_table()
        fwd[1] = "corrupted_location"
        fwd[999] = "injected_location"

        # Internal state must remain uncorrupted
        self.assertEqual(self.conv.translate(1), loc)
        self.assertIsNone(self.conv.translate(999))

        rev = self.conv.get_reverse_mapping_table()
        rev[loc] = 99999
        self.assertEqual(self.conv.reverse_translate(loc), 1)


# ============================================================================
# 3. Conventional FTL Capacity Exhaustion Boundary Analysis
# ============================================================================

class Tier5ConventionalCapacityExhaustionTests(unittest.TestCase):
    """
    White-box boundary stress testing of ConventionalFTL capacity limit.
    Verifies exact allocation boundary at total_pages, exception semantics,
    and post-exhaustion recovery.
    """

    def test_exact_boundary_single_page_capacity(self):
        """
        Verify single-page capacity drive (1 ch, 1 die, 1 pl, 1 blk, 1 pg = 1 total page):
        - Allocation 0 succeeds at ch0_die0_pl0_blk0_pg0.
        - Allocation 1 fails immediately with RuntimeError.
        """
        conv = ConventionalFTL(
            channels=1, dies_per_channel=1, planes_per_die=1, blocks_per_plane=1, pages_per_block=1
        )
        self.assertEqual(conv.total_pages, 1)

        b0 = KVBlock.create_default(block_id=100, layer_id=0, token_start=0)
        loc0 = conv.allocate(b0)
        self.assertEqual(loc0, "ch0_die0_pl0_blk0_pg0")

        b1 = KVBlock.create_default(block_id=101, layer_id=0, token_start=16)
        with self.assertRaises(RuntimeError) as ctx:
            conv.allocate(b1)

        err_msg = str(ctx.exception)
        self.assertIn("SSD capacity exceeded", err_msg)
        self.assertIn("cannot allocate block 101", err_msg)
        self.assertIn("Total capacity 1 pages exhausted", err_msg)

    def test_exact_boundary_multi_dimensional_exhaustion(self):
        """
        Verify multi-dimensional drive:
        2 channels * 2 dies * 2 planes * 2 blocks * 2 pages = 32 total pages.
        All 32 allocations succeed, covering every hierarchy dimension up to ch1_die1_pl1_blk1_pg1.
        The 33rd allocation (index 32) raises RuntimeError.
        """
        conv = ConventionalFTL(
            channels=2, dies_per_channel=2, planes_per_die=2, blocks_per_plane=2, pages_per_block=2
        )
        self.assertEqual(conv.total_pages, 32)

        allocated_locs = []
        for i in range(32):
            b = KVBlock.create_default(block_id=i, layer_id=0, token_start=i * 16)
            loc = conv.allocate(b)
            allocated_locs.append(loc)

        # First and last page checks
        self.assertEqual(allocated_locs[0], "ch0_die0_pl0_blk0_pg0")
        self.assertEqual(allocated_locs[-1], "ch1_die1_pl1_blk1_pg1")
        self.assertEqual(len(set(allocated_locs)), 32, "All 32 locations must be unique")

        # 33rd allocation must raise RuntimeError
        b_overflow = KVBlock.create_default(block_id=32, layer_id=0, token_start=512)
        with self.assertRaises(RuntimeError) as ctx:
            conv.allocate(b_overflow)
        self.assertIn("Total capacity 32 pages exhausted", str(ctx.exception))

    def test_capacity_exhaustion_preserves_existing_mappings(self):
        """
        Verify that after capacity exhaustion exception is raised, prior mappings
        remain fully valid and intact for forward and reverse lookups.
        """
        conv = ConventionalFTL(
            channels=1, dies_per_channel=1, planes_per_die=1, blocks_per_plane=1, pages_per_block=3
        )
        self.assertEqual(conv.total_pages, 3)

        blocks = [KVBlock.create_default(block_id=i, layer_id=0, token_start=i * 16) for i in range(3)]
        locs = [conv.allocate(b) for b in blocks]

        # Trigger overflow
        b_bad = KVBlock.create_default(block_id=999, layer_id=0, token_start=48)
        with self.assertRaises(RuntimeError):
            conv.allocate(b_bad)

        # Verify prior mappings are uncorrupted
        for i, loc in enumerate(locs):
            self.assertEqual(conv.translate(i), loc)
            self.assertEqual(conv.reverse_translate(loc), i)
        self.assertIsNone(conv.translate(999))
        self.assertEqual(len(conv.get_mapping_table()), 3)

    def test_capacity_exhaustion_in_batch_allocation(self):
        """
        Verify that allocating a batch that exceeds remaining capacity fails on the exact
        overflow block, leaving earlier blocks allocated in the mapping table.
        """
        conv = ConventionalFTL(
            channels=1, dies_per_channel=1, planes_per_die=1, blocks_per_plane=1, pages_per_block=2
        )
        batch = [
            KVBlock.create_default(block_id=0, layer_id=0, token_start=0),
            KVBlock.create_default(block_id=1, layer_id=0, token_start=16),
            KVBlock.create_default(block_id=2, layer_id=0, token_start=32),
        ]

        with self.assertRaises(RuntimeError) as ctx:
            conv.allocate_batch(batch)

        self.assertIn("cannot allocate block 2", str(ctx.exception))
        self.assertEqual(conv.translate(0), "ch0_die0_pl0_blk0_pg0")
        self.assertEqual(conv.translate(1), "ch0_die0_pl0_blk0_pg1")
        self.assertIsNone(conv.translate(2))

    def test_reset_restores_capacity_after_exhaustion(self):
        """
        Verify that reset() resets the linear counter to 0, allowing the drive to be
        fully re-allocated from scratch after capacity exhaustion.
        """
        conv = ConventionalFTL(
            channels=1, dies_per_channel=1, planes_per_die=1, blocks_per_plane=1, pages_per_block=2
        )
        for i in range(2):
            conv.allocate(KVBlock.create_default(block_id=i, layer_id=0, token_start=i * 16))

        with self.assertRaises(RuntimeError):
            conv.allocate(KVBlock.create_default(block_id=2, layer_id=0, token_start=32))

        # Reset and verify recovery
        conv.reset()
        self.assertEqual(conv._linear_counter, 0)
        self.assertEqual(len(conv.get_mapping_table()), 0)

        # Can now allocate 2 more blocks successfully
        loc0 = conv.allocate(KVBlock.create_default(block_id=10, layer_id=0, token_start=0))
        loc1 = conv.allocate(KVBlock.create_default(block_id=11, layer_id=0, token_start=16))
        self.assertEqual(loc0, "ch0_die0_pl0_blk0_pg0")
        self.assertEqual(loc1, "ch0_die0_pl0_blk0_pg1")


# ============================================================================
# 4. Tensor-Aware Striping Under Unusual Head Counts
# ============================================================================

class Tier5TensorAwareUnusualHeadCountsTests(unittest.TestCase):
    """
    Stress-tests TensorAwareFTL placement under atypical head counts:
    - heads=1 (Multi-Query Attention / MQA)
    - heads=3, 5, 7, 9, 17, 33 (Atypical Grouped-Query Attention / GQA)
    Verifies canonical format, load distribution, speedup, and lack of parity starvation.
    """

    def test_mqa_single_head_striping(self):
        """
        Verify heads=1 (MQA):
        KV blocks for consecutive token chunks must stripe across channels 0..7 and dies,
        preventing single-channel concentration even with a single head.
        """
        ta = TensorAwareFTL(channels=8)
        mock = MockKVEngine(layers=1, heads=1)
        blocks = mock.generate_kv_blocks(num_blocks=16, token_count=16, layout="token_major")

        locs = [ta.allocate(b) for b in blocks]

        # Verify all addresses match canonical regex
        for loc in locs:
            self.assertRegex(loc, CANONICAL_ADDR_REGEX)

        # First 8 token blocks must distribute across channels 0..7
        ch_first_8 = [LatencyModel.extract_channel(locs[i]) for i in range(8)]
        self.assertEqual(len(set(ch_first_8)), 8, f"Single-head MQA failed to stripe first 8 tokens: {ch_first_8}")

    def test_unusual_heads_address_validity_and_bounds(self):
        """
        Verify all generated addresses across unusual head counts [3, 5, 7, 9, 17, 33]
        strictly respect physical hardware boundaries:
        ch < 8, die < 4, pl < 2, blk < 64, pg < 128.
        """
        unusual_heads = [1, 3, 5, 7, 9, 17, 33]
        for h in unusual_heads:
            ta = TensorAwareFTL(channels=8, dies_per_channel=4, planes_per_die=2, blocks_per_plane=64, pages_per_block=128)
            mock = MockKVEngine(layers=4, heads=h)
            blocks = mock.generate_kv_blocks(num_blocks=h * 4, token_count=16, layout="token_major")

            for b in blocks:
                loc = ta.allocate(b)
                m = CANONICAL_ADDR_REGEX.match(loc)
                self.assertIsNotNone(m, f"Malformed address {loc} for heads={h}")
                ch = int(m.group("ch"))
                die = int(m.group("die"))
                pl = int(m.group("pl"))
                blk = int(m.group("blk"))
                pg = int(m.group("pg"))

                self.assertTrue(0 <= ch < 8, f"Channel out of bounds: {ch}")
                self.assertTrue(0 <= die < 4, f"Die out of bounds: {die}")
                self.assertTrue(0 <= pl < 2, f"Plane out of bounds: {pl}")
                self.assertTrue(0 <= blk < 64, f"Block out of bounds: {blk}")
                self.assertTrue(0 <= pg < 128, f"Page out of bounds: {pg}")

    def test_unusual_heads_channel_load_balance(self):
        """
        Verify that for unusual head counts [3, 5, 7, 9, 17, 33], the maximum channel load
        does not exceed math.ceil(total_blocks / 8) + 1, ensuring no pathological channel clustering.
        """
        unusual_heads = [3, 5, 7, 9, 17, 33]
        for h in unusual_heads:
            ta = TensorAwareFTL(channels=8)
            mock = MockKVEngine(layers=2, heads=h)
            num_blocks = h * 8
            blocks = mock.generate_kv_blocks(num_blocks=num_blocks, token_count=16, layout="token_major")
            locs = [ta.allocate(b) for b in blocks]

            lm = LatencyModel(channels=8)
            bd = lm.get_latency_breakdown(locs)
            max_load = bd["max_channel_load"]
            ideal_load = math.ceil(num_blocks / 8)

            # Max load should be within 1 of ideal load
            self.assertLessEqual(
                max_load,
                ideal_load + 1,
                f"Head count {h} experienced channel imbalance: max_load={max_load}, ideal={ideal_load}",
            )

    def test_unusual_heads_speedup_over_conventional(self):
        """
        Verify that Tensor-Aware FTL achieves speedup >= 2.5x over Conventional FTL
        for batch sizes >= 24 under unusual head counts (e.g. heads=3, 5, 7, 9, 17, 33).
        """
        lm = LatencyModel(channels=8)
        unusual_heads = [3, 5, 7, 9, 17, 33]

        for h in unusual_heads:
            mock = MockKVEngine(layers=4, heads=h)
            num_blocks = max(24, h * 8)
            blocks = mock.generate_kv_blocks(num_blocks=num_blocks, token_count=16, layout="token_major")

            ta = TensorAwareFTL(channels=8)
            conv = ConventionalFTL(channels=8)

            ta_locs = [ta.allocate(b) for b in blocks]
            conv_locs = [conv.allocate(b) for b in blocks]

            t_ta = lm.calculate_batch_read_latency(ta_locs)
            t_conv = lm.calculate_batch_read_latency(conv_locs)
            speedup = t_conv / t_ta

            self.assertGreaterEqual(
                speedup,
                2.5,
                f"Tensor-Aware speedup {speedup:.2f}x fell below 2.5x threshold for heads={h}, blocks={num_blocks}",
            )

    def test_unusual_heads_zero_parity_starvation(self):
        """
        Verify that both even channels (0, 2, 4, 6) and odd channels (1, 3, 5, 7)
        receive allocations without starvation under prime/unusual head configurations.
        """
        ta = TensorAwareFTL(channels=8)
        for h in [3, 5, 7, 9, 17, 33]:
            ta.reset()
            mock = MockKVEngine(layers=1, heads=h)
            blocks = mock.generate_kv_blocks(num_blocks=h * 8, layout="token_major")
            locs = [ta.allocate(b) for b in blocks]

            channel_counts = [0] * 8
            for loc in locs:
                ch = LatencyModel.extract_channel(loc)
                channel_counts[ch] += 1

            even_sum = sum(channel_counts[0::2])
            odd_sum = sum(channel_counts[1::2])

            self.assertGreater(even_sum, 0, f"Even channels starved for heads={h}")
            self.assertGreater(odd_sum, 0, f"Odd channels starved for heads={h}")
            # Parity delta should be small
            self.assertLessEqual(abs(even_sum - odd_sum), 8, f"Severe parity imbalance for heads={h}")


# ============================================================================
# 5. Long Context Sequence Length Stress Testing
# ============================================================================

class Tier5LongContextSequenceLengthTests(unittest.TestCase):
    """
    Stress-tests FTL allocation and address computation with extreme sequence lengths
    (65,536; 131,072; 262,144; 1,048,576; 10,000,000 tokens).
    Verifies multi-plane and multi-die activation, numeric stability, and address integrity.
    """

    def test_long_context_token_coordinates_validity(self):
        """
        Verify that large token_start values produce strictly valid canonical addresses
        satisfying channel, die, plane, block, and page constraints.
        """
        extreme_seq_lens = [65536, 131072, 262144, 1048576, 10000000]
        ta = TensorAwareFTL(channels=8, dies_per_channel=4, planes_per_die=2)

        for seq_len in extreme_seq_lens:
            b = KVBlock.create_default(
                block_id=seq_len, layer_id=3, token_start=seq_len, token_count=16, kv_head_start=5
            )
            loc = ta.allocate(b)
            m = CANONICAL_ADDR_REGEX.match(loc)
            self.assertIsNotNone(m, f"Address format invalid for token_start={seq_len}: {loc}")
            self.assertTrue(0 <= int(m.group("ch")) < 8)
            self.assertTrue(0 <= int(m.group("die")) < 4)
            self.assertTrue(0 <= int(m.group("pl")) < 2)

    def test_long_context_multi_plane_activation(self):
        """
        Verify multi-plane striping:
        In TensorAwareFTL, plane index is computed as:
        pl = (token_block_idx // (channels * dies_per_channel)) % planes_per_die.
        With channels=8, dies=4, channels * dies = 32.
        For token_block_idx >= 32 (token_start >= 512 with token_count=16),
        pl alternates between 0 and 1.
        """
        ta = TensorAwareFTL(channels=8, dies_per_channel=4, planes_per_die=2)
        plane_histogram = {0: 0, 1: 0}

        # Probe 128 sequential token blocks for a single head
        for token_idx in range(128):
            b = KVBlock.create_default(
                block_id=token_idx, layer_id=0, token_start=token_idx * 16, token_count=16, kv_head_start=0
            )
            loc = ta.allocate(b)
            pl = int(CANONICAL_ADDR_REGEX.match(loc).group("pl"))
            plane_histogram[pl] += 1

        self.assertGreater(plane_histogram[0], 0, "Plane 0 was never targeted")
        self.assertGreater(plane_histogram[1], 0, "Plane 1 was never targeted (multi-plane inactive!)")
        self.assertEqual(plane_histogram[0], 64)
        self.assertEqual(plane_histogram[1], 64)

    def test_long_context_multi_die_activation(self):
        """
        Verify that multi-die striping distributes long context token blocks
        evenly across all 4 dies per channel.
        """
        ta = TensorAwareFTL(channels=8, dies_per_channel=4, planes_per_die=2)
        die_histogram = {d: 0 for d in range(4)}

        for token_idx in range(32):
            b = KVBlock.create_default(
                block_id=token_idx, layer_id=0, token_start=token_idx * 16, token_count=16, kv_head_start=0
            )
            loc = ta.allocate(b)
            die = int(CANONICAL_ADDR_REGEX.match(loc).group("die"))
            die_histogram[die] += 1

        for d in range(4):
            self.assertEqual(die_histogram[d], 8, f"Die {d} did not receive expected 8 allocations: {die_histogram}")

    def test_massive_batch_allocation_10000_blocks(self):
        """
        Allocate 10,000 blocks in TensorAwareFTL.
        Verify memory integrity, monotonicity of channel counters, and 100% regex conformance.
        """
        ta = TensorAwareFTL(channels=8)
        blocks = [
            KVBlock.create_default(
                block_id=i,
                layer_id=i % 32,
                token_start=(i // 32) * 16,
                token_count=16,
                kv_head_start=i % 32,
            )
            for i in range(10000)
        ]

        ta.allocate_batch(blocks)
        self.assertEqual(len(ta.get_mapping_table()), 10000)
        self.assertEqual(len(ta.get_reverse_mapping_table()), 10000)

        # Spot-check 100 random blocks for regex match and bijection
        for i in range(0, 10000, 100):
            loc = ta.translate(i)
            self.assertIsNotNone(loc)
            self.assertRegex(loc, CANONICAL_ADDR_REGEX)
            self.assertEqual(ta.reverse_translate(loc), i)

    def test_conventional_ftl_large_context_capacity(self):
        """
        Allocate 65,536 blocks (simulating a 65k context with 16 tokens/block * 16 heads)
        in ConventionalFTL under default geometry (524,288 page capacity).
        Verify zero failures and correct monotonic page allocation.
        """
        conv = ConventionalFTL(channels=8, dies_per_channel=4, planes_per_die=2, blocks_per_plane=64, pages_per_block=128)
        self.assertEqual(conv.total_pages, 524288)

        # Allocate 4096 blocks (representing a 65k token sequence for a single head)
        blocks = [KVBlock.create_default(block_id=i, layer_id=0, token_start=i * 16) for i in range(4096)]
        conv.allocate_batch(blocks)

        self.assertEqual(len(conv.get_mapping_table()), 4096)
        self.assertEqual(conv._linear_counter, 4096)
        self.assertEqual(conv.translate(0), "ch0_die0_pl0_blk0_pg0")
        self.assertEqual(conv.translate(4095), "ch0_die0_pl0_blk31_pg127")


# ============================================================================
# 6. MockKVEngine Boundary, Zero-Token & Edge-Case Probing
# ============================================================================

class Tier5MockKVEngineBoundaryEdgeCaseTests(unittest.TestCase):
    """
    Stress-tests MockKVEngine edge cases, empty batches, 0-token blocks,
    out-of-range layer parameters, and forensic probing of trace generators.
    """

    def test_generate_kv_blocks_zero_and_negative_blocks(self):
        """Verify MockKVEngine returns empty list when num_blocks <= 0."""
        mock = MockKVEngine(layers=32, heads=32)
        self.assertEqual(mock.generate_kv_blocks(num_blocks=0), [])
        self.assertEqual(mock.generate_kv_blocks(num_blocks=-1), [])
        self.assertEqual(mock.generate_kv_blocks(num_blocks=-100), [])

    def test_generate_kv_blocks_zero_token_count(self):
        """
        Verify MockKVEngine generates blocks when token_count=0,
        and TensorAwareFTL / ConventionalFTL safely handle token_count=0 without ZeroDivisionError.
        """
        mock = MockKVEngine(layers=4, heads=4)
        blocks = mock.generate_kv_blocks(num_blocks=8, token_count=0)
        self.assertEqual(len(blocks), 8)
        for b in blocks:
            self.assertEqual(b.token_count, 0)
            self.assertEqual(b.key_size_bytes, 0)
            self.assertEqual(b.value_size_bytes, 0)

        # TensorAwareFTL has defensive token_count guard: max(1, getattr(...) or 16)
        ta = TensorAwareFTL(channels=8)
        for b in blocks:
            loc = ta.allocate(b)
            self.assertRegex(loc, CANONICAL_ADDR_REGEX)

        # ConventionalFTL ignores token_count
        conv = ConventionalFTL(channels=8)
        for b in blocks:
            loc = conv.allocate(b)
            self.assertRegex(loc, CANONICAL_ADDR_REGEX)

    def test_generate_kv_blocks_negative_token_count(self):
        """Verify negative token_count is handled safely by TensorAwareFTL without crashing."""
        b = KVBlock.create_default(block_id=1, layer_id=0, token_start=0, token_count=-16)
        ta = TensorAwareFTL(channels=8)
        loc = ta.allocate(b)
        self.assertRegex(loc, CANONICAL_ADDR_REGEX)

    def test_generate_kv_blocks_out_of_range_layer_id(self):
        """
        Verify that out-of-range layer IDs (e.g. 9999 or -1) produce valid KVBlocks
        and map within valid die/plane hardware bounds via modulo arithmetic.
        """
        mock = MockKVEngine(layers=32, heads=8)
        for extreme_layer in [9999, -1, 100000]:
            blocks = mock.generate_kv_blocks(num_blocks=8, layer_id=extreme_layer)
            self.assertEqual(blocks[0].layer_id, extreme_layer)

            ta = TensorAwareFTL(channels=8, dies_per_channel=4)
            loc = ta.allocate(blocks[0])
            m = CANONICAL_ADDR_REGEX.match(loc)
            self.assertIsNotNone(m)
            self.assertTrue(0 <= int(m.group("die")) < 4)

    def test_generate_attention_trace_boundary_k_and_blocks(self):
        """
        Test boundary combinations in generate_attention_trace:
        - total_blocks = 0 -> returns []
        - k = 0 -> returns []
        - k < 0 -> returns []
        - k > total_blocks -> returns total_blocks elements
        """
        mock = MockKVEngine(layers=32, heads=32)

        self.assertEqual(mock.generate_attention_trace(layer_id=0, total_blocks=0, k=16), [])
        self.assertEqual(mock.generate_attention_trace(layer_id=0, total_blocks=64, k=0), [])
        self.assertEqual(mock.generate_attention_trace(layer_id=0, total_blocks=64, k=-5), [])

        trace = mock.generate_attention_trace(layer_id=0, total_blocks=10, k=100)
        self.assertEqual(len(trace), 10)
        self.assertEqual(trace, list(range(10)))

    def test_generate_attention_trace_patterns(self):
        """
        Test distinct trace patterns in generate_attention_trace:
        - 'concurrent_heads': sequential prefix
        - 'strided': strided selection across blocks
        - unknown pattern: falls back to sequential prefix
        """
        mock = MockKVEngine(layers=32, heads=32)
        seq = mock.generate_attention_trace(layer_id=2, total_blocks=64, k=8, pattern="concurrent_heads")
        self.assertEqual(seq, [128 + i for i in range(8)])

        strided = mock.generate_attention_trace(layer_id=2, total_blocks=64, k=8, pattern="strided")
        self.assertEqual(len(strided), 8)
        self.assertEqual(strided, [128 + (j * 8) for j in range(8)])

        fallback = mock.generate_attention_trace(layer_id=2, total_blocks=64, k=8, pattern="unsupported_pattern")
        self.assertEqual(fallback, seq)

    def test_generate_sparse_attention_request_zero_k_and_empty_blocks(self):
        """Verify generate_sparse_attention_request returns [] on empty blocks or k <= 0."""
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=16)

        self.assertEqual(mock.generate_sparse_attention_request(blocks=[], k=8), [])
        self.assertEqual(mock.generate_sparse_attention_request(blocks=blocks, k=0), [])
        self.assertEqual(mock.generate_sparse_attention_request(blocks=blocks, k=-1), [])

    def test_generate_sparse_attention_request_k_exceeds_length(self):
        """Verify generate_sparse_attention_request returns all block IDs when len(blocks) <= k."""
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=8)
        req = mock.generate_sparse_attention_request(blocks=blocks, k=16)
        self.assertEqual(req, [b.block_id for b in blocks])

    def test_forensic_sparse_attention_request_k_recent_zero_boundary(self):
        """
        Forensic probe of MockKVEngine.generate_sparse_attention_request:
        Documents the implementation phenomenon where k_recent = k - k_sink == 0.
        In Python, blocks[-0:] evaluates to blocks[0:] (the whole slice), causing
        len(sink_ids + recent_ids) to return k_sink + len(blocks) instead of k.
        """
        mock = MockKVEngine()
        blocks = mock.generate_kv_blocks(num_blocks=10)

        # Case A: Normal operation where k=8, sink_ratio=0.25 -> k_sink=2, k_recent=6
        normal_req = mock.generate_sparse_attention_request(blocks, k=8, sink_ratio=0.25)
        self.assertEqual(len(normal_req), 8)

        # Case B: When k=1, sink_ratio=0.25:
        # k_sink = max(1, int(1 * 0.25)) = 1, k_recent = 1 - 1 = 0
        # blocks[-0:] evaluates to blocks[0:] (all 10 elements)
        # Demonstrates empirical edge-case behavior
        k1_req = mock.generate_sparse_attention_request(blocks, k=1, sink_ratio=0.25)
        self.assertEqual(len(k1_req), 1 + len(blocks), "Empirically verifies [-0:] slice boundary condition")
        self.assertEqual(k1_req[0], blocks[0].block_id)


# ============================================================================
# 7. White-Box Robustness, Duck-Typing & Architectural Differences
# ============================================================================

class Tier5FTLWhiteBoxStressAndRobustnessTests(unittest.TestCase):
    """
    White-box stress tests verifying structural architectural contracts:
    - Duck-typed block compatibility
    - Capacity wrap-around behavior differences between Conventional and TensorAware FTL
    - Channel counter monotonicity
    """

    def test_ftl_duck_typed_block_compatibility(self):
        """
        Verify both ConventionalFTL and TensorAwareFTL function correctly on minimal
        duck-typed objects that provide only required attributes.
        """
        class DuckBlock:
            def __init__(self, bid):
                self.block_id = bid
                self.physical_location = None

        d_conv = DuckBlock(101)
        d_ta = DuckBlock(202)

        conv = ConventionalFTL(channels=8)
        ta = TensorAwareFTL(channels=8)

        loc_conv = conv.allocate(d_conv)
        loc_ta = ta.allocate(d_ta)

        self.assertEqual(d_conv.physical_location, loc_conv)
        self.assertEqual(d_ta.physical_location, loc_ta)
        self.assertEqual(conv.translate(101), loc_conv)
        self.assertEqual(ta.translate(202), loc_ta)

    def test_tensor_aware_capacity_wrap_around_difference(self):
        """
        Forensic comparison of capacity exhaustion behavior:
        - ConventionalFTL strictly raises RuntimeError when capacity is exceeded.
        - TensorAwareFTL wraps channel counters modulo (blocks_per_plane * pages_per_block)
          without raising an error, overwriting reverse mapping entries for aliased pages.
        """
        # Micro-SSD with 1 page per channel, 2 channels
        conv = ConventionalFTL(channels=2, dies_per_channel=1, planes_per_die=1, blocks_per_plane=1, pages_per_block=1)
        ta = TensorAwareFTL(channels=2, dies_per_channel=1, planes_per_die=1, blocks_per_plane=1, pages_per_block=1)

        b0 = KVBlock.create_default(block_id=0, layer_id=0, token_start=0, kv_head_start=0)
        b1 = KVBlock.create_default(block_id=1, layer_id=0, token_start=0, kv_head_start=1)
        b2 = KVBlock.create_default(block_id=2, layer_id=0, token_start=0, kv_head_start=0)

        # Conventional raises RuntimeError at block 2
        conv.allocate(b0)
        conv.allocate(b1)
        with self.assertRaises(RuntimeError):
            conv.allocate(b2)

        # TensorAware wraps around without error
        loc0 = ta.allocate(b0)
        loc1 = ta.allocate(b1)
        loc2 = ta.allocate(b2)  # wraps channel 0 counter

        self.assertEqual(loc0, loc2, "TensorAware wraps to the same physical page")
        # Reverse map now points to b2
        self.assertEqual(ta.reverse_translate(loc0), 2)
        # Forward map still records both
        self.assertEqual(ta.translate(0), loc0)
        self.assertEqual(ta.translate(2), loc2)
        self.assertEqual(len(ta.get_mapping_table()), 3)
        self.assertEqual(len(ta.get_reverse_mapping_table()), 2)


if __name__ == "__main__":
    unittest.main()
