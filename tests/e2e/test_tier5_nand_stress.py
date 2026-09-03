"""
Tier 5 White-Box Adversarial Stress Testing Suite for AI-SSD & NAND Flash Physics.

Focus Areas:
1. Wear tracking and bad block marking under high erase cycles (>3000).
2. Invalidation of unprogrammed pages, invalidation of already invalid pages, and block impact.
3. Queue starvation under large request bursts, channel queue overflow, and die/bus contention.
4. Malformed address string inputs to LatencyModel and StorageSimulator.
5. Zero-block batch requests and massive multi-thousand block batch scaling.
6. End-to-end StorageSimulator physical state and read disturb lifecycle stress.

Pure Python 3 standard library: zero external dependencies.
"""

import math
import re
import sys
import time
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
from person2_ssd.nand.nand import FlashPlane, FlashDie
from person2_ssd.channels.channel import FlashChannel, ChannelTransferRequest
from person2_ssd.storage_model.latency import LatencyModel
from person2_ssd.storage_model.io_model import StorageSimulator, parse_physical_location


# ============================================================================
# 1. Wear Tracking & Bad Block Marking Under High Erase Cycles
# ============================================================================

class Tier5WearAndBadBlockStressTests(unittest.TestCase):
    """
    White-box stress testing of FlashBlock wear tracking, erase limits,
    bad block transitions, and boundary states.
    """

    def test_erase_cycles_boundary_transition_at_3000(self):
        """
        Verify exact threshold transition for is_bad_block at 3,000 erase cycles.
        Cycles 0..2999 -> is_bad_block == False
        Cycle 3000 -> is_bad_block == True
        """
        block = FlashBlock(block_id=1, pages_count=128, max_erase_cycles=3000)
        self.assertEqual(block.erase_count, 0)
        self.assertFalse(block.is_bad_block)

        # Fast-forward to 2,999 erases
        block.erase_count = 2999
        self.assertFalse(block.is_bad_block)

        # 3,000th erase via official API
        block.erase()
        self.assertEqual(block.erase_count, 3000)
        self.assertTrue(block.is_bad_block)

        # Subsequent erases remain bad block
        block.erase()
        self.assertEqual(block.erase_count, 3001)
        self.assertTrue(block.is_bad_block)

    def test_extreme_high_erase_cycles_stress(self):
        """
        Subject FlashBlock to 5,000 consecutive erase operations.
        Verify counter monotonicity and bad block persistence.
        """
        block = FlashBlock(block_id=42, pages_count=16, max_erase_cycles=3000)
        for i in range(1, 3501):
            block.erase()
            if i < 3000:
                self.assertFalse(block.is_bad_block)
            else:
                self.assertTrue(block.is_bad_block)
            self.assertEqual(block.erase_count, i)
            self.assertEqual(block.free_page_index, 0)

        self.assertEqual(block.erase_count, 3500)
        self.assertTrue(block.is_bad_block)

    def test_custom_and_zero_max_erase_cycles(self):
        """
        Verify custom wear thresholds including immediate retirement thresholds.
        """
        # Threshold of 50 cycles
        b50 = FlashBlock(block_id=10, pages_count=8, max_erase_cycles=50)
        for _ in range(49):
            b50.erase()
        self.assertFalse(b50.is_bad_block)
        b50.erase()
        self.assertTrue(b50.is_bad_block)

        # Immediate retirement (max_erase_cycles=0)
        b0 = FlashBlock(block_id=11, pages_count=8, max_erase_cycles=0)
        self.assertTrue(b0.is_bad_block)

        # Negative threshold (max_erase_cycles=-1)
        b_neg = FlashBlock(block_id=12, pages_count=8, max_erase_cycles=-1)
        self.assertTrue(b_neg.is_bad_block)

    def test_wear_isolation_across_multiple_blocks(self):
        """
        Verify wear tracking in each FlashBlock is strictly independent and
        does not leak across blocks in the same plane.
        """
        plane = FlashPlane(plane_id=0, blocks_per_plane=10)
        # Erase block 0 3000 times
        for _ in range(3000):
            plane[0].erase()

        # Erase block 1 500 times
        for _ in range(500):
            plane[1].erase()

        # Block 2 un-erased
        self.assertEqual(plane[0].erase_count, 3000)
        self.assertTrue(plane[0].is_bad_block)

        self.assertEqual(plane[1].erase_count, 500)
        self.assertFalse(plane[1].is_bad_block)

        self.assertEqual(plane[2].erase_count, 0)
        self.assertFalse(plane[2].is_bad_block)

    def test_zero_page_block_edge_case(self):
        """
        Adversarially instantiate a FlashBlock with pages_count=0.
        Verify property safety (no ZeroDivisionError, no IndexError).
        """
        empty_block = FlashBlock(block_id=99, pages_count=0)
        self.assertEqual(len(empty_block), 0)
        self.assertTrue(empty_block.is_full)
        self.assertTrue(empty_block.is_empty)
        self.assertEqual(empty_block.garbage_ratio, 0.0)
        self.assertEqual(empty_block.valid_page_count, 0)
        self.assertEqual(empty_block.invalid_page_count, 0)
        self.assertEqual(empty_block.free_page_count, 0)

        # Allocation returns None
        alloc = empty_block.allocate_page(101)
        self.assertIsNone(alloc)

        # Erase works safely
        empty_block.erase()
        self.assertEqual(empty_block.erase_count, 1)

    def test_allocation_and_read_behavior_on_bad_block(self):
        """
        Verify physical FlashBlock state transitions when operations continue on bad blocks.
        The block model tracks physical wear without crashing.
        """
        bad_block = FlashBlock(block_id=7, pages_count=4, max_erase_cycles=1)
        bad_block.erase()
        self.assertTrue(bad_block.is_bad_block)

        # Can still program pages physically
        p0 = bad_block.allocate_page(data_block_id=999)
        self.assertIsNotNone(p0)
        self.assertEqual(p0.state, PageState.VALID)
        self.assertEqual(p0.read(), 999)


# ============================================================================
# 2. Invalidation of Unprogrammed Pages & Already Invalid Pages
# ============================================================================

class Tier5PageInvalidationAdversarialTests(unittest.TestCase):
    """
    White-box stress testing of page invalidation edge cases:
    - Invalidation of unprogrammed FREE pages.
    - Repeated invalidation of INVALID pages.
    - Consequences of invalidating unprogrammed pages on sequential allocation.
    - Garbage ratio monotonicity under adversarial invalidation patterns.
    """

    def test_invalidation_of_unprogrammed_free_page(self):
        """
        Directly invalidate a FREE (unprogrammed) page.
        Verify state changes to INVALID and data_block_id is None.
        """
        page = FlashPage(page_id=0)
        self.assertEqual(page.state, PageState.FREE)
        self.assertIsNone(page.data_block_id)

        page.invalidate()
        self.assertEqual(page.state, PageState.INVALID)
        self.assertIsNone(page.data_block_id)

        # Reading an invalidated unprogrammed page returns None but increments read disturb
        val = page.read(current_time_us=5.0)
        self.assertIsNone(val)
        self.assertEqual(page.read_count, 1)
        self.assertEqual(page.last_accessed_us, 5.0)

    def test_programming_invalidated_unprogrammed_page_raises_value_error(self):
        """
        Attempting to program a page that was marked INVALID (even if never programmed)
        must raise ValueError. Flash requires an erase before re-programming.
        """
        page = FlashPage(page_id=1)
        page.invalidate()
        self.assertEqual(page.state, PageState.INVALID)

        with self.assertRaises(ValueError) as ctx:
            page.program(data_block_id=123)
        self.assertIn("Cannot program non-free page", str(ctx.exception))

    def test_double_and_repeated_invalidation_idempotence(self):
        """
        Repeated calls to invalidate() on an already INVALID page must be safe and idempotent.
        """
        page = FlashPage(page_id=2)
        page.program(data_block_id=55)
        self.assertEqual(page.state, PageState.VALID)

        # First invalidation
        page.invalidate()
        self.assertEqual(page.state, PageState.INVALID)
        self.assertIsNone(page.data_block_id)

        # Repeated invalidations
        for _ in range(10):
            page.invalidate()
            self.assertEqual(page.state, PageState.INVALID)
            self.assertIsNone(page.data_block_id)

    def test_erase_recovers_invalidated_unprogrammed_page(self):
        """
        Verifies that erase() resets an INVALID unprogrammed page back to FREE,
        clears read disturb counts, and allows subsequent programming.
        """
        page = FlashPage(page_id=3)
        page.invalidate()
        page.read()  # read_count = 1
        self.assertEqual(page.read_count, 1)

        page.erase()
        self.assertEqual(page.state, PageState.FREE)
        self.assertEqual(page.read_count, 0)
        self.assertIsNone(page.data_block_id)

        # Now program succeeds
        page.program(data_block_id=777)
        self.assertEqual(page.state, PageState.VALID)
        self.assertEqual(page.data_block_id, 777)

    def test_block_allocation_crashes_if_next_page_is_pre_invalidated(self):
        """
        White-box architectural observation:
        FlashBlock.allocate_page() indexes pages sequentially using free_page_index
        and immediately calls page.program(). If an unprogrammed page at free_page_index
        was marked INVALID out-of-band, allocate_page() raises ValueError.
        """
        block = FlashBlock(block_id=5, pages_count=4)
        # Invalidate page 0 directly before allocation
        block[0].invalidate()
        self.assertEqual(block[0].state, PageState.INVALID)

        # Attempt to allocate page on this block
        with self.assertRaises(ValueError) as ctx:
            block.allocate_page(data_block_id=99)
        self.assertIn("Cannot program non-free page", str(ctx.exception))

    def test_monotonic_garbage_ratio_scaling(self):
        """
        Verify that sequentially invalidating pages across a block strictly increases
        invalid_page_count and garbage_ratio monotonically from 0.0 to 1.0.
        """
        pages_count = 128
        block = FlashBlock(block_id=8, pages_count=pages_count)
        # Program all pages
        for i in range(pages_count):
            p = block.allocate_page(data_block_id=1000 + i)
            self.assertIsNotNone(p)

        self.assertEqual(block.valid_page_count, pages_count)
        self.assertEqual(block.invalid_page_count, 0)
        self.assertEqual(block.garbage_ratio, 0.0)

        # Invalidate pages one by one
        for i in range(pages_count):
            block[i].invalidate()
            expected_invalid = i + 1
            expected_valid = pages_count - (i + 1)
            self.assertEqual(block.invalid_page_count, expected_invalid)
            self.assertEqual(block.valid_page_count, expected_valid)
            self.assertAlmostEqual(block.garbage_ratio, (i + 1) / pages_count, delta=1e-6)

        self.assertEqual(block.garbage_ratio, 1.0)


# ============================================================================
# 3. Queue Starvation Under Request Bursts, Overflow, & Contention
# ============================================================================

class Tier5ChannelQueueBurstAndContentionTests(unittest.TestCase):
    """
    White-box stress testing of FlashChannel queueing, request bursts,
    die parallelism vs die contention, and mixed-op serialization.
    """

    def test_massive_request_burst_fifo_processing(self):
        """
        Enqueue a burst of 1,000 read requests into a single FlashChannel queue.
        Verify strict FIFO processing, zero dropped requests, and complete queue clearing.
        """
        channel = FlashChannel(channel_id=0, dies_per_channel=4)
        burst_size = 1000

        for i in range(burst_size):
            req = ChannelTransferRequest(
                request_id=f"burst_{i}",
                die_id=i % 4,
                plane_id=0,
                block_id=0,
                page_id=i % 128,
                op_type="READ",
                arrival_time_us=0.0,
            )
            channel.enqueue_request(req)

        self.assertEqual(len(channel.queue), burst_size)
        self.assertEqual(len(channel.completed_transfers), 0)

        finish_time = channel.process_queue(base_time_us=0.0)

        self.assertEqual(len(channel.queue), 0)
        self.assertEqual(len(channel.completed_transfers), burst_size)
        self.assertGreater(finish_time, 0.0)

        # Verify FIFO order and monotonic completion times
        for idx in range(burst_size):
            req = channel.completed_transfers[idx]
            self.assertEqual(req.request_id, f"burst_{idx}")
            if idx > 0:
                prev_req = channel.completed_transfers[idx - 1]
                self.assertGreaterEqual(req.completion_time_us, prev_req.completion_time_us)

    def test_die_parallelism_vs_die_contention_speedup(self):
        """
        Physical contention proof:
        Compare 4 concurrent reads targeting 4 DISTINCT dies vs 4 concurrent reads on the SAME die.
        - Distinct dies: sensing overlaps (25us), then bus transfers serialize (4 * 5us = 20us) -> 45us.
        - Same die: sensing serializes (4 * 25us = 100us), then bus serializes -> 105us.
        Die parallelism achieves > 2.3x speedup on the channel!
        """
        # Case A: 4 reads on 4 distinct dies (Die 0, 1, 2, 3)
        ch_parallel = FlashChannel(channel_id=0, dies_per_channel=4)
        for d in range(4):
            ch_parallel.enqueue_request(
                ChannelTransferRequest(request_id=d, die_id=d, op_type="READ", arrival_time_us=0.0)
            )
        time_parallel = ch_parallel.process_queue()

        # Expected: Die sensing finishes at 25us for all dies.
        # Bus transfers: Die 0 (25-30), Die 1 (30-35), Die 2 (35-40), Die 3 (40-45). Total = 45us.
        self.assertAlmostEqual(time_parallel, 45.0, delta=1e-6)

        # Case B: 4 reads on the SAME die (Die 0)
        ch_contended = FlashChannel(channel_id=0, dies_per_channel=4)
        for i in range(4):
            ch_contended.enqueue_request(
                ChannelTransferRequest(request_id=i, die_id=0, op_type="READ", arrival_time_us=0.0)
            )
        time_contended = ch_contended.process_queue()

        # Expected: Die 0 senses req 0 (0-25), bus (25-30)
        # Die 0 senses req 1 (25-50), bus (50-55)
        # Die 0 senses req 2 (50-75), bus (75-80)
        # Die 0 senses req 3 (75-100), bus (100-105). Total = 105us.
        self.assertAlmostEqual(time_contended, 105.0, delta=1e-6)

        # Die parallelism speedup ratio
        speedup = time_contended / time_parallel
        self.assertAlmostEqual(speedup, 105.0 / 45.0, delta=1e-3)
        self.assertGreater(speedup, 2.3)

    def test_erase_read_contention_same_die_vs_different_die(self):
        """
        Verify physical serialization when an ERASE (t_BERS=2000us) is followed by a READ.
        - On the SAME die: READ must wait for ERASE to complete (2000us) + t_R (25us) + t_bus (5us) = 2030us.
        - On a DIFFERENT die: READ senses in parallel with erase, completing at 30us!
        """
        # Same Die
        ch_same = FlashChannel(channel_id=1, dies_per_channel=2)
        ch_same.enqueue_request(
            ChannelTransferRequest(request_id="erase0", die_id=0, op_type="ERASE", arrival_time_us=0.0)
        )
        ch_same.enqueue_request(
            ChannelTransferRequest(request_id="read0", die_id=0, op_type="READ", arrival_time_us=0.0)
        )
        time_same = ch_same.process_queue()
        self.assertAlmostEqual(time_same, 2030.0, delta=1e-6)

        # Different Die
        ch_diff = FlashChannel(channel_id=1, dies_per_channel=2)
        ch_diff.enqueue_request(
            ChannelTransferRequest(request_id="erase0", die_id=0, op_type="ERASE", arrival_time_us=0.0)
        )
        ch_diff.enqueue_request(
            ChannelTransferRequest(request_id="read1", die_id=1, op_type="READ", arrival_time_us=0.0)
        )
        time_diff = ch_diff.process_queue()
        # Die 1 read completes at 30us (Die 0 is still erasing in background)
        self.assertAlmostEqual(time_diff, 30.0, delta=1e-6)
        self.assertAlmostEqual(ch_diff.dies[0].busy_until_us, 2000.0, delta=1e-6)

    def test_program_read_contention_same_die(self):
        """
        Verify PROGRAM followed by READ on the same die:
        PROGRAM: bus transfer (0-5us), die program (5-205us).
        READ: die busy until 205us, senses (205-230us), bus transfer (230-235us).
        Total completion = 235us.
        """
        ch = FlashChannel(channel_id=2, dies_per_channel=2)
        ch.enqueue_request(
            ChannelTransferRequest(request_id="prog0", die_id=0, op_type="PROGRAM", arrival_time_us=0.0)
        )
        ch.enqueue_request(
            ChannelTransferRequest(request_id="read0", die_id=0, op_type="READ", arrival_time_us=0.0)
        )
        completion = ch.process_queue()
        self.assertAlmostEqual(completion, 235.0, delta=1e-6)

    def test_out_of_bounds_die_id_graceful_fallback(self):
        """
        Adversarially enqueue requests with out-of-bounds die IDs (-1, 999).
        Verify FlashChannel gracefully falls back to default timing without raising IndexError.
        """
        ch = FlashChannel(channel_id=3, dies_per_channel=4)
        ch.enqueue_request(
            ChannelTransferRequest(request_id="bad_die_neg", die_id=-1, op_type="READ", arrival_time_us=0.0)
        )
        ch.enqueue_request(
            ChannelTransferRequest(request_id="bad_die_hi", die_id=99, op_type="READ", arrival_time_us=0.0)
        )
        completion = ch.process_queue()
        self.assertGreater(completion, 0.0)
        self.assertEqual(len(ch.completed_transfers), 2)

    def test_unknown_op_type_fallback(self):
        """
        Adversarially enqueue a request with an unrecognized op_type (e.g. "CUSTOM_DMA").
        Verify process_queue safely executes default bus transfer branch.
        """
        ch = FlashChannel(channel_id=4)
        ch.enqueue_request(
            ChannelTransferRequest(request_id="custom", die_id=0, op_type="CUSTOM_DMA", arrival_time_us=10.0)
        )
        completion = ch.process_queue()
        self.assertAlmostEqual(completion, 15.0, delta=1e-6)

    def test_empty_queue_processing_and_reset(self):
        """
        Verify process_queue() on an empty queue returns 0.0 without error,
        and reset() completely restores initial state.
        """
        ch = FlashChannel(channel_id=5)
        self.assertEqual(ch.process_queue(), 0.0)

        # Schedule activity and verify reset
        ch.schedule_transfer(0.0, 50.0)
        self.assertEqual(ch.bus_busy_until_us, 50.0)
        ch.reset()
        self.assertEqual(ch.bus_busy_until_us, 0.0)
        self.assertEqual(len(ch.queue), 0)
        self.assertEqual(len(ch.completed_transfers), 0)


# ============================================================================
# 4. Malformed Address String Inputs to LatencyModel & StorageSimulator
# ============================================================================

class Tier5MalformedAddressStressTests(unittest.TestCase):
    """
    White-box stress testing of address parsing robustness across LatencyModel
    and StorageSimulator against malformed, adversarial, and edge-case strings.
    """

    def test_latency_model_extract_channel_non_string_types(self):
        """
        Pass non-string types (None, int, float, list, dict, bool) to extract_channel.
        Verify fallback to channel 0 without exceptions.
        """
        non_strings = [None, 0, 123, -5, 3.1415, ["ch2"], {"ch": 3}, True, False]
        for val in non_strings:
            res = LatencyModel.extract_channel(val)
            self.assertEqual(res, 0, f"Expected 0 for non-string input: {type(val)}")

    def test_latency_model_extract_channel_malformed_and_edge_strings(self):
        """
        Pass adversarial string inputs: empty, whitespace, missing tokens,
        negative tokens, huge numbers, mixed case, delimiters.
        """
        cases = [
            ("", 0),
            ("    ", 0),
            ("\t\n", 0),
            ("no_channel_here", 0),
            ("channel_5", 0),  # Not 'ch<N>'
            ("ch", 0),
            ("ch-3", 0),       # Negative digit not matched by \d+
            ("ch_die_pl_blk_pg", 0),
            ("ch0", 0),
            ("ch7", 7),
            ("CH3_DIE0_PL1_BLK2_PG3", 3),
            ("ssd_ch4_plane1", 4),
            ("PREFIX_ch6_SUFFIX", 6),
            ("ch999999", 999999),
            ("ch00005", 5),
            ("ch1\x00_die0", 1),  # Embedded null byte
        ]
        for s, expected in cases:
            res = LatencyModel.extract_channel(s)
            self.assertEqual(res, expected, f"Failed on input '{s}'")

    def test_parse_physical_location_malformed_inputs(self):
        """
        Test StorageSimulator.parse_physical_location on malformed addresses.
        """
        invalid_inputs = [
            None,
            "",
            "   ",
            12345,
            ["ch0_die0_pl0_blk0_pg0"],
            "ch0_die0_pl0_blk0",          # Missing page
            "die0_pl0_blk0_pg0",          # Missing channel
            "chA_die0_pl0_blk0_pg0",      # Non-numeric channel
            "ch0_dieB_pl0_blk0_pg0",      # Non-numeric die
            "ch-1_die0_pl0_blk0_pg0",     # Negative channel
            "garbage_string",
        ]
        for inv in invalid_inputs:
            res = parse_physical_location(inv)
            self.assertIsNone(res, f"Expected None for invalid address: {inv}")

    def test_parse_physical_location_valid_variations(self):
        """
        Test StorageSimulator.parse_physical_location on valid format variants:
        canonical, 'plane' instead of 'pl', uppercase, and embedded delimiters.
        """
        # Canonical
        res = parse_physical_location("ch0_die1_pl0_blk10_pg5")
        self.assertEqual(res, (0, 1, 0, 10, 5))

        # 'plane' variant
        res_pl = parse_physical_location("ch2_die3_plane1_blk20_pg30")
        self.assertEqual(res_pl, (2, 3, 1, 20, 30))

        # Uppercase
        res_upper = parse_physical_location("CH7_DIE0_PL1_BLK511_PG127")
        self.assertEqual(res_upper, (7, 0, 1, 511, 127))

        # Embedded in prefix/suffix
        res_emb = parse_physical_location("prefix_ch3_die2_pl0_blk5_pg6_suffix")
        self.assertEqual(res_emb, (3, 2, 0, 5, 6))

    def test_storage_simulator_out_of_bounds_address_tolerance(self):
        """
        Adversarially simulate StorageSimulator behavior when physical locations
        reference out-of-bounds channels, dies, planes, blocks, or pages.
        Verify system tolerates them without unhandled IndexError crashes.
        """
        sim = StorageSimulator(mode="tensor_aware", channels=8)

        # Location with channel 99 (beyond 8 channels)
        out_of_bounds_loc = "ch99_die99_pl99_blk9999_pg9999"
        parsed = parse_physical_location(out_of_bounds_loc)
        self.assertEqual(parsed, (99, 99, 99, 9999, 9999))

        # Calculate latency on out-of-bounds locations: LatencyModel extracts ch 99
        lat = sim.latency_model.calculate_batch_read_latency([out_of_bounds_loc])
        self.assertAlmostEqual(lat, PCIE_OVERHEAD_US + (T_R_US + BUS_TRANSFER_US_PER_PAGE), delta=1e-6)

    def test_storage_simulator_reading_unmapped_or_malformed_blocks(self):
        """
        Verify StorageSimulator.read_blocks with non-existent IDs, negative IDs,
        or malformed inputs returns 0.0 latency without crashing.
        """
        sim = StorageSimulator(mode="tensor_aware")
        # Blocks that were never stored
        lat = sim.read_blocks([99999, -1, 404])
        self.assertEqual(lat, 0.0)

        # Mix of non-block objects
        lat_mix = sim.read_blocks(["not_a_block", None, object()])  # type: ignore
        self.assertEqual(lat_mix, 0.0)


# ============================================================================
# 5. Zero-Block & Massive Multi-Thousand Block Requests
# ============================================================================

class Tier5BatchRequestBoundaryStressTests(unittest.TestCase):
    """
    White-box stress testing of extreme batch boundaries:
    - Zero-block batch requests across all latency and simulator APIs.
    - Massive multi-thousand block requests (1,000, 4,000, 10,000 blocks).
    - Analytical speedup verification at scale.
    """

    def test_zero_block_batch_requests(self):
        """
        Verify that passing an empty list to batch read, write, erase,
        and diagnostic breakdown returns 0.0 and cleanly initialized structures.
        """
        lm = LatencyModel()
        self.assertEqual(lm.calculate_batch_read_latency([]), 0.0)
        self.assertEqual(lm.calculate_batch_write_latency([]), 0.0)
        self.assertEqual(lm.calculate_batch_erase_latency([]), 0.0)

        breakdown = lm.get_latency_breakdown([])
        self.assertEqual(breakdown["total_requests"], 0)
        self.assertEqual(breakdown["channel_loads"], {})
        self.assertIsNone(breakdown["bottleneck_channel"])
        self.assertEqual(breakdown["max_channel_load"], 0)
        self.assertEqual(breakdown["total_read_latency_us"], 0.0)
        self.assertEqual(breakdown["total_write_latency_us"], 0.0)
        self.assertEqual(breakdown["total_erase_latency_us"], 0.0)

        sim = StorageSimulator()
        self.assertEqual(sim.read_blocks([]), 0.0)

    def test_single_block_batch_request(self):
        """
        Verify single-block batch request latency:
        T = t_pcie + 1 * (t_R + t_bus) = 10 + 30 = 40 us.
        """
        lm = LatencyModel()
        lat = lm.calculate_batch_read_latency(["ch0_die0_pl0_blk0_pg0"])
        self.assertAlmostEqual(lat, 40.0, delta=1e-6)

    def test_massive_1000_block_batch_contention_vs_striping(self):
        """
        Verify 1,000 block batch request scaling:
        - 1,000 blocks on channel 0 (worst-case): T = 10 + 1000 * 30 = 30,010 us.
        - 1,000 blocks striped evenly across 8 channels (125 per channel):
          T = 10 + 125 * 30 = 3,760 us.
        Speedup: 30010 / 3760 = 7.981x.
        """
        lm = LatencyModel()
        # Worst-case hot-spot
        hotspot_locs = [f"ch0_die0_pl0_blk{i // 128}_pg{i % 128}" for i in range(1000)]
        hotspot_lat = lm.calculate_batch_read_latency(hotspot_locs)
        self.assertAlmostEqual(hotspot_lat, 30010.0, delta=1e-6)

        # Striped across 8 channels
        striped_locs = [f"ch{i % 8}_die0_pl0_blk0_pg{i // 8}" for i in range(1000)]
        striped_lat = lm.calculate_batch_read_latency(striped_locs)
        self.assertAlmostEqual(striped_lat, 3760.0, delta=1e-6)

        speedup = hotspot_lat / striped_lat
        self.assertAlmostEqual(speedup, 30010.0 / 3760.0, delta=1e-3)
        self.assertGreater(speedup, 7.9)

    def test_massive_4000_block_batch_scaling(self):
        """
        Verify 4,000 block batch request scaling:
        - Worst-case: 10 + 4000 * 30 = 120,010 us.
        - Striped (500 per channel): 10 + 500 * 30 = 15,010 us.
        Speedup: 120010 / 15010 = 7.995x.
        """
        lm = LatencyModel()
        hotspot_locs = [f"ch1_die0_pl0_blk0_pg0" for _ in range(4000)]
        hotspot_lat = lm.calculate_batch_read_latency(hotspot_locs)
        self.assertAlmostEqual(hotspot_lat, 120010.0, delta=1e-6)

        striped_locs = [f"ch{i % 8}_die0_pl0_blk0_pg0" for i in range(4000)]
        striped_lat = lm.calculate_batch_read_latency(striped_locs)
        self.assertAlmostEqual(striped_lat, 15010.0, delta=1e-6)

        speedup = hotspot_lat / striped_lat
        self.assertAlmostEqual(speedup, 120010.0 / 15010.0, delta=1e-3)
        self.assertGreater(speedup, 7.99)

    def test_massive_10000_block_execution_efficiency(self):
        """
        Stress test LatencyModel with 10,000 blocks to verify linear runtime complexity.
        Execution must complete within 100 milliseconds.
        """
        lm = LatencyModel()
        locs = [f"ch{i % 8}_die{(i // 8) % 4}_pl0_blk0_pg0" for i in range(10000)]

        t_start = time.perf_counter()
        lat = lm.calculate_batch_read_latency(locs)
        t_elapsed = time.perf_counter() - t_start

        # 10,000 / 8 = 1,250 per channel. Latency = 10 + 1250 * 30 = 37,510 us.
        self.assertAlmostEqual(lat, 37510.0, delta=1e-6)
        self.assertLess(t_elapsed, 0.1, f"10k latency calculation took {t_elapsed:.4f}s (exceeded 0.1s threshold)")


# ============================================================================
# 6. End-to-End StorageSimulator State & Read Disturb Lifecycle
# ============================================================================

class Tier5StorageSimulatorEndToEndStressTests(unittest.TestCase):
    """
    White-box stress testing of StorageSimulator integrating FTL allocation,
    physical FlashBlock page state transitions, read disturb tracking,
    and simulator reset lifecycle.
    """

    def test_storage_simulator_physical_page_programming_verification(self):
        """
        Store blocks via StorageSimulator, parse physical address, and directly
        inspect underlying physical FlashPage to verify state is VALID and
        data_block_id matches logical block ID.
        """
        sim = StorageSimulator(mode="tensor_aware", channels=8)
        block = KVBlock.create_default(
            block_id=501,
            layer_id=0,
            token_start=0,
            token_count=16,
            kv_head_start=0,
            kv_head_count=1,
        )
        loc = sim.store_block(block)
        parsed = parse_physical_location(loc)
        self.assertIsNotNone(parsed)
        ch, die, pl, blk, pg = parsed  # type: ignore

        # Inspect physical FlashPage inside the hierarchy
        physical_page = sim.channels[ch].dies[die].planes[pl].blocks[blk].pages[pg]
        self.assertEqual(physical_page.state, PageState.VALID)
        self.assertEqual(physical_page.data_block_id, 501)
        self.assertEqual(physical_page.program_count, 1)

    def test_read_disturb_counter_stress_on_repeated_reads(self):
        """
        Store a block and issue 50 consecutive read_blocks calls.
        Verify that physical page read_count increments to exactly 50 (read disturb metric).
        """
        sim = StorageSimulator(mode="tensor_aware", channels=8)
        block = KVBlock.create_default(
            block_id=700,
            layer_id=1,
            token_start=0,
            token_count=16,
            kv_head_start=2,
            kv_head_count=1,
        )
        loc = sim.store_block(block)
        ch, die, pl, blk, pg = parse_physical_location(loc)  # type: ignore
        physical_page = sim.channels[ch].dies[die].planes[pl].blocks[blk].pages[pg]

        self.assertEqual(physical_page.read_count, 0)

        for _ in range(50):
            sim.read_blocks([block])

        self.assertEqual(physical_page.read_count, 50)

    def test_partial_hit_batch_read_latency(self):
        """
        Issue a batch read with 8 blocks where only 4 were actually stored in SSD.
        Verify latency reflects only the 4 mapped locations.
        """
        sim = StorageSimulator(mode="tensor_aware", channels=8)
        stored_blocks = []
        for i in range(4):
            b = KVBlock.create_default(
                block_id=i,
                layer_id=0,
                token_start=0,
                token_count=16,
                kv_head_start=i,
                kv_head_count=1,
            )
            sim.store_block(b)
            stored_blocks.append(b)

        # Batch of 8: 4 stored + 4 non-existent
        query_ids = [0, 1, 2, 3, 100, 101, 102, 103]
        lat = sim.read_blocks(query_ids)

        # Only 4 locations resolved across 4 distinct channels:
        # Max channel load = 1. Latency = 10 + 1 * 30 = 40 us.
        self.assertAlmostEqual(lat, 40.0, delta=1e-6)

    def test_simulator_reset_lifecycle(self):
        """
        Verify sim.reset() resets channel bus and die timers while preserving
        previously stored metadata and physical page states.
        """
        sim = StorageSimulator(mode="tensor_aware", channels=8)
        block = KVBlock.create_default(
            block_id=888,
            layer_id=0,
            token_start=0,
            token_count=16,
            kv_head_start=0,
            kv_head_count=1,
        )
        sim.store_block(block)
        sim.read_blocks([block])

        # Channel bus has activity
        loc = sim.get_location(888)
        ch = parse_physical_location(loc)[0]  # type: ignore
        self.assertGreater(sim.channels[ch].bus_busy_until_us, 0.0)

        # Reset simulator
        sim.reset()
        self.assertEqual(sim.channels[ch].bus_busy_until_us, 0.0)
        self.assertEqual(len(sim.channels[ch].queue), 0)

        # Stored metadata is still retrievable
        self.assertEqual(sim.get_location(888), loc)
        self.assertIsNotNone(sim.load_block(888))


# ============================================================================
# Main Execution Entrypoint
# ============================================================================

def load_tier5_suite() -> unittest.TestSuite:
    """Builds and returns the complete Tier 5 unittest suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(Tier5WearAndBadBlockStressTests))
    suite.addTests(loader.loadTestsFromTestCase(Tier5PageInvalidationAdversarialTests))
    suite.addTests(loader.loadTestsFromTestCase(Tier5ChannelQueueBurstAndContentionTests))
    suite.addTests(loader.loadTestsFromTestCase(Tier5MalformedAddressStressTests))
    suite.addTests(loader.loadTestsFromTestCase(Tier5BatchRequestBoundaryStressTests))
    suite.addTests(loader.loadTestsFromTestCase(Tier5StorageSimulatorEndToEndStressTests))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(load_tier5_suite())
    sys.exit(0 if result.wasSuccessful() else 1)
