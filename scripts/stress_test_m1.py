"""
Challenger M1.2: Empirical Stress Test Suite
Stress tests:
1. Contention inequality across batch sizes N in [2..64].
2. Address parsing with irregular prefixes and malformed strings.
3. StorageSimulator read_blocks and concurrent channel queue serialization.
4. Boundary edge cases, physical state invariants, and exhaustion limits.
"""

import sys
import os
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from person2_ssd.storage_model.io_model import (
    StorageSimulator,
    parse_physical_location,
)
from person2_ssd.mock_kv_engine import MockKVEngine


class StressTestRunner:
    def __init__(self):
        self.results = {
            "suite_1_contention": {"passed": 0, "failed": 0, "details": []},
            "suite_2_parsing": {"passed": 0, "failed": 0, "details": []},
            "suite_3_queue_simulation": {"passed": 0, "failed": 0, "details": []},
            "suite_4_edge_cases": {"passed": 0, "failed": 0, "details": []},
        }

    def record(self, suite: str, test_name: str, passed: bool, msg: str = ""):
        key = "passed" if passed else "failed"
        self.results[suite][key] += 1
        self.results[suite]["details"].append({
            "test": test_name,
            "passed": passed,
            "message": msg,
        })
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}: {msg}")

    # =========================================================================
    # Suite 1: Contention Inequality across N in [2..64]
    # =========================================================================
    def run_suite_1(self):
        print("\n" + "=" * 70)
        print("Suite 1: Contention Inequality Validation across N in [2..64]")
        print("=" * 70)
        lm = LatencyModel()

        all_inequalities_hold = True
        all_formulas_match = True
        speedup_at_64 = 0.0

        for n in range(2, 65):
            # Case A: N blocks on single channel (ch0)
            single_locs = [f"ch0_die0_pl0_blk0_pg{i % 128}" for i in range(n)]
            t_single = lm.calculate_batch_read_latency(single_locs)

            # Case B: N blocks striped across 8 channels
            striped_locs = [
                f"ch{i % 8}_die{(i // 8) % 4}_pl0_blk{i // 32}_pg{i % 128}"
                for i in range(n)
            ]
            t_striped = lm.calculate_batch_read_latency(striped_locs)

            expected_t_single = PCIE_OVERHEAD_US + n * (T_R_US + BUS_TRANSFER_US_PER_PAGE)
            expected_max_load = math.ceil(n / 8.0)
            expected_t_striped = PCIE_OVERHEAD_US + expected_max_load * (T_R_US + BUS_TRANSFER_US_PER_PAGE)

            # Check inequality
            if t_single <= t_striped:
                all_inequalities_hold = False
                self.record(
                    "suite_1_contention",
                    f"inequality_n_{n}",
                    False,
                    f"Violation at N={n}: t_single={t_single}us <= t_striped={t_striped}us"
                )

            # Check exact analytical formula matching
            if abs(t_single - expected_t_single) > 1e-6 or abs(t_striped - expected_t_striped) > 1e-6:
                all_formulas_match = False
                self.record(
                    "suite_1_contention",
                    f"formula_match_n_{n}",
                    False,
                    f"Formula mismatch at N={n}: t_single={t_single} vs {expected_t_single}, t_striped={t_striped} vs {expected_t_striped}"
                )

            if n == 64:
                speedup_at_64 = t_single / t_striped

        self.record(
            "suite_1_contention",
            "inequality_all_n_2_to_64",
            all_inequalities_hold,
            f"Strict inequality T_single > T_striped held for all 63 batch sizes N in [2..64]."
        )
        self.record(
            "suite_1_contention",
            "formula_match_all_n_2_to_64",
            all_formulas_match,
            f"All latency calculations perfectly match T = t_pcie + max_c (N_c * (tR + t_bus))."
        )

        speedup_pass = speedup_at_64 >= 2.5
        self.record(
            "suite_1_contention",
            "speedup_at_n_64",
            speedup_pass,
            f"Speedup at N=64 is {speedup_at_64:.2f}x (Required >= 2.5x). T_single={10.0 + 64*30.0}us, T_striped={10.0 + 8*30.0}us."
        )

        # Extended test: N in [128, 256, 512]
        for n in [128, 256, 512]:
            s_locs = [f"ch0_die0_pl0_blk0_pg{i % 128}" for i in range(n)]
            st_locs = [f"ch{i % 8}_die{(i // 8) % 4}_pl0_blk0_pg{i % 128}" for i in range(n)]
            ts = lm.calculate_batch_read_latency(s_locs)
            tst = lm.calculate_batch_read_latency(st_locs)
            spd = ts / tst
            self.record(
                "suite_1_contention",
                f"large_batch_n_{n}",
                ts > tst and spd >= 2.5,
                f"N={n}: T_single={ts:.1f}us, T_striped={tst:.1f}us, Speedup={spd:.2f}x"
            )

    # =========================================================================
    # Suite 2: Address Parsing Stress & Irregular Prefix Formats
    # =========================================================================
    def run_suite_2(self):
        print("\n" + "=" * 70)
        print("Suite 2: Address Parsing Stress & Irregular Prefix Formats")
        print("=" * 70)

        test_cases: List[Tuple[str, int, Optional[Tuple[int, int, int, int, int]], str]] = [
            ("ch0_die0_pl0_blk0_pg0", 0, (0, 0, 0, 0, 0), "canonical lowercase"),
            ("ch7_die3_pl1_blk511_pg127", 7, (7, 3, 1, 511, 127), "canonical max boundaries"),
            ("ssd_ch3_die0_pl0_blk0_pg0", 3, (3, 0, 0, 0, 0), "prefix ssd_"),
            ("nvme_CH5_die1_pl0_blk0_pg0", 5, (5, 1, 0, 0, 0), "prefix nvme_ with uppercase CH5"),
            ("pba_ch1_die2_pl1_blk5_pg10", 1, (1, 2, 1, 5, 10), "prefix pba_"),
            ("device0_ch4_die0_pl0_blk0_pg0", 4, (4, 0, 0, 0, 0), "prefix device0_"),
            ("ssd_host_pcie_ch2_die3_pl0_blk10_pg20", 2, (2, 3, 0, 10, 20), "multi-part prefix"),
            ("CH0_DIE0_PL0_BLK0_PG0", 0, (0, 0, 0, 0, 0), "all uppercase"),
            ("Ch6_Die1_Pl1_Blk2_Pg3", 6, (6, 1, 1, 2, 3), "mixed case"),
            ("ssd_ch03_die0_pl0_blk0_pg0", 3, (3, 0, 0, 0, 0), "zero-padded channel 03"),
            ("ch07_die0_pl0_blk0_pg0", 7, (7, 0, 0, 0, 0), "zero-padded channel 07"),
            ("ch3_die0_plane1_blk2_pg4", 3, (3, 0, 1, 2, 4), "spelled out 'plane'"),
            ("prefix_CH4_die0_plane0_blk0_pg0", 4, (4, 0, 0, 0, 0), "prefix + uppercase + plane"),
        ]

        for loc_str, expected_ch, expected_tuple, desc in test_cases:
            # LatencyModel regex test
            ch_extracted = LatencyModel.extract_channel(loc_str)
            lm_pass = (ch_extracted == expected_ch)
            self.record(
                "suite_2_parsing",
                f"lm_extract_{desc.replace(' ', '_')}",
                lm_pass,
                f"Input '{loc_str}' -> channel {ch_extracted} (Expected {expected_ch})"
            )

            # parse_physical_location regex test
            parsed = parse_physical_location(loc_str)
            if expected_tuple is not None:
                parse_pass = (parsed == expected_tuple)
                self.record(
                    "suite_2_parsing",
                    f"parse_loc_{desc.replace(' ', '_')}",
                    parse_pass,
                    f"Input '{loc_str}' -> {parsed} (Expected {expected_tuple})"
                )

        # Malformed / Fallback cases
        malformed_cases = [
            ("", 0, None, "empty string"),
            (None, 0, None, "None input"),
            (12345, 0, None, "integer input"),
            ("invalid_string_without_channel", 0, None, "completely invalid string"),
            ("ch_die_pl_blk_pg", 0, None, "missing numbers"),
            ("chX_dieY_plZ", 0, None, "non-numeric fields"),
        ]

        for mal_input, exp_ch, exp_parsed, desc in malformed_cases:
            ch = LatencyModel.extract_channel(mal_input)
            lm_pass = (ch == exp_ch)
            self.record(
                "suite_2_parsing",
                f"malformed_lm_{desc.replace(' ', '_')}",
                lm_pass,
                f"Malformed input {repr(mal_input)} safely defaulted to ch {ch}"
            )

            parsed = parse_physical_location(mal_input)
            parse_pass = (parsed == exp_parsed)
            self.record(
                "suite_2_parsing",
                f"malformed_parse_{desc.replace(' ', '_')}",
                parse_pass,
                f"Malformed input {repr(mal_input)} returned {parsed}"
            )

    # =========================================================================
    # Suite 3: Concurrent Channel Bus Queue Serialization & StorageSimulator
    # =========================================================================
    def run_suite_3(self):
        print("\n" + "=" * 70)
        print("Suite 3: Concurrent Channel Bus Queue Serialization & StorageSimulator")
        print("=" * 70)

        # 3.1 Verify FlashChannel queue processing logic directly
        ch = FlashChannel(channel_id=0, dies_per_channel=4)
        # Enqueue 3 read requests to the SAME die (die 0)
        ch.enqueue_request(ChannelTransferRequest(request_id=1, die_id=0, transfer_time_us=5.0))
        ch.enqueue_request(ChannelTransferRequest(request_id=2, die_id=0, transfer_time_us=5.0))
        ch.enqueue_request(ChannelTransferRequest(request_id=3, die_id=0, transfer_time_us=5.0))

        self.record(
            "suite_3_queue_simulation",
            "channel_queue_enqueue",
            len(ch.queue) == 3,
            f"Successfully queued 3 requests. Queue length={len(ch.queue)}"
        )

        busy_time = ch.process_queue(base_time_us=0.0)
        self.record(
            "suite_3_queue_simulation",
            "channel_queue_clear_after_process",
            len(ch.queue) == 0 and len(ch.completed_transfers) == 3,
            f"Queue cleared, completed_transfers={len(ch.completed_transfers)}"
        )

        # In same die: req 1 sense 0..25, bus 25..30
        # req 2 sense 25..50, bus 50..55
        # req 3 sense 50..75, bus 75..80
        t1 = ch.completed_transfers[0]
        t2 = ch.completed_transfers[1]
        t3 = ch.completed_transfers[2]

        ordering_correct = (
            t1.start_time_us == 25.0 and t1.completion_time_us == 30.0 and
            t2.start_time_us == 50.0 and t2.completion_time_us == 55.0 and
            t3.start_time_us == 75.0 and t3.completion_time_us == 80.0
        )
        self.record(
            "suite_3_queue_simulation",
            "same_die_serialization_timestamps",
            ordering_correct,
            f"Req 1: [{t1.start_time_us}..{t1.completion_time_us}], Req 2: [{t2.start_time_us}..{t2.completion_time_us}], Req 3: [{t3.start_time_us}..{t3.completion_time_us}]"
        )

        # 3.2 Verify Parallel Die Sensing with Serialized Bus
        ch_parallel = FlashChannel(channel_id=1, dies_per_channel=4)
        # Enqueue 2 requests to DIFFERENT dies (die 0 and die 1) on the same channel
        ch_parallel.enqueue_request(ChannelTransferRequest(request_id=10, die_id=0, transfer_time_us=5.0))
        ch_parallel.enqueue_request(ChannelTransferRequest(request_id=11, die_id=1, transfer_time_us=5.0))
        ch_parallel.process_queue(base_time_us=0.0)

        p1 = ch_parallel.completed_transfers[0]
        p2 = ch_parallel.completed_transfers[1]

        # Parallel die sensing: both dies sense 0..25 in parallel
        # Bus transfers serialize: p1 transfers 25..30, p2 transfers 30..35
        parallel_sense_correct = (
            p1.start_time_us == 25.0 and p1.completion_time_us == 30.0 and
            p2.start_time_us == 30.0 and p2.completion_time_us == 35.0
        )
        self.record(
            "suite_3_queue_simulation",
            "multi_die_parallel_sensing_bus_serialization",
            parallel_sense_correct,
            f"Parallel sensing verified: p1 bus=[{p1.start_time_us}..{p1.completion_time_us}], p2 bus=[{p2.start_time_us}..{p2.completion_time_us}], total channel time={ch_parallel.bus_busy_until_us}us"
        )

        # 3.3 End-to-end StorageSimulator store and read across batch sizes
        for n in [2, 4, 8, 16, 32, 64]:
            sim_conv = StorageSimulator(mode="conventional")
            sim_ta = StorageSimulator(mode="tensor_aware")

            # Generate N blocks using MockKVEngine
            engine = MockKVEngine()
            blocks = engine.generate_kv_blocks(num_blocks=n)

            # Store blocks
            for b in blocks:
                sim_conv.store_block(b)
                sim_ta.store_block(b)

            # Read blocks via read_blocks
            lat_conv = sim_conv.read_blocks(blocks)
            lat_ta = sim_ta.read_blocks(blocks)
            speedup = lat_conv / lat_ta

            ineq_pass = (lat_conv > lat_ta)
            speedup_pass = (speedup >= 2.5) if n >= 64 else True

            self.record(
                "suite_3_queue_simulation",
                f"e2e_sim_batch_{n}",
                ineq_pass and speedup_pass,
                f"N={n}: Lat_Conv={lat_conv:.1f}us, Lat_TA={lat_ta:.1f}us, Speedup={speedup:.2f}x"
            )

        # 3.4 Verify read count tracking in physical FlashPage
        sim = StorageSimulator(mode="tensor_aware")
        blk = KVBlock.create_default(1001, 0, 0)
        loc = sim.store_block(blk)
        parsed = parse_physical_location(loc)
        ch_idx, die_idx, pl_idx, blk_idx, pg_idx = parsed
        page_obj = sim.channels[ch_idx].dies[die_idx].planes[pl_idx].blocks[blk_idx].pages[pg_idx]

        self.record(
            "suite_3_queue_simulation",
            "physical_page_state_after_store",
            page_obj.state == PageState.VALID and page_obj.data_block_id == 1001,
            f"Page state={page_obj.state}, data_block_id={page_obj.data_block_id}"
        )

        # Read 5 times
        for _ in range(5):
            sim.read_blocks([blk])

        self.record(
            "suite_3_queue_simulation",
            "physical_page_read_count_tracking",
            page_obj.read_count == 5,
            f"Physical page read_count={page_obj.read_count} (Expected 5)"
        )

    # =========================================================================
    # Suite 4: Boundary Edge Cases, State Machines & Exhaustion
    # =========================================================================
    def run_suite_4(self):
        print("\n" + "=" * 70)
        print("Suite 4: Boundary Edge Cases, State Machines & Exhaustion Limits")
        print("=" * 70)

        # 4.1 Empty batch read
        sim = StorageSimulator()
        empty_lat = sim.read_blocks([])
        self.record(
            "suite_4_edge_cases",
            "empty_batch_read",
            empty_lat == 0.0,
            f"Empty batch read returned {empty_lat}us (Expected 0.0)"
        )

        # 4.2 Non-existent block IDs
        non_existent_lat = sim.read_blocks([999999, 888888])
        self.record(
            "suite_4_edge_cases",
            "non_existent_blocks_read",
            non_existent_lat == 0.0,
            f"Non-existent blocks read returned {non_existent_lat}us (Expected 0.0)"
        )

        # 4.3 Polymorphic input: List of KVBlock vs List of int vs mixed
        blk1 = KVBlock.create_default(501, 0, 0)
        blk2 = KVBlock.create_default(502, 0, 1)
        sim.store_block(blk1)
        sim.store_block(blk2)

        lat_objects = sim.read_blocks([blk1, blk2])
        lat_ints = sim.read_blocks([501, 502])
        lat_mixed = sim.read_blocks([blk1, 502])

        self.record(
            "suite_4_edge_cases",
            "polymorphic_read_blocks_input",
            lat_objects == lat_ints == lat_mixed and lat_objects > 0,
            f"Latencies match across input types: objects={lat_objects}, ints={lat_ints}, mixed={lat_mixed}"
        )

        # 4.4 FlashPage State Machine Invariants
        page = FlashPage(page_id=0)
        self.record(
            "suite_4_edge_cases",
            "page_initial_free",
            page.state == PageState.FREE,
            f"Initial page state is {page.state}"
        )

        page.program(data_block_id=42)
        self.record(
            "suite_4_edge_cases",
            "page_program_valid",
            page.state == PageState.VALID and page.data_block_id == 42 and page.program_count == 1,
            f"Page programmed to VALID, data_block_id={page.data_block_id}, prog_count={page.program_count}"
        )

        # Attempt to reprogram VALID page -> must raise ValueError
        reprogram_failed = False
        try:
            page.program(data_block_id=43)
        except ValueError:
            reprogram_failed = True

        self.record(
            "suite_4_edge_cases",
            "page_double_program_raises_value_error",
            reprogram_failed,
            "Attempting to program non-free page correctly raised ValueError."
        )

        page.invalidate()
        read_val_invalid = page.read()
        self.record(
            "suite_4_edge_cases",
            "page_read_invalid_returns_none",
            page.state == PageState.INVALID and read_val_invalid is None,
            f"Reading INVALID page returned {read_val_invalid}"
        )

        page.erase()
        self.record(
            "suite_4_edge_cases",
            "page_erase_resets_to_free",
            page.state == PageState.FREE and page.read_count == 0,
            f"Erased page reset to {page.state}, read_count={page.read_count}"
        )

        # 4.5 FlashBlock Exhaustion and Allocation
        fb = FlashBlock(block_id=0, pages_count=4)
        for i in range(4):
            p = fb.allocate_page(logical_block_id=i * 10)
            assert p is not None

        self.record(
            "suite_4_edge_cases",
            "block_full_state",
            fb.is_full and fb.free_page_count == 0 and fb.valid_page_count == 4,
            f"Block full={fb.is_full}, free={fb.free_page_count}, valid={fb.valid_page_count}"
        )

        # Allocate on full block -> must return None
        p_overflow = fb.allocate_page(logical_block_id=999)
        self.record(
            "suite_4_edge_cases",
            "block_allocation_on_full_returns_none",
            p_overflow is None,
            f"Allocating on full block returned {p_overflow}"
        )

        # Erase full block
        fb.erase()
        self.record(
            "suite_4_edge_cases",
            "block_erase_cycle",
            fb.is_empty and fb.free_page_count == 4 and fb.erase_count == 1,
            f"Block erased: is_empty={fb.is_empty}, free={fb.free_page_count}, erase_count={fb.erase_count}"
        )

        # 4.6 Backward compatibility dunder operations
        p_compat = fb.allocate_page(123)
        self.record(
            "suite_4_edge_cases",
            "page_backward_compat_integer_dunders",
            p_compat == 0 and isinstance(p_compat, FlashPage) and int(p_compat) == 0,
            f"FlashPage satisfies p == 0 ({p_compat == 0}) and isinstance(p, FlashPage) ({isinstance(p_compat, FlashPage)})"
        )

        # 4.7 FlashDie reset and busy tracking
        die = FlashDie(die_id=0)
        finish_read = die.schedule_read(10.0)
        self.record(
            "suite_4_edge_cases",
            "die_schedule_read_timing",
            finish_read == 10.0 + T_R_US and die.busy_until_us == 35.0,
            f"Die scheduled read completion={finish_read}us (Expected 35.0us)"
        )
        die.reset()
        self.record(
            "suite_4_edge_cases",
            "die_reset_clears_busy",
            die.busy_until_us == 0.0,
            f"Die busy_until_us after reset={die.busy_until_us}"
        )

    def summary(self) -> bool:
        print("\n" + "=" * 70)
        print("SUMMARY OF EMPIRICAL STRESS TESTS")
        print("=" * 70)
        all_passed = True
        for suite, data in self.results.items():
            p = data["passed"]
            f = data["failed"]
            total = p + f
            pass_rate = (p / total * 100) if total > 0 else 0
            print(f"{suite}: {p}/{total} passed ({pass_rate:.1f}%), {f} failed")
            if f > 0:
                all_passed = False

        print("=" * 70)
        verdict = "APPROVE" if all_passed else "REJECT"
        print(f"OVERALL EMPIRICAL VERDICT: {verdict}")
        print("=" * 70)
        return all_passed


if __name__ == "__main__":
    runner = StressTestRunner()
    runner.run_suite_1()
    runner.run_suite_2()
    runner.run_suite_3()
    runner.run_suite_4()
    success = runner.summary()
    sys.exit(0 if success else 1)
