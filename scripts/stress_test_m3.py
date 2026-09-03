"""
Challenger M3.2: Empirical Stress Test Suite
Stress tests:
1. Benchmark repeatability, bitwise determinism across repeated executions, clean file overwrite.
2. CSV output integrity: DictReader parsing, schema conformance, types, numerical precision, analytical equations.
3. Extended batch size scaling (up to 1024 and 2048): memory stability, channel load balance, speedup retention (>= 2.5x).
4. Edge cases & adversarial scaling: odd/prime batch sizes (N=1, 3, 7, 15, 33, 65, 127, 255, 513, 1023), zero batch, capacity limits.
"""

import sys
import os
import csv
import math
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.schemas.kv_block import KVBlock, StorageTier
from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    SSD_PAGES_PER_BLOCK,
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
from person2_ssd.ftl.conventional import ConventionalFTL
from person2_ssd.ftl.tensor_aware import TensorAwareFTL
from person2_ssd.storage_model.latency import LatencyModel
from person2_ssd.storage_model.io_model import StorageSimulator, parse_physical_location
from person2_ssd.mock_kv_engine import MockKVEngine
from benchmarks.run_ftl import run_ftl_benchmark


class M3StressTestRunner:
    def __init__(self):
        self.results = {
            "suite_1_repeatability": {"passed": 0, "failed": 0, "details": []},
            "suite_2_csv_integrity": {"passed": 0, "failed": 0, "details": []},
            "suite_3_extended_scaling": {"passed": 0, "failed": 0, "details": []},
            "suite_4_edge_and_adversarial": {"passed": 0, "failed": 0, "details": []},
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
    # Suite 1: Benchmark Repeatability & Determinism Across Multiple Executions
    # =========================================================================
    def run_suite_1(self):
        print("\n" + "=" * 75)
        print("Suite 1: Benchmark Repeatability & Determinism Across Multiple Executions")
        print("=" * 75)

        csv_path = PROJECT_ROOT / "results" / "raw" / "ftl_results.csv"

        # 1.1 Test truncation & overwrite cleaniness: write garbage trailing data
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("GARBAGE_HEADER_CORRUPTION_TEST" + "X" * 2048 + "\n")

        # Run benchmark
        rows_run1 = run_ftl_benchmark()

        # Check that file exists and doesn't contain garbage
        with open(csv_path, "r", encoding="utf-8") as f:
            raw_content_1 = f.read()

        no_garbage = "GARBAGE_HEADER_CORRUPTION_TEST" not in raw_content_1
        self.record(
            "suite_1_repeatability",
            "file_overwrite_truncation_clean",
            no_garbage,
            "run_ftl_benchmark() cleanly truncates and overwrites pre-existing file without corruption."
        )

        # 1.2 Repeat benchmark 5 times and check bitwise identical output
        runs_identical = True
        raw_runs = [raw_content_1]
        for run_idx in range(2, 6):
            rows_curr = run_ftl_benchmark()
            with open(csv_path, "r", encoding="utf-8") as f:
                raw_curr = f.read()
            raw_runs.append(raw_curr)

            if raw_curr != raw_content_1:
                runs_identical = False
                self.record(
                    "suite_1_repeatability",
                    f"determinism_run_{run_idx}",
                    False,
                    f"Run {run_idx} output differed from Run 1!"
                )
            if rows_curr != rows_run1:
                runs_identical = False
                self.record(
                    "suite_1_repeatability",
                    f"return_rows_determinism_{run_idx}",
                    False,
                    f"Return rows from Run {run_idx} differed from Run 1!"
                )

        self.record(
            "suite_1_repeatability",
            "bitwise_determinism_5_runs",
            runs_identical,
            f"All 5 benchmark runs produced 100% bitwise identical CSV content and return values."
        )

        # 1.3 External Subprocess Invocation Determinism
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "benchmarks" / "run_ftl.py")],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        proc_pass = (proc.returncode == 0) and ("Speedup >= 2.5x achieved" in proc.stdout)
        with open(csv_path, "r", encoding="utf-8") as f:
            raw_proc = f.read()

        self.record(
            "suite_1_repeatability",
            "subprocess_execution_determinism",
            proc_pass and (raw_proc == raw_content_1),
            f"Subprocess exit_code={proc.returncode}, bitwise match with in-process execution."
        )

    # =========================================================================
    # Suite 2: CSV Output Parsing Integrity & Numerical Precision
    # =========================================================================
    def run_suite_2(self):
        print("\n" + "=" * 75)
        print("Suite 2: CSV Output Parsing Integrity & Numerical Precision")
        print("=" * 75)

        csv_path = PROJECT_ROOT / "results" / "raw" / "ftl_results.csv"
        expected_fields = [
            "experiment",
            "batch_size",
            "conventional_latency_us",
            "tensor_aware_latency_us",
            "speedup_x",
        ]
        expected_batches = [16, 32, 64, 128, 256]

        # 2.1 Standard csv.DictReader validation
        rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for r in reader:
                rows.append(r)

        header_match = (fieldnames == expected_fields)
        self.record(
            "suite_2_csv_integrity",
            "csv_dictreader_header_conformance",
            header_match,
            f"Columns: {fieldnames} (Expected: {expected_fields})"
        )

        row_count_match = len(rows) == len(expected_batches)
        self.record(
            "suite_2_csv_integrity",
            "csv_row_count",
            row_count_match,
            f"Found {len(rows)} data rows (Expected {len(expected_batches)})"
        )

        # 2.2 Datatype and precision assertions
        types_valid = True
        precision_valid = True
        speedup_math_valid = True
        analytical_match_valid = True

        for idx, row in enumerate(rows):
            try:
                exp = row["experiment"]
                bs = int(row["batch_size"])
                conv_lat = float(row["conventional_latency_us"])
                ta_lat = float(row["tensor_aware_latency_us"])
                speedup = float(row["speedup_x"])

                if exp != "ftl_comparison" or bs != expected_batches[idx]:
                    types_valid = False

                # Check string formatting precision (should have .X format)
                conv_str = row["conventional_latency_us"]
                ta_str = row["tensor_aware_latency_us"]
                spd_str = row["speedup_x"]

                # Conventional and TA latency: rounded to 1 decimal place
                if "." not in conv_str or len(conv_str.split(".")[1]) > 1:
                    precision_valid = False
                if "." not in ta_str or len(ta_str.split(".")[1]) > 1:
                    precision_valid = False

                # Speedup math check: speedup_x == round(conv / ta, 2)
                expected_speedup = round(conv_lat / ta_lat, 2)
                if abs(speedup - expected_speedup) > 0.01:
                    speedup_math_valid = False

                # Analytical formula exact check
                # Conv: 10 + 30 * bs
                # TA: 10 + 30 * (bs / 8)
                expected_conv_lat = 10.0 + 30.0 * bs
                expected_ta_lat = 10.0 + 30.0 * (bs / 8.0)
                if abs(conv_lat - expected_conv_lat) > 1e-5 or abs(ta_lat - expected_ta_lat) > 1e-5:
                    analytical_match_valid = False

            except Exception as e:
                types_valid = False
                print(f"Error parsing row {idx}: {e}")

        self.record(
            "suite_2_csv_integrity",
            "column_datatypes_and_values",
            types_valid,
            "All fields parsed cleanly to str, int, float matching expected values."
        )
        self.record(
            "suite_2_csv_integrity",
            "numerical_precision_formatting",
            precision_valid,
            "Latencies correctly rounded to 1 decimal place; speedup rounded to 2 decimal places."
        )
        self.record(
            "suite_2_csv_integrity",
            "speedup_ratio_arithmetic_integrity",
            speedup_math_valid,
            "speedup_x exactly equals round(conventional_latency_us / tensor_aware_latency_us, 2)."
        )
        self.record(
            "suite_2_csv_integrity",
            "analytical_formula_identity",
            analytical_match_valid,
            "Latencies exactly match physical contention formula: T = 10.0 + max_load * 30.0 us."
        )

    # =========================================================================
    # Suite 3: Extended Batch Size Scaling (up to 1024 and 2048)
    # =========================================================================
    def run_suite_3(self):
        print("\n" + "=" * 75)
        print("Suite 3: Extended Batch Size Scaling (up to 1024 and 2048)")
        print("=" * 75)

        extended_batches = [16, 32, 64, 128, 256, 512, 1024, 2048]
        mock_kv = MockKVEngine(layers=32, heads=32)

        prev_speedup = 0.0
        all_speedups_pass = True
        load_balance_pass = True
        scale_performance_pass = True

        for bs in extended_batches:
            t0 = time.perf_counter()
            mock_kv.reset()

            conv_ssd = StorageSimulator(mode="conventional", channels=8)
            ta_ssd = StorageSimulator(mode="tensor_aware", channels=8)

            blocks = mock_kv.generate_kv_blocks(num_blocks=bs, layer_id=0, layout="token_major")

            for b in blocks:
                conv_ssd.store_block(b)
                ta_ssd.store_block(b)

            b_ids = [b.block_id for b in blocks]
            conv_lat = conv_ssd.estimate_read_latency(b_ids)
            ta_lat = ta_ssd.estimate_read_latency(b_ids)
            speedup = conv_lat / ta_lat if ta_lat > 0 else 1.0
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Verify speedup >= 2.5x for all bs >= 64
            if bs >= 64 and speedup < 2.5:
                all_speedups_pass = False

            # Verify monotonic increase of speedup (approaching 8.0x)
            if speedup < prev_speedup:
                all_speedups_pass = False
            prev_speedup = speedup

            # Verify perfect channel load balance for TA
            # bs / 8 blocks per channel
            expected_load = bs // 8
            model = LatencyModel(channels=8)
            ta_locs = [ta_ssd.get_location(bid) for bid in b_ids]
            loads = model._get_channel_loads(ta_locs)
            for c in range(8):
                if loads.get(c, 0) != expected_load:
                    load_balance_pass = False

            # Check that simulation finishes quickly (no quadratic blowup)
            if elapsed_ms > 5000.0:  # > 5 seconds
                scale_performance_pass = False

            self.record(
                "suite_3_extended_scaling",
                f"scaling_batch_{bs}",
                speedup >= (2.5 if bs >= 64 else 1.0) and loads.get(0, 0) == expected_load,
                f"Batch {bs:4d}: Conv={conv_lat:8.1f}us, TA={ta_lat:7.1f}us, Speedup={speedup:5.2f}x, Time={elapsed_ms:.1f}ms"
            )

        self.record(
            "suite_3_extended_scaling",
            "speedup_monotonicity_and_asymptotic_limit",
            all_speedups_pass,
            f"Speedup increases monotonically towards theoretical 8.0x limit ({prev_speedup:.2f}x at 2048 blocks)."
        )
        self.record(
            "suite_3_extended_scaling",
            "perfect_load_balance_all_extended_batches",
            load_balance_pass,
            "Every channel receives exactly B/8 blocks across all extended batches [16..2048]."
        )
        self.record(
            "suite_3_extended_scaling",
            "runtime_efficiency_no_quadratic_blowup",
            scale_performance_pass,
            "All batch simulations completed within normal linear operational envelopes."
        )

    # =========================================================================
    # Suite 4: Edge Cases, Odd Batch Sizes, Zero Batch & Capacity Limits
    # =========================================================================
    def run_suite_4(self):
        print("\n" + "=" * 75)
        print("Suite 4: Edge Cases, Odd Batch Sizes, Zero Batch & Capacity Limits")
        print("=" * 75)

        mock_kv = MockKVEngine(layers=32, heads=32)

        # 4.1 Zero Batch Size (Empty batch)
        conv_ssd = StorageSimulator(mode="conventional", channels=8)
        ta_ssd = StorageSimulator(mode="tensor_aware", channels=8)
        lat_zero_conv = conv_ssd.estimate_read_latency([])
        lat_zero_ta = ta_ssd.estimate_read_latency([])
        zero_pass = (lat_zero_conv == 0.0) and (lat_zero_ta == 0.0)
        self.record(
            "suite_4_edge_and_adversarial",
            "zero_batch_size_handling",
            zero_pass,
            f"Zero blocks read returns exactly 0.0 us: Conv={lat_zero_conv}, TA={lat_zero_ta}"
        )

        # 4.2 Single Block (Batch Size 1)
        mock_kv.reset()
        blocks_1 = mock_kv.generate_kv_blocks(num_blocks=1)
        conv_ssd.store_block(blocks_1[0])
        ta_ssd.store_block(blocks_1[0])
        lat_1_conv = conv_ssd.estimate_read_latency([blocks_1[0].block_id])
        lat_1_ta = ta_ssd.estimate_read_latency([blocks_1[0].block_id])
        # For 1 block: T = 10.0 + 1 * 30.0 = 40.0 us on both
        single_pass = (lat_1_conv == 40.0) and (lat_1_ta == 40.0)
        self.record(
            "suite_4_edge_and_adversarial",
            "single_block_batch_size_1",
            single_pass,
            f"Batch 1 returns 40.0 us for both (Conv={lat_1_conv}, TA={lat_1_ta}, Speedup={lat_1_conv/lat_1_ta:.2f}x)"
        )

        # 4.3 Odd / Prime Batch Sizes: [3, 7, 15, 33, 65, 127, 255, 513, 1023]
        odd_batches = [3, 7, 15, 33, 65, 127, 255, 513, 1023]
        all_odd_pass = True
        odd_speedups_64_plus = True

        for n in odd_batches:
            mock_kv.reset()
            c_sim = StorageSimulator(mode="conventional", channels=8)
            t_sim = StorageSimulator(mode="tensor_aware", channels=8)

            blks = mock_kv.generate_kv_blocks(num_blocks=n, layer_id=0, layout="token_major")
            for b in blks:
                c_sim.store_block(b)
                t_sim.store_block(b)

            b_ids = [b.block_id for b in blks]
            c_lat = c_sim.estimate_read_latency(b_ids)
            t_lat = t_sim.estimate_read_latency(b_ids)
            spd = c_lat / t_lat

            # Expected for Conv: 10 + n * 30
            exp_c_lat = 10.0 + n * 30.0
            # Expected for TA: max load is ceil(n / 8)
            exp_max_load = math.ceil(n / 8.0)
            exp_t_lat = 10.0 + exp_max_load * 30.0

            if abs(c_lat - exp_c_lat) > 1e-5 or abs(t_lat - exp_t_lat) > 1e-5:
                all_odd_pass = False

            if n >= 65 and spd < 2.5:
                odd_speedups_64_plus = False

            self.record(
                "suite_4_edge_and_adversarial",
                f"odd_batch_{n}",
                c_lat > t_lat and (spd >= 2.5 if n >= 65 else True),
                f"N={n:4d}: Conv={c_lat:8.1f}us, TA={t_lat:7.1f}us, Speedup={spd:5.2f}x, MaxLoad={exp_max_load}"
            )

        self.record(
            "suite_4_edge_and_adversarial",
            "odd_batch_analytical_accuracy",
            all_odd_pass,
            "Analytical formulas strictly match for all non-power-of-2 and prime batch sizes."
        )
        self.record(
            "suite_4_edge_and_adversarial",
            "odd_batch_speedup_retention",
            odd_speedups_64_plus,
            "Speedup >= 2.5x strictly retained for all odd batch sizes >= 65."
        )

        # 4.4 Polymorphic read_blocks vs estimate_read_latency on large batches (1024)
        mock_kv.reset()
        test_blocks = mock_kv.generate_kv_blocks(num_blocks=1024, layer_id=0)
        sim_poly = StorageSimulator(mode="tensor_aware", channels=8)
        for b in test_blocks:
            sim_poly.store_block(b)

        lat_objects = sim_poly.read_blocks(test_blocks)
        lat_integers = sim_poly.read_blocks([b.block_id for b in test_blocks])
        lat_direct = sim_poly.estimate_read_latency([b.block_id for b in test_blocks])

        poly_pass = (lat_objects == lat_integers == lat_direct == 3850.0)
        self.record(
            "suite_4_edge_and_adversarial",
            "polymorphic_read_blocks_at_1024",
            poly_pass,
            f"lat_objects={lat_objects}, lat_integers={lat_integers}, lat_direct={lat_direct} (All exactly 3850.0 us)"
        )

        # 4.5 Physical Page Validation at Scale (N=1024)
        # Check that physical pages were properly programmed to VALID
        pages_valid = True
        for b in test_blocks[:32]:  # Sample first 32 blocks
            loc = sim_poly.get_location(b.block_id)
            ch, die, pl, blk, pg = parse_physical_location(loc)
            page_obj = sim_poly.channels[ch].dies[die].planes[pl].blocks[blk].pages[pg]
            if page_obj.state != PageState.VALID or page_obj.data_block_id != b.block_id:
                pages_valid = False
                break

        self.record(
            "suite_4_edge_and_adversarial",
            "physical_page_programming_integrity_at_1024",
            pages_valid,
            "Physical pages correctly marked VALID with matching data_block_id for N=1024 batch."
        )

    def summary(self) -> bool:
        print("\n" + "=" * 75)
        print("SUMMARY OF CHALLENGER M3.2 EMPIRICAL STRESS TESTS")
        print("=" * 75)
        all_passed = True
        total_tests = 0
        total_passed = 0
        total_failed = 0

        for suite, data in self.results.items():
            p = data["passed"]
            f = data["failed"]
            total = p + f
            total_tests += total
            total_passed += p
            total_failed += f
            pass_rate = (p / total * 100) if total > 0 else 0
            print(f"{suite}: {p}/{total} passed ({pass_rate:.1f}%), {f} failed")
            if f > 0:
                all_passed = False

        print("=" * 75)
        print(f"TOTAL: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%), {total_failed} failed")
        verdict = "APPROVE" if all_passed else "REJECT"
        print(f"CHALLENGER M3.2 FINAL VERDICT: {verdict}")
        print("=" * 75)
        return all_passed


if __name__ == "__main__":
    runner = M3StressTestRunner()
    runner.run_suite_1()
    runner.run_suite_2()
    runner.run_suite_3()
    runner.run_suite_4()
    success = runner.summary()
    sys.exit(0 if success else 1)
