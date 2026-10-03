#!/usr/bin/env python3
"""
Phase 8 Benchmark Suite: Asynchronous Storage, DMA/Pipelining & Prefetch Optimization.

Executes and measures:
1. Run A: Baseline (Dense PyTorch in Host DRAM)
2. Run B: File-Backed Phase 7 (In-Storage Computational Filtering, No Async)
3. Run C: QEMU/NVMe Phase 7 Baseline (In-Storage Computational Filtering, No Prefetch, No Async)
4. Run D: QEMU/NVMe Phase 7 with Prefetch (In-Storage Computational Filtering, Synchronous Prefetch)
5. Run E: QEMU/NVMe Phase 8 Async Pipeline (Prefetch OFF, Contiguous 8 KiB block retrieval)
6. Run F: QEMU/NVMe Phase 8 Async Pipeline (Prefetch ON, Inter-layer pipelined async prefetch)

Strict Constraints:
- Qwen3-4B-Instruct-2507
- Context: 4,096 tokens
- Decode: 16 tokens
- FP32 precision
- 4 CPU threads
- Seed: 42
- Zero sleep, zero hardcoded numbers, 100% measured real execution
- Critical-path reconciliation error < 2%
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "live_inference" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = RESULTS_DIR / "phase8_async_storage_results.json"

EXPECTED_TOKENS = [11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13]

PYTHON_BIN = sys.executable


def run_worker_subproc(cmd: list) -> Dict[str, Any]:
    print(f"\n[RUN] Executing: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL] Return code: {proc.returncode}")
        print(f"STDOUT:\n{proc.stdout}")
        print(f"STDERR:\n{proc.stderr}")
        raise RuntimeError(f"Worker failed: {cmd}")
    print(f"[OK] Completed in {elapsed:.2f}s")
    print(proc.stdout.strip().split("\n")[-1])
    return proc


def main():
    print("=" * 80)
    print("AI-SSD V2 Phase 8: Asynchronous Storage, DMA/Pipelining & Prefetch Benchmarks")
    print("=" * 80)

    worker_script = str(PROJECT_ROOT / "scripts" / "context_scaling_worker.py")

    tmp_files = {
        "run_a_baseline": "/tmp/p8_run_a_baseline.json",
        "run_b_file_phase7": "/tmp/p8_run_b_file_phase7.json",
        "run_c_nvme_phase7_noprefetch": "/tmp/p8_run_c_nvme_phase7_noprefetch.json",
        "run_d_nvme_phase7_prefetch": "/tmp/p8_run_d_nvme_phase7_prefetch.json",
        "run_e_nvme_async_noprefetch": "/tmp/p8_run_e_nvme_async_noprefetch.json",
        "run_f_nvme_async_prefetch": "/tmp/p8_run_f_nvme_async_prefetch.json",
    }

    # Run A: Baseline (Dense PyTorch in Host DRAM)
    run_worker_subproc([
        PYTHON_BIN, worker_script,
        "--mode", "baseline",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", tmp_files["run_a_baseline"],
    ])

    # Run B: File-backed Phase 7 (In-Storage Computational Filtering, No Async)
    run_worker_subproc([
        PYTHON_BIN, worker_script,
        "--mode", "aissd",
        "--storage-mode", "file",
        "--enable-computational-storage",
        "--disable-prefetch",
        "--disable-async-pipeline",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", tmp_files["run_b_file_phase7"],
    ])

    # Run C: QEMU/NVMe Phase 7 Baseline (In-Storage Computational Filtering, Prefetch OFF, No Async)
    run_worker_subproc([
        PYTHON_BIN, worker_script,
        "--mode", "aissd",
        "--storage-mode", "nvme_qemu",
        "--enable-computational-storage",
        "--disable-prefetch",
        "--disable-async-pipeline",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", tmp_files["run_c_nvme_phase7_noprefetch"],
    ])

    # Run D: QEMU/NVMe Phase 7 with Prefetch (In-Storage Computational Filtering, Prefetch ON, No Async)
    run_worker_subproc([
        PYTHON_BIN, worker_script,
        "--mode", "aissd",
        "--storage-mode", "nvme_qemu",
        "--enable-computational-storage",
        "--enable-prefetch",
        "--disable-async-pipeline",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", tmp_files["run_d_nvme_phase7_prefetch"],
    ])

    # Run E: QEMU/NVMe Phase 8 Async Pipeline (Prefetch OFF, Contiguous 8 KiB block retrieval)
    run_worker_subproc([
        PYTHON_BIN, worker_script,
        "--mode", "aissd",
        "--storage-mode", "nvme_qemu",
        "--enable-computational-storage",
        "--disable-prefetch",
        "--enable-async-pipeline",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", tmp_files["run_e_nvme_async_noprefetch"],
    ])

    # Run F: QEMU/NVMe Phase 8 Async Pipeline (Prefetch ON, Inter-layer pipelined async prefetch)
    run_worker_subproc([
        PYTHON_BIN, worker_script,
        "--mode", "aissd",
        "--storage-mode", "nvme_qemu",
        "--enable-computational-storage",
        "--enable-prefetch",
        "--enable-async-pipeline",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", tmp_files["run_f_nvme_async_prefetch"],
    ])

    # Collect data
    collected = {}
    for key, path in tmp_files.items():
        with open(path, "r") as f:
            collected[key] = json.load(f)

    # Verification of Exact Token Matches
    print("\n" + "=" * 80)
    print("VERIFICATION: Exact Token ID Matching")
    print("=" * 80)
    all_tokens_match = True
    for key, data in collected.items():
        tokens = data.get("token_ids", [])
        matches = (tokens == EXPECTED_TOKENS)
        all_tokens_match = all_tokens_match and matches
        print(f"  [{key:<30}] Match: {matches} | Tokens: {tokens[:8]}... (len={len(tokens)})")
        if not matches:
            print(f"  WARNING: Token mismatch for {key}!")
            print(f"    Expected: {EXPECTED_TOKENS}")
            print(f"    Got:      {tokens}")

    # Summary table
    print("\n" + "=" * 105)
    print(f"{'Run / Mode':<32} | {'tok/s':<7} | {'Wall(s)':<8} | {'Storage(s)':<10} | {'Compute(s)':<10} | {'Error%':<7} | {'Peak RSS':<9}")
    print("-" * 105)
    for key, data in collected.items():
        tps = data.get("tokens_per_second", 0.0)
        wall = data.get("wall_time_s", 0.0)
        vis_stor = data.get("visible_storage_s", 0.0)
        vis_comp = data.get("visible_compute_s", 0.0)
        recon_err = data.get("reconciliation_error_pct", 0.0)
        rss = data.get("peak_rss_mb", 0.0)
        print(f"{key:<32} | {tps:<7.3f} | {wall:<8.2f} | {vis_stor:<10.3f} | {vis_comp:<10.3f} | {recon_err:<7.2f}% | {rss:<9.1f}")
    print("=" * 105)

    final_report = {
        "benchmark_name": "Phase 8 - Asynchronous Storage, DMA/Pipelining & Prefetch Optimization",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workload": {
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "context_length": 4096,
            "decode_tokens": 16,
            "precision": "FP32",
            "cpu_threads": 4,
            "seed": 42,
        },
        "expected_token_ids": EXPECTED_TOKENS,
        "token_match_status": {
            k: (v.get("token_ids") == EXPECTED_TOKENS) for k, v in collected.items()
        },
        "all_tokens_match": all_tokens_match,
        "results": collected,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n[REPORT] Saved full Phase 8 results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
