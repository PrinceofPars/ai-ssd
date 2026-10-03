#!/usr/bin/env python3
"""
Phase 7 Benchmark Suite: In-Storage Computational Top-K Candidate Filtering.

Executes and measures:
1. Baseline (Dense PyTorch in Host DRAM)
2. File-Backed AI-SSD (Host-Side Filtering baseline)
3. File-Backed AI-SSD (In-Storage Computational Filtering)
4. QEMU/NVMe AI-SSD (Host-Side Filtering baseline from Phase 6)
5. QEMU/NVMe AI-SSD (In-Storage Computational Filtering, Prefetch OFF)
6. QEMU/NVMe AI-SSD (In-Storage Computational Filtering, Prefetch ON)

Strict Constraints:
- Qwen3-4B-Instruct-2507
- Context: 4,096 tokens
- Decode: 16 tokens
- FP32 precision
- 4 CPU threads
- Seed: 42
- Zero sleep, zero hardcoded numbers, 100% measured real execution
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
OUTPUT_JSON = RESULTS_DIR / "phase7_computational_storage_results.json"

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
    print("=" * 70)
    print("AI-SSD V2 Phase 7: Computational Storage In-Storage Top-K Benchmarks")
    print("=" * 70)

    worker_script = str(PROJECT_ROOT / "scripts" / "context_scaling_worker.py")
    runs = {}

    tmp_files = {
        "baseline": "/tmp/p7_baseline.json",
        "file_host_side": "/tmp/p7_file_host_side.json",
        "file_computational": "/tmp/p7_file_comp.json",
        "nvme_host_side": "/tmp/p7_nvme_host_side.json",
        "nvme_comp_noprefetch": "/tmp/p7_nvme_comp_noprefetch.json",
        "nvme_comp_prefetch": "/tmp/p7_nvme_comp_prefetch.json",
    }

    # 1. Baseline
    run_worker_subproc([
        PYTHON_BIN, worker_script,
        "--mode", "baseline",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", tmp_files["baseline"],
    ])

    # 2. File-backed Host-side Filtering (Phase 6 baseline)
    run_worker_subproc([
        PYTHON_BIN, worker_script,
        "--mode", "aissd",
        "--storage-mode", "file",
        "--disable-computational-storage",
        "--enable-prefetch",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", tmp_files["file_host_side"],
    ])

    # 3. File-backed Computational Storage
    run_worker_subproc([
        PYTHON_BIN, worker_script,
        "--mode", "aissd",
        "--storage-mode", "file",
        "--enable-computational-storage",
        "--enable-prefetch",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", tmp_files["file_computational"],
    ])

    # 4. QEMU/NVMe Host-side Filtering (Phase 6 baseline)
    run_worker_subproc([
        PYTHON_BIN, worker_script,
        "--mode", "aissd",
        "--storage-mode", "nvme_qemu",
        "--disable-computational-storage",
        "--enable-prefetch",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", tmp_files["nvme_host_side"],
    ])

    # 5. QEMU/NVMe Computational Storage (Prefetch OFF)
    run_worker_subproc([
        PYTHON_BIN, worker_script,
        "--mode", "aissd",
        "--storage-mode", "nvme_qemu",
        "--enable-computational-storage",
        "--disable-prefetch",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", tmp_files["nvme_comp_noprefetch"],
    ])

    # 6. QEMU/NVMe Computational Storage (Prefetch ON)
    run_worker_subproc([
        PYTHON_BIN, worker_script,
        "--mode", "aissd",
        "--storage-mode", "nvme_qemu",
        "--enable-computational-storage",
        "--enable-prefetch",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", tmp_files["nvme_comp_prefetch"],
    ])

    # Collect data
    collected = {}
    for key, path in tmp_files.items():
        with open(path, "r") as f:
            collected[key] = json.load(f)

    # Verification of Exact Token Matches
    print("\n" + "=" * 70)
    print("VERIFICATION: Exact Token ID Matching")
    print("=" * 70)
    for key, data in collected.items():
        tokens = data.get("token_ids", [])
        matches = (tokens == EXPECTED_TOKENS)
        print(f"  [{key:<22}] Match: {matches} | Tokens: {tokens[:8]}... (len={len(tokens)})")
        if not matches:
            print(f"  WARNING: Token mismatch for {key}!")
            print(f"    Expected: {EXPECTED_TOKENS}")
            print(f"    Got:      {tokens}")

    # Summary table
    print("\n" + "=" * 85)
    print(f"{'Mode':<30} | {'tok/s':<8} | {'Wall (s)':<9} | {'Peak RSS':<10} | {'Cand K (B)':<10} | {'Total MV (B)':<12}")
    print("-" * 85)
    for key, data in collected.items():
        tps = data.get("tokens_per_second", 0.0)
        wall = data.get("wall_time_s", 0.0)
        rss = data.get("peak_rss_mb", 0.0)
        cand_k = data.get("candidate_k_bytes_to_host", 0)
        total_mv = data.get("total_data_movement_bytes", 0)
        print(f"{key:<30} | {tps:<8.3f} | {wall:<9.2f} | {rss:<10.1f} | {cand_k:<10} | {total_mv:<12}")
    print("=" * 85)

    final_report = {
        "benchmark_name": "Phase 7 - Computational Storage: In-Storage Top-K Candidate Filtering",
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
        "results": collected,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n[REPORT] Saved full Phase 7 results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
