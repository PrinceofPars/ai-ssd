#!/usr/bin/env python3
"""
AI-SSD V2 — Final Release Candidate Independent Validation Suite.
Executes:
1. Thread-scaling sanity check (2, 4, 8 threads).
2. Context-scaling validation (4096, 8192, 16384, 32768).
3. Detailed component breakdown & memory audit.
4. Generates rc_validation_complete.json.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "live_inference" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = RESULTS_DIR / "rc_validation_complete.json"

PYTHON_BIN = "/home/ubuntu/ai-ssd-p1/.venv/bin/python3"
WORKER_SCRIPT = str(PROJECT_ROOT / "scripts" / "context_scaling_worker.py")
EXPECTED_TOKENS = [11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13]


def run_worker(context: int, threads: int = 4, rep: int = 0) -> Dict[str, Any]:
    out_file = f"/tmp/rc_val_ctx{context}_th{threads}_rep{rep}.json"
    if os.path.exists(out_file):
        os.remove(out_file)

    cmd = [
        PYTHON_BIN, WORKER_SCRIPT,
        "--mode", "aissd",
        "--storage-mode", "nvme_qemu",
        "--enable-computational-storage",
        "--disable-prefetch",
        "--enable-async-pipeline",
        "--context", str(context),
        "--decode", "16",
        "--seed", "42",
        "--rep", str(rep),
        "--output-json", out_file,
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["TORCH_NUM_THREADS"] = str(threads)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    t0 = time.time()
    print(f"[*] Executing Context={context}, Threads={threads}...")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"[FAIL] Context={context}, Threads={threads} failed after {elapsed:.2f}s")
        print(proc.stderr)
        raise RuntimeError(f"Run failed for ctx={context}, th={threads}")

    with open(out_file, "r") as f:
        data = json.load(f)

    tokens = data.get("token_ids", [])
    matches = (tokens == EXPECTED_TOKENS)
    print(f"[OK] Ctx={context}, Th={threads} completed in {data['wall_time_s']:.2f}s ({data['tokens_per_second']:.3f} tok/s) | Peak RSS: {data['peak_rss_mb']:.1f} MB | Match: {matches}")
    return data


def main():
    print("=" * 80)
    print("AI-SSD V2 — Release Candidate Independent Validation Suite")
    print("=" * 80)

    # 1. Thread Scaling Sanity Check (2, 4, 8 threads)
    print("\n--- Phase 1: Thread-Scaling Sanity Check (2, 4, 8 threads) ---")
    thread_results = {}
    for th in [2, 4, 8]:
        data = run_worker(context=4096, threads=th, rep=0)
        thread_results[th] = {
            "threads": th,
            "wall_time_s": data["wall_time_s"],
            "tokens_per_second": data["tokens_per_second"],
            "peak_rss_mb": data["peak_rss_mb"],
            "timing_breakdown": data.get("timing_breakdown", {}),
            "token_match": data.get("token_ids") == EXPECTED_TOKENS
        }

    # 2. Context Scaling Validation (4096, 8192, 16384, 32768)
    print("\n--- Phase 2: Context-Scaling Validation (4096, 8192, 16384, 32768) ---")
    context_results = {}
    for ctx in [4096, 8192, 16384, 32768]:
        data = run_worker(context=ctx, threads=4, rep=0)
        context_results[ctx] = {
            "context_length": ctx,
            "wall_time_s": data["wall_time_s"],
            "tokens_per_second": data["tokens_per_second"],
            "peak_rss_mb": data["peak_rss_mb"],
            "active_kv_mb": data["active_kv_mb"],
            "candidate_k_bytes_to_host": data.get("candidate_k_bytes_to_host", 0),
            "winning_k_bytes_to_host": data.get("winning_k_bytes_to_host", 0),
            "winning_v_bytes_to_host": data.get("winning_v_bytes_to_host", 0),
            "total_data_movement_bytes": data.get("total_data_movement_bytes", 0),
            "p2_resident_payload_mb": data.get("p2_resident_payload_mb", 0.0),
            "p3_resident_payload_mb": data.get("p3_resident_payload_mb", 0.0),
            "token_match": data.get("token_ids") == EXPECTED_TOKENS,
            "nvme_telemetry": data.get("nvme_telemetry", {}),
            "timing_breakdown": data.get("timing_breakdown", {})
        }

    validation_pkg = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "release_candidate_commit": "cc3972e",
        "thread_scaling_sanity": thread_results,
        "context_scaling_validation": context_results,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(validation_pkg, f, indent=2)

    print("\n" + "=" * 80)
    print("RELEASE CANDIDATE VALIDATION SUITE COMPLETE")
    print(f"Saved results to: {OUT_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
