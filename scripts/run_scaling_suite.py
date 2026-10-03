#!/usr/bin/env python3
"""Scaling Suite Driver for 4096 -> 8192 -> 16384 -> 32768 Contexts.

Orchestrates sequential runs, performs safety checks, enforces exact token match,
collects real OS telemetry from /proc/self/status and psutil, and generates
both structured machine-readable outputs and Markdown reports.
"""

import sys
import os
import json
import csv
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "live_inference" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUTPUT = RESULTS_DIR / "context_scaling_results.json"
CSV_OUTPUT = RESULTS_DIR / "context_scaling_results.csv"

CSV_FIELDS = [
    "context_length",
    "configuration",
    "repetition",
    "wall_time_s",
    "decode_tokens",
    "tokens_per_second",
    "min_rss_mb",
    "avg_rss_mb",
    "peak_rss_mb",
    "vm_peak_mb",
    "rss_anon_mb",
    "active_kv_mb",
    "cold_kv_mb",
    "storage_bytes",
    "storage_read_bytes",
    "storage_write_bytes",
    "storage_requests",
    "p2_resident_payload_mb",
    "p3_resident_payload_mb",
    "staging_mb",
    "p2_metadata_bytes",
    "exact_token_match",
]

# Scaling matrix: context -> (baseline_reps, aissd_reps)
MATRIX = [
    (4096, 3, 3),
    (8192, 3, 3),
    (16384, 2, 2),
    (32768, 1, 1),
]

def check_safety(context: int):
    # Check RAM
    total_ram_gb = 0.0
    free_ram_gb = 0.0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total_ram_gb = float(line.split()[1]) / (1024 * 1024)
            elif line.startswith("MemAvailable:"):
                free_ram_gb = float(line.split()[1]) / (1024 * 1024)

    # Check disk on /tmp
    stat = shutil.disk_usage("/tmp")
    free_disk_gb = stat.free / (1024 * 1024 * 1024)

    # Estimate required storage: 36 layers * 8 kv_heads * 128 dim * 4 bytes * 2 * context
    est_kv_bytes = 36 * 8 * 128 * 4 * 2 * context
    est_kv_gb = est_kv_bytes / (1024 * 1024 * 1024)

    print(f"\n[SAFETY CHECK for Context {context}]")
    print(f"  Total RAM: {total_ram_gb:.1f} GB | Available RAM: {free_ram_gb:.1f} GB")
    print(f"  Available Disk on /tmp: {free_disk_gb:.1f} GB")
    print(f"  Estimated KV storage:   {est_kv_gb:.2f} GB")

    if free_ram_gb < 20.0 and context >= 16384:
        raise RuntimeError(f"Insufficient RAM ({free_ram_gb:.1f} GB) for context {context}")
    if free_disk_gb < est_kv_gb * 2:
        raise RuntimeError(f"Insufficient disk ({free_disk_gb:.1f} GB) on /tmp for context {context}")
    print("  [SAFETY PASS] Sufficient host resources verified.\n")


def append_to_csv(row: Dict[str, Any]):
    file_exists = CSV_OUTPUT.exists() and CSV_OUTPUT.stat().st_size > 0
    with open(CSV_OUTPUT, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_single(mode: str, context: int, rep: int, decode: int = 16, seed: int = 42) -> Dict[str, Any]:
    out_file = f"/tmp/run_{mode}_{context}_rep{rep}.json"
    if os.path.exists(out_file):
        os.remove(out_file)

    cmd = [
        "/home/ubuntu/ai-ssd-p1/.venv/bin/python3",
        str(PROJECT_ROOT / "scripts" / "context_scaling_worker.py"),
        "--mode", mode,
        "--context", str(context),
        "--rep", str(rep),
        "--decode", str(decode),
        "--seed", str(seed),
        "--output-json", out_file,
    ]

    print(f"--> Executing: {' '.join(cmd)}")
    p = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"Worker failed with return code {p.returncode} for mode={mode} context={context} rep={rep}")

    with open(out_file, "r") as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("   AI-SSD V2: LARGE-CONTEXT TRUE OFFLOAD SCALING BENCHMARK")
    print("   Contexts: 4096 -> 8192 -> 16384 -> 32768")
    print("   Model:    Qwen3-4B-Instruct-2507 (CPU, 4 threads, FP32)")
    print("   Decode:   16 tokens per configuration")
    print("=" * 70)

    # Initialize / clean results
    all_results = []
    if JSON_OUTPUT.exists():
        try:
            with open(JSON_OUTPUT, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = []

    for context, b_reps, a_reps in MATRIX:
        print("\n" + "#" * 70)
        print(f"          CONTEXT LENGTH: {context} TOKENS")
        print("#" * 70)

        # Check if already completed in all_results
        existing_b = [x for x in all_results if x.get("context_length") == context and x.get("configuration") == "BASELINE"]
        existing_a = [x for x in all_results if x.get("context_length") == context and x.get("configuration") == "AI-SSD"]
        if len(existing_b) >= b_reps and len(existing_a) >= a_reps:
            print(f"  [ALREADY COMPLETE] Found {len(existing_b)} Baseline and {len(existing_a)} AI-SSD runs. Skipping execution.")
            baseline_runs = existing_b[:b_reps]
            aissd_runs = existing_a[:a_reps]
            # print summary
            b_peaks = [x["peak_rss_mb"] for x in baseline_runs]
            a_peaks = [x["peak_rss_mb"] for x in aissd_runs]
            b_thrs = [x["tokens_per_second"] for x in baseline_runs]
            a_thrs = [x["tokens_per_second"] for x in aissd_runs]
            rss_savings_mb = np.mean(b_peaks) - np.mean(a_peaks)
            rss_savings_pct = (rss_savings_mb / np.mean(b_peaks)) * 100.0
            b_active_kv = baseline_runs[0]["active_kv_mb"]
            a_active_kv = aissd_runs[0]["active_kv_mb"]
            kv_reduct_pct = ((b_active_kv - a_active_kv) / b_active_kv) * 100.0
            print("\n" + "=" * 60)
            print(f"          CONTEXT {context} SUMMARY")
            print("=" * 60)
            print(f"Baseline Peak RSS  : {np.mean(b_peaks):.1f} MB (std: {np.std(b_peaks):.1f})")
            print(f"AI-SSD Peak RSS    : {np.mean(a_peaks):.1f} MB (std: {np.std(a_peaks):.1f})")
            print(f"Physical RSS Savings: {rss_savings_mb:.1f} MB ({rss_savings_pct:.1f}%)")
            print(f"Baseline Active KV : {b_active_kv:.1f} MB")
            print(f"AI-SSD Active KV   : {a_active_kv:.1f} MB (Reduction: {kv_reduct_pct:.1f}%)")
            print(f"Cold KV in Storage : {aissd_runs[0]['cold_kv_mb']:.1f} MB")
            print(f"P2 Resident Payload: {aissd_runs[0]['p2_resident_payload_mb']:.2f} MB")
            print(f"P3 Resident Payload: {aissd_runs[0]['p3_resident_payload_mb']:.2f} MB")
            print(f"Throughput Baseline: {np.mean(b_thrs):.2f} tok/s")
            print(f"Throughput AI-SSD  : {np.mean(a_thrs):.2f} tok/s")
            print(f"Exact Token Match  : 100.0%")
            print("=" * 60)
            continue

        check_safety(context)

        # 1. Run Baseline
        baseline_runs = []
        for r in range(b_reps):
            print(f"\n--- Baseline Rep {r+1}/{b_reps} (Context {context}) ---")
            res = run_single("baseline", context, r, decode=16)
            baseline_runs.append(res)
            res_row = dict(res)
            res_row["exact_token_match"] = True
            res_row["configuration"] = "BASELINE"
            append_to_csv(res_row)
            all_results.append(res_row)
            with open(JSON_OUTPUT, "w") as f:
                json.dump(all_results, f, indent=2)

        # 2. Run AI-SSD
        aissd_runs = []
        for r in range(a_reps):
            print(f"\n--- AI-SSD Rep {r+1}/{a_reps} (Context {context}) ---")
            res = run_single("aissd", context, r, decode=16)
            aissd_runs.append(res)

            # Correctness Check
            b_tokens = baseline_runs[r % len(baseline_runs)]["token_ids"]
            a_tokens = res["token_ids"]
            match = (b_tokens == a_tokens)
            print(f"  -> Correctness Check Rep {r+1}: Exact Token Match = {match}")
            if not match:
                first_mismatch = None
                for idx, (bt, at) in enumerate(zip(b_tokens, a_tokens)):
                    if bt != at:
                        first_mismatch = idx
                        break
                print(f"  [ERROR] Token mismatch at index {first_mismatch}!")
                print(f"  Baseline: {b_tokens}")
                print(f"  AI-SSD:   {a_tokens}")
                res["exact_token_match"] = False
                res_row = dict(res)
                res_row["configuration"] = "AI-SSD"
                append_to_csv(res_row)
                raise RuntimeError(f"Exact token match failed at context {context}, rep {r+1}!")
            
            res["exact_token_match"] = True
            res_row = dict(res)
            res_row["configuration"] = "AI-SSD"
            append_to_csv(res_row)
            all_results.append(res_row)
            with open(JSON_OUTPUT, "w") as f:
                json.dump(all_results, f, indent=2)

        # Summarize this context
        b_peaks = [x["peak_rss_mb"] for x in baseline_runs]
        a_peaks = [x["peak_rss_mb"] for x in aissd_runs]
        b_thrs = [x["tokens_per_second"] for x in baseline_runs]
        a_thrs = [x["tokens_per_second"] for x in aissd_runs]
        
        rss_savings_mb = np.mean(b_peaks) - np.mean(a_peaks)
        rss_savings_pct = (rss_savings_mb / np.mean(b_peaks)) * 100.0

        b_active_kv = baseline_runs[0]["active_kv_mb"]
        a_active_kv = aissd_runs[0]["active_kv_mb"]
        kv_reduct_pct = ((b_active_kv - a_active_kv) / b_active_kv) * 100.0

        print("\n" + "=" * 60)
        print(f"          CONTEXT {context} SUMMARY")
        print("=" * 60)
        print(f"Baseline Peak RSS  : {np.mean(b_peaks):.1f} MB (std: {np.std(b_peaks):.1f})")
        print(f"AI-SSD Peak RSS    : {np.mean(a_peaks):.1f} MB (std: {np.std(a_peaks):.1f})")
        print(f"Physical RSS Savings: {rss_savings_mb:.1f} MB ({rss_savings_pct:.1f}%)")
        print(f"Baseline Active KV : {b_active_kv:.1f} MB")
        print(f"AI-SSD Active KV   : {a_active_kv:.1f} MB (Reduction: {kv_reduct_pct:.1f}%)")
        print(f"Cold KV in Storage : {aissd_runs[0]['cold_kv_mb']:.1f} MB")
        print(f"P2 Resident Payload: {aissd_runs[0]['p2_resident_payload_mb']:.2f} MB")
        print(f"P3 Resident Payload: {aissd_runs[0]['p3_resident_payload_mb']:.2f} MB")
        print(f"Throughput Baseline: {np.mean(b_thrs):.2f} tok/s")
        print(f"Throughput AI-SSD  : {np.mean(a_thrs):.2f} tok/s (Retention: {(np.mean(a_thrs)/np.mean(b_thrs))*100.0:.1f}%)")
        print(f"Exact Token Match  : 100.0% ({b_reps}x Baseline, {a_reps}x AI-SSD verified)")
        print("=" * 60)

    print("\n[ALL CONTEXTS COMPLETE] Results saved to:")
    print(f"  CSV : {CSV_OUTPUT}")
    print(f"  JSON: {JSON_OUTPUT}")

if __name__ == "__main__":
    main()
