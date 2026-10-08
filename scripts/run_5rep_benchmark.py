#!/usr/bin/env python3
"""
AI-SSD V2 — Multi-Repetition Statistical Benchmark Runner
Runs N repetitions of Baseline and AI-SSD modes, collecting comprehensive
phase memory and timing metrics, and reporting mean, std, min, max distributions.
"""

import sys
import os
import time
import json
import numpy as np
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_single_rep(mode: str, model_name: str, context: int, decode: int, threads: int, rep: int, storage_mode: str = "file", dtype: str = "float32"):
    worker_script = PROJECT_ROOT / "scripts" / "context_scaling_worker.py"
    temp_json = f"/tmp/rep_{mode}_{rep}_{int(time.time()*1000)}.json"
    
    cmd = [
        sys.executable,
        str(worker_script),
        "--mode", mode,
        "--model-name", model_name,
        "--dtype", dtype,
        "--context", str(context),
        "--decode", str(decode),
        "--threads", str(threads),
        "--rep", str(rep),
        "--seed", "42",
        "--output-json", temp_json,
        "--disable-progress",
    ]
    if mode == "aissd":
        cmd.extend([
            "--storage-mode", storage_mode,
            "--top-k-pct", "10.0",
            "--enable-computational-storage",
        ])

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"[ERROR] Rep {rep} failed:\n{res.stderr}")
        return None

    if not os.path.exists(temp_json):
        return None

    with open(temp_json, "r") as f:
        data = json.load(f)
    try:
        os.remove(temp_json)
    except Exception:
        pass

    return data


def compute_stats(values):
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--storage-mode", default="file")
    parser.add_argument("--out", default="/tmp/5rep_summary.json")
    args = parser.parse_args()

    print(f"Running {args.reps}-repetition benchmark on {args.model} (ctx={args.context}, decode={args.decode}, threads={args.threads})...")

    results = {"baseline": [], "aissd": []}

    print("\n--- BASELINE REPETITIONS ---")
    for r in range(args.reps):
        print(f"Rep {r+1}/{args.reps} (Baseline)...", flush=True)
        data = run_single_rep("baseline", args.model, args.context, args.decode, args.threads, r, dtype=args.dtype)
        if data:
            results["baseline"].append(data)

    print("\n--- AI-SSD REPETITIONS ---")
    for r in range(args.reps):
        print(f"Rep {r+1}/{args.reps} (AI-SSD)...", flush=True)
        data = run_single_rep("aissd", args.model, args.context, args.decode, args.threads, r, storage_mode=args.storage_mode, dtype=args.dtype)
        if data:
            results["aissd"].append(data)

    summary = {"meta": vars(args), "baseline": {}, "aissd": {}}
    metric_keys = [
        "model_load_peak_rss_mb",
        "prefill_peak_rss_mb",
        "post_prefill_rss_mb",
        "decode_peak_rss_mb",
        "overall_peak_rss_mb",
        "prefill_time_s",
        "decode_time_s",
        "wall_time_s",
        "tokens_per_second",
        "active_kv_mb",
        "candidate_k_bytes_to_host",
        "total_read_mb",
        "total_write_mb",
    ]

    for mode in ["baseline", "aissd"]:
        runs = results[mode]
        summary[mode]["runs_count"] = len(runs)
        for k in metric_keys:
            vals = [r.get(k, 0.0) for r in runs]
            summary[mode][k] = compute_stats(vals)
        if runs:
            summary[mode]["token_ids"] = runs[0].get("token_ids", [])
            summary[mode]["generated_text"] = runs[0].get("generated_text", "")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print(f"                  {args.reps}-REPETITION BENCHMARK STATISTICAL REPORT                  ")
    print("=" * 90)
    print(f"{'Metric':<30} | {'Baseline (Mean ± Std)':<26} | {'AI-SSD (Mean ± Std)':<26}")
    print("-" * 90)
    for k in metric_keys:
        b_st = summary["baseline"][k]
        a_st = summary["aissd"][k]
        b_str = f"{b_st['mean']:.2f} ± {b_st['std']:.2f} [{b_st['min']:.1f}-{b_st['max']:.1f}]"
        a_str = f"{a_st['mean']:.2f} ± {a_st['std']:.2f} [{a_st['min']:.1f}-{a_st['max']:.1f}]"
        print(f"{k:<30} | {b_str:<26} | {a_str:<26}")
    print("=" * 90)
    print(f"Token parity check: {summary['baseline']['token_ids'] == summary['aissd']['token_ids']}")
    print(f"Summary written to: {args.out}")


if __name__ == "__main__":
    main()
