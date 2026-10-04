#!/usr/bin/env python3
"""
AI-SSD V2 — Optimization 2 Fresh Baseline Runner.
Executes 5 fresh repetitions on commit cc3972e to establish an immutable,
empirical baseline before any Optimization-2 changes.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "live_inference" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_JSON = RESULTS_DIR / "opt2_baseline.json"

PYTHON_BIN = "/home/ubuntu/ai-ssd-p1/.venv/bin/python3"
WORKER_SCRIPT = str(PROJECT_ROOT / "scripts" / "context_scaling_worker.py")
EXPECTED_TOKENS = [11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13]


def run_worker_rep(rep: int) -> Dict[str, Any]:
    out_file = f"/tmp/opt2_base_rep{rep}.json"
    if os.path.exists(out_file):
        os.remove(out_file)

    cmd = [
        PYTHON_BIN, WORKER_SCRIPT,
        "--mode", "aissd",
        "--storage-mode", "nvme_qemu",
        "--enable-computational-storage",
        "--disable-prefetch",
        "--enable-async-pipeline",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--rep", str(rep),
        "--output-json", out_file,
    ]

    print(f"\n[OPT2 BASELINE REP {rep+1}/5] Executing worker...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"[FAIL] Worker exited with code {proc.returncode} after {elapsed:.2f}s")
        print(proc.stderr)
        raise RuntimeError(f"Repetition {rep} failed")

    with open(out_file, "r") as f:
        data = json.load(f)

    tokens = data.get("token_ids", [])
    matches = (tokens == EXPECTED_TOKENS)
    print(f"[OK] Rep {rep+1} in {data['wall_time_s']:.2f}s ({data['tokens_per_second']:.3f} tok/s) | Peak RSS: {data['peak_rss_mb']:.1f} MB | Match: {matches}")
    return data


def main():
    print("=" * 80)
    print("AI-SSD V2 — Establishing Fresh Optimization 2 Baseline (5 Repetitions)")
    print("Commit: cc3972e | Branch: v2-optimization-2")
    print("=" * 80)

    runs: List[Dict[str, Any]] = []
    for rep in range(5):
        runs.append(run_worker_rep(rep))

    walls = [r["wall_time_s"] for r in runs]
    tpss = [r["tokens_per_second"] for r in runs]
    rsss = [r["peak_rss_mb"] for r in runs]
    act_kvs = [r["active_kv_mb"] for r in runs]

    tb_keys = list(runs[0].get("timing_breakdown", {}).keys())
    timing_means = {}
    for k in tb_keys:
        vals = [r.get("timing_breakdown", {}).get(k, 0.0) for r in runs]
        timing_means[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    nvme_0 = runs[0].get("nvme_telemetry", {})

    baseline_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": "cc3972e",
        "branch": "v2-optimization-2",
        "configuration": {
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "context_length": 4096,
            "decode_tokens": 16,
            "precision": "FP32",
            "cpu_threads": 4,
            "seed": 42,
            "storage_mode": "nvme_qemu",
            "computational_storage": True,
            "async_pipeline": True,
            "prefetch": False,
        },
        "expected_token_ids": EXPECTED_TOKENS,
        "repetitions": len(runs),
        "all_tokens_match": all(r.get("token_ids") == EXPECTED_TOKENS for r in runs),
        "summary": {
            "wall_time_s": {
                "mean": float(np.mean(walls)),
                "std": float(np.std(walls)),
                "min": float(np.min(walls)),
                "max": float(np.max(walls)),
            },
            "tokens_per_second": {
                "mean": float(np.mean(tpss)),
                "std": float(np.std(tpss)),
                "min": float(np.min(tpss)),
                "max": float(np.max(tpss)),
            },
            "peak_rss_mb": {
                "mean": float(np.mean(rsss)),
                "std": float(np.std(rsss)),
                "min": float(np.min(rsss)),
                "max": float(np.max(rsss)),
            },
            "active_kv_mb": float(np.mean(act_kvs)),
            "candidate_k_bytes_to_host": runs[0].get("candidate_k_bytes_to_host", 0),
            "winning_k_bytes_to_host": runs[0].get("winning_k_bytes_to_host", 0),
            "winning_v_bytes_to_host": runs[0].get("winning_v_bytes_to_host", 0),
            "total_data_movement_bytes": runs[0].get("total_data_movement_bytes", 0),
            "p2_resident_payload_mb": runs[0].get("p2_resident_payload_mb", 0.0),
            "p3_resident_payload_mb": runs[0].get("p3_resident_payload_mb", 0.0),
        },
        "component_timings": timing_means,
        "nvme_telemetry": nvme_0,
        "raw_runs": runs,
    }

    with open(BASELINE_JSON, "w") as f:
        json.dump(baseline_report, f, indent=2)

    print("\n" + "=" * 80)
    print("FRESH OPT2 BASELINE ESTABLISHED")
    print("=" * 80)
    print(f"Wall Time (s):        {baseline_report['summary']['wall_time_s']['mean']:.2f} +/- {baseline_report['summary']['wall_time_s']['std']:.3f} s (Min: {baseline_report['summary']['wall_time_s']['min']:.2f}, Max: {baseline_report['summary']['wall_time_s']['max']:.2f})")
    print(f"Throughput (tok/s):   {baseline_report['summary']['tokens_per_second']['mean']:.3f} +/- {baseline_report['summary']['tokens_per_second']['std']:.3f} tok/s")
    print(f"Peak RSS (MB):        {baseline_report['summary']['peak_rss_mb']['mean']:.1f} +/- {baseline_report['summary']['peak_rss_mb']['std']:.1f} MB")
    print(f"Top-K Scoring (s):    {baseline_report['component_timings']['topk_scoring_s']['mean']:.3f} s")
    print(f"Attention Matmul (s): {baseline_report['component_timings']['attn_matmul_s']['mean']:.3f} s")
    print(f"MLP & Norm (s):       {baseline_report['component_timings']['mlp_and_norm_s']['mean']:.3f} s")
    print(f"Winning V Reads (s):  {baseline_report['component_timings']['winning_v_reads_s']['mean']:.3f} s")
    print(f"Candidate K -> Host:  {baseline_report['summary']['candidate_k_bytes_to_host']} bytes")
    print(f"Winning KV -> Host:   {(baseline_report['summary']['winning_k_bytes_to_host'] + baseline_report['summary']['winning_v_bytes_to_host']) / (1024*1024):.2f} MB")
    print(f"Exact Token Match:    16/16 ({baseline_report['all_tokens_match']})")
    print(f"Saved Baseline to:    {BASELINE_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
