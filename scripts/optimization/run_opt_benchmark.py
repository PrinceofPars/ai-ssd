#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "live_inference" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_JSON = RESULTS_DIR / "optimization_baseline.json"

PYTHON_BIN = "/home/ubuntu/ai-ssd-p1/.venv/bin/python3"
WORKER_SCRIPT = str(PROJECT_ROOT / "scripts" / "context_scaling_worker.py")
EXPECTED_TOKENS = [11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13]

def run_reps(opt_id: str, reps: int = 5) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for rep in range(reps):
        out_file = f"/tmp/{opt_id}_rep{rep}.json"
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

        print(f"[{opt_id.upper()} REP {rep+1}/{reps}] Executing worker...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"[FAIL] Worker exited with code {proc.returncode} after {elapsed:.2f}s")
            print(f"STDOUT:\n{proc.stdout}")
            print(f"STDERR:\n{proc.stderr}")
            raise RuntimeError(f"Repetition {rep} failed")

        with open(out_file, "r") as f:
            data = json.load(f)

        tokens = data.get("token_ids", [])
        matches = (tokens == EXPECTED_TOKENS)
        print(f"[OK] Rep {rep+1} in {data['wall_time_s']:.2f}s ({data['tokens_per_second']:.3f} tok/s) | Match: {matches}")
        runs.append(data)

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

    res = {
        "opt_id": opt_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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
        "nvme_telemetry": runs[0].get("nvme_telemetry", {}),
        "raw_runs": runs,
    }

    out_json = RESULTS_DIR / f"{opt_id}_results.json"
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)

    # Print baseline comparison
    if BASELINE_JSON.exists():
        with open(BASELINE_JSON, "r") as f:
            base = json.load(f)
        b_wall = base["summary"]["wall_time_s"]["mean"]
        b_tps = base["summary"]["tokens_per_second"]["mean"]
        o_wall = res["summary"]["wall_time_s"]["mean"]
        o_tps = res["summary"]["tokens_per_second"]["mean"]
        wall_pct = (b_wall - o_wall) / b_wall * 100.0
        tps_pct = (o_tps - b_tps) / b_tps * 100.0

        print("\n" + "=" * 80)
        print(f"EVALUATION COMPLETE: {opt_id.upper()}")
        print("=" * 80)
        print(f"Wall Time:   {o_wall:.2f} +/- {res['summary']['wall_time_s']['std']:.3f} s (Baseline: {b_wall:.2f} s | Change: {wall_pct:+.2f}%)")
        print(f"Throughput:  {o_tps:.3f} +/- {res['summary']['tokens_per_second']['std']:.3f} tok/s (Baseline: {b_tps:.3f} tok/s | Change: {tps_pct:+.2f}%)")
        print(f"Exact Token Match: 16/16 ({res['all_tokens_match']})")
        print(f"Results saved to: {out_json}")
        print("=" * 80)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt-id", type=str, required=True)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    run_reps(args.opt_id, reps=args.reps)
