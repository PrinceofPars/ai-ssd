#!/usr/bin/env python3
"""
AI-SSD V2 — Thread-Scaling & Hardware Metric Profiler.

Evaluates inference performance across:
1, 2, 4, 6, 8 CPU threads.

Measures for each thread count:
- Wall time & Throughput (tok/s)
- Peak RSS & Active KV
- Hardware counters via perf stat:
  - CPU utilization
  - CPU cycles
  - Instructions retired
  - Instructions per cycle (IPC)
  - Branch misses
  - Context switches
"""

import os
import sys
import json
import time
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "live_inference" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = RESULTS_DIR / "thread_scaling_results.json"

PYTHON_BIN = "/home/ubuntu/ai-ssd-p1/.venv/bin/python3"
WORKER_SCRIPT = str(PROJECT_ROOT / "scripts" / "context_scaling_worker.py")
THREAD_COUNTS = [1, 2, 4, 6, 8]


def run_threaded_test(threads: int) -> Dict[str, Any]:
    out_json = f"/tmp/opt_thread_{threads}.json"
    perf_log = f"/tmp/perf_thread_{threads}.log"
    for p in [out_json, perf_log]:
        if os.path.exists(p):
            os.remove(p)

    worker_cmd = [
        PYTHON_BIN, WORKER_SCRIPT,
        "--mode", "aissd",
        "--storage-mode", "nvme_qemu",
        "--enable-computational-storage",
        "--disable-prefetch",
        "--enable-async-pipeline",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", out_json,
    ]

    perf_cmd = [
        "sudo", "perf", "stat",
        "-e", "cycles,instructions,branches,branch-misses,task-clock,context-switches,page-faults",
        "-o", perf_log,
        "--",
    ] + worker_cmd

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["TORCH_NUM_THREADS"] = str(threads)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    print(f"\n[THREADS = {threads}] Running worker under perf stat...")
    t0 = time.time()
    proc = subprocess.run(perf_cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"[FAIL] Run with {threads} threads failed:")
        print(proc.stderr)
        raise RuntimeError(f"Thread test {threads} failed")

    with open(out_json, "r") as f:
        data = json.load(f)

    # Parse perf output
    perf_metrics = {
        "cycles": 0,
        "instructions": 0,
        "ipc": 0.0,
        "branch_misses_pct": 0.0,
        "cpu_utilization": 0.0,
        "context_switches": 0,
    }
    if os.path.exists(perf_log):
        with open(perf_log, "r") as f:
            perf_text = f.read()
            for line in perf_text.splitlines():
                if "cycles" in line and "GHz" in line:
                    m = re.search(r"([\d,]+)\s+cycles", line)
                    if m:
                        perf_metrics["cycles"] = int(m.group(1).replace(",", ""))
                elif "instructions" in line and "insn per cycle" in line:
                    m = re.search(r"([\d,]+)\s+instructions\s+#\s+([\d.]+)", line)
                    if m:
                        perf_metrics["instructions"] = int(m.group(1).replace(",", ""))
                        perf_metrics["ipc"] = float(m.group(2))
                elif "branch-misses" in line:
                    m = re.search(r"#\s+([\d.]+)%", line)
                    if m:
                        perf_metrics["branch_misses_pct"] = float(m.group(1))
                elif "task-clock" in line:
                    m = re.search(r"#\s+([\d.]+)\s+CPUs", line)
                    if m:
                        perf_metrics["cpu_utilization"] = float(m.group(1))
                elif "context-switches" in line:
                    m = re.search(r"([\d,]+)\s+context-switches", line)
                    if m:
                        perf_metrics["context_switches"] = int(m.group(1).replace(",", ""))

    print(f"[OK] {threads} threads -> {data['wall_time_s']:.2f}s ({data['tokens_per_second']:.3f} tok/s) | IPC: {perf_metrics['ipc']:.2f} | CPU Util: {perf_metrics['cpu_utilization']:.2f} CPUs")
    return {
        "threads": threads,
        "wall_time_s": data["wall_time_s"],
        "tokens_per_second": data["tokens_per_second"],
        "peak_rss_mb": data["peak_rss_mb"],
        "timing_breakdown": data.get("timing_breakdown", {}),
        "perf_metrics": perf_metrics,
        "token_match": data.get("token_ids") == [11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13],
    }


def main():
    print("=" * 80)
    print("AI-SSD V2 — Thread Scaling & Hardware Bottleneck Investigation")
    print("=" * 80)

    results = {}
    for th in THREAD_COUNTS:
        results[th] = run_threaded_test(th)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thread_counts": THREAD_COUNTS,
        "results": results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print("THREAD SCALING SUMMARY")
    print("=" * 90)
    print(f"{'Threads':<8} | {'tok/s':<7} | {'Wall (s)':<8} | {'Visible Storage':<16} | {'Visible Compute':<16} | {'IPC':<6} | {'CPU Util':<9}")
    print("-" * 90)
    for th in THREAD_COUNTS:
        r = results[th]
        t = r["timing_breakdown"]
        p = r["perf_metrics"]
        print(f"{th:<8} | {r['tokens_per_second']:<7.3f} | {r['wall_time_s']:<8.2f} | {t.get('visible_storage_s', 0.0):<16.2f} | {t.get('visible_compute_s', 0.0):<16.2f} | {p['ipc']:<6.2f} | {p['cpu_utilization']:<9.2f}")
    print("=" * 90)
    print(f"Results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
