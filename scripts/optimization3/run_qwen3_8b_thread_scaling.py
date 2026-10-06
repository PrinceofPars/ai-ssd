#!/usr/bin/env python3
"""AI-SSD V2 — Optimization 3: Qwen3-8B FP16 Thread Scaling Benchmark (2, 4, 8 Threads).

Evaluates CPU scaling behavior and memory-bandwidth saturation for Qwen3-8B FP16:
  - Threads: 2, 4, 8
  - Context: 4096 tokens
  - Decode: 16 tokens
  - Seed: 42
  - Mode: AI-SSD on QEMU/NVMe (In-storage Top-K, Async ON)

Saves results to: benchmarks/live_inference/results/optimization3/qwen3_8b_fp16_thread_scaling.json
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON_BIN = "/home/ubuntu/ai-ssd-p1/.venv/bin/python3"
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "context_scaling_worker.py"
OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "live_inference" / "results" / "optimization3"

def main():
    print("=" * 80)
    print("AI-SSD V2 — Optimization 3: Qwen3-8B FP16 Thread Scaling Benchmark")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threads_list = [2, 4, 8]
    results = {}

    for th in threads_list:
        temp_out = f"/tmp/qwen3_8b_fp16_threads_{th}.json"
        if os.path.exists(temp_out):
            os.remove(temp_out)

        cmd = [
            PYTHON_BIN,
            str(WORKER_SCRIPT),
            "--mode", "aissd",
            "--storage-mode", "nvme_qemu",
            "--model-name", "Qwen/Qwen3-8B",
            "--dtype", "float16",
            "--threads", str(th),
            "--enable-computational-storage",
            "--disable-prefetch",
            "--enable-async-pipeline",
            "--context", "4096",
            "--decode", "16",
            "--seed", "42",
            "--rep", "0",
            "--output-json", temp_out,
        ]

        print(f"\n[THREADS = {th}] Launching worker under {th} threads...")
        t0 = time.time()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - t0

        if p.returncode != 0:
            print(f"[FAIL] Threads={th} failed with code {p.returncode}:")
            print(p.stdout)
            sys.exit(1)

        with open(temp_out, "r") as f:
            data = json.load(f)

        wall = data["wall_time_s"]
        tps = data["tokens_per_second"]
        rss = data["peak_rss_mb"]
        vis_stor = data.get("visible_storage_s", 0.0)
        vis_comp = data.get("visible_compute_s", 0.0)

        print(f"[OK] Threads={th} finished in {elapsed:.1f}s | Decode: {wall:.3f}s ({tps:.3f} tok/s) | Storage: {vis_stor:.3f}s | Compute: {vis_comp:.3f}s | Peak RSS: {rss:.1f} MB")
        results[str(th)] = data

    out_file = OUTPUT_DIR / "qwen3_8b_fp16_thread_scaling.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("THREAD SCALING SUMMARY (Qwen3-8B FP16)")
    print("=" * 80)
    print(f"{'Threads':<10} | {'tok/s':<10} | {'Wall (s)':<10} | {'Visible Storage':<16} | {'Visible Compute':<16}")
    print("-" * 70)
    for th in threads_list:
        d = results[str(th)]
        print(f"{th:<10} | {d['tokens_per_second']:<10.3f} | {d['wall_time_s']:<10.3f} | {d.get('visible_storage_s', 0.0):<16.3f} | {d.get('visible_compute_s', 0.0):<16.3f}")
    print("=" * 80)
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    main()
