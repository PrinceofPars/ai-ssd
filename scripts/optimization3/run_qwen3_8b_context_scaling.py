#!/usr/bin/env python3
"""AI-SSD V2 — Optimization 3: Qwen3-8B FP16 Context Scaling Benchmark (4096 -> 8192 -> 16384).

Measures KV cache offload efficiency, peak RSS, and inference throughput as
context scales up on Qwen3-8B FP16 with QEMU/NVMe computational storage:
  - Context lengths: 4096, 8192, 16384
  - Decode: 16 tokens
  - Threads: 4
  - Precision: FP16

Saves results to: benchmarks/live_inference/results/optimization3/qwen3_8b_fp16_context_scaling.json
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
    print("AI-SSD V2 — Optimization 3: Qwen3-8B FP16 Context Scaling Benchmark")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contexts = [4096, 8192, 16384]
    results = {}

    for ctx in contexts:
        temp_out = f"/tmp/qwen3_8b_fp16_ctx_{ctx}.json"
        if os.path.exists(temp_out):
            os.remove(temp_out)

        cmd = [
            PYTHON_BIN,
            str(WORKER_SCRIPT),
            "--mode", "aissd",
            "--storage-mode", "nvme_qemu",
            "--model-name", "Qwen/Qwen3-8B",
            "--dtype", "float16",
            "--threads", "4",
            "--enable-computational-storage",
            "--disable-prefetch",
            "--enable-async-pipeline",
            "--context", str(ctx),
            "--decode", "16",
            "--seed", "42",
            "--rep", "0",
            "--output-json", temp_out,
        ]

        print(f"\n[CONTEXT = {ctx}] Launching AI-SSD worker (4 threads, FP16)...")
        t0 = time.time()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - t0

        if p.returncode != 0:
            print(f"[FAIL] Context={ctx} failed with code {p.returncode}:")
            print(p.stdout)
            sys.exit(1)

        with open(temp_out, "r") as f:
            data = json.load(f)

        wall = data["wall_time_s"]
        tps = data["tokens_per_second"]
        rss = data["peak_rss_mb"]
        act_kv = data["active_kv_mb"]
        cold_kv = data.get("cold_kv_mb", 0.0)
        total_kv = act_kv + cold_kv
        offload_pct = (cold_kv / max(1e-6, total_kv)) * 100.0 if total_kv > 0 else 0.0

        print(f"[OK] Context={ctx} finished in {elapsed:.1f}s | Decode: {wall:.3f}s ({tps:.3f} tok/s) | Peak RSS: {rss:.1f} MB | Active KV: {act_kv:.1f} MB | Cold KV: {cold_kv:.1f} MB | Offload: {offload_pct:.1f}%")
        results[str(ctx)] = data

    out_file = OUTPUT_DIR / "qwen3_8b_fp16_context_scaling.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("CONTEXT SCALING SUMMARY (Qwen3-8B FP16)")
    print("=" * 80)
    print(f"{'Context':<10} | {'tok/s':<10} | {'Wall (s)':<10} | {'Active KV (MB)':<16} | {'Cold KV (MB)':<16} | {'Offload %':<10} | {'Peak RSS (MB)':<14}")
    print("-" * 90)
    for ctx in contexts:
        d = results[str(ctx)]
        act = d["active_kv_mb"]
        cold = d.get("cold_kv_mb", 0.0)
        tot = act + cold
        off_pct = (cold / max(1e-6, tot)) * 100.0
        print(f"{ctx:<10} | {d['tokens_per_second']:<10.3f} | {d['wall_time_s']:<10.3f} | {act:<16.1f} | {cold:<16.1f} | {off_pct:<10.1f}% | {d['peak_rss_mb']:<14.1f}")
    print("=" * 80)
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    main()
