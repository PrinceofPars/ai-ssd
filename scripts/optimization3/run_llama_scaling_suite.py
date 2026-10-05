#!/usr/bin/env python3
"""
AI-SSD V2 — Optimization 3: Llama 3 8B FP16 Scaling Suite.
Measures:
1. Thread Scaling: 2, 4, 8 threads at Context=4096, Decode=16.
2. Context Scaling: 4096, 8192, 16384 tokens.
3. Component Profiling: Detailed timing breakdown of all decode operations.
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "live_inference" / "results" / "optimization3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THREAD_SCALING_JSON = RESULTS_DIR / "llama8b_fp16_thread_scaling.json"
CONTEXT_SCALING_JSON = RESULTS_DIR / "llama8b_fp16_context_scaling.json"
COMPONENT_PROFILE_JSON = RESULTS_DIR / "llama8b_fp16_component_profile.json"

MODEL_NAME = "NousResearch/Meta-Llama-3-8B"
DTYPE = "float16"
SEED = 42

def run_worker_cmd(mode: str, context: int, threads: int, decode: int = 16) -> Dict[str, Any]:
    tmp_json = f"/tmp/scaling_worker_{mode}_{context}_{threads}_{os.getpid()}.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "context_scaling_worker.py"),
        "--mode", mode,
        "--model", MODEL_NAME,
        "--dtype", DTYPE,
        "--num-threads", str(threads),
        "--context", str(context),
        "--decode", str(decode),
        "--rep", "0",
        "--seed", str(SEED),
        "--storage-mode", "nvme_qemu",
        "--mapping-mode", "tensor_aware",
        "--enable-batching",
        "--enable-computational-storage",
        "--enable-async-pipeline",
        "--disable-prefetch",
        "--top-k-pct", "10.0",
        "--output-json", tmp_json,
    ]
    print(f"\n[RUN] Mode={mode} | Context={context} | Threads={threads} | Decode={decode}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed: {' '.join(cmd)}")
    with open(tmp_json, "r") as f:
        data = json.load(f)
    try:
        os.remove(tmp_json)
    except OSError:
        pass
    return data

def run_thread_scaling():
    print("\n" + "=" * 80)
    print("THREAD SCALING EXPERIMENT (Context=4096, Threads=[2, 4, 8])")
    print("=" * 80)
    results = {}
    for t in [2, 4, 8]:
        data = run_worker_cmd(mode="aissd", context=4096, threads=t, decode=16)
        results[f"threads_{t}"] = {
            "threads": t,
            "wall_time_s": data["wall_time_s"],
            "tokens_per_second": data["tokens_per_second"],
            "peak_rss_mb": data["peak_rss_mb"],
            "active_kv_mb": data["active_kv_mb"],
            "timing_breakdown": data.get("timing_breakdown", {}),
        }
        print(f"Threads={t}: Wall={data['wall_time_s']:.3f}s | Throughput={data['tokens_per_second']:.3f} tok/s")

    with open(THREAD_SCALING_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] Thread scaling results -> {THREAD_SCALING_JSON}")
    return results

def run_context_scaling():
    print("\n" + "=" * 80)
    print("CONTEXT SCALING EXPERIMENT (Context=[4096, 8192, 16384], Threads=4)")
    print("=" * 80)
    results = {}
    for ctx in [4096, 8192, 16384]:
        # Run AI-SSD
        data_aissd = run_worker_cmd(mode="aissd", context=ctx, threads=4, decode=16)
        results[f"context_{ctx}"] = {
            "context_tokens": ctx,
            "aissd": {
                "wall_time_s": data_aissd["wall_time_s"],
                "tokens_per_second": data_aissd["tokens_per_second"],
                "peak_rss_mb": data_aissd["peak_rss_mb"],
                "active_kv_mb": data_aissd["active_kv_mb"],
                "cold_kv_mb": data_aissd.get("cold_kv_mb", 0.0),
                "storage_read_bytes": data_aissd.get("storage_read_bytes", 0),
                "candidate_k_bytes": data_aissd.get("candidate_k_bytes_to_host", 0),
                "winning_kv_bytes": data_aissd.get("winning_k_bytes_to_host", 0) + data_aissd.get("winning_v_bytes_to_host", 0),
            },
        }
        print(f"Context={ctx}: AI-SSD Wall={data_aissd['wall_time_s']:.3f}s | tok/s={data_aissd['tokens_per_second']:.3f} | Active KV={data_aissd['active_kv_mb']:.1f} MB")

    with open(CONTEXT_SCALING_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] Context scaling results -> {CONTEXT_SCALING_JSON}")
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "threads":
        run_thread_scaling()
    elif len(sys.argv) > 1 and sys.argv[1] == "context":
        run_context_scaling()
    else:
        run_thread_scaling()
        run_context_scaling()
