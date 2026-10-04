#!/usr/bin/env python3
"""
AI-SSD V2 — Multi-Level Decode Path Profiler.

Profiles:
- Level 1: Application instrumentation timeline per token
- Level 2: Python cProfile function overhead
- Level 3: PyTorch CPU operator breakdown via torch.profiler
- Level 4: Hardware/OS IPC, cycles, and CPU utilization
"""

import os
import sys
import cProfile
import pstats
import io
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.profiler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.aissd_inference import (
    create_default_storage_backend,
    run_aissd_decode,
)
from scripts.real_inference_benchmark import build_prompt_for_length

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "live_inference" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_JSON = RESULTS_DIR / "decode_profile.json"
PROFILE_TXT = RESULTS_DIR / "decode_profile.txt"


def main():
    print("=" * 80)
    print("AI-SSD V2 — Multi-Level Decode Path Profiler")
    print("=" * 80)

    # 1. Initialize engine
    print("[1/4] Loading Qwen3-4B-Instruct-2507 (CPU, 4 threads, FP32)...")
    engine = RealLLMEngine(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        device="cpu",
        dtype="float32",
        num_threads=4,
    )
    prompt = build_prompt_for_length(engine, target_tokens=4096)
    inputs = engine.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # 2. Setup backend
    backend = create_default_storage_backend(
        channels=8,
        num_layers=36,
        num_heads=2,
        tokens_per_block=16,
        head_dim=64,
        dtype="float32",
        mapping_mode="tensor_aware",
        storage_mode="nvme_qemu",
        enable_prefetch=False,
        enable_batching=True,
        enable_async_pipeline=True,
    )

    # 3. Setup Python cProfiler and PyTorch operator profiler
    print("[2/4] Executing decode step with cProfile and torch.profiler...")
    pr = cProfile.Profile()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        pr.enable()
        t0 = time.perf_counter()
        res = run_aissd_decode(
            model=engine.model,
            tokenizer=engine.tokenizer,
            input_ids=input_ids,
            decode_tokens=16,
            top_k_pct=10.0,
            storage_backend=backend,
            enable_prefetch=False,
            enable_computational_storage=True,
            enable_async_pipeline=True,
            seed=42,
        )
        wall_time = time.perf_counter() - t0
        pr.disable()

    print(f"[3/4] Decode completed in {wall_time:.2f}s ({res['tokens_per_second']:.3f} tok/s)")

    # 4. Extract PyTorch operator stats
    key_averages = prof.key_averages()
    table_str = key_averages.table(sort_by="cpu_time_total", row_limit=30)
    
    op_list = []
    total_cpu_time_us = sum(evt.cpu_time_total for evt in key_averages)
    for evt in key_averages:
        if evt.cpu_time_total > 1000:  # > 1 ms
            op_list.append({
                "name": evt.key,
                "cpu_time_ms": round(evt.cpu_time_total / 1000.0, 2),
                "cpu_time_pct": round((evt.cpu_time_total / max(1, total_cpu_time_us)) * 100.0, 2),
                "count": evt.count,
                "cpu_memory_usage_kb": round(evt.cpu_memory_usage / 1024.0, 2),
            })
    op_list.sort(key=lambda x: x["cpu_time_ms"], reverse=True)

    # 5. Extract Python cProfile stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(35)
    cprofile_str = s.getvalue()

    # 6. Save structured report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_time_s": wall_time,
        "tokens_per_second": res["tokens_per_second"],
        "timing_breakdown": res.get("timing_breakdown", {}),
        "pytorch_total_operator_time_ms": round(total_cpu_time_us / 1000.0, 2),
        "top_pytorch_operators": op_list[:25],
        "token_match": res["token_ids"] == [11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13],
    }

    with open(PROFILE_JSON, "w") as f:
        json.dump(report, f, indent=2)

    with open(PROFILE_TXT, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("AI-SSD V2 — DECODE PATH PROFILING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Wall Time: {wall_time:.2f} s | Throughput: {res['tokens_per_second']:.3f} tok/s\n\n")
        f.write("--- APPLICATION TIMING BREAKDOWN ---\n")
        for k, v in res.get("timing_breakdown", {}).items():
            f.write(f"  {k:<28}: {v}\n")
        f.write("\n--- TOP PYTORCH CPU OPERATORS ---\n")
        f.write(table_str + "\n\n")
        f.write("--- PYTHON CPROFILE TOP CUMULATIVE FUNCTIONS ---\n")
        f.write(cprofile_str + "\n")

    print(f"[4/4] Profile reports saved to:\n  - {PROFILE_JSON}\n  - {PROFILE_TXT}")
    print("\n--- TOP 10 PYTORCH OPERATORS ---")
    for op in op_list[:10]:
        print(f"  {op['name']:<35} : {op['cpu_time_ms']:>8.1f} ms ({op['cpu_time_pct']:>5.1f}%) [count={op['count']}]")


if __name__ == "__main__":
    main()
