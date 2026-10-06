#!/usr/bin/env python3
"""AI-SSD V2 — Optimization 3: Qwen3-8B FP16 Dense In-Memory Baseline Reference.

Runs dense in-memory inference on Qwen3-8B FP16 with 4 CPU threads:
  - Context = 4096 tokens
  - Decode = 16 tokens
  - Seed = 42
  - Dtype = float16
  - Device = CPU (4 threads)

Measures and records:
  - Ground-truth generated token IDs (16 tokens)
  - Generated text
  - Wall time (decode only)
  - Throughput (tokens/s)
  - Memory telemetry: min_rss_mb, avg_rss_mb, peak_rss_mb
  - Active KV cache size (MB)

Saves output to: benchmarks/live_inference/results/optimization3/qwen3_8b_fp16_dense_reference.json
"""

import sys
import os
import gc
import json
import time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.real_llm.aissd_inference import (
    run_baseline_decode,
    ProcessMemorySampler,
    get_current_rss_mb,
)

MODEL_ID = "Qwen/Qwen3-8B"
CONTEXT_LENGTH = 4096
DECODE_TOKENS = 16
SEED = 42
NUM_THREADS = 4

def build_prompt(tokenizer, target_tokens=4096) -> str:
    base_paragraph = (
        "AI-SSD computational storage architecture disaggregates Key and Value tensors into 4 KiB flash pages. "
        "The host processor issues top-k search requests to the solid state drive controller over the PCIe NVMe bus. "
        "The controller embedded processing unit reads Key pages directly from flash memory into on-chip SRAM buffers. "
        "The hardware accelerated attention scoring engine computes dot products between the query vector and candidate keys. "
        "Only the highest scoring key and value blocks are transferred back across the host interface. "
    )
    tokens = tokenizer(base_paragraph)["input_ids"]
    reps = (target_tokens // len(tokens)) + 2
    full_text = base_paragraph * reps
    token_ids = tokenizer(full_text, max_length=target_tokens, truncation=True)["input_ids"]
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def main():
    print("=" * 80)
    print("AI-SSD V2 — Optimization 3: Qwen3-8B FP16 Dense Reference Benchmark")
    print("=" * 80)

    # 1. System & Thread Configuration
    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(SEED)
    print(f"Config: Threads={NUM_THREADS}, Seed={SEED}, Context={CONTEXT_LENGTH}, Decode={DECODE_TOKENS}")

    # 2. Load Model & Tokenizer
    print(f"\n[1/4] Loading Tokenizer & Model ({MODEL_ID} in FP16 on CPU)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s | RSS: {get_current_rss_mb():.1f} MB")

    # 3. Build Prompt
    print(f"\n[2/4] Constructing deterministic prompt of {CONTEXT_LENGTH} tokens...")
    prompt = build_prompt(tokenizer, target_tokens=CONTEXT_LENGTH)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    actual_tokens = input_ids.shape[1]
    print(f"Input tokens: {actual_tokens}")

    # 4. Execute Dense Baseline Decode
    print(f"\n[3/4] Executing Dense Baseline Decode ({DECODE_TOKENS} tokens, Seed={SEED})...")
    t_start = time.perf_counter()
    result = run_baseline_decode(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        decode_tokens=DECODE_TOKENS,
        seed=SEED,
    )
    total_time = time.perf_counter() - t_start

    print("\n[4/4] Execution Completed:")
    print(f"  Wall time (decode):    {result['wall_time_s']:.3f} s")
    print(f"  Tokens per second:     {result['tokens_per_second']:.3f} tok/s")
    print(f"  Peak RSS:              {result['peak_rss_mb']:.1f} MB")
    print(f"  Active KV Cache:       {result['kv_memory_mb']:.1f} MB")
    print(f"  Generated Tokens:      {result['token_ids']}")
    print(f"  Generated Text:        {repr(result['generated_text'])}")

    # 5. Save Output JSON
    output_dir = PROJECT_ROOT / "benchmarks" / "live_inference" / "results" / "optimization3"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "qwen3_8b_fp16_dense_reference.json"

    data = {
        "model_id": MODEL_ID,
        "dtype": "float16",
        "num_threads": NUM_THREADS,
        "seed": SEED,
        "context_length": CONTEXT_LENGTH,
        "actual_input_tokens": actual_tokens,
        "decode_tokens": DECODE_TOKENS,
        "wall_time_s": result["wall_time_s"],
        "tokens_per_second": result["tokens_per_second"],
        "min_rss_mb": result["min_rss_mb"],
        "avg_rss_mb": result["avg_rss_mb"],
        "peak_rss_mb": result["peak_rss_mb"],
        "kv_memory_mb": result["kv_memory_mb"],
        "token_ids": result["token_ids"],
        "generated_text": result["generated_text"],
        "load_time_s": load_time,
        "total_benchmark_time_s": total_time,
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nReference results saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
