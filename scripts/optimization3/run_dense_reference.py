#!/usr/bin/env python3
"""
AI-SSD V2 — Optimization 3: Llama 3 8B FP16 Dense In-Memory Reference Benchmark.
Measures standard PyTorch in-memory DynamicCache execution with high-resolution RSS sampling.
"""

import sys
import gc
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.real_llm.aissd_inference import ProcessMemorySampler, get_current_rss_mb

MODEL_PATH = "/home/ubuntu/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B/snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "live_inference" / "results" / "optimization3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DENSE_JSON = RESULTS_DIR / "llama8b_fp16_dense_reference.json"


def build_llama_prompt_for_length(tokenizer: Any, target_tokens: int = 4096) -> torch.Tensor:
    """Builds a semantically coherent prompt that tokenizes to exactly target_tokens."""
    base_text = (
        "The development of computational storage architectures represents a fundamental paradigm shift "
        "in modern data-intensive computing systems. Traditional von Neumann computer architectures suffer "
        "from the classic memory wall problem, where the latency and energy required to transfer data "
        "across the memory bus significantly constrain end-to-end application throughput. "
        "In Large Language Model (LLM) autoregressive inference, the key-value (KV) cache grows linearly "
        "with sequence length, consuming tens of gigabytes of host DRAM and saturating PCIe bandwidth during retrieval. "
    )
    base_ids = tokenizer.encode(base_text, add_special_tokens=True)
    repeats = (target_tokens // len(base_ids)) + 2
    full_text = " ".join([base_text] * repeats)
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)
    input_ids = torch.tensor([full_ids[:target_tokens]], dtype=torch.long)
    return input_ids


def run_dense_benchmark(
    context_tokens: int = 4096,
    decode_tokens: int = 16,
    num_threads: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"AI-SSD V2 — Llama 3 8B FP16 Dense Reference (Context={context_tokens}, Decode={decode_tokens})")
    print("=" * 80)

    torch.set_num_threads(num_threads)
    torch.manual_seed(seed)

    print("[1/4] Loading Tokenizer & Model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    print(f"[2/4] Constructing prompt for exactly {context_tokens} tokens...")
    input_ids = build_llama_prompt_for_length(tokenizer, target_tokens=context_tokens)
    actual_prompt_tokens = input_ids.shape[1]
    print(f"Prompt verified: {actual_prompt_tokens} tokens")
    assert actual_prompt_tokens == context_tokens, f"Expected {context_tokens}, got {actual_prompt_tokens}"

    gc.collect()
    rss_before = get_current_rss_mb()

    # 1. Prefill step (excluded from decode timer)
    print("[3/4] Executing Prefill Step...")
    t_pref0 = time.perf_counter()
    with torch.no_grad():
        prefill_out = model(input_ids=input_ids, use_cache=True)
    prefill_time = time.perf_counter() - t_pref0
    pkv = prefill_out.past_key_values
    next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)

    generated_tokens = [next_token.item()]
    cur_seq_len = input_ids.shape[1]
    print(f"Prefill completed in {prefill_time:.2f}s | First token: {next_token.item()} ({repr(tokenizer.decode([next_token.item()]))})")

    # 2. Generation execution interval (measured strictly with high-res RSS sampler)
    print(f"[4/4] Executing Decode Generation ({decode_tokens} tokens)...")
    sampler = ProcessMemorySampler(sample_interval_s=0.002)
    sampler.start()
    t_start = time.perf_counter()
    for step in range(1, decode_tokens):
        pos_ids = torch.tensor([[cur_seq_len + step - 1]], device=input_ids.device)
        with torch.no_grad():
            step_out = model(input_ids=next_token, position_ids=pos_ids, past_key_values=pkv, use_cache=True)
        pkv = step_out.past_key_values
        next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())
    t_end = time.perf_counter()
    proc_mem = sampler.stop()

    wall_time = t_end - t_start
    tps = len(generated_tokens) / max(1e-6, wall_time)

    # Dynamic KV-cache computation based on Llama 3 architecture
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    bytes_per_elem = 2  # FP16
    total_tokens = actual_prompt_tokens + decode_tokens
    total_kv_bytes = total_tokens * num_layers * num_kv_heads * head_dim * bytes_per_elem * 2
    total_kv_mb = total_kv_bytes / (1024.0 * 1024.0)

    generated_text = tokenizer.decode(generated_tokens)
    print(f"\n[DONE] Decode completed in {wall_time:.3f}s ({tps:.3f} tok/s)")
    print(f"Peak RSS: {proc_mem['peak_rss_mb']:.1f} MB | Dense KV in DRAM: {total_kv_mb:.2f} MB")
    print(f"Generated Token IDs: {generated_tokens}")
    print(f"Generated Text: {repr(generated_text)}")

    results = {
        "model": "NousResearch/Meta-Llama-3-8B",
        "precision": "FP16",
        "mode": "DENSE_ALL_DRAM",
        "cpu_threads": num_threads,
        "context_tokens": context_tokens,
        "decode_tokens": decode_tokens,
        "seed": seed,
        "wall_time_s": wall_time,
        "tokens_per_second": tps,
        "prefill_time_s": prefill_time,
        "load_time_s": load_time,
        "peak_rss_mb": proc_mem["peak_rss_mb"],
        "min_rss_mb": proc_mem["min_rss_mb"],
        "avg_rss_mb": proc_mem["avg_rss_mb"],
        "dense_kv_mb": total_kv_mb,
        "token_ids": generated_tokens,
        "generated_text": generated_text,
    }

    with open(DENSE_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {DENSE_JSON}")
    return results


if __name__ == "__main__":
    run_dense_benchmark()
