"""Real Qwen2.5-0.5B Inference Benchmark for AI-SSD V2.

Measures actual wall-clock Qwen2.5-0.5B inference execution across genuine modes:
1. BASELINE: Standard causal inference with in-memory DynamicCache.
2. AI-SSD: Causal inference executing in-storage Top-k KV block selection and retrieval
   using Person 3's RealInferencePrefetchAdapter wrapping Person 2's RealInferenceStorageBackend.

Measures strictly the generation execution interval (16 decode steps).
Zero analytical timing injection; zero sleep; zero fabricated speedup.
Outputs results to /opt/ai-ssd-v2/results/p1/
"""

import sys
import os
import argparse
import json
import time
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import numpy as np
import torch

PROJECT_ROOT = "/home/ubuntu/ai-ssd-p1"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.aissd_inference import (
    run_baseline_decode,
    run_aissd_decode,
    get_current_rss_mb,
    is_english_text,
)


def build_prompt_for_length(engine: RealLLMEngine, target_tokens: int = 512) -> str:
    """Builds a deterministic realistic technical prompt of exact target token length in English."""
    base_paragraph = (
        "AI-SSD computational storage architecture disaggregates Key and Value tensors into 4 KiB flash pages. "
        "The host processor issues top-k search requests to the solid state drive controller over the PCIe NVMe bus. "
        "The controller embedded processing unit reads Key pages directly from flash memory into on-chip SRAM buffers. "
        "The hardware accelerated attention scoring engine computes dot products between the query vector and candidate keys. "
        "Only the highest scoring key and value blocks are transferred back across the host interface. "
    )
    tokens = engine.tokenizer(base_paragraph)["input_ids"]
    reps = (target_tokens // len(tokens)) + 2
    full_text = base_paragraph * reps
    token_ids = engine.tokenizer(full_text, max_length=target_tokens, truncation=True)["input_ids"]
    prompt = engine.tokenizer.decode(token_ids, skip_special_tokens=True)
    if not is_english_text(prompt):
        raise ValueError("Generated prompt fails English language verification.")
    return prompt


def parse_args():
    parser = argparse.ArgumentParser(description="AI-SSD V2 Real Inference Benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["baseline", "ai_ssd", "compare"],
        help="Inference mode: 'baseline' (in-memory DRAM), 'ai_ssd' (in-storage KV retrieval), or 'compare' (both)",
    )
    parser.add_argument("--context", type=int, default=512, help="Context length in tokens (default: 512)")
    parser.add_argument("--decode", type=int, default=16, help="Tokens to generate during decode (default: 16)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads budget (default: 4)")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of benchmark repetitions (default: 3)")
    parser.add_argument("--topk", type=float, default=10.0, help="Top-k sparsity percentage (default: 10.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable Person 3 prefetch adapter and run Person 2 backend directly",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/ai-ssd-v2/results/p1",
        help="Directory to save machine-readable results",
    )
    return parser.parse_args()


def print_benchmark_banner(
    mode_str: str,
    context: int,
    decode_tokens: int,
    threads: int,
    wall_time_s: float,
    throughput_tps: float,
    kv_memory_mb: float,
    kv_offloaded_pct: float,
    kv_blocks_read: int,
    backend_str: str,
    bytes_read: int,
    requests: int,
    classification_str: str,
    telemetry: Optional[Dict[str, Any]] = None,
):
    print("============================================================")
    print("AI-SSD V2 REAL INFERENCE BENCHMARK")
    print("============================================================")
    print(f"Mode             : {mode_str}")
    print("Model            : Qwen2.5-0.5B")
    print(f"Context          : {context}")
    print(f"Decode tokens    : {decode_tokens}")
    print(f"CPU threads      : {threads}")
    print("")
    print("------------------------------------------------------------")
    print("Inference")
    print("------------------------------------------------------------")
    print(f"Wall time        : {wall_time_s:.4f} s")
    print(f"Throughput       : {throughput_tps:.2f} tok/s")
    print("")
    print("------------------------------------------------------------")
    print("KV")
    print("------------------------------------------------------------")
    print(f"KV memory        : {kv_memory_mb:.2f} MB")
    print(f"KV offloaded     : {kv_offloaded_pct:.1f} %")
    print(f"KV blocks read   : {kv_blocks_read}")
    print("")
    print("------------------------------------------------------------")
    print("Storage")
    print("------------------------------------------------------------")
    print(f"Backend          : {backend_str}")
    print(f"Bytes read       : {bytes_read:,} bytes ({bytes_read / (1024.0 * 1024.0):.2f} MB)")
    print(f"Requests         : {requests}")

    if telemetry:
        if "demand_hits" in telemetry:
            print("")
            print("------------------------------------------------------------")
            print("Prefetch Telemetry (Person 3 DRAM Staging & Speculative Prefetch)")
            print("------------------------------------------------------------")
            print(f"Demand requests  : {telemetry.get('demand_reads')}")
            print(f"Demand hits      : {telemetry.get('demand_hits')} ({telemetry.get('demand_hit_rate_pct', 0.0):.2f}%)")
            print(f"Demand misses    : {telemetry.get('demand_misses')}")
            print(f"Prefetch requests: {telemetry.get('prefetch_requests')}")
            print(f"Useful prefetches: {telemetry.get('useful_prefetches')} ({telemetry.get('prefetch_accuracy_pct', 0.0):.2f}%)")
            print(f"Useful bytes     : {telemetry.get('useful_bytes', 0):,} bytes ({telemetry.get('useful_bytes', 0)/(1024.0*1024.0):.2f} MB)")
            print(f"Wasted bytes     : {telemetry.get('wasted_bytes', 0):,} bytes ({telemetry.get('wasted_bytes', 0)/(1024.0*1024.0):.4f} MB)")
            print(f"Staging memory   : {telemetry.get('staging_memory_mb', 0.0):.2f} MB (Peak: {telemetry.get('peak_memory_mb', 0.0):.2f} MB)")
            print(f"Hit avg latency  : {telemetry.get('demand_hit_avg_latency_us', 0.0):.2f} us")
            print(f"Miss avg latency : {telemetry.get('demand_miss_avg_latency_us', 0.0):.2f} us")

        p2_telem = telemetry.get("storage_backend", telemetry)
        if isinstance(p2_telem, dict) and "channel_distribution" in p2_telem:
            ch_dist = p2_telem["channel_distribution"]
            print("")
            print("------------------------------------------------------------")
            print("Hardware Telemetry (Person 2 FTL Multi-Channel Distribution)")
            print("------------------------------------------------------------")
            print(f"Per-channel reads: {ch_dist.get('channel_read_counts', {})}")
            print(f"Per-channel load : {ch_dist.get('per_channel_total_requests', {})}")
            print(f"Load imbalance   : {ch_dist.get('load_imbalance_percent', 0.0):.2f} %")
            print(f"Contention ratio : {ch_dist.get('contention_ratio', 1.0):.2f} (vs 8.0x serial conventional)")
            print(f"Sleep latency    : {p2_telem.get('simulated_metrics', {}).get('sleep_latency_injected', False)} (Zero artificial delay)")

    print("")
    print("------------------------------------------------------------")
    print("Classification")
    print("------------------------------------------------------------")
    print(f"{classification_str}")
    print("============================================================")


def compare_outputs(baseline_res: Dict[str, Any], aissd_res: Dict[str, Any]) -> Dict[str, Any]:
    b_tokens = baseline_res["token_ids"]
    a_tokens = aissd_res["token_ids"]

    matches = sum(1 for b, a in zip(b_tokens, a_tokens) if b == a)
    match_pct = (matches / len(b_tokens)) * 100.0

    b_logits = baseline_res["final_logits"]
    a_logits = aissd_res["final_logits"]

    # Compute cosine similarity of final logits
    b_flat = b_logits.flatten()
    a_flat = a_logits.flatten()
    norm_b = np.linalg.norm(b_flat)
    norm_a = np.linalg.norm(a_flat)
    cos_sim = float(np.dot(b_flat, a_flat) / max(1e-12, norm_b * norm_a))
    mse = float(np.mean((b_flat - a_flat) ** 2))
    max_abs_diff = float(np.max(np.abs(b_flat - a_flat)))

    print("\n------------------------------------------------------------")
    print("CORRECTNESS & OUTPUT COMPARISON (Baseline vs AI-SSD)")
    print("------------------------------------------------------------")
    print(f"Baseline Tokens  : {b_tokens}")
    print(f"AI-SSD Tokens    : {a_tokens}")
    print(f"Token Match      : {matches}/{len(b_tokens)} ({match_pct:.1f}%)")
    print(f"Logits Cosine Sim: {cos_sim:.5f}")
    print(f"Logits MSE       : {mse:.6f}")
    print(f"Logits Max Diff  : {max_abs_diff:.6f}")
    print(f"Baseline Text    : {baseline_res['generated_text']!r}")
    print(f"AI-SSD Text      : {aissd_res['generated_text']!r}")
    print("------------------------------------------------------------\n")

    return {
        "token_matches": matches,
        "token_count": len(b_tokens),
        "token_match_percent": match_pct,
        "logits_cosine_similarity": cos_sim,
        "logits_mse": mse,
        "logits_max_abs_diff": max_abs_diff,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Initializing Qwen2.5-0.5B on CPU (threads={args.threads})...")
    engine = RealLLMEngine(
        model_name="Qwen/Qwen2.5-0.5B",
        device="cpu",
        dtype="FP32",
        num_threads=args.threads,
    )
    model = engine.model
    tokenizer = engine.tokenizer

    prompt = build_prompt_for_length(engine, target_tokens=args.context)
    input_ids = tokenizer(prompt, max_length=args.context, truncation=True, return_tensors="pt")["input_ids"]
    actual_context = input_ids.shape[1]

    # Classification strings
    baseline_classification = "REAL INFERENCE (HOST CPU + IN-MEMORY DRAM)"
    aissd_classification = "REAL MODEL + REAL IN-STORAGE KV RETRIEVAL"

    baseline_payload = None
    aissd_payload = None

    if args.mode in ("baseline", "compare"):
        print(f"\nExecuting BASELINE Benchmark ({args.repetitions} repetitions)...")
        rep_times = []
        rep_tps = []
        rep_rss = []
        last_base_res = None

        for r in range(args.repetitions):
            res = run_baseline_decode(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                decode_tokens=args.decode,
                seed=args.seed + r,
            )
            rep_times.append(res["wall_time_s"])
            rep_tps.append(res["tokens_per_second"])
            rep_rss.append(res["peak_rss_mb"])
            last_base_res = res
            print(f"  Rep {r+1}/{args.repetitions}: {res['wall_time_s']:.4f}s ({res['tokens_per_second']:.2f} tok/s)")

        mean_wall = float(np.mean(rep_times))
        std_wall = float(np.std(rep_times))
        mean_tps = float(np.mean(rep_tps))
        std_tps = float(np.std(rep_tps))
        mean_rss = float(np.mean(rep_rss))

        print_benchmark_banner(
            mode_str="BASELINE",
            context=actual_context,
            decode_tokens=args.decode,
            threads=args.threads,
            wall_time_s=mean_wall,
            throughput_tps=mean_tps,
            kv_memory_mb=last_base_res["kv_memory_mb"],
            kv_offloaded_pct=0.0,
            kv_blocks_read=0,
            backend_str=last_base_res["storage_backend"],
            bytes_read=0,
            requests=0,
            classification_str=baseline_classification,
        )

        baseline_payload = {
            "mode": "BASELINE",
            "model": "Qwen/Qwen2.5-0.5B",
            "context_length": actual_context,
            "decode_tokens": args.decode,
            "cpu_threads": args.threads,
            "repetitions": args.repetitions,
            "wall_time_s": {
                "mean": mean_wall,
                "std": std_wall,
                "raw": rep_times,
            },
            "tokens_per_second": {
                "mean": mean_tps,
                "std": std_tps,
                "raw": rep_tps,
            },
            "peak_rss_mb": mean_rss,
            "kv_memory_mb": last_base_res["kv_memory_mb"],
            "kv_offloaded_pct": 0.0,
            "kv_blocks_read": 0,
            "storage": {
                "backend": last_base_res["storage_backend"],
                "bytes_read": 0,
                "requests": 0,
            },
            "classification": baseline_classification,
            "token_ids": last_base_res["token_ids"],
            "generated_text": last_base_res["generated_text"],
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        }

        baseline_out_path = os.path.join(args.output_dir, "real_inference_baseline.json")
        with open(baseline_out_path, "w", encoding="utf-8") as f:
            json.dump(baseline_payload, f, indent=2)
        print(f"Saved baseline results to: {baseline_out_path}")

    if args.mode in ("ai_ssd", "compare"):
        enable_prefetch = not args.no_prefetch
        prefetch_str = "with P3 DRAM Staging & Prefetch" if enable_prefetch else "Direct P2 Multi-Channel FTL"
        print(f"\nExecuting AI-SSD Benchmark ({args.repetitions} repetitions, Top-k={args.topk}%, {prefetch_str})...")
        rep_times = []
        rep_tps = []
        rep_rss = []
        last_aissd_res = None

        for r in range(args.repetitions):
            res = run_aissd_decode(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                decode_tokens=args.decode,
                top_k_pct=args.topk,
                seed=args.seed + r,
                enable_prefetch=enable_prefetch,
            )
            rep_times.append(res["wall_time_s"])
            rep_tps.append(res["tokens_per_second"])
            rep_rss.append(res["peak_rss_mb"])
            last_aissd_res = res
            print(f"  Rep {r+1}/{args.repetitions}: {res['wall_time_s']:.4f}s ({res['tokens_per_second']:.2f} tok/s)")

        mean_wall = float(np.mean(rep_times))
        std_wall = float(np.std(rep_times))
        mean_tps = float(np.mean(rep_tps))
        std_tps = float(np.std(rep_tps))
        mean_rss = float(np.mean(rep_rss))

        telemetry = last_aissd_res.get("telemetry")

        print_benchmark_banner(
            mode_str="AI-SSD",
            context=actual_context,
            decode_tokens=args.decode,
            threads=args.threads,
            wall_time_s=mean_wall,
            throughput_tps=mean_tps,
            kv_memory_mb=last_aissd_res["kv_memory_mb"],
            kv_offloaded_pct=last_aissd_res["kv_offloaded_pct"],
            kv_blocks_read=last_aissd_res["kv_blocks_read"],
            backend_str=last_aissd_res["storage_backend"],
            bytes_read=last_aissd_res["storage_bytes_read"],
            requests=last_aissd_res["storage_requests"],
            classification_str=aissd_classification,
            telemetry=telemetry,
        )

        aissd_payload = {
            "mode": "AI-SSD",
            "model": "Qwen/Qwen2.5-0.5B",
            "context_length": actual_context,
            "decode_tokens": args.decode,
            "cpu_threads": args.threads,
            "top_k_percent": args.topk,
            "prefetch_enabled": enable_prefetch,
            "repetitions": args.repetitions,
            "wall_time_s": {
                "mean": mean_wall,
                "std": std_wall,
                "raw": rep_times,
            },
            "tokens_per_second": {
                "mean": mean_tps,
                "std": std_tps,
                "raw": rep_tps,
            },
            "peak_rss_mb": mean_rss,
            "kv_memory_mb": last_aissd_res["kv_memory_mb"],
            "kv_offloaded_pct": last_aissd_res["kv_offloaded_pct"],
            "kv_blocks_read": last_aissd_res["kv_blocks_read"],
            "storage": {
                "backend": last_aissd_res["storage_backend"],
                "bytes_read": last_aissd_res["storage_bytes_read"],
                "requests": last_aissd_res["storage_requests"],
            },
            "classification": aissd_classification,
            "token_ids": last_aissd_res["token_ids"],
            "generated_text": last_aissd_res["generated_text"],
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        }

        if telemetry:
            aissd_payload["storage"]["telemetry"] = telemetry
            if "demand_hits" in telemetry:
                aissd_payload["prefetch"] = {
                    "demand_requests": telemetry.get("demand_reads"),
                    "demand_hits": telemetry.get("demand_hits"),
                    "demand_misses": telemetry.get("demand_misses"),
                    "demand_hit_rate_pct": telemetry.get("demand_hit_rate_pct"),
                    "prefetch_requests": telemetry.get("prefetch_requests"),
                    "useful_prefetches": telemetry.get("useful_prefetches"),
                    "prefetch_accuracy_pct": telemetry.get("prefetch_accuracy_pct"),
                    "useful_bytes": telemetry.get("useful_bytes"),
                    "wasted_bytes": telemetry.get("wasted_bytes"),
                    "staging_memory_mb": telemetry.get("staging_memory_mb"),
                    "peak_memory_mb": telemetry.get("peak_memory_mb"),
                }
            p2_telem = telemetry.get("storage_backend", telemetry)
            if isinstance(p2_telem, dict) and "channel_distribution" in p2_telem:
                aissd_payload["channel_distribution"] = p2_telem["channel_distribution"]

        aissd_out_path = os.path.join(args.output_dir, "real_inference_ai_ssd.json")
        with open(aissd_out_path, "w", encoding="utf-8") as f:
            json.dump(aissd_payload, f, indent=2)
        print(f"Saved AI-SSD results to: {aissd_out_path}")

    # If both modes ran or if files exist, perform comparison
    if args.mode == "compare" or (baseline_payload and aissd_payload):
        cmp_metrics = compare_outputs(last_base_res, last_aissd_res)
        aissd_payload["correctness_vs_baseline"] = cmp_metrics
        aissd_out_path = os.path.join(args.output_dir, "real_inference_ai_ssd.json")
        with open(aissd_out_path, "w", encoding="utf-8") as f:
            json.dump(aissd_payload, f, indent=2)


if __name__ == "__main__":
    main()
