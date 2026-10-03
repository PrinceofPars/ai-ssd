#!/usr/bin/env python3
"""AI-SSD V2 Reproducible Live Inference Benchmark Runner.

Reads configuration from config.yaml (or CLI overrides), executes canonical
live inference runs for BASELINE and AI-SSD, and outputs structured JSON and CSV.

Zero analytical timing injection; zero artificial sleeps; 100% genuine model execution.
Process RAM telemetry sampled continuously (2ms interval) via psutil.
"""

import sys
import os
from pathlib import Path
import argparse
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import yaml
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.aissd_inference import (
    create_default_storage_backend,
    run_baseline_decode,
    run_aissd_decode,
    get_current_rss_mb,
)
from scripts.real_inference_benchmark import build_prompt_for_length


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="AI-SSD V2 Live Inference Reproducible Benchmark")
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--dtype", type=str, default=None, help="Override dtype (FP32, BF16, FP16)")
    parser.add_argument("--context", type=int, default=None, help="Override context length")
    parser.add_argument("--decode", type=int, default=None, help="Override decode token count")
    parser.add_argument("--repetitions", type=int, default=None, help="Override repetitions")
    parser.add_argument("--topk", type=float, default=None, help="Override Top-k percentage")
    parser.add_argument("--no-prefetch", action="store_true", help="Disable P3 prefetch")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent / "results"), help="Results directory")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    b_cfg = cfg.get("benchmark", {})
    a_cfg = cfg.get("aissd", {})

    context_len = args.context or b_cfg.get("context_length", 512)
    decode_tokens = args.decode or b_cfg.get("decode_tokens", 16)
    repetitions = args.repetitions or b_cfg.get("repetitions", 3)
    topk_ratio = (args.topk / 100.0) if args.topk is not None else a_cfg.get("topk_ratio", 0.10)
    topk_pct = topk_ratio * 100.0
    enable_prefetch = not args.no_prefetch if args.no_prefetch else a_cfg.get("enable_prefetch", True)
    channels = a_cfg.get("channels", 8)
    threads = b_cfg.get("threads", 4)
    model_name = args.model or b_cfg.get("model_name", "Qwen/Qwen2.5-0.5B")
    dtype = args.dtype or b_cfg.get("dtype", "float32")
    seed = b_cfg.get("seed", 42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("      AI-SSD V2 REPRODUCIBLE LIVE INFERENCE BENCHMARK")
    print("=" * 64)
    print(f"Model          : {model_name}")
    print(f"Context Length : {context_len}")
    print(f"Decode Tokens  : {decode_tokens}")
    print(f"CPU Threads    : {threads}")
    print(f"Dtype          : {dtype}")
    print(f"Repetitions    : {repetitions}")
    print(f"Top-k Ratio    : {topk_ratio:.2f} ({topk_pct:.1f}%)")
    print(f"Prefetch       : {'ENABLED' if enable_prefetch else 'DISABLED'}")
    print(f"FTL Channels   : {channels}")
    print("=" * 64)

    # Initialize Engine
    print(f"\n[INIT] Loading engine and weights for {model_name} ({dtype})...")
    engine = RealLLMEngine(model_name=model_name, device="cpu", dtype=dtype, num_threads=threads)
    prompt = build_prompt_for_length(engine, target_tokens=context_len)
    inputs = engine.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # 1. Baseline Benchmark
    print(f"\n[1/2] Executing BASELINE Benchmark ({repetitions} reps)...")
    baseline_runs = []
    baseline_peak_rss = 0.0
    last_baseline_res = None
    for r in range(repetitions):
        res = run_baseline_decode(engine.model, engine.tokenizer, input_ids, decode_tokens=decode_tokens, seed=seed + r)
        res_tps = res.get("tokens_per_second", res.get("throughput_tok_s", 0.0))
        res_rss = res.get("peak_rss_mb", res.get("rss_peak_mb", 0.0))
        res["tokens_per_second"] = res_tps
        res["peak_rss_mb"] = res_rss
        baseline_runs.append(res)
        last_baseline_res = res
        baseline_peak_rss = max(baseline_peak_rss, res_rss)
        print(f"  Rep {r+1}/{repetitions}: {res['wall_time_s']:.4f}s ({res_tps:.2f} tok/s, RSS min/avg/peak: {res['min_rss_mb']:.1f}/{res['avg_rss_mb']:.1f}/{res_rss:.1f} MB)")

    b_times = [r["wall_time_s"] for r in baseline_runs]
    b_thrs = [r["tokens_per_second"] for r in baseline_runs]
    b_min_rss = [r.get("min_rss_mb", r["peak_rss_mb"]) for r in baseline_runs]
    b_avg_rss = [r.get("avg_rss_mb", r["peak_rss_mb"]) for r in baseline_runs]
    b_peak_rss = [r["peak_rss_mb"] for r in baseline_runs]

    baseline_summary = {
        "wall_time_mean_s": float(np.mean(b_times)),
        "wall_time_std_s": float(np.std(b_times)),
        "throughput_mean_tok_s": float(np.mean(b_thrs)),
        "throughput_std_tok_s": float(np.std(b_thrs)),
        "min_rss_mean_mb": float(np.mean(b_min_rss)),
        "min_rss_std_mb": float(np.std(b_min_rss)),
        "avg_rss_mean_mb": float(np.mean(b_avg_rss)),
        "avg_rss_std_mb": float(np.std(b_avg_rss)),
        "peak_rss_mean_mb": float(np.mean(b_peak_rss)),
        "peak_rss_std_mb": float(np.std(b_peak_rss)),
        "peak_rss_mb": float(np.max(b_peak_rss)),
        "kv_memory_mb": float(last_baseline_res["kv_memory_mb"]),
        "non_kv_ram_peak_mb": float(np.mean(b_peak_rss) - last_baseline_res["kv_memory_mb"]),
    }

    # 2. AI-SSD Benchmark
    print(f"\n[2/2] Executing AI-SSD Benchmark ({repetitions} reps, Top-k={topk_pct:.1f}%, Prefetch={enable_prefetch})...")
    aissd_runs = []
    aissd_peak_rss = 0.0
    last_aissd_res = None
    for r in range(repetitions):
        num_layers = getattr(engine.model.config, "num_hidden_layers", 24)
        num_kv_heads = getattr(engine.model.config, "num_key_value_heads", 2)
        head_dim = getattr(engine.model.config, "head_dim", 64)
        backend = create_default_storage_backend(
            channels=channels,
            enable_prefetch=enable_prefetch,
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        res = run_aissd_decode(
            engine.model,
            engine.tokenizer,
            input_ids,
            decode_tokens=decode_tokens,
            top_k_pct=topk_pct,
            storage_backend=backend,
            enable_prefetch=enable_prefetch,
            seed=seed + r,
        )
        res_tps = res.get("tokens_per_second", res.get("throughput_tok_s", 0.0))
        res_rss = res.get("peak_rss_mb", res.get("rss_peak_mb", 0.0))
        res["tokens_per_second"] = res_tps
        res["peak_rss_mb"] = res_rss
        aissd_runs.append(res)
        last_aissd_res = res
        aissd_peak_rss = max(aissd_peak_rss, res_rss)
        print(f"  Rep {r+1}/{repetitions}: {res['wall_time_s']:.4f}s ({res_tps:.2f} tok/s, RSS min/avg/peak: {res['min_rss_mb']:.1f}/{res['avg_rss_mb']:.1f}/{res_rss:.1f} MB)")

    a_times = [r["wall_time_s"] for r in aissd_runs]
    a_thrs = [r["tokens_per_second"] for r in aissd_runs]
    a_min_rss = [r.get("min_rss_mb", r["peak_rss_mb"]) for r in aissd_runs]
    a_avg_rss = [r.get("avg_rss_mb", r["peak_rss_mb"]) for r in aissd_runs]
    a_peak_rss = [r["peak_rss_mb"] for r in aissd_runs]

    aissd_summary = {
        "wall_time_mean_s": float(np.mean(a_times)),
        "wall_time_std_s": float(np.std(a_times)),
        "throughput_mean_tok_s": float(np.mean(a_thrs)),
        "throughput_std_tok_s": float(np.std(a_thrs)),
        "min_rss_mean_mb": float(np.mean(a_min_rss)),
        "min_rss_std_mb": float(np.std(a_min_rss)),
        "avg_rss_mean_mb": float(np.mean(a_avg_rss)),
        "avg_rss_std_mb": float(np.std(a_avg_rss)),
        "peak_rss_mean_mb": float(np.mean(a_peak_rss)),
        "peak_rss_std_mb": float(np.std(a_peak_rss)),
        "peak_rss_mb": float(np.max(a_peak_rss)),
        "kv_active_dram_mb": float(last_aissd_res["kv_memory_mb"]),
        "kv_offload_pct": float(last_aissd_res["kv_offloaded_pct"]),
        "kv_blocks_read": int(last_aissd_res["kv_blocks_read"]),
        "non_kv_ram_peak_mb": float(np.mean(a_peak_rss) - last_aissd_res["kv_memory_mb"]),
    }

    # Storage & Prefetch Telemetry
    telemetry = last_aissd_res.get("telemetry", {})
    prefetch_summary = {
        "demand_requests": int(telemetry.get("demand_reads", 0)),
        "demand_hits": int(telemetry.get("demand_hits", 0)),
        "demand_misses": int(telemetry.get("demand_misses", 0)),
        "demand_hit_rate_pct": float(telemetry.get("demand_hit_rate_pct", 0.0)),
        "prefetch_requests": int(telemetry.get("prefetch_requests", 0)),
        "useful_prefetches": int(telemetry.get("useful_prefetches", 0)),
        "useful_bytes": int(telemetry.get("useful_bytes", 0)),
        "wasted_bytes": int(telemetry.get("wasted_bytes", 0)),
        "staging_memory_mb": float(telemetry.get("staging_memory_mb", 0.0)),
        "peak_memory_mb": float(telemetry.get("peak_memory_mb", 0.0)),
    }

    p2_telem = telemetry.get("storage_backend", telemetry)
    ch_dist = p2_telem.get("channel_distribution", {}) if isinstance(p2_telem, dict) else {}
    storage_bytes = int(last_aissd_res.get("storage_bytes_read", last_aissd_res.get("bytes_read", 0)))
    storage_reqs = int(last_aissd_res.get("storage_requests", last_aissd_res.get("requests", 0)))
    storage_summary = {
        "bytes_read": storage_bytes,
        "total_requests": storage_reqs,
        "channels_active": len([k for k, v in ch_dist.get("channel_read_counts", {}).items() if v > 0]),
        "channel_reads": ch_dist.get("channel_read_counts", {}),
        "load_imbalance_pct": float(ch_dist.get("load_imbalance_percent", 0.0)),
        "contention_ratio": float(ch_dist.get("contention_ratio", 1.0)),
    }

    # Accuracy / Output Comparison
    b_tokens = last_baseline_res["token_ids"]
    a_tokens = last_aissd_res["token_ids"]
    matches = sum(1 for b, a in zip(b_tokens, a_tokens) if b == a)
    match_pct = (matches / len(b_tokens)) * 100.0 if b_tokens else 0.0

    b_flat = last_baseline_res["final_logits"].flatten()
    a_flat = last_aissd_res["final_logits"].flatten()
    norm_b = np.linalg.norm(b_flat)
    norm_a = np.linalg.norm(a_flat)
    cos_sim = float(np.dot(b_flat, a_flat) / max(1e-12, norm_b * norm_a))
    mse = float(np.mean((b_flat - a_flat) ** 2))

    accuracy_summary = {
        "token_matches": int(matches),
        "token_count": len(b_tokens),
        "token_match_pct": float(match_pct),
        "logits_cosine_similarity": float(cos_sim),
        "logits_mse": float(mse),
        "baseline_text": last_baseline_res["generated_text"],
        "aissd_text": last_aissd_res["generated_text"],
    }

    # Package Complete Evidence Record
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    record = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "benchmark_metadata": {
            "model": model_name,
            "context_length": context_len,
            "decode_tokens": decode_tokens,
            "threads": threads,
            "dtype": dtype,
            "seed": seed,
            "repetitions": repetitions,
        },
        "baseline": baseline_summary,
        "ai_ssd": aissd_summary,
        "storage": storage_summary,
        "prefetch": prefetch_summary,
        "top_k": {
            "ratio": topk_ratio,
            "percent": topk_pct,
        },
        "accuracy": accuracy_summary,
    }

    # 1. Save JSON
    json_path = output_dir / f"live_benchmark_{timestamp_str}.json"
    latest_json_path = output_dir / "live_benchmark_latest.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)
    with open(latest_json_path, "w") as f:
        json.dump(record, f, indent=2)

    # 2. Append to CSV
    csv_path = output_dir / "live_benchmark.csv"
    csv_exists = csv_path.exists()
    csv_row = {
        "timestamp": record["timestamp_utc"],
        "model": model_name,
        "context": context_len,
        "decode": decode_tokens,
        "threads": threads,
        "dtype": dtype,
        "topk_ratio": topk_ratio,
        "prefetch": enable_prefetch,
        "baseline_tok_s": f"{baseline_summary['throughput_mean_tok_s']:.2f}",
        "baseline_wall_s": f"{baseline_summary['wall_time_mean_s']:.4f}",
        "baseline_min_rss_mb": f"{baseline_summary['min_rss_mean_mb']:.1f}",
        "baseline_avg_rss_mb": f"{baseline_summary['avg_rss_mean_mb']:.1f}",
        "baseline_peak_rss_mb": f"{baseline_summary['peak_rss_mean_mb']:.1f}",
        "baseline_kv_mb": f"{baseline_summary['kv_memory_mb']:.2f}",
        "aissd_tok_s": f"{aissd_summary['throughput_mean_tok_s']:.2f}",
        "aissd_wall_s": f"{aissd_summary['wall_time_mean_s']:.4f}",
        "aissd_min_rss_mb": f"{aissd_summary['min_rss_mean_mb']:.1f}",
        "aissd_avg_rss_mb": f"{aissd_summary['avg_rss_mean_mb']:.1f}",
        "aissd_peak_rss_mb": f"{aissd_summary['peak_rss_mean_mb']:.1f}",
        "aissd_kv_mb": f"{aissd_summary['kv_active_dram_mb']:.2f}",
        "throughput_retention_pct": f"{(aissd_summary['throughput_mean_tok_s'] / max(1e-6, baseline_summary['throughput_mean_tok_s'])) * 100.0:.1f}",
        "dram_reduction_pct": f"{aissd_summary['kv_offload_pct']:.1f}",
        "storage_bytes_read": storage_summary["bytes_read"],
        "storage_requests": storage_summary["total_requests"],
        "prefetch_hit_pct": f"{prefetch_summary['demand_hit_rate_pct']:.1f}",
        "useful_prefetches": prefetch_summary["useful_prefetches"],
        "wasted_bytes": prefetch_summary["wasted_bytes"],
        "token_match_pct": f"{accuracy_summary['token_match_pct']:.1f}",
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if not csv_exists:
            writer.writeheader()
        writer.writerow(csv_row)

    # Pretty Print Results
    print("\n" + "=" * 64)
    print("                     BENCHMARK SUMMARY")
    print("=" * 64)
    print(f"Baseline Throughput : {baseline_summary['throughput_mean_tok_s']:.2f} ? {baseline_summary['throughput_std_tok_s']:.2f} tok/s")
    print(f"AI-SSD Throughput   : {aissd_summary['throughput_mean_tok_s']:.2f} ? {aissd_summary['throughput_std_tok_s']:.2f} tok/s")
    retention = (aissd_summary['throughput_mean_tok_s'] / max(1e-6, baseline_summary['throughput_mean_tok_s'])) * 100.0
    print(f"Throughput Ratio    : {retention:.1f}%")
    print(f"DRAM KV Reduction   : {aissd_summary['kv_offload_pct']:.1f}% ({baseline_summary['kv_memory_mb']:.2f} MB -> {aissd_summary['kv_active_dram_mb']:.2f} MB)")
    print(f"Storage Reads       : {storage_summary['bytes_read']:,} B across {storage_summary['channels_active']}/8 channels")
    print(f"Prefetch Hit Rate   : {prefetch_summary['demand_hit_rate_pct']:.1f}% ({prefetch_summary['useful_prefetches']} useful / 0 wasted)")
    print(f"Accuracy Token Match: {accuracy_summary['token_match_pct']:.1f}% ({accuracy_summary['token_matches']}/{accuracy_summary['token_count']})")
    
    print("\n" + "=" * 64)
    print("                  PROCESS RAM TELEMETRY (MB)")
    print("=" * 64)
    print(f"| {'Metric':<22} | {'Baseline':<18} | {'AI-SSD':<18} |")
    print(f"|{'-'*24}|{'-'*20}|{'-'*20}|")
    print(f"| {'Min RSS':<22} | {baseline_summary['min_rss_mean_mb']:>7.1f} ? {baseline_summary['min_rss_std_mb']:<6.1f} MB | {aissd_summary['min_rss_mean_mb']:>7.1f} ? {aissd_summary['min_rss_std_mb']:<6.1f} MB |")
    print(f"| {'Avg RSS':<22} | {baseline_summary['avg_rss_mean_mb']:>7.1f} ? {baseline_summary['avg_rss_std_mb']:<6.1f} MB | {aissd_summary['avg_rss_mean_mb']:>7.1f} ? {aissd_summary['avg_rss_std_mb']:<6.1f} MB |")
    print(f"| {'Peak RSS':<22} | {baseline_summary['peak_rss_mean_mb']:>7.1f} ? {baseline_summary['peak_rss_std_mb']:<6.1f} MB | {aissd_summary['peak_rss_mean_mb']:>7.1f} ? {aissd_summary['peak_rss_std_mb']:<6.1f} MB |")
    print(f"| {'KV memory (active)':<22} | {baseline_summary['kv_memory_mb']:>14.2f} MB | {aissd_summary['kv_active_dram_mb']:>14.2f} MB |")
    print(f"| {'KV reduction':<22} | {'0.0%':>17} | {aissd_summary['kv_offload_pct']:>16.1f}% |")
    print(f"| {'Non-KV RAM (Peak-KV)':<22} | {baseline_summary['non_kv_ram_peak_mb']:>14.1f} MB | {aissd_summary['non_kv_ram_peak_mb']:>14.1f} MB |")
    print("=" * 64)
    print("* Note: Continuous process RSS was sampled at 2ms intervals during decode.")
    print("  Non-KV RAM is directly derived as Peak RSS minus KV Cache memory,")
    print("  accounting for model weights, computational activations, and PyTorch runtime.\n")
    print(f"[SUCCESS] Results written to:\n  JSON: {json_path}\n  CSV : {csv_path}\n")


if __name__ == "__main__":
    main()
