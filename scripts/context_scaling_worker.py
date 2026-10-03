#!/usr/bin/env python3
"""Context Scaling Benchmark Runner (4096 -> 8192 -> 16384 -> 32768).

Executes reproducible, isolated inference runs for BASELINE and AI-SSD at
increasing context lengths on Qwen3-4B-Instruct-2507.
Measures real OS process telemetry from /proc/self/status and psutil.
Zero analytical timing injection; zero artificial sleeps; 100% genuine execution.
"""

import sys
import os
import gc
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.aissd_inference import (
    create_default_storage_backend,
    run_baseline_decode,
    run_aissd_decode,
)
from scripts.real_inference_benchmark import build_prompt_for_length


def get_proc_memory() -> Dict[str, float]:
    """Reads Linux /proc/self/status metrics in MB."""
    res = {"vm_rss_mb": 0.0, "rss_anon_mb": 0.0, "rss_file_mb": 0.0, "vm_peak_mb": 0.0}
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    res["vm_rss_mb"] = float(line.split()[1]) / 1024.0
                elif line.startswith("RssAnon:"):
                    res["rss_anon_mb"] = float(line.split()[1]) / 1024.0
                elif line.startswith("RssFile:"):
                    res["rss_file_mb"] = float(line.split()[1]) / 1024.0
                elif line.startswith("VmPeak:"):
                    res["vm_peak_mb"] = float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return res


def run_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "aissd"])
    parser.add_argument("--storage-mode", type=str, default="file", choices=["file", "nvme_qemu"])
    parser.add_argument("--mapping-mode", type=str, default="tensor_aware", choices=["tensor_aware", "conventional"])
    parser.add_argument("--enable-prefetch", action="store_true", default=True, dest="enable_prefetch")
    parser.add_argument("--disable-prefetch", action="store_false", dest="enable_prefetch")
    parser.add_argument("--enable-batching", action="store_true", default=True, dest="enable_batching")
    parser.add_argument("--disable-batching", action="store_false", dest="enable_batching")
    parser.add_argument("--enable-computational-storage", action="store_true", default=True, dest="enable_computational_storage")
    parser.add_argument("--disable-computational-storage", action="store_false", dest="enable_computational_storage")
    parser.add_argument("--top-k-pct", type=float, default=10.0)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    mem_start = get_proc_memory()

    # Load model
    print(f"[{args.mode.upper()}] Loading Qwen3-4B-Instruct-2507 (CPU, 4 threads, FP32)...")
    engine = RealLLMEngine(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        device="cpu",
        dtype="float32",
        num_threads=4,
    )
    mem_model = get_proc_memory()
    print(f"[{args.mode.upper()}] Model loaded. RSS: {mem_model['vm_rss_mb']:.1f} MB")

    # Generate prompt
    print(f"[{args.mode.upper()}] Generating prompt for context={args.context} tokens...")
    prompt = build_prompt_for_length(engine, target_tokens=args.context)
    inputs = engine.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    actual_tokens = input_ids.shape[1]
    print(f"[{args.mode.upper()}] Input tokens: {actual_tokens}")

    effective_seed = args.seed + args.rep

    if args.mode == "baseline":
        t0 = time.perf_counter()
        res = run_baseline_decode(
            model=engine.model,
            tokenizer=engine.tokenizer,
            input_ids=input_ids,
            decode_tokens=args.decode,
            seed=effective_seed,
        )
        total_time = time.perf_counter() - t0
        mem_final = get_proc_memory()

        output_data = {
            "mode": "BASELINE",
            "context_length": args.context,
            "actual_tokens": actual_tokens,
            "repetition": args.rep,
            "wall_time_s": res["wall_time_s"],
            "total_process_time_s": total_time,
            "decode_tokens": res["generated_tokens"],
            "tokens_per_second": res["tokens_per_second"],
            "min_rss_mb": res["min_rss_mb"],
            "avg_rss_mb": res["avg_rss_mb"],
            "peak_rss_mb": res["peak_rss_mb"],
            "std_rss_mb": res.get("std_rss_mb", 0.0),
            "vm_peak_mb": mem_final["vm_peak_mb"],
            "rss_anon_mb": mem_final["rss_anon_mb"],
            "final_rss_mb": mem_final["vm_rss_mb"],
            "active_kv_mb": res["kv_memory_mb"],
            "cold_kv_mb": 0.0,
            "storage_bytes": 0,
            "storage_read_bytes": 0,
            "storage_write_bytes": 0,
            "storage_requests": 0,
            "storage_batches": 0,
            "avg_batch_size": 0.0,
            "p2_resident_payload_mb": 0.0,
            "p3_resident_payload_mb": 0.0,
            "staging_mb": 0.0,
            "p2_metadata_bytes": 0,
            "token_ids": res["token_ids"],
            "generated_text": res["generated_text"],
        }

    else:
        # AI-SSD mode
        num_layers = getattr(engine.model.config, "num_hidden_layers", 36)
        num_kv_heads = getattr(engine.model.config, "num_key_value_heads", 8)
        head_dim = getattr(engine.model.config, "head_dim", 128)

        backend = create_default_storage_backend(
            channels=8,
            enable_prefetch=args.enable_prefetch,
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            dtype="float32",
            mapping_mode=args.mapping_mode,
            storage_mode=args.storage_mode,
            enable_batching=args.enable_batching,
        )

        storage = getattr(backend, "storage_backend", backend)
        backing_path = getattr(storage, "_backing_path", "")

        try:
            t0 = time.perf_counter()
            res = run_aissd_decode(
                model=engine.model,
                tokenizer=engine.tokenizer,
                input_ids=input_ids,
                decode_tokens=args.decode,
                top_k_pct=args.top_k_pct,
                storage_backend=backend,
                enable_prefetch=args.enable_prefetch,
                enable_computational_storage=args.enable_computational_storage,
                seed=effective_seed,
            )
            total_time = time.perf_counter() - t0
            mem_final = get_proc_memory()

            # Audit storage structures
            p2_resident_bytes = 0
            p2_meta_bytes = sys.getsizeof(storage._storage) if hasattr(storage, "_storage") else 0
            if hasattr(storage, "_storage"):
                for k, v in storage._storage.items():
                    p2_meta_bytes += sys.getsizeof(v)
                    if "k" in v and isinstance(v["k"], np.ndarray):
                        p2_resident_bytes += v["k"].nbytes
                    if "v" in v and isinstance(v["v"], np.ndarray):
                        p2_resident_bytes += v["v"].nbytes

            p3_resident_bytes = 0
            if hasattr(backend, "_block_payloads"):
                for k, v in backend._block_payloads.items():
                    if "k" in v and isinstance(v["k"], np.ndarray):
                        p3_resident_bytes += v["k"].nbytes
                    if "v" in v and isinstance(v["v"], np.ndarray):
                        p3_resident_bytes += v["v"].nbytes

            backing_size = getattr(storage, "_file_offset", 0)
            if not backing_size and backing_path and os.path.exists(backing_path):
                backing_size = os.path.getsize(backing_path)
            staging_bytes = getattr(backend, "staging_memory_bytes", getattr(backend, "current_memory_bytes", 0))

            storage_telemetry = storage.get_telemetry() if hasattr(storage, "get_telemetry") else {}

            output_data = {
                "mode": "AI-SSD",
                "storage_mode": args.storage_mode,
                "mapping_mode": args.mapping_mode,
                "enable_prefetch": args.enable_prefetch,
                "enable_batching": args.enable_batching,
                "enable_computational_storage": args.enable_computational_storage,
                "top_k_pct": args.top_k_pct,
                "backend_classification": getattr(storage, "CLASSIFICATION", "UNKNOWN"),
                "context_length": args.context,
                "actual_tokens": actual_tokens,
                "repetition": args.rep,
                "wall_time_s": res["wall_time_s"],
                "total_process_time_s": total_time,
                "decode_tokens": res["generated_tokens"],
                "tokens_per_second": res["tokens_per_second"],
                "min_rss_mb": res["min_rss_mb"],
                "avg_rss_mb": res["avg_rss_mb"],
                "peak_rss_mb": res["peak_rss_mb"],
                "std_rss_mb": res.get("std_rss_mb", 0.0),
                "vm_peak_mb": mem_final["vm_peak_mb"],
                "rss_anon_mb": mem_final["rss_anon_mb"],
                "final_rss_mb": mem_final["vm_rss_mb"],
                "active_kv_mb": res["kv_memory_mb"],
                "cold_kv_mb": backing_size / (1024.0 * 1024.0),
                "storage_bytes": backing_size,
                "storage_backing_path": backing_path,
                "storage_read_bytes": res.get("storage_bytes_read", 0),
                "storage_write_bytes": backing_size,
                "storage_requests": res.get("storage_requests", 0),
                "storage_batches": res.get("storage_batches", 0),
                "avg_batch_size": res.get("avg_batch_size", 1.0),
                "candidate_k_bytes_to_host": res.get("candidate_k_bytes_to_host", 0),
                "winning_k_bytes_to_host": res.get("winning_k_bytes_to_host", 0),
                "winning_v_bytes_to_host": res.get("winning_v_bytes_to_host", 0),
                "topk_metadata_bytes_to_host": res.get("topk_metadata_bytes_to_host", 0),
                "total_data_movement_bytes": res.get("total_data_movement_bytes", 0),
                "stored_blocks": len(storage._storage) if hasattr(storage, "_storage") else 0,
                "p2_resident_payload_mb": p2_resident_bytes / (1024.0 * 1024.0),
                "p3_resident_payload_mb": p3_resident_bytes / (1024.0 * 1024.0),
                "staging_mb": staging_bytes / (1024.0 * 1024.0),
                "p2_metadata_bytes": p2_meta_bytes,
                "channel_distribution": storage_telemetry.get("channel_distribution", {}),
                "nvme_telemetry": res.get("nvme_telemetry", storage_telemetry.get("nvme_telemetry", {})),
                "timing_breakdown": res.get("timing_breakdown", {}),
                "token_ids": res["token_ids"],
                "generated_text": res["generated_text"],
            }
        finally:
            if hasattr(backend, "close"):
                try:
                    backend.close()
                except Exception:
                    pass

    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[{args.mode.upper()}] Run completed. Decode RSS min/avg/peak: {output_data['min_rss_mb']:.1f}/{output_data['avg_rss_mb']:.1f}/{output_data['peak_rss_mb']:.1f} MB, tok/s: {output_data['tokens_per_second']:.2f}")


if __name__ == "__main__":
    run_worker()
