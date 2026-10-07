#!/usr/bin/env python3
"""
AI-SSD Full Correctness Regression Matrix across Supported Models.
Produces standardized Token-ID based comparison table and storage telemetry.
"""

import sys
import os
import json
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.adapters.registry import ModelRegistry
from person1_kv_engine.adapters.descriptor import CompatibilityLevel
from person1_kv_engine.real_llm.aissd_inference import (
    create_default_storage_backend,
    run_baseline_decode,
    run_aissd_decode,
)

MODELS = [
    {"name": "qwen2.5-0.5b", "id": "Qwen/Qwen2.5-0.5B", "dtype": "float32", "decode": 16},
    {"name": "tiny-mistral", "id": "openaccess-ai-collective/tiny-mistral", "dtype": "float32", "decode": 16},
    {"name": "qwen3-4b", "id": "Qwen/Qwen3-4B-Instruct-2507", "dtype": "float32", "decode": 16},
    {"name": "qwen3-8b", "id": "Qwen/Qwen3-8B", "dtype": "float16", "decode": 16},
    {"name": "qwen3.5-9b", "id": "Qwen/Qwen3.5-9B", "dtype": "float16", "decode": 8},
    {"name": "mistral-7b", "id": "mistralai/Mistral-7B-v0.1", "dtype": "float16", "decode": 16},
    {"name": "jamba", "id": "ai21labs/AI21-Jamba-1.5-Mini", "dtype": "float16", "decode": 16},
]

PROMPT = "The PCI Express (PCIe) standard defines high-speed serial computer expansion bus technology for high performance solid state drive storage controllers."

def run_matrix():
    results = []

    print("=" * 110)
    print("                AI-SSD FORMAL CORRECTNESS REGRESSION MATRIX ACROSS SUPPORTED MODELS")
    print("=" * 110)
    print(f"Deterministic Prompt: {repr(PROMPT)}")
    print(f"CPU Threads: 4 | Seed: 42 | Storage: AI-SSD FTL / QEMU NVMe (Top-K Filter Enabled)")
    print("-" * 110)

    for entry in MODELS:
        m_name = entry["name"]
        m_id = entry["id"]
        dtype_str = entry["dtype"]
        decode_len = entry["decode"]

        print(f"\n[*] Evaluating Model: {m_name} ({m_id}, {dtype_str.upper()})...", flush=True)

        # Check weights availability
        try:
            cfg_obj, comp_level, comp_reason = ModelRegistry.detect_model_config(m_id)
        except Exception as e:
            print(f"    [SKIP] Configuration detection failed: {e}")
            results.append({
                "model": m_name,
                "status": "NOT_AVAILABLE",
                "reason": str(e),
            })
            continue

        try:
            engine = RealLLMEngine(model_name=m_id, device="cpu", dtype=dtype_str, num_threads=4)
        except Exception as e:
            print(f"    [SKIP] Model weights not available or gated: {e}")
            results.append({
                "model": m_name,
                "status": "WEIGHTS_UNAVAILABLE",
                "reason": "Weights not cached locally / gating required",
            })
            continue

        input_ids = engine.tokenizer(PROMPT, return_tensors="pt")["input_ids"]

        # Run Baseline (Firmware Disabled)
        print("    Running Baseline (Firmware Disabled)...", flush=True)
        b_res = run_baseline_decode(
            engine.model, engine.tokenizer, input_ids,
            decode_tokens=decode_len, seed=42, show_progress=False, enforce_english=False
        )

        # Run AI-SSD (Firmware Enabled)
        print("    Running AI-SSD (Firmware Enabled)...", flush=True)
        adapter = ModelRegistry.get_adapter(cfg_obj)
        backend = create_default_storage_backend(
            channels=8,
            num_layers=engine.num_layers,
            num_heads=engine.num_kv_heads,
            head_dim=engine.head_dim,
            dtype=dtype_str,
            mapping_mode="tensor_aware",
            storage_mode="file",
            enable_batching=True,
        )
        a_res = run_aissd_decode(
            engine.model, engine.tokenizer, input_ids,
            decode_tokens=decode_len, top_k_pct=10.0, storage_backend=backend,
            model_adapter=adapter,
            enable_prefetch=True, enable_computational_storage=True, seed=42, show_progress=False, enforce_english=False
        )

        match = (b_res["token_ids"] == a_res["token_ids"])
        b_tokens = b_res["token_ids"]
        a_tokens = a_res["token_ids"]

        rec = {
            "model": m_name,
            "status": "EVALUATED",
            "match": match,
            "tokens_count": len(b_tokens),
            "baseline_tokens": b_tokens,
            "aissd_tokens": a_tokens,
            "baseline_text": b_res["generated_text"],
            "aissd_text": a_res["generated_text"],
            "baseline_tps": round(b_res["tokens_per_second"], 2),
            "aissd_tps": round(a_res["tokens_per_second"], 2),
            "baseline_peak_rss_mb": round(b_res["peak_rss_mb"], 1),
            "aissd_peak_rss_mb": round(a_res["peak_rss_mb"], 1),
            "baseline_read_mb": round(b_res.get("total_read_mb", 0.0), 2),
            "baseline_write_mb": round(b_res.get("total_write_mb", 0.0), 2),
            "aissd_read_mb": round(a_res.get("total_read_mb", 0.0), 2),
            "aissd_write_mb": round(a_res.get("total_write_mb", 0.0), 2),
            "candidate_k_bytes_to_host": a_res.get("candidate_k_bytes_to_host", 0),
        }
        results.append(rec)

        print(f"    Result: {'EXACT MATCH (PASS)' if match else 'DIVERGED'}")
        print(f"    Baseline Tokens: {b_tokens}")
        print(f"    AI-SSD   Tokens: {a_tokens}")
        print(f"    Baseline Text  : {repr(b_res['generated_text'][:60])}")
        print(f"    AI-SSD   Text  : {repr(a_res['generated_text'][:60])}")
        print(f"    Storage I/O    : Read={a_res.get('total_read_mb', 0.0):.2f} MB, Write={a_res.get('total_write_mb', 0.0):.2f} MB, Cand-K={a_res.get('candidate_k_bytes_to_host', 0)} B")

        del engine
        del backend
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time.sleep(1)

    # Print Standard Table
    print("\n" + "=" * 120)
    print(f"{'Model':<16} | {'Firmware':<10} | {'Token IDs (First 8)':<36} | {'Match':<7} | {'Total Read':<11} | {'Total Write'}")
    print("-" * 120)
    for r in results:
        m = r["model"]
        if r["status"] != "EVALUATED":
            print(f"{m:<16} | {'N/A':<10} | {r['reason']:<36} | {'SKIP':<7} | {'-':<11} | {'-'}")
            continue

        b_t_str = str(r["baseline_tokens"][:8])
        a_t_str = str(r["aissd_tokens"][:8])
        match_str = "PASS" if r["match"] else "FAIL"

        print(f"{m:<16} | {'Disabled':<10} | {b_t_str:<36} | {match_str:<7} | {r['baseline_read_mb']:>7.2f} MB | {r['baseline_write_mb']:>7.2f} MB")
        print(f"{'':<16} | {'Enabled':<10} | {a_t_str:<36} | {match_str:<7} | {r['aissd_read_mb']:>7.2f} MB | {r['aissd_write_mb']:>7.2f} MB")
        print("-" * 120)

    # Save to json
    out_path = Path(PROJECT_ROOT) / "results" / "correctness_regression_matrix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Matrix results saved to: {out_path}\n")

if __name__ == "__main__":
    run_matrix()
