#!/usr/bin/env python3
"""
AI-SSD V2 — Automated Quick Comparison & Staged Comprehensive Benchmark Runner

Executes both BASELINE (in-DRAM) and AI-SSD (storage offload + Top-K)
across models, precisions, and context lengths:
    2048 (2K) -> 4096 (4K) -> 8192 (8K) -> 16384 (16K) -> 32768 (32K)

Supports:
- Single execution (legacy CLI interface preserved):
    python scripts/run_quick_comparison.py --model Qwen/Qwen2.5-0.5B --context 512
- Staged full matrix execution:
    python scripts/run_quick_comparison.py --all
- Single stage execution:
    python scripts/run_quick_comparison.py --stage 2048
- Smoke verification test:
    python scripts/run_quick_comparison.py --smoke
- Matrix inspection without running:
    python scripts/run_quick_comparison.py --list-matrix
"""

from __future__ import annotations
import sys
import os
import time
import json
import uuid
import argparse
import tempfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.live_inference.result_schema import (
    STAGE_SPECS,
    build_benchmark_record,
    save_benchmark_result,
    save_quick_comparison_record,
    load_current_benchmark_records,
    load_historical_benchmark_records,
    resolve_benchmark_matrix,
    archive_current_benchmark,
    get_known_models_catalog,
)
from person1_kv_engine.adapters.registry import ModelRegistry
from person1_kv_engine.adapters.descriptor import CompatibilityLevel


def ensure_c_kernel_built():
    """Builds instorage_attention.so if running on Linux and missing."""
    if sys.platform != "win32":
        so_path = PROJECT_ROOT / "person1_kv_engine" / "c_kernel" / "instorage_attention.so"
        if not so_path.exists():
            print("[INFO] Compiling Linux native SIMD kernel (instorage_attention.so)...")
            compile_script = PROJECT_ROOT / "person1_kv_engine" / "c_kernel" / "compile_kernel.py"
            try:
                subprocess.run([sys.executable, str(compile_script)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                pass


def get_git_commit() -> str:
    """Retrieves current git commit hash if available."""
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def is_model_cached(model_id: str) -> bool:
    """Checks whether the specified Hugging Face model has non-empty cached weights."""
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    folder_name = "models--" + model_id.replace("/", "--")
    model_path = hub_dir / folder_name
    if not model_path.exists():
        return False
    snapshots = model_path / "snapshots"
    if not snapshots.exists():
        return False
    try:
        for snap in snapshots.iterdir():
            if snap.is_dir() and any(snap.iterdir()):
                return True
    except Exception:
        return False
    return False


def run_worker_process(
    mode: str,
    model_name: str,
    prompt_tokens: int,
    decode: int,
    threads: int,
    storage_mode: str,
    output_file: str,
    precision: str = "fp32",
    timeout_s: int = 1800,
) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    """
    Executes context_scaling_worker.py in an isolated child process.
    Returns (returncode, error_message, result_dict_or_None).
    """
    worker_script = PROJECT_ROOT / "scripts" / "context_scaling_worker.py"
    dtype_str = "float16" if "16" in precision.lower() else "float32"
    cmd = [
        sys.executable,
        str(worker_script),
        "--mode", mode,
        "--model-name", model_name,
        "--dtype", dtype_str,
        "--context", str(prompt_tokens),
        "--decode", str(decode),
        "--threads", str(threads),
        "--output-json", output_file,
        "--disable-progress",
    ]
    if mode == "aissd":
        cmd.extend([
            "--storage-mode", storage_mode,
            "--top-k-pct", "10.0",
            "--enable-computational-storage",
        ])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        retcode = proc.returncode
        stderr = proc.stderr
        stdout = proc.stdout
    except subprocess.TimeoutExpired:
        return -1, f"TimeoutExpired ({timeout_s}s exceeded)", None
    except Exception as e:
        return -1, str(e), None

    if retcode != 0:
        err = stderr.strip() if stderr.strip() else stdout.strip()
        # Check for OOM indicators
        if retcode == 137 or retcode == -9 or "MemoryError" in err or "OutOfMemory" in err:
            return 137, f"OOM: Process terminated by system out-of-memory killer ({err[:200]})", None
        return retcode, f"Process failed with exit code {retcode}: {err[:300]}", None

    if not os.path.exists(output_file):
        return -1, "Worker process exited with code 0 but produced no JSON output file", None

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return 0, "", data
    except Exception as e:
        return -1, f"Failed to parse worker output JSON: {e}", None


def print_comparison(baseline_json: str, aissd_json: str, model_name: str):
    """Prints the rich console side-by-side comparison table."""
    with open(baseline_json, "r") as f:
        b = json.load(f)
    with open(aissd_json, "r") as f:
        a = json.load(f)

    b_load_rss = b.get("model_load_peak_rss_mb", 0.0)
    a_load_rss = a.get("model_load_peak_rss_mb", 0.0)
    b_prefill_rss = b.get("prefill_peak_rss_mb", b.get("peak_rss_mb", 0.0))
    a_prefill_rss = a.get("prefill_peak_rss_mb", a.get("peak_rss_mb", 0.0))
    b_post_rss = b.get("post_prefill_rss_mb", b.get("min_rss_mb", 0.0))
    a_post_rss = a.get("post_prefill_rss_mb", a.get("min_rss_mb", 0.0))
    b_dec_rss = b.get("decode_peak_rss_mb", b.get("peak_rss_mb", 0.0))
    a_dec_rss = a.get("decode_peak_rss_mb", a.get("peak_rss_mb", 0.0))
    b_peak_rss = b.get("overall_peak_rss_mb", b.get("peak_rss_mb", 0.0))
    a_peak_rss = a.get("overall_peak_rss_mb", a.get("peak_rss_mb", 0.0))

    b_act_kv = b.get("active_kv_mb", 0.0)
    a_act_kv = a.get("active_kv_mb", 0.0)
    b_cold_kv = b.get("cold_kv_mb", 0.0)
    a_cold_kv = a.get("cold_kv_mb", 0.0)
    kv_red = ((b_act_kv - a_act_kv) / max(1e-6, b_act_kv)) * 100.0 if b_act_kv > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"             AI-SSD vs BASELINE BENCHMARK COMPARISON ({model_name})           ")
    print("=" * 80)
    print(f"{'Metric':<32} | {'Baseline (Without AI-SSD)':<20} | {'AI-SSD (With Offload)'}")
    print("-" * 80)
    print(f"--- Process Memory Telemetry ---")
    if b_load_rss > 0 or a_load_rss > 0:
        print(f"{'Model Load Peak RSS':<32} | {b_load_rss:>17.1f} MB | {a_load_rss:>16.1f} MB")
    print(f"{'Prefill Peak RSS':<32} | {b_prefill_rss:>17.1f} MB | {a_prefill_rss:>16.1f} MB")
    print(f"{'Post-Prefill RSS':<32} | {b_post_rss:>17.1f} MB | {a_post_rss:>16.1f} MB")
    print(f"{'Decode Peak RSS':<32} | {b_dec_rss:>17.1f} MB | {a_dec_rss:>16.1f} MB")
    print(f"{'Overall Peak RSS':<32} | {b_peak_rss:>17.1f} MB | {a_peak_rss:>16.1f} MB")
    print("-" * 80)
    print(f"--- KV Cache Telemetry ---")
    print(f"{'Active KV in Host RAM':<32} | {b_act_kv:>17.2f} MB | {a_act_kv:>16.2f} MB")
    print(f"{'Cold KV on Disk':<32} | {b_cold_kv:>17.2f} MB | {a_cold_kv:>16.2f} MB")
    print(f"{'KV RAM Reduction':<32} | {'0.0%':>20} | {kv_red:>19.1f}%")
    print("-" * 80)
    print(f"--- Performance & Throughput ---")
    b_prefill_s = b.get("prefill_time_s", 0.0)
    a_prefill_s = a.get("prefill_time_s", 0.0)
    if b_prefill_s > 0 or a_prefill_s > 0:
        print(f"{'Prefill Wall Time':<32} | {b_prefill_s:>17.3f} s  | {a_prefill_s:>16.3f} s")
    print(f"{'Decode Wall Time':<32} | {b.get('wall_time_s', 0):>17.3f} s  | {a.get('wall_time_s', 0):>16.3f} s")
    b_tps = b.get("tokens_per_second", 0)
    a_tps = a.get("tokens_per_second", 0)
    print(f"{'Decode Throughput':<32} | {b_tps:>15.2f} tok/s | {a_tps:>14.2f} tok/s")
    print("-" * 80)
    print(f"--- Storage & Hardware Bus ---")
    b_read_mb = b.get("total_read_mb", 0.0)
    a_read_mb = a.get("total_read_mb", round(a.get("storage_read_bytes", 0) / (1024.0 * 1024.0), 2))
    b_write_mb = b.get("total_write_mb", 0.0)
    a_write_mb = a.get("total_write_mb", round(a.get("storage_write_bytes", 0) / (1024.0 * 1024.0), 2))

    print(f"{'Total Read MB':<32} | {b_read_mb:>17.2f} MB | {a_read_mb:>16.2f} MB")
    print(f"{'Total Write MB':<32} | {b_write_mb:>17.2f} MB | {a_write_mb:>16.2f} MB")

    cand_bytes = a.get("candidate_k_bytes_to_host", 0)
    cand_str = f"{cand_bytes} B (Zero-Bus)" if cand_bytes == 0 else f"{cand_bytes} B"
    print(f"{'Candidate K to Host Bus':<32} | {'0 B (Host RAM)':>20} | {cand_str:>20}")
    print("=" * 80)

    # Verification
    b_tokens = b.get("token_ids", [])
    a_tokens = a.get("token_ids", [])
    if b_tokens and a_tokens:
        matches = sum(1 for x, y in zip(b_tokens, a_tokens) if x == y)
        total = min(len(b_tokens), len(a_tokens))
        pct = (matches / total) * 100.0 if total > 0 else 0.0
        status = "EXACT MATCH (PASS)" if matches == total else f"DIVERGENT ({matches}/{total})"
        print(f"Token Parity       : {matches}/{total} tokens identical ({pct:.1f}% match) [{status}]")
    print(f"Baseline Generated : {repr(b.get('generated_text', ''))[:80]}")
    print(f"AI-SSD Generated   : {repr(a.get('generated_text', ''))[:80]}")
    print("=" * 80 + "\n")


def make_comparison_record(
    stage_ctx: int,
    stage_name: str,
    model_key: str,
    model_info: Dict[str, Any],
    prec_req: str,
    status: str,
    status_reason: str,
    prompt_tokens: int,
    decode_tokens: int,
    threads: int,
    seed: int,
    b_data: Optional[Dict[str, Any]] = None,
    a_data: Optional[Dict[str, Any]] = None,
    storage_mode: str = "nvme_qemu",
    git_commit: str = "unknown",
    generation_id: str = "",
) -> Dict[str, Any]:
    """Constructs a normalized quick_comparison record matching CSV_FIELDNAMES and schema."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cfg_key = f"{model_key}|{prec_req.lower()}|{stage_ctx}"
    m_id = model_info.get("model_id", model_key)
    arch = model_info.get("architecture", "unknown")
    family = model_info.get("model_family", "transformer")
    params = model_info.get("params", "N/A")

    rec = {
        "config_key": cfg_key,
        "stage": int(stage_ctx),
        "stage_name": stage_name,
        "model_key": model_key,
        "model_id": m_id,
        "architecture": arch,
        "model_family": family,
        "params": params,
        "precision_requested": prec_req.lower(),
        "precision_actual": prec_req.lower(),
        "requested_context": int(stage_ctx),
        "actual_context": int(stage_ctx),
        "prompt_tokens": int(prompt_tokens),
        "decode_tokens": int(decode_tokens),
        "threads": int(threads),
        "seed": int(seed),
        "status": status,
        "status_reason": status_reason,
        "result_source": "CURRENT",
        "benchmark_generation_id": generation_id or "gen_manual",
        "benchmark_timestamp": now_utc,
        "git_commit": git_commit,
        "exact_token_match": None,
        "token_match_rate": 0.0,
        "matching_tokens": 0,
        "total_tokens": 0,
        "first_divergent_token": None,
        "candidate_k_zero_bus": None,
        "candidate_k_bytes_to_host": 0,
        "winning_k_bytes_to_host": 0,
        "winning_v_bytes_to_host": 0,
        "baseline_wall_time_s": 0.0,
        "baseline_prefill_time_s": 0.0,
        "baseline_decode_time_s": 0.0,
        "baseline_tps": 0.0,
        "baseline_peak_rss_mb": 0.0,
        "baseline_post_prefill_rss_mb": 0.0,
        "baseline_decode_rss_mb": 0.0,
        "baseline_active_kv_mb": 0.0,
        "baseline_read_mb": 0.0,
        "baseline_write_mb": 0.0,
        "aissd_wall_time_s": 0.0,
        "aissd_prefill_time_s": 0.0,
        "aissd_decode_time_s": 0.0,
        "aissd_tps": 0.0,
        "aissd_peak_rss_mb": 0.0,
        "aissd_post_prefill_rss_mb": 0.0,
        "aissd_decode_rss_mb": 0.0,
        "aissd_active_kv_mb": 0.0,
        "aissd_cold_kv_mb": 0.0,
        "aissd_read_mb": 0.0,
        "aissd_write_mb": 0.0,
        "memory_saved_mb": 0.0,
        "kv_memory_saved_mb": 0.0,
        "kv_memory_reduction_pct": 0.0,
        "storage_mode": storage_mode,
    }

    if status in ("PASS", "FAIL") and b_data and a_data:
        b_tokens = b_data.get("token_ids", [])
        a_tokens = a_data.get("token_ids", [])
        matching = sum(1 for x, y in zip(b_tokens, a_tokens) if x == y)
        total = min(len(b_tokens), len(a_tokens))
        match_rate = (matching / total) * 100.0 if total > 0 else 0.0
        exact_match = (matching == total and total > 0)
        first_div = None
        for i, (x, y) in enumerate(zip(b_tokens, a_tokens)):
            if x != y:
                first_div = i
                break

        b_peak_rss = float(b_data.get("peak_rss_mb", 0.0))
        a_peak_rss = float(a_data.get("peak_rss_mb", 0.0))
        b_act_kv = float(b_data.get("active_kv_mb", 0.0))
        a_act_kv = float(a_data.get("active_kv_mb", 0.0))
        b_act_toks = int(b_data.get("actual_tokens", prompt_tokens))
        b_dec_toks = int(b_data.get("decode_tokens", decode_tokens))

        cand_bytes = int(a_data.get("candidate_k_bytes_to_host", 0))
        win_k = int(a_data.get("winning_k_bytes_to_host", 0))
        win_v = int(a_data.get("winning_v_bytes_to_host", 0))

        rec.update({
            "actual_context": b_act_toks + b_dec_toks,
            "prompt_tokens": b_act_toks,
            "decode_tokens": b_dec_toks,
            "exact_token_match": exact_match,
            "token_match_rate": round(match_rate, 2),
            "matching_tokens": matching,
            "total_tokens": total,
            "first_divergent_token": first_div,
            "candidate_k_zero_bus": (cand_bytes == 0),
            "candidate_k_bytes_to_host": cand_bytes,
            "winning_k_bytes_to_host": win_k,
            "winning_v_bytes_to_host": win_v,
            "baseline_wall_time_s": round(float(b_data.get("wall_time_s", 0.0)), 3),
            "baseline_prefill_time_s": round(float(b_data.get("prefill_time_s", 0.0)), 3),
            "baseline_decode_time_s": round(float(b_data.get("decode_time_s", b_data.get("wall_time_s", 0.0))), 3),
            "baseline_tps": round(float(b_data.get("tokens_per_second", 0.0)), 3),
            "baseline_peak_rss_mb": round(b_peak_rss, 2),
            "baseline_post_prefill_rss_mb": round(float(b_data.get("post_prefill_rss_mb", b_data.get("min_rss_mb", 0.0))), 2),
            "baseline_decode_rss_mb": round(float(b_data.get("decode_peak_rss_mb", b_peak_rss)), 2),
            "baseline_active_kv_mb": round(b_act_kv, 2),
            "baseline_read_mb": round(float(b_data.get("total_read_mb", 0.0)), 2),
            "baseline_write_mb": round(float(b_data.get("total_write_mb", 0.0)), 2),
            "aissd_wall_time_s": round(float(a_data.get("wall_time_s", 0.0)), 3),
            "aissd_prefill_time_s": round(float(a_data.get("prefill_time_s", 0.0)), 3),
            "aissd_decode_time_s": round(float(a_data.get("decode_time_s", a_data.get("wall_time_s", 0.0))), 3),
            "aissd_tps": round(float(a_data.get("tokens_per_second", 0.0)), 3),
            "aissd_peak_rss_mb": round(a_peak_rss, 2),
            "aissd_post_prefill_rss_mb": round(float(a_data.get("post_prefill_rss_mb", a_data.get("min_rss_mb", 0.0))), 2),
            "aissd_decode_rss_mb": round(float(a_data.get("decode_peak_rss_mb", a_peak_rss)), 2),
            "aissd_active_kv_mb": round(a_act_kv, 2),
            "aissd_cold_kv_mb": round(float(a_data.get("cold_kv_mb", 0.0)), 2),
            "aissd_read_mb": round(float(a_data.get("total_read_mb", a_data.get("storage_read_bytes", 0) / (1024.0 * 1024.0))), 2),
            "aissd_write_mb": round(float(a_data.get("total_write_mb", a_data.get("storage_write_bytes", 0) / (1024.0 * 1024.0))), 2),
            "memory_saved_mb": round(b_peak_rss - a_peak_rss, 2),
            "kv_memory_saved_mb": round(b_act_kv - a_act_kv, 2),
            "kv_memory_reduction_pct": round(((b_act_kv - a_act_kv) / max(1e-6, b_act_kv)) * 100.0, 2) if b_act_kv > 0 else 0.0,
        })

    return rec


def parse_stage_spec(stage_arg: str) -> Optional[int]:
    """Parses stage argument like '2048', '2k', '1', 'stage1' into integer context."""
    s = stage_arg.strip().lower()
    if s in ("1", "stage1", "2048", "2k"):
        return 2048
    elif s in ("2", "stage2", "4096", "4k"):
        return 4096
    elif s in ("3", "stage3", "8192", "8k"):
        return 8192
    elif s in ("4", "stage4", "16384", "16k"):
        return 16384
    elif s in ("5", "stage5", "32768", "32k"):
        return 32768
    try:
        val = int(s)
        if val in [2048, 4096, 8192, 16384, 32768]:
            return val
    except ValueError:
        pass
    return None


def run_staged_benchmark(
    stages: List[int],
    models: List[str],
    precisions: List[str],
    decode_tokens: int,
    threads: int,
    storage_mode: str,
    resume: bool = True,
    force: bool = False,
    git_commit: str = "unknown",
    generation_id: str = "",
) -> Dict[str, Any]:
    """
    Executes the benchmark strictly by context stage across all models and precisions:
    Stage 1 (2K) -> Stage 2 (4K) -> Stage 3 (8K) -> Stage 4 (16K) -> Stage 5 (32K).
    Persists results atomically after every individual configuration.
    """
    reg_models = ModelRegistry.list_models()
    current_records = load_current_benchmark_records()
    current_dict = {r["config_key"]: r for r in current_records if "config_key" in r}

    total_cells = len(stages) * len(models) * len(precisions)
    completed_cells = 0

    print("\n" + "=" * 80)
    print("           AI-SSD V2 — STAGED FULL MODEL × PRECISION BENCHMARK MATRIX         ")
    print("=" * 80)
    print(f"Stages      : {[f'{s//1024}K' for s in stages]}")
    print(f"Models ({len(models)}) : {models}")
    print(f"Precisions  : {precisions}")
    print(f"Threads     : {threads} | Storage Mode: {storage_mode} | Decode: {decode_tokens}")
    print(f"Resume Mode : {resume} (Force: {force})")
    print(f"Total Cells : {total_cells} configurations")
    print("=" * 80 + "\n")

    stage_summary_table: List[Dict[str, Any]] = []

    for stage_idx, stage_ctx in enumerate(stages, 1):
        stage_label = f"{stage_ctx // 1024}K" if stage_ctx >= 1024 else f"{stage_ctx}"
        prompt_target = stage_ctx - decode_tokens
        print(f"\n{'='*30} STAGE {stage_idx} / {len(stages)} — {stage_ctx} TOKENS ({stage_label}) {'='*30}")
        print(f"Target Sequence: {stage_ctx} tokens (Prompt: ~{prompt_target} tokens + Decode: {decode_tokens} tokens)\n")

        for m_key in models:
            m_info = reg_models.get(m_key, {})
            m_id = m_info.get("model_id", m_key)
            comp_level = m_info.get("compatibility_level")
            comp_reason = m_info.get("compatibility_reason", "")
            supp_precs = [p.lower() for p in m_info.get("supported_precisions", ["fp32", "fp16"])]
            supp_contexts = m_info.get("supported_contexts", [])

            for prec in precisions:
                cfg_key = f"{m_key}|{prec.lower()}|{stage_ctx}"
                completed_cells += 1
                cell_tag = f"[{stage_label} | {completed_cells}/{total_cells}] {m_key} ({prec.upper()})"

                # 1. Resumability check
                if resume and not force and cfg_key in current_dict:
                    existing = current_dict[cfg_key]
                    existing_status = existing.get("status", "UNKNOWN")
                    print(f"[RESUME] Skipping {cell_tag}: Already evaluated with status [{existing_status}]")
                    stage_summary_table.append(existing)
                    continue

                # 2. Check Architecture Compatibility (e.g. Mamba SSM)
                if comp_level == CompatibilityLevel.UNSUPPORTED:
                    rec = make_comparison_record(
                        stage_ctx=stage_ctx,
                        stage_name=stage_label,
                        model_key=m_key,
                        model_info=m_info,
                        prec_req=prec,
                        status="UNSUPPORTED",
                        status_reason=f"Architecture unsupported: {comp_reason}",
                        prompt_tokens=prompt_target,
                        decode_tokens=decode_tokens,
                        threads=threads,
                        seed=42,
                        storage_mode=storage_mode,
                        git_commit=git_commit,
                        generation_id=generation_id,
                    )
                    save_quick_comparison_record(rec)
                    current_dict[cfg_key] = rec
                    stage_summary_table.append(rec)
                    print(f"[SKIP] {cell_tag}: UNSUPPORTED ({comp_reason})")
                    continue

                # 3. Check Precision Compatibility
                if prec.lower() not in supp_precs:
                    reason = f"Precision {prec.upper()} not supported by {m_key} (supports {supp_precs})"
                    rec = make_comparison_record(
                        stage_ctx=stage_ctx,
                        stage_name=stage_label,
                        model_key=m_key,
                        model_info=m_info,
                        prec_req=prec,
                        status="UNSUPPORTED",
                        status_reason=reason,
                        prompt_tokens=prompt_target,
                        decode_tokens=decode_tokens,
                        threads=threads,
                        seed=42,
                        storage_mode=storage_mode,
                        git_commit=git_commit,
                        generation_id=generation_id,
                    )
                    save_quick_comparison_record(rec)
                    current_dict[cfg_key] = rec
                    stage_summary_table.append(rec)
                    print(f"[SKIP] {cell_tag}: UNSUPPORTED ({reason})")
                    continue

                # 4. Check Weight Cache Availability
                if not is_model_cached(m_id):
                    reason = f"Model weights not cached locally on host ({m_id})"
                    rec = make_comparison_record(
                        stage_ctx=stage_ctx,
                        stage_name=stage_label,
                        model_key=m_key,
                        model_info=m_info,
                        prec_req=prec,
                        status="UNAVAILABLE",
                        status_reason=reason,
                        prompt_tokens=prompt_target,
                        decode_tokens=decode_tokens,
                        threads=threads,
                        seed=42,
                        storage_mode=storage_mode,
                        git_commit=git_commit,
                        generation_id=generation_id,
                    )
                    save_quick_comparison_record(rec)
                    current_dict[cfg_key] = rec
                    stage_summary_table.append(rec)
                    print(f"[SKIP] {cell_tag}: UNAVAILABLE ({reason})")
                    continue

                # 5. Execute Live Evaluation (Baseline & AI-SSD)
                print(f"[RUN ] {cell_tag}: Executing Baseline and AI-SSD...")
                tmp_dir = tempfile.gettempdir()
                b_file = os.path.join(tmp_dir, f"base_{uuid.uuid4().hex[:8]}.json")
                a_file = os.path.join(tmp_dir, f"aissd_{uuid.uuid4().hex[:8]}.json")

                try:
                    # Run Baseline
                    ret_b, err_b, b_data = run_worker_process(
                        mode="baseline",
                        model_name=m_id,
                        prompt_tokens=prompt_target,
                        decode=decode_tokens,
                        threads=threads,
                        storage_mode=storage_mode,
                        output_file=b_file,
                        precision=prec,
                    )

                    if ret_b != 0:
                        status_b = "OOM" if ret_b == 137 else "ERROR"
                        rec = make_comparison_record(
                            stage_ctx=stage_ctx,
                            stage_name=stage_label,
                            model_key=m_key,
                            model_info=m_info,
                            prec_req=prec,
                            status=status_b,
                            status_reason=f"Baseline execution failed: {err_b}",
                            prompt_tokens=prompt_target,
                            decode_tokens=decode_tokens,
                            threads=threads,
                            seed=42,
                            storage_mode=storage_mode,
                            git_commit=git_commit,
                            generation_id=generation_id,
                        )
                        save_quick_comparison_record(rec)
                        current_dict[cfg_key] = rec
                        stage_summary_table.append(rec)
                        print(f"[FAIL] {cell_tag}: Baseline {status_b} -> {err_b[:120]}")
                        continue

                    # Run AI-SSD
                    ret_a, err_a, a_data = run_worker_process(
                        mode="aissd",
                        model_name=m_id,
                        prompt_tokens=prompt_target,
                        decode=decode_tokens,
                        threads=threads,
                        storage_mode=storage_mode,
                        output_file=a_file,
                        precision=prec,
                    )

                    if ret_a != 0:
                        status_a = "OOM" if ret_a == 137 else "ERROR"
                        rec = make_comparison_record(
                            stage_ctx=stage_ctx,
                            stage_name=stage_label,
                            model_key=m_key,
                            model_info=m_info,
                            prec_req=prec,
                            status=status_a,
                            status_reason=f"AI-SSD execution failed: {err_a}",
                            prompt_tokens=prompt_target,
                            decode_tokens=decode_tokens,
                            threads=threads,
                            seed=42,
                            storage_mode=storage_mode,
                            git_commit=git_commit,
                            generation_id=generation_id,
                        )
                        save_quick_comparison_record(rec)
                        current_dict[cfg_key] = rec
                        stage_summary_table.append(rec)
                        print(f"[FAIL] {cell_tag}: AI-SSD {status_a} -> {err_a[:120]}")
                        continue

                    # Parity check
                    b_toks = b_data.get("token_ids", [])
                    a_toks = a_data.get("token_ids", [])
                    matches = sum(1 for x, y in zip(b_toks, a_toks) if x == y)
                    tot = min(len(b_toks), len(a_toks))
                    exact = (matches == tot and tot > 0)
                    status_final = "PASS" if exact else "FAIL"
                    status_reason = f"Exact match ({matches}/{tot} tokens)" if exact else f"Token divergence ({matches}/{tot} matched)"

                    rec = make_comparison_record(
                        stage_ctx=stage_ctx,
                        stage_name=stage_label,
                        model_key=m_key,
                        model_info=m_info,
                        prec_req=prec,
                        status=status_final,
                        status_reason=status_reason,
                        prompt_tokens=prompt_target,
                        decode_tokens=decode_tokens,
                        threads=threads,
                        seed=42,
                        b_data=b_data,
                        a_data=a_data,
                        storage_mode=storage_mode,
                        git_commit=git_commit,
                        generation_id=generation_id,
                    )
                    save_quick_comparison_record(rec)
                    current_dict[cfg_key] = rec
                    stage_summary_table.append(rec)

                    b_rss = rec.get("baseline_peak_rss_mb", 0.0)
                    a_rss = rec.get("aissd_peak_rss_mb", 0.0)
                    saved = rec.get("memory_saved_mb", 0.0)
                    tps = rec.get("aissd_tps", 0.0)
                    match_rate_disp = f"{rec.get('token_match_rate', 0.0):.1f}%"
                    print(f"[{status_final}] {cell_tag}: {matches}/{tot} tokens ({match_rate_disp}) | Peak RSS: Base {b_rss:.1f} MB -> AI-SSD {a_rss:.1f} MB ({saved:.1f} MB saved) | {tps:.2f} tok/s")

                finally:
                    for p in (b_file, a_file):
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass

    print("\n" + "=" * 80)
    print("                  STAGED BENCHMARK RUN COMPLETE                       ")
    print("=" * 80)
    res = resolve_benchmark_matrix()
    stats = res.get("stats", {})
    print(f"Total Evaluated       : {stats.get('total_evaluated', 0)} / {stats.get('total_matrix_cells', 0)}")
    print(f"Current Evaluated     : {stats.get('current_evaluated', 0)}")
    print(f"Historical Evaluated  : {stats.get('historical_evaluated', 0)}")
    print(f"Not Evaluated         : {stats.get('not_evaluated', 0)}")
    print(f"PASS: {stats.get('pass', 0)} | FAIL: {stats.get('fail', 0)} | OOM: {stats.get('oom', 0)} | ERROR: {stats.get('error', 0)} | UNSUPPORTED: {stats.get('unsupported', 0)} | UNAVAILABLE: {stats.get('unavailable', 0)}")
    print("=" * 80 + "\n")
    return res


def list_matrix_configurations(stages: List[int], models: List[str], precisions: List[str]):
    """Prints the matrix of configurations and current resolution status without executing."""
    reg_models = ModelRegistry.list_models()
    resolved = resolve_benchmark_matrix(models=models, stages=stages, precisions=precisions)
    records = resolved["records"]
    stats = resolved["stats"]

    print("\n" + "=" * 90)
    print("                    AI-SSD V2 — BENCHMARK CONFIGURATION MATRIX                       ")
    print("=" * 90)
    print(f"{'Stage':<8} | {'Model Key':<15} | {'Prec':<6} | {'Status':<14} | {'Source':<12} | {'Reason / Notes'}")
    print("-" * 90)
    for r in records:
        s_name = r.get("stage_name", "")
        m_key = r.get("model_key", "")
        prec = r.get("precision_requested", "").upper()
        stat = r.get("status", "")
        src = r.get("result_source", "")
        reason = r.get("status_reason", "")[:35]
        print(f"{s_name:<8} | {m_key:<15} | {prec:<6} | {stat:<14} | {src:<12} | {reason}")
    print("=" * 90)
    print(f"Summary: Cells={stats['total_matrix_cells']} | Current={stats['current_evaluated']} | Hist={stats['historical_evaluated']} | NotEval={stats['not_evaluated']}")
    print(f"Statuses: PASS={stats['pass']} | FAIL={stats['fail']} | OOM={stats['oom']} | ERROR={stats['error']} | UNSUPP={stats['unsupported']} | UNAVAIL={stats['unavailable']}")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AI-SSD Quick & Staged Comprehensive Benchmark Runner")
    # Staged benchmark flags
    parser.add_argument("--all", action="store_true", help="Execute full staged matrix across all 5 context stages")
    parser.add_argument("--stage", type=str, default=None, help="Execute a specific stage (e.g. '2048', '2k', '1')")
    parser.add_argument("--stages", type=str, default=None, help="Comma-separated context stages (e.g. '2048,4096')")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys (e.g. 'qwen3-4b,tiny-mistral')")
    parser.add_argument("--smoke", action="store_true", help="Run fast 2-model smoke benchmark test at 2K context")
    parser.add_argument("--list-matrix", action="store_true", help="Print planned benchmark matrix with current status and exit")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume benchmark, skipping completed cells (default: True)")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Disable resuming")
    parser.add_argument("--force", action="store_true", default=False, help="Force re-running already completed configurations")
    parser.add_argument("--archive", action="store_true", default=False, help="Archive existing current results before starting")

    # Single-run compatibility flags
    parser.add_argument("--model", type=str, default=None, help="Model name or registry key for single run (default: Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--context", type=int, default=512, help="Context length in tokens for single run (default: 512)")
    parser.add_argument("--decode", type=int, default=16, help="Tokens to generate during decode (default: 16)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    parser.add_argument("--precision", type=str, default="fp32", help="Model precision: fp32 (default), fp16")
    parser.add_argument("--storage-mode", type=str, default="nvme_qemu", choices=["file", "nvme_qemu"], help="Storage backend mode (default: nvme_qemu)")

    args = parser.parse_args()

    ensure_c_kernel_built()
    git_commit = get_git_commit()
    generation_id = f"gen_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # Handle archiving if requested
    if args.archive:
        arch_dir = archive_current_benchmark(archive_reason="cli_flag")
        if arch_dir:
            print(f"[INFO] Archived current benchmark records to: {arch_dir}")
        else:
            print("[INFO] No current benchmark records found to archive.")

    # Determine execution mode: Staged vs Single
    is_staged_run = args.all or args.stage or args.stages or args.smoke or args.list_matrix

    if is_staged_run:
        # Determine stages
        if args.smoke:
            stages_to_run = [2048]
        elif args.stage:
            parsed_s = parse_stage_spec(args.stage)
            if not parsed_s:
                print(f"[ERROR] Invalid stage specification: {args.stage}. Choose from 2048/2k, 4096/4k, 8192/8k, 16384/16k, 32768/32k")
                sys.exit(1)
            stages_to_run = [parsed_s]
        elif args.stages:
            stages_to_run = []
            for s_str in args.stages.split(","):
                parsed = parse_stage_spec(s_str.strip())
                if parsed and parsed not in stages_to_run:
                    stages_to_run.append(parsed)
            if not stages_to_run:
                print(f"[ERROR] No valid stages parsed from: {args.stages}")
                sys.exit(1)
        else:
            stages_to_run = [2048, 4096, 8192, 16384, 32768]

        # Determine models
        reg_models = ModelRegistry.list_models()
        if args.smoke:
            # 2 lightweight cached models
            models_to_run = ["qwen2.5-0.5b", "tiny-mistral"]
        elif args.models:
            models_to_run = [m.strip().lower() for m in args.models.split(",") if m.strip()]
        else:
            models_to_run = list(reg_models.keys())

        # Determine precisions
        if args.smoke:
            precisions_to_run = ["fp32"]
        elif args.precision and args.precision != "fp32":
            precisions_to_run = [args.precision.lower()]
        else:
            precisions_to_run = ["fp32", "fp16"]

        if args.list_matrix:
            list_matrix_configurations(stages=stages_to_run, models=models_to_run, precisions=precisions_to_run)
            sys.exit(0)

        run_staged_benchmark(
            stages=stages_to_run,
            models=models_to_run,
            precisions=precisions_to_run,
            decode_tokens=args.decode,
            threads=args.threads,
            storage_mode=args.storage_mode,
            resume=args.resume,
            force=args.force,
            git_commit=git_commit,
            generation_id=generation_id,
        )

    else:
        # Legacy single comparison execution
        model_name = args.model or "Qwen/Qwen2.5-0.5B"
        print(f"\n[AI-SSD] Single Comparison Run: {model_name} (Context: {args.context}, Threads: {args.threads}, Precision: {args.precision})")
        tmp_dir = tempfile.gettempdir()
        base_out = os.path.join(tmp_dir, f"baseline_{os.getpid()}.json")
        aissd_out = os.path.join(tmp_dir, f"aissd_{os.getpid()}.json")

        try:
            # 1. Run Baseline
            ret_b, err_b, b_data = run_worker_process(
                mode="baseline",
                model_name=model_name,
                prompt_tokens=args.context,
                decode=args.decode,
                threads=args.threads,
                storage_mode=args.storage_mode,
                output_file=base_out,
                precision=args.precision,
            )
            if ret_b != 0:
                print(f"[ERROR] Baseline failed: {err_b}")
                sys.exit(ret_b)

            # 2. Run AI-SSD
            ret_a, err_a, a_data = run_worker_process(
                mode="aissd",
                model_name=model_name,
                prompt_tokens=args.context,
                decode=args.decode,
                threads=args.threads,
                storage_mode=args.storage_mode,
                output_file=aissd_out,
                precision=args.precision,
            )
            if ret_a != 0:
                print(f"[ERROR] AI-SSD failed: {err_a}")
                sys.exit(ret_a)

            # 3. Print Comparison
            print_comparison(base_out, aissd_out, model_name)

            # 4. Save to results/current/ atomically
            reg_models = ModelRegistry.list_models()
            model_key = model_name.split("/")[-1].lower()
            m_info = reg_models.get(model_key, {"model_id": model_name})

            b_toks = b_data.get("token_ids", []) if b_data else []
            a_toks = a_data.get("token_ids", []) if a_data else []
            matches = sum(1 for x, y in zip(b_toks, a_toks) if x == y)
            tot = min(len(b_toks), len(a_toks))
            status = "PASS" if (matches == tot and tot > 0) else "FAIL"

            rec = make_comparison_record(
                stage_ctx=args.context,
                stage_name=f"{args.context // 1024}K" if args.context >= 1024 else str(args.context),
                model_key=model_key,
                model_info=m_info,
                prec_req=args.precision,
                status=status,
                status_reason=f"Exact match ({matches}/{tot})" if status == "PASS" else "Token divergence",
                prompt_tokens=args.context,
                decode_tokens=args.decode,
                threads=args.threads,
                seed=42,
                b_data=b_data,
                a_data=a_data,
                storage_mode=args.storage_mode,
                git_commit=git_commit,
                generation_id=generation_id,
            )
            json_p, csv_p = save_quick_comparison_record(rec)
            print(f"[AI-SSD Telemetry] Saved comparison record to:\n  JSON: {json_p}\n  CSV:  {csv_p}")

            # Also save legacy format for backward compatibility
            try:
                b_dtype = b_data.get("dtype", args.precision).lower() if b_data else args.precision.lower()
                b_prec = "fp16" if "16" in b_dtype else "fp32"
                a_dtype = a_data.get("dtype", args.precision).lower() if a_data else args.precision.lower()
                a_prec = "fp16" if "16" in a_dtype else "fp32"

                b_rec = build_benchmark_record(
                    source="scripts/run_quick_comparison.py",
                    mode="baseline",
                    model_name=model_name,
                    precision=b_prec,
                    context_length=args.context,
                    decode_tokens=args.decode,
                    threads=args.threads,
                    raw_metrics=b_data or {},
                    storage_mode="none",
                    computational_storage=False,
                )
                a_rec = build_benchmark_record(
                    source="scripts/run_quick_comparison.py",
                    mode="aissd",
                    model_name=model_name,
                    precision=a_prec,
                    context_length=args.context,
                    decode_tokens=args.decode,
                    threads=args.threads,
                    raw_metrics=a_data or {},
                    storage_mode=args.storage_mode,
                    computational_storage=True,
                )
                save_benchmark_result(b_rec, prefix="quick_comp")
                save_benchmark_result(a_rec, prefix="quick_comp")
            except Exception as e:
                print(f"[WARNING] Legacy format save warning: {e}")

        finally:
            for p in (base_out, aissd_out):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
