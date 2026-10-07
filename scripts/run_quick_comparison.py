#!/usr/bin/env python3
"""
AI-SSD V2 — Automated Quick Comparison Runner

Executes both BASELINE (in-DRAM) and AI-SSD (storage offload + Top-K)
for a chosen model (default: Qwen/Qwen2.5-0.5B), collects telemetry,
and prints a side-by-side benchmark comparison table.

Usage:
    python scripts/run_quick_comparison.py
    python scripts/run_quick_comparison.py --model Qwen/Qwen2.5-0.5B --context 512
"""

import sys
import os
import time
import json
import tempfile
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.live_inference.result_schema import build_benchmark_record, save_benchmark_result


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


def run_benchmark(mode: str, model_name: str, context: int, decode: int, threads: int, storage_mode: str, output_file: str, precision: str = "fp32"):
    worker_script = PROJECT_ROOT / "scripts" / "context_scaling_worker.py"
    dtype_str = "float16" if "16" in precision.lower() else "float32"
    cmd = [
        sys.executable,
        str(worker_script),
        "--mode", mode,
        "--model-name", model_name,
        "--dtype", dtype_str,
        "--context", str(context),
        "--decode", str(decode),
        "--threads", str(threads),
        "--output-json", output_file,
    ]
    if mode == "aissd":
        cmd.extend([
            "--storage-mode", storage_mode,
            "--top-k-pct", "10.0",
            "--enable-computational-storage",
        ])

    print(f"\n>>> Running {mode.upper()} mode ({model_name}, context={context}, decode={decode})...")
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"\n[ERROR] {mode.upper()} run failed with exit code {res.returncode}")
        sys.exit(res.returncode)

    print(f"[OK] {mode.upper()} completed in {elapsed:.2f}s")


def print_comparison(baseline_json: str, aissd_json: str, model_name: str):
    with open(baseline_json, "r") as f:
        b = json.load(f)
    with open(aissd_json, "r") as f:
        a = json.load(f)

    print("\n" + "=" * 80)
    print(f"             AI-SSD vs BASELINE BENCHMARK COMPARISON ({model_name})           ")
    print("=" * 80)
    print(f"{'Metric':<32} | {'Baseline (Without AI-SSD)':<20} | {'AI-SSD (With Offload)'}")
    print("-" * 80)
    print(f"{'Peak Process RAM (RSS)':<32} | {b.get('peak_rss_mb', 0):>17.1f} MB | {a.get('peak_rss_mb', 0):>16.1f} MB")
    print(f"{'Active KV in Host RAM':<32} | {b.get('active_kv_mb', 0):>17.2f} MB | {a.get('active_kv_mb', 0):>16.2f} MB")
    print(f"{'Cold KV on Disk':<32} | {b.get('cold_kv_mb', 0):>17.2f} MB | {a.get('cold_kv_mb', 0):>16.2f} MB")
    
    b_read_mb = b.get("total_read_mb", 0.0)
    a_read_mb = a.get("total_read_mb", round(a.get("storage_read_bytes", 0) / (1024.0 * 1024.0), 2))
    b_write_mb = b.get("total_write_mb", 0.0)
    a_write_mb = a.get("total_write_mb", round(a.get("storage_write_bytes", 0) / (1024.0 * 1024.0), 2))

    print(f"{'Total Read MB':<32} | {b_read_mb:>17.2f} MB | {a_read_mb:>16.2f} MB")
    print(f"{'Total Write MB':<32} | {b_write_mb:>17.2f} MB | {a_write_mb:>16.2f} MB")

    cand_bytes = a.get("candidate_k_bytes_to_host", 0)
    cand_str = f"{cand_bytes} B (Zero-Bus)" if cand_bytes == 0 else f"{cand_bytes} B"
    print(f"{'Candidate K to Host Bus':<32} | {'0 B (Host RAM)':>20} | {cand_str:>20}")

    b_tps = b.get("tokens_per_second", 0)
    a_tps = a.get("tokens_per_second", 0)
    print(f"{'Decode Throughput':<32} | {b_tps:>15.2f} tok/s | {a_tps:>14.2f} tok/s")
    print(f"{'Decode Wall Time':<32} | {b.get('wall_time_s', 0):>17.3f} s  | {a.get('wall_time_s', 0):>16.3f} s")
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


def main():
    parser = argparse.ArgumentParser(description="AI-SSD Quick Automated Comparison Runner")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name (default: Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--context", type=int, default=512, help="Context length in tokens (default: 512)")
    parser.add_argument("--decode", type=int, default=16, help="Tokens to generate during decode (default: 16)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    parser.add_argument("--precision", type=str, default="fp32", help="Model precision: fp32 (default), fp16")
    parser.add_argument("--storage-mode", type=str, default="file", choices=["file", "nvme_qemu"], help="Storage backend mode (default: file)")
    args = parser.parse_args()

    ensure_c_kernel_built()

    tmp_dir = tempfile.gettempdir()
    base_out = os.path.join(tmp_dir, f"baseline_{os.getpid()}.json")
    aissd_out = os.path.join(tmp_dir, f"aissd_{os.getpid()}.json")

    try:
        # 1. Run Baseline
        run_benchmark("baseline", args.model, args.context, args.decode, args.threads, args.storage_mode, base_out, precision=args.precision)

        # 2. Run AI-SSD
        run_benchmark("aissd", args.model, args.context, args.decode, args.threads, args.storage_mode, aissd_out, precision=args.precision)

        # 3. Print Comparison
        print_comparison(base_out, aissd_out, args.model)

        # 4. Save standardized results to benchmarks/live_inference/results/
        try:
            with open(base_out, "r") as f:
                b_data = json.load(f)
            with open(aissd_out, "r") as f:
                a_data = json.load(f)

            b_dtype = b_data.get("dtype", args.precision).lower()
            b_prec = "fp16" if "16" in b_dtype else "fp32"
            a_dtype = a_data.get("dtype", args.precision).lower()
            a_prec = "fp16" if "16" in a_dtype else "fp32"

            b_rec = build_benchmark_record(
                source="scripts/run_quick_comparison.py",
                mode="baseline",
                model_name=args.model,
                precision=b_prec,
                context_length=args.context,
                decode_tokens=args.decode,
                threads=args.threads,
                raw_metrics=b_data,
                storage_mode="none",
                computational_storage=False,
            )
            a_rec = build_benchmark_record(
                source="scripts/run_quick_comparison.py",
                mode="aissd",
                model_name=args.model,
                precision=a_prec,
                context_length=args.context,
                decode_tokens=args.decode,
                threads=args.threads,
                raw_metrics=a_data,
                storage_mode=args.storage_mode,
                computational_storage=True,
            )
            p_b = save_benchmark_result(b_rec, prefix="quick_comp")
            p_a = save_benchmark_result(a_rec, prefix="quick_comp")
            print(f"[AI-SSD Telemetry] Saved baseline benchmark to: {p_b}")
            print(f"[AI-SSD Telemetry] Saved AI-SSD benchmark to:   {p_a}")
        except Exception as e:
            print(f"[WARNING] Failed to save comparison results: {e}")

    finally:
        for p in (base_out, aissd_out):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
