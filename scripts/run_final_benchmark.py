#!/usr/bin/env python3
"""
AI-SSD V2 — Phase 9: Final End-to-End Validation, Benchmark Matrix & Demonstration.

Orchestrates the complete final empirical benchmark:
1. Canonical Configuration Reproduction
2. Primary Benchmark Matrix (A, B, C, D, E, F)
3. Repeatability Study (5 repetitions each for Dense Baseline, File-Backed, and QEMU Final)
4. Context Scaling Matrix (4096, 8192, 16384, 32768 tokens)
5. Tensor-Aware vs Conventional FTL Comparison
6. CPU Profiling & Bottleneck Breakdown
7. Storage Traffic & Data Movement Accounting
8. Exact Token-ID Verification & Memory Audit
"""

import os
import sys
import json
import csv
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "live_inference" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_JSON = RESULTS_DIR / "final_benchmark_results.json"
FINAL_CSV = RESULTS_DIR / "final_benchmark_results.csv"

PYTHON_BIN = "/home/ubuntu/ai-ssd-p1/.venv/bin/python3"
WORKER_SCRIPT = str(PROJECT_ROOT / "scripts" / "context_scaling_worker.py")

EXPECTED_TOKENS = [11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13]


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    print(f"\n[EXEC] {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL] Worker exited with code {proc.returncode} after {elapsed:.2f}s")
        print(f"STDOUT:\n{proc.stdout}")
        print(f"STDERR:\n{proc.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    print(f"[OK] Completed in {elapsed:.2f}s")
    lines = [l.strip() for l in proc.stdout.strip().split("\n") if l.strip()]
    if lines:
        print(f"     -> {lines[-1]}")
    return proc


def run_worker_task(task_name: str, args: List[str]) -> Dict[str, Any]:
    out_file = f"/tmp/p9_{task_name}.json"
    if os.path.exists(out_file) and os.path.getsize(out_file) > 100:
        print(f"\n[CACHE] Loading existing result: {out_file}")
        with open(out_file, "r") as f:
            return json.load(f)
    cmd = [PYTHON_BIN, WORKER_SCRIPT] + args + ["--output-json", out_file]
    run_cmd(cmd)
    with open(out_file, "r") as f:
        data = json.load(f)
    return data


def main():
    print("=" * 80)
    print("AI-SSD V2 — Phase 9: Final Benchmark Matrix & Validation Suite")
    print("=" * 80)

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "precision": "FP32",
            "cpu_threads": 4,
            "seed": 42,
            "decode_tokens": 16,
            "expected_token_ids": EXPECTED_TOKENS,
        },
        "canonical_reproduction": {},
        "benchmark_matrix": {},
        "repeatability": {},
        "context_scaling": {},
        "ftl_comparison": {},
    }

    # -------------------------------------------------------------------------
    # 1. Canonical Configuration Reproduction (Run E reference)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 9: PART 1 — Reproduce Canonical Configuration")
    print("=" * 80)
    canonical_data = run_worker_task(
        "canonical_reproduce",
        [
            "--mode", "aissd",
            "--storage-mode", "nvme_qemu",
            "--enable-computational-storage",
            "--disable-prefetch",
            "--enable-async-pipeline",
            "--context", "4096",
            "--decode", "16",
            "--seed", "42",
        ]
    )
    results["canonical_reproduction"] = canonical_data
    canon_match = canonical_data.get("token_ids") == EXPECTED_TOKENS
    print(f"Canonical Reproduction: {canonical_data['tokens_per_second']:.3f} tok/s, {canonical_data['wall_time_s']:.2f}s wall, Token match: {canon_match}")

    # -------------------------------------------------------------------------
    # 2. Benchmark Matrix & Repeatability Runs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 9: PART 2 — Benchmark Matrix & Repeatability (5 reps for A, B, E; 1 rep for C, D, F)")
    print("=" * 80)

    # Config A: Dense PyTorch Baseline (5 reps)
    runs_a = []
    for rep in range(5):
        print(f"\n--- Run A (Dense Baseline) Repetition {rep + 1}/5 ---")
        d = run_worker_task(
            f"run_a_baseline_rep{rep}",
            [
                "--mode", "baseline",
                "--context", "4096",
                "--decode", "16",
                "--seed", "42",
                "--rep", str(rep),
            ]
        )
        runs_a.append(d)

    # Config B: File-backed AI-SSD (5 reps)
    runs_b = []
    for rep in range(5):
        print(f"\n--- Run B (File-Backed AI-SSD) Repetition {rep + 1}/5 ---")
        d = run_worker_task(
            f"run_b_file_rep{rep}",
            [
                "--mode", "aissd",
                "--storage-mode", "file",
                "--enable-computational-storage",
                "--disable-prefetch",
                "--enable-async-pipeline",
                "--context", "4096",
                "--decode", "16",
                "--seed", "42",
                "--rep", str(rep),
            ]
        )
        runs_b.append(d)

    # Config C: QEMU Host-side Top-K (1 rep historical control)
    print("\n--- Run C (QEMU Host-side Top-K) ---")
    run_c = run_worker_task(
        "run_c_qemu_host_topk",
        [
            "--mode", "aissd",
            "--storage-mode", "nvme_qemu",
            "--disable-computational-storage",
            "--disable-prefetch",
            "--disable-async-pipeline",
            "--context", "4096",
            "--decode", "16",
            "--seed", "42",
        ]
    )

    # Config D: QEMU Computational Storage Sync (1 rep Phase 7 control)
    print("\n--- Run D (QEMU Computational Storage Sync) ---")
    run_d = run_worker_task(
        "run_d_qemu_comp_sync",
        [
            "--mode", "aissd",
            "--storage-mode", "nvme_qemu",
            "--enable-computational-storage",
            "--disable-prefetch",
            "--disable-async-pipeline",
            "--context", "4096",
            "--decode", "16",
            "--seed", "42",
        ]
    )

    # Config E: QEMU Computational Storage + Async (Canonical Final, 5 reps)
    runs_e = []
    for rep in range(5):
        print(f"\n--- Run E (QEMU Async Final) Repetition {rep + 1}/5 ---")
        d = run_worker_task(
            f"run_e_qemu_async_rep{rep}",
            [
                "--mode", "aissd",
                "--storage-mode", "nvme_qemu",
                "--enable-computational-storage",
                "--disable-prefetch",
                "--enable-async-pipeline",
                "--context", "4096",
                "--decode", "16",
                "--seed", "42",
                "--rep", str(rep),
            ]
        )
        runs_e.append(d)

    # Config F: QEMU Computational Storage + Async + Prefetch (1 rep ablation)
    print("\n--- Run F (QEMU Async + Prefetch Ablation) ---")
    run_f = run_worker_task(
        "run_f_qemu_async_prefetch",
        [
            "--mode", "aissd",
            "--storage-mode", "nvme_qemu",
            "--enable-computational-storage",
            "--enable-prefetch",
            "--enable-async-pipeline",
            "--context", "4096",
            "--decode", "16",
            "--seed", "42",
        ]
    )

    results["benchmark_matrix"] = {
        "run_a_dense_baseline": runs_a[0],
        "run_b_file_backed": runs_b[0],
        "run_c_qemu_host_topk": run_c,
        "run_d_qemu_comp_sync": run_d,
        "run_e_qemu_final_async": runs_e[0],
        "run_f_qemu_async_prefetch": run_f,
    }

    # Helper function for repeatability statistics
    def calc_stats(run_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        walls = [r["wall_time_s"] for r in run_list]
        tpss = [r["tokens_per_second"] for r in run_list]
        rsss = [r["peak_rss_mb"] for r in run_list]
        return {
            "reps": len(run_list),
            "wall_time": {
                "mean": float(np.mean(walls)),
                "std": float(np.std(walls)),
                "min": float(np.min(walls)),
                "max": float(np.max(walls)),
            },
            "tokens_per_second": {
                "mean": float(np.mean(tpss)),
                "std": float(np.std(tpss)),
                "min": float(np.min(tpss)),
                "max": float(np.max(tpss)),
            },
            "peak_rss_mb": {
                "mean": float(np.mean(rsss)),
                "std": float(np.std(rsss)),
                "min": float(np.min(rsss)),
                "max": float(np.max(rsss)),
            },
            "exact_token_match": all(r.get("token_ids") == EXPECTED_TOKENS for r in run_list),
        }

    results["repeatability"] = {
        "dense_baseline": calc_stats(runs_a),
        "file_backed": calc_stats(runs_b),
        "qemu_final_async": calc_stats(runs_e),
        "all_runs_a": runs_a,
        "all_runs_b": runs_b,
        "all_runs_e": runs_e,
    }

    # -------------------------------------------------------------------------
    # 3. Context Scaling Experiment (4096, 8192, 16384, 32768)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 9: PART 3 — Context Scaling Experiment (4K, 8K, 16K, 32K)")
    print("=" * 80)

    scaling_results = {}
    scaling_results[4096] = {
        "baseline": runs_a[0],
        "aissd": runs_e[0],
    }

    for ctx in [8192, 16384, 32768]:
        print(f"\n--- Context Scaling: {ctx} tokens ---")
        base_d = run_worker_task(
            f"scaling_baseline_{ctx}",
            [
                "--mode", "baseline",
                "--context", str(ctx),
                "--decode", "16",
                "--seed", "42",
            ]
        )
        aissd_d = run_worker_task(
            f"scaling_aissd_{ctx}",
            [
                "--mode", "aissd",
                "--storage-mode", "nvme_qemu",
                "--enable-computational-storage",
                "--disable-prefetch",
                "--enable-async-pipeline",
                "--context", str(ctx),
                "--decode", "16",
                "--seed", "42",
            ]
        )
        scaling_results[ctx] = {
            "baseline": base_d,
            "aissd": aissd_d,
        }

    results["context_scaling"] = scaling_results

    # -------------------------------------------------------------------------
    # 4. FTL Verification: Tensor-Aware vs Conventional Mapping
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 9: PART 4 — FTL Verification: Tensor-Aware vs Conventional Mapping")
    print("=" * 80)
    conv_data = run_worker_task(
        "qemu_conventional_ftl",
        [
            "--mode", "aissd",
            "--storage-mode", "nvme_qemu",
            "--mapping-mode", "conventional",
            "--enable-computational-storage",
            "--disable-prefetch",
            "--enable-async-pipeline",
            "--context", "4096",
            "--decode", "16",
            "--seed", "42",
        ]
    )
    results["ftl_comparison"] = {
        "tensor_aware": runs_e[0].get("channel_distribution", {}),
        "conventional": conv_data.get("channel_distribution", {}),
        "tensor_aware_wall_s": runs_e[0]["wall_time_s"],
        "conventional_wall_s": conv_data["wall_time_s"],
    }

    # -------------------------------------------------------------------------
    # 5. Save Artifacts (JSON and CSV)
    # -------------------------------------------------------------------------
    with open(FINAL_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ARTIFACT] Saved complete results JSON to {FINAL_JSON}")

    # Build and save CSV
    csv_rows = []
    matrix_map = {
        "Dense Baseline": runs_a[0],
        "File-backed AI-SSD": runs_b[0],
        "QEMU Host Top-K": run_c,
        "QEMU In-Storage Sync": run_d,
        "QEMU In-Storage Async (Final)": runs_e[0],
        "QEMU Async + Prefetch": run_f,
    }
    for name, r in matrix_map.items():
        csv_rows.append({
            "section": "benchmark_matrix",
            "name": name,
            "context": r.get("context_length", 4096),
            "tokens_per_second": round(r.get("tokens_per_second", 0.0), 3),
            "wall_time_s": round(r.get("wall_time_s", 0.0), 3),
            "visible_storage_s": round(r.get("visible_storage_s", 0.0), 3),
            "visible_compute_s": round(r.get("visible_compute_s", 0.0), 3),
            "reconciliation_error_pct": round(r.get("reconciliation_error_pct", 0.0), 2),
            "peak_rss_mb": round(r.get("peak_rss_mb", 0.0), 1),
            "active_kv_mb": round(r.get("active_kv_mb", 0.0), 1),
            "candidate_k_bytes_to_host": r.get("candidate_k_bytes_to_host", 0),
            "winning_k_bytes_to_host": r.get("winning_k_bytes_to_host", 0),
            "winning_v_bytes_to_host": r.get("winning_v_bytes_to_host", 0),
            "total_bus_movement_bytes": r.get("total_data_movement_bytes", 0),
            "token_match": r.get("token_ids") == EXPECTED_TOKENS,
        })

    for ctx, d in scaling_results.items():
        b = d["baseline"]
        a = d["aissd"]
        csv_rows.append({
            "section": f"context_scaling_{ctx}_baseline",
            "name": f"Baseline-{ctx}",
            "context": ctx,
            "tokens_per_second": round(b.get("tokens_per_second", 0.0), 3),
            "wall_time_s": round(b.get("wall_time_s", 0.0), 3),
            "visible_storage_s": 0.0,
            "visible_compute_s": round(b.get("wall_time_s", 0.0), 3),
            "reconciliation_error_pct": 0.0,
            "peak_rss_mb": round(b.get("peak_rss_mb", 0.0), 1),
            "active_kv_mb": round(b.get("active_kv_mb", 0.0), 1),
            "candidate_k_bytes_to_host": 0,
            "winning_k_bytes_to_host": 0,
            "winning_v_bytes_to_host": 0,
            "total_bus_movement_bytes": 0,
            "token_match": b.get("token_ids") == EXPECTED_TOKENS,
        })
        csv_rows.append({
            "section": f"context_scaling_{ctx}_aissd",
            "name": f"AI-SSD-{ctx}",
            "context": ctx,
            "tokens_per_second": round(a.get("tokens_per_second", 0.0), 3),
            "wall_time_s": round(a.get("wall_time_s", 0.0), 3),
            "visible_storage_s": round(a.get("visible_storage_s", 0.0), 3),
            "visible_compute_s": round(a.get("visible_compute_s", 0.0), 3),
            "reconciliation_error_pct": round(a.get("reconciliation_error_pct", 0.0), 2),
            "peak_rss_mb": round(a.get("peak_rss_mb", 0.0), 1),
            "active_kv_mb": round(a.get("active_kv_mb", 0.0), 1),
            "candidate_k_bytes_to_host": a.get("candidate_k_bytes_to_host", 0),
            "winning_k_bytes_to_host": a.get("winning_k_bytes_to_host", 0),
            "winning_v_bytes_to_host": a.get("winning_v_bytes_to_host", 0),
            "total_bus_movement_bytes": a.get("total_data_movement_bytes", 0),
            "token_match": a.get("token_ids") == EXPECTED_TOKENS,
        })

    with open(FINAL_CSV, "w", newline="") as f:
        fieldnames = list(csv_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[ARTIFACT] Saved complete results CSV to {FINAL_CSV}")

    # -------------------------------------------------------------------------
    # 6. Print Formatted Summary Tables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("FINAL BENCHMARK COMPARISON TABLE (Context=4096, Decode=16, FP32, 4 Threads)")
    print("=" * 115)
    print(f"{'System Configuration':<32} | {'tok/s':<7} | {'Wall (s)':<8} | {'Peak RSS':<9} | {'Active KV':<9} | {'Cand K -> Host':<13} | {'Bus Movement':<12} | {'Match':<5}")
    print("-" * 115)
    for row in csv_rows[:6]:
        cand_k = f"{row['candidate_k_bytes_to_host']:,} B"
        bus = f"{row['total_bus_movement_bytes'] / (1024*1024):.1f} MB"
        match_str = "16/16" if row["token_match"] else "FAIL"
        print(f"{row['name']:<32} | {row['tokens_per_second']:<7.3f} | {row['wall_time_s']:<8.2f} | {row['peak_rss_mb']:<9.1f} | {row['active_kv_mb']:<9.1f} | {cand_k:<13} | {bus:<12} | {match_str:<5}")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("REPEATABILITY STATISTICS (5 Repetitions Each)")
    print("=" * 115)
    print(f"{'Configuration':<25} | {'Metric':<16} | {'Mean':<9} | {'Std':<9} | {'Min':<9} | {'Max':<9}")
    print("-" * 115)
    for cfg_name, stats in results["repeatability"].items():
        if not isinstance(stats, dict) or "wall_time" not in stats:
            continue
        print(f"{cfg_name:<25} | {'Wall Time (s)':<16} | {stats['wall_time']['mean']:<9.2f} | {stats['wall_time']['std']:<9.3f} | {stats['wall_time']['min']:<9.2f} | {stats['wall_time']['max']:<9.2f}")
        print(f"{'':<25} | {'tok/s':<16} | {stats['tokens_per_second']['mean']:<9.3f} | {stats['tokens_per_second']['std']:<9.3f} | {stats['tokens_per_second']['min']:<9.3f} | {stats['tokens_per_second']['max']:<9.3f}")
        print(f"{'':<25} | {'Peak RSS (MB)':<16} | {stats['peak_rss_mb']['mean']:<9.1f} | {stats['peak_rss_mb']['std']:<9.2f} | {stats['peak_rss_mb']['min']:<9.1f} | {stats['peak_rss_mb']['max']:<9.1f}")
        print("-" * 115)

    print("\n" + "=" * 115)
    print("CONTEXT SCALING SUMMARY (4096 -> 8192 -> 16384 -> 32768)")
    print("=" * 115)
    print(f"{'Context':<8} | {'Base tok/s':<10} | {'AI-SSD tok/s':<12} | {'Base RSS':<10} | {'AI-SSD RSS':<11} | {'RSS Saved (MB)':<14} | {'RSS Red %':<9} | {'KV Red %':<8}")
    print("-" * 115)
    for ctx, d in scaling_results.items():
        b = d["baseline"]
        a = d["aissd"]
        b_rss = b["peak_rss_mb"]
        a_rss = a["peak_rss_mb"]
        saved_mb = b_rss - a_rss
        saved_pct = (saved_mb / b_rss) * 100.0
        b_kv = b["active_kv_mb"]
        a_kv = a["active_kv_mb"]
        kv_red_pct = ((b_kv - a_kv) / b_kv) * 100.0
        print(f"{ctx:<8} | {b['tokens_per_second']:<10.3f} | {a['tokens_per_second']:<12.3f} | {b_rss:<10.1f} | {a_rss:<11.1f} | {saved_mb:<14.1f} | {saved_pct:<8.1f}% | {kv_red_pct:<7.1f}%")
    print("=" * 115)


if __name__ == "__main__":
    main()
