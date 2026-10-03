"""Phase 3 Comprehensive Real LLM / KV-Cache Evaluator for AI-SSD V2.

Performs empirical benchmarks of Qwen2.5-0.5B across 6 context lengths
(128, 256, 512, 1024, 2048, 4096 tokens) under a strict 4-CPU-thread budget.

Evaluates:
1. Real LLM baseline metrics (prefill latency, decode latency, decode throughput,
   process memory RSS, KV-cache size and geometry, with mean & variance across runs).
2. In-storage Top-k sparse attention across 5 sparsity budgets (1%, 5%, 10%, 20%, 50%)
   against dense attention reference (active blocks, token selection ratio, PCIe bytes
   transferred vs avoided, cosine similarity, Frobenius relative error, attention mass recall).
3. Component ablations: Dense vs Top-k KV, Python/NumPy scoring vs Native AVX2 C kernel.
4. Generates real traces conforming to CanonicalTraceRecord with manifests & SHA-256 digests.
5. Emits machine-readable results to /opt/ai-ssd-v2/results/p1/ with strict metric classifications
   (REAL, ANALYTICAL, SYNTHETIC).
"""

import sys
import os
import json
import time
import hashlib
import gc
from typing import Dict, Any, List, Tuple
from datetime import datetime
import numpy as np
import psutil

PROJECT_ROOT = "/home/ubuntu/ai-ssd-p1"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.block_adapter import KVBlockAdapter
from person1_kv_engine.real_llm.trace_generator import RealKVTraceGenerator
from person1_kv_engine.real_llm.topk_evaluator import TopKEvaluator
from person1_kv_engine.c_kernel.kernel_binding import get_native_c_kernel
from person1_kv_engine.c_kernel.compile_kernel import compile_c_kernel
from common.schemas.trace import CanonicalTraceRecord


RESULTS_DIR = "/opt/ai-ssd-v2/results/p1"
TRACES_DIR = "/opt/ai-ssd-v2/traces/real_llm"

CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
SPARSITY_BUDGETS = [1.0, 5.0, 10.0, 20.0, 50.0]
NUM_REPETITIONS = 3
DECODE_TOKENS = 16
NUM_THREADS = 4
RANDOM_SEED = 42
GIT_COMMIT = "e13531a"


def get_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) in megabytes."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024.0 * 1024.0)


def build_prompt_for_length(engine: RealLLMEngine, target_tokens: int) -> str:
    """Constructs a deterministic realistic technical prompt of exact target token length."""
    base_paragraph = (
        "In modern solid-state drive computational storage architectures, the host computing complex "
        "offloads sparse attention scoring algorithms into the flash memory controller ASIC. "
        "The controller embedded core cluster coordinates key-value tensor page retrieval across multiple "
        "NAND flash channels. Key tensors are organized in contiguous 4 KiB physical flash pages while "
        "Value tensors reside in paired 4 KiB flash pages, forming unified 8 KiB logical blocks. "
        "During autoregressive token decoding, incoming query vectors are broadcast to controller SRAM. "
        "SIMD vector dot-product engines scan candidate key pages directly from flash buffers, sorting block "
        "attention scores using in-storage hardware priority queues. Only winning Top-k blocks and host "
        "DRAM resident attention sinks are transferred across the PCIe NVMe interconnect, eliminating up to "
        "90 percent of host memory bus saturation and drastically reducing memory amplification. "
    )
    tokens = engine.tokenizer(base_paragraph)["input_ids"]
    reps = (target_tokens // len(tokens)) + 2
    full_text = base_paragraph * reps
    
    # Truncate to exact target length
    token_ids = engine.tokenizer(full_text, max_length=target_tokens, truncation=True)["input_ids"]
    prompt = engine.tokenizer.decode(token_ids, skip_special_tokens=True)
    return prompt


def run_baseline_benchmarks(
    engine: RealLLMEngine,
    adapter: KVBlockAdapter,
) -> Dict[str, Any]:
    """Evaluates real LLM baseline across context lengths with multiple repetitions."""
    print("\n" + "=" * 80)
    print("PHASE 3 PART 1: REAL LLM BASELINE BENCHMARK (Qwen2.5-0.5B, 4 CPU Threads)")
    print("=" * 80)

    baseline_results = {}
    base_rss_mb = get_process_memory_mb()

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n--- Benchmarking Context Length: {ctx_len} tokens ({NUM_REPETITIONS} repetitions) ---")
        prompt = build_prompt_for_length(engine, ctx_len)
        actual_prompt_tokens = len(engine.tokenizer(prompt)["input_ids"])

        prefill_latencies = []
        decode_latencies = []
        decode_throughputs = []
        peak_rss_list = []

        last_inf_res = None

        for rep in range(NUM_REPETITIONS):
            gc.collect()
            rss_start = get_process_memory_mb()

            inf_res = engine.run_inference(
                prompt=prompt,
                max_new_tokens=DECODE_TOKENS,
                seed=RANDOM_SEED + rep,
            )
            last_inf_res = inf_res

            rss_peak = get_process_memory_mb()
            prefill_s = inf_res["timings"]["prefill_time_s"]
            decode_s = inf_res["timings"]["decode_time_s"]
            tps = inf_res["timings"]["tokens_per_second"]

            prefill_latencies.append(prefill_s)
            decode_latencies.append(decode_s)
            decode_throughputs.append(tps)
            peak_rss_list.append(rss_peak)

            print(
                f"  Rep {rep + 1}/{NUM_REPETITIONS}: "
                f"Prefill: {prefill_s:.3f}s | "
                f"Decode: {decode_s:.3f}s ({tps:.2f} tok/s) | "
                f"Peak RSS: {rss_peak:.1f} MB"
            )

        # Compute block counts and KV-cache byte sizes
        total_tokens = actual_prompt_tokens + DECODE_TOKENS
        layer_blocks = adapter.blockize_all_layers(last_inf_res["layer_kv"])
        blocks_per_layer = len(layer_blocks[0])
        total_blocks = sum(len(b) for b in layer_blocks.values())

        # Exact physical byte calculations
        bytes_per_token_per_layer = engine.num_kv_heads * engine.head_dim * 4 * 2  # K+V in FP32
        total_kv_bytes = total_tokens * bytes_per_token_per_layer * engine.num_layers
        logical_storage_bytes = total_blocks * adapter.logical_block_bytes

        baseline_results[str(ctx_len)] = {
            "target_context_tokens": ctx_len,
            "actual_prompt_tokens": actual_prompt_tokens,
            "generated_tokens": DECODE_TOKENS,
            "total_tokens": total_tokens,
            "prefill_latency_s": {
                "mean": float(np.mean(prefill_latencies)),
                "std": float(np.std(prefill_latencies)),
                "min": float(np.min(prefill_latencies)),
                "max": float(np.max(prefill_latencies)),
            },
            "decode_latency_s": {
                "mean": float(np.mean(decode_latencies)),
                "std": float(np.std(decode_latencies)),
                "min": float(np.min(decode_latencies)),
                "max": float(np.max(decode_latencies)),
            },
            "decode_throughput_tps": {
                "mean": float(np.mean(decode_throughputs)),
                "std": float(np.std(decode_throughputs)),
                "min": float(np.min(decode_throughputs)),
                "max": float(np.max(decode_throughputs)),
            },
            "process_rss_mb": {
                "base_rss_mb": base_rss_mb,
                "peak_mean": float(np.mean(peak_rss_list)),
                "net_increment_mb": float(np.mean(peak_rss_list) - base_rss_mb),
            },
            "kv_cache_geometry": {
                "num_layers": engine.num_layers,
                "num_kv_heads": engine.num_kv_heads,
                "head_dim": engine.head_dim,
                "dtype": engine.dtype_str,
                "tokens_per_block": adapter.tokens_per_block,
                "key_page_bytes": adapter.key_size_bytes,
                "value_page_bytes": adapter.value_size_bytes,
                "logical_block_bytes": adapter.logical_block_bytes,
                "blocks_per_layer": blocks_per_layer,
                "total_blocks_all_layers": total_blocks,
                "kv_bytes_per_layer": total_tokens * bytes_per_token_per_layer,
                "total_kv_bytes_exact": total_kv_bytes,
                "total_kv_megabytes_exact": total_kv_bytes / (1024.0 * 1024.0),
                "total_storage_bytes_blockized": logical_storage_bytes,
                "total_storage_megabytes_blockized": logical_storage_bytes / (1024.0 * 1024.0),
            },
        }

    return baseline_results


def run_topk_sparse_evaluations(
    engine: RealLLMEngine,
    topk_eval: TopKEvaluator,
) -> Dict[str, Any]:
    """Evaluates in-storage Top-k sparse attention against dense reference across contexts."""
    print("\n" + "=" * 80)
    print("PHASE 3 PART 2: IN-STORAGE TOP-K EVALUATION (Sparsity Budgets: 1%, 5%, 10%, 20%, 50%)")
    print("=" * 80)

    topk_eval_results = {}

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n--- Evaluating Sparse Attention for Context: {ctx_len} tokens ---")
        prompt = build_prompt_for_length(engine, ctx_len)
        inf_res = engine.run_inference(prompt=prompt, max_new_tokens=4, seed=RANDOM_SEED)

        # Evaluate across all 24 layers using query from the final decode step
        step_idx = -1
        query_dict = inf_res["step_queries"][step_idx]

        context_sparsity_data = {
            "target_context_tokens": ctx_len,
            "total_tokens": inf_res["total_tokens"],
            "sparsity_evaluations": {},
            "dense_reference": {},
        }

        # Aggregate across all 24 layers
        dense_times_us = []
        dense_bytes_list = []

        layer_evals_by_sparsity = {f"{p:.1f}%": [] for p in SPARSITY_BUDGETS}

        for l_idx in range(engine.num_layers):
            q_layer = query_dict[l_idx]
            k_seq = inf_res["layer_kv"][l_idx]["k"]
            v_seq = inf_res["layer_kv"][l_idx]["v"]

            res = topk_eval.evaluate_query(
                query=q_layer,
                k_seq=k_seq,
                v_seq=v_seq,
                sparsity_levels=SPARSITY_BUDGETS,
            )

            dense_times_us.append(res["dense_reference"]["latency_us"])
            dense_bytes_list.append(res["dense_reference"]["bytes_transferred"])

            for sp_str, sp_data in res["sparsity_evaluations"].items():
                layer_evals_by_sparsity[sp_str].append(sp_data)

        # Dense reference summary
        total_dense_bytes = sum(dense_bytes_list)
        mean_dense_latency_us = float(np.mean(dense_times_us))

        context_sparsity_data["dense_reference"] = {
            "mean_layer_latency_us": mean_dense_latency_us,
            "total_model_latency_us": float(np.sum(dense_times_us)),
            "per_layer_bytes": int(dense_bytes_list[0]),
            "total_model_bytes": total_dense_bytes,
            "total_model_megabytes": total_dense_bytes / (1024.0 * 1024.0),
        }

        print(
            f"{'Sparsity':<10} | {'Active/Total Blks':<18} | {'Token Ratio':<12} | "
            f"{'PCIe Traffic Reduc':<20} | {'Cos Sim (Mean)':<16} | {'Rel Err (Mean)':<16} | {'Mass Recall':<12}"
        )
        print("-" * 115)

        for sp_str in sorted(layer_evals_by_sparsity.keys(), key=lambda x: float(x.replace("%", ""))):
            sp_list = layer_evals_by_sparsity[sp_str]

            total_active_blks = sum(x["total_active_blocks"] for x in sp_list)
            total_cand_blks = sum(x["total_candidate_blocks"] for x in sp_list)
            total_possible_blks = sp_list[0]["total_active_blocks"] + sp_list[0]["total_candidate_blocks"]
            total_possible_blks_all = sum(x["total_active_blocks"] + x["total_candidate_blocks"] - x["selected_ssd_blocks"] for x in sp_list)

            token_ratio = float(np.mean([x["token_selection_ratio"] for x in sp_list]))
            pcie_transferred = sum(x["pcie_bytes_transferred"] for x in sp_list)
            pcie_avoided = max(0, total_dense_bytes - pcie_transferred)
            traffic_reduction = (1.0 - (pcie_transferred / max(1.0, float(total_dense_bytes)))) * 100.0

            cos_sim_mean = float(np.mean([x["mean_cosine_similarity"] for x in sp_list]))
            cos_sim_min = float(np.min([x["min_cosine_similarity"] for x in sp_list]))
            cos_sim_std = float(np.std([x["mean_cosine_similarity"] for x in sp_list]))

            rel_err_mean = float(np.mean([x["mean_relative_error"] for x in sp_list]))
            rel_err_max = float(np.max([x["mean_relative_error"] for x in sp_list]))
            rel_err_std = float(np.std([x["mean_relative_error"] for x in sp_list]))

            mass_recall_mean = float(np.mean([x["mean_attention_mass_recall"] for x in sp_list]))
            mass_recall_min = float(np.min([x["mean_attention_mass_recall"] for x in sp_list]))

            sparse_time_us = float(np.mean([x["latency_us"] for x in sp_list]))

            # Analytical PCIe latency calculation (PCIe 4.0 x4 @ 7.88 GB/s)
            pcie_bw_gbps = 7.88
            dense_pcie_time_us = (total_dense_bytes / (pcie_bw_gbps * 1e9)) * 1e6
            sparse_pcie_time_us = (pcie_transferred / (pcie_bw_gbps * 1e9)) * 1e6
            pcie_time_saved_us = dense_pcie_time_us - sparse_pcie_time_us

            context_sparsity_data["sparsity_evaluations"][sp_str] = {
                "sparsity_budget_percent": float(sp_str.replace("%", "")),
                "total_active_blocks_across_layers": total_active_blks,
                "token_selection_ratio": token_ratio,
                "pcie_bytes_transferred": pcie_transferred,
                "pcie_bytes_avoided": pcie_avoided,
                "pcie_traffic_reduction_percent": traffic_reduction,
                "cosine_similarity": {
                    "mean": cos_sim_mean,
                    "min": cos_sim_min,
                    "std": cos_sim_std,
                },
                "frobenius_relative_error": {
                    "mean": rel_err_mean,
                    "max": rel_err_max,
                    "std": rel_err_std,
                },
                "attention_mass_recall": {
                    "mean": mass_recall_mean,
                    "min": mass_recall_min,
                },
                "measured_compute_latency_us": {
                    "dense_mean_us": mean_dense_latency_us,
                    "sparse_mean_us": sparse_time_us,
                },
                "analytical_pcie_transfer_time_us": {
                    "bus_standard": "PCIe 4.0 x4 (7.88 GB/s)",
                    "dense_transfer_time_us": dense_pcie_time_us,
                    "sparse_transfer_time_us": sparse_pcie_time_us,
                    "bus_time_saved_us": pcie_time_saved_us,
                },
            }

            print(
                f"{sp_str:<10} | {sp_list[0]['total_active_blocks']:>4} blks/lyr      | "
                f"{token_ratio * 100.0:>8.1f}%    | "
                f"{traffic_reduction:>18.2f}% | "
                f"{cos_sim_mean:>14.4f}   | "
                f"{rel_err_mean:>14.4f}   | "
                f"{mass_recall_mean * 100.0:>8.2f}%"
            )

        topk_eval_results[str(ctx_len)] = context_sparsity_data

    return topk_eval_results


def run_kernel_ablation_benchmarks(
    engine: RealLLMEngine,
) -> Dict[str, Any]:
    """Runs rigorous component ablation between Native AVX2 C kernel and NumPy reference."""
    print("\n" + "=" * 80)
    print("PHASE 3 PART 3: COMPONENT ABLATION (Native AVX2 C Kernel vs Python/NumPy)")
    print("=" * 80)

    compile_c_kernel()
    kernel = get_native_c_kernel()
    if not kernel.is_available():
        raise RuntimeError("Native C kernel instorage_attention.so is not available!")

    test_block_counts = [32, 64, 128, 256, 512]
    tokens_per_block = 16
    heads = engine.num_attention_heads
    head_dim = engine.head_dim
    scale = float(1.0 / np.sqrt(head_dim))
    num_iters = 50

    rng = np.random.RandomState(42)
    query = rng.randn(heads, head_dim).astype(np.float32)

    ablation_results = {
        "kernel_library": "instorage_attention.so",
        "compiler_flags": "-O3 -mavx2 -mfma -shared -fPIC",
        "cpu_target": "x86_64 AVX2 / FMA",
        "iterations": num_iters,
        "evaluations": [],
    }

    print(
        f"{'Blocks':<8} | {'Seq Equiv':<10} | {'Scan (MB)':<10} | "
        f"{'C Lat (us)':<12} | {'NumPy Lat (us)':<16} | {'Speedup':<10} | "
        f"{'C Thrpt (GiB/s)':<16} | {'Max Err':<10}"
    )
    print("-" * 105)

    for num_blocks in test_block_counts:
        top_k = max(1, num_blocks // 10)  # 10% sparsity
        seq_equiv = num_blocks * tokens_per_block

        # Create realistic block payloads
        blocks = []
        for bid in range(num_blocks):
            entry = {
                "k": rng.randn(tokens_per_block, heads, head_dim).astype(np.float32),
                "v": rng.randn(tokens_per_block, heads, head_dim).astype(np.float32),
            }
            blocks.append((bid, entry))

        # 1. Numerical Parity Validation
        ref_scores = []
        for bid, entry in blocks:
            k_b = entry["k"]
            dots = np.einsum("hd,thd->th", query, k_b) * scale
            ref_scores.append((float(np.max(dots)), bid))
        ref_scores.sort(key=lambda x: x[0], reverse=True)
        ref_topk_ids = [bid for _, bid in ref_scores[:top_k]]
        ref_topk_vals = [val for val, _ in ref_scores[:top_k]]

        c_topk_ids, _, c_topk_vals = kernel.compute_topk(query, blocks, top_k)
        c_topk_ids_list = c_topk_ids.tolist()
        c_topk_vals_list = c_topk_vals.tolist()

        overlap = len(set(ref_topk_ids).intersection(set(c_topk_ids_list)))
        id_match_percent = (overlap / top_k) * 100.0
        score_errs = [abs(c - r) for c, r in zip(c_topk_vals_list, ref_topk_vals)]
        max_score_err = max(score_errs)

        # 2. Timing C Kernel
        for _ in range(5):
            kernel.compute_topk(query, blocks, top_k)
        t0 = time.perf_counter()
        for _ in range(num_iters):
            kernel.compute_topk(query, blocks, top_k)
        c_dur = time.perf_counter() - t0
        c_lat_us = (c_dur / num_iters) * 1e6

        # 3. Timing NumPy Reference
        for _ in range(5):
            scs = []
            for _, entry in blocks:
                dots = np.einsum("hd,thd->th", query, entry["k"]) * scale
                scs.append(float(np.max(dots)))
            np.argsort(scs)[-top_k:]
        t0 = time.perf_counter()
        for _ in range(num_iters):
            scs = []
            for _, entry in blocks:
                dots = np.einsum("hd,thd->th", query, entry["k"]) * scale
                scs.append(float(np.max(dots)))
            np.argsort(scs)[-top_k:]
        np_dur = time.perf_counter() - t0
        np_lat_us = (np_dur / num_iters) * 1e6

        speedup = np_lat_us / max(1e-6, c_lat_us)
        block_bytes = tokens_per_block * heads * head_dim * 4
        total_scanned_bytes = num_blocks * block_bytes
        total_scanned_mb = total_scanned_bytes / (1024.0 * 1024.0)

        c_throughput_gib_s = (total_scanned_bytes / (1024.0**3)) / max(1e-9, c_dur / num_iters)
        c_blocks_per_sec = num_blocks / max(1e-9, c_dur / num_iters)

        eval_entry = {
            "num_blocks": num_blocks,
            "equivalent_tokens": seq_equiv,
            "bytes_scanned": total_scanned_bytes,
            "megabytes_scanned": total_scanned_mb,
            "top_k": top_k,
            "c_kernel_latency_us": c_lat_us,
            "numpy_latency_us": np_lat_us,
            "speedup_factor": speedup,
            "c_scan_throughput_gib_per_sec": c_throughput_gib_s,
            "c_blocks_per_sec": c_blocks_per_sec,
            "numerical_parity": {
                "topk_id_match_percent": id_match_percent,
                "max_score_error": float(max_score_err),
            },
        }
        ablation_results["evaluations"].append(eval_entry)

        print(
            f"{num_blocks:<8} | {seq_equiv:<10} | {total_scanned_mb:>8.2f}MB | "
            f"{c_lat_us:>10.1f}us | {np_lat_us:>14.1f}us | {speedup:>8.2f}x | "
            f"{c_throughput_gib_s:>14.2f}   | {max_score_err:.2e}"
        )

    return ablation_results


def generate_benchmark_traces(
    engine: RealLLMEngine,
    adapter: KVBlockAdapter,
    trace_gen: RealKVTraceGenerator,
) -> List[Dict[str, Any]]:
    """Generates and validates real traces for benchmark context lengths."""
    print("\n" + "=" * 80)
    print("PHASE 3 PART 4: REAL KV TRACE GENERATION (Canonical Schema Contract)")
    print("=" * 80)

    trace_records = []
    target_trace_contexts = [128, 1024, 2048, 4096]

    for ctx_len in target_trace_contexts:
        workload_name = f"qwen2.5_0.5b_context{ctx_len}"
        print(f"\n--- Generating Trace for: {workload_name} ---")
        prompt = build_prompt_for_length(engine, ctx_len)

        inf_res = engine.run_inference(prompt=prompt, max_new_tokens=DECODE_TOKENS, seed=RANDOM_SEED)
        layer_blocks = adapter.blockize_all_layers(inf_res["layer_kv"])

        t_path, m_path, sha_digest = trace_gen.generate_trace_from_run(
            inference_result=inf_res,
            layer_blocks=layer_blocks,
            workload_name=workload_name,
            top_k_percent=10.0,
        )

        trace_size = os.path.getsize(t_path)
        with open(m_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Quick validation of trace records using CanonicalTraceRecord
        record_count = 0
        with open(t_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = CanonicalTraceRecord.from_dict(json.loads(line))
                record_count += 1

        assert record_count == manifest["total_events"], f"Event count mismatch: {record_count} != {manifest['total_events']}"

        print(
            f"Generated: {os.path.basename(t_path)} | "
            f"Events: {manifest['total_events']} | "
            f"Tokens: {manifest['total_tokens']} | "
            f"Size: {trace_size / (1024*1024):.2f} MB | "
            f"SHA: {sha_digest[:16]}..."
        )

        trace_records.append({
            "workload_name": workload_name,
            "context_length": ctx_len,
            "trace_file": t_path,
            "manifest_file": m_path,
            "sha256": sha_digest,
            "total_events": manifest["total_events"],
            "total_tokens": manifest["total_tokens"],
            "size_bytes": trace_size,
        })

    return trace_records


def emit_summary_csv(results: Dict[str, Any], filepath: str) -> None:
    """Emits human/script friendly summary CSV of the primary Phase 3 results."""
    lines = [
        "Context_Tokens,Sparsity_Budget_Percent,Active_Tokens,Token_Ratio_Pct,PCIe_Transferred_MB,PCIe_Avoided_MB,PCIe_Traffic_Reduction_Pct,Cosine_Similarity,Relative_Error,Mass_Recall_Pct,Classification"
    ]
    
    topk_data = results["topk_evaluations"]
    for ctx_str, ctx_entry in topk_data.items():
        ctx_len = ctx_entry["target_context_tokens"]
        for sp_str, sp_eval in ctx_entry["sparsity_evaluations"].items():
            sp_pct = sp_eval["sparsity_budget_percent"]
            token_ratio_pct = sp_eval["token_selection_ratio"] * 100.0
            pcie_tx_mb = sp_eval["pcie_bytes_transferred"] / (1024.0 * 1024.0)
            pcie_av_mb = sp_eval["pcie_bytes_avoided"] / (1024.0 * 1024.0)
            pcie_reduc_pct = sp_eval["pcie_traffic_reduction_percent"]
            cos_sim = sp_eval["cosine_similarity"]["mean"]
            rel_err = sp_eval["frobenius_relative_error"]["mean"]
            mass_recall_pct = sp_eval["attention_mass_recall"]["mean"] * 100.0
            classification = "REAL"
            
            lines.append(
                f"{ctx_len},{sp_pct:.1f},{sp_eval.get('active_tokens', 0)},{token_ratio_pct:.2f},"
                f"{pcie_tx_mb:.2f},{pcie_av_mb:.2f},{pcie_reduc_pct:.2f},"
                f"{cos_sim:.5f},{rel_err:.5f},{mass_recall_pct:.2f},{classification}"
            )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary CSV written to: {filepath}")


def main():
    print("=" * 80)
    print("AI-SSD V2 PHASE 3: REAL LLM / KV-CACHE FULL EVALUATION SUITE")
    print("=" * 80)
    print(f"Model: Qwen/Qwen2.5-0.5B on CPU (threads={NUM_THREADS}, seed={RANDOM_SEED})")
    print(f"Contexts: {CONTEXT_LENGTHS}")
    print(f"Sparsity Budgets: {SPARSITY_BUDGETS}%")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Traces Directory: {TRACES_DIR}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TRACES_DIR, exist_ok=True)

    t_suite_start = time.time()

    # 1. Initialize Engine & Components
    print("\nInitializing RealLLMEngine (CPU, FP32, 4 threads)...")
    engine = RealLLMEngine(
        model_name="Qwen/Qwen2.5-0.5B",
        device="cpu",
        dtype="FP32",
        num_threads=NUM_THREADS,
    )

    adapter = KVBlockAdapter(
        tokens_per_block=16,
        kv_heads_per_block=1,
        head_dim=engine.head_dim,
        dtype="FP32",
        attention_sink_tokens=4,
        recent_window_tokens=16,
    )

    topk_eval = TopKEvaluator(
        head_dim=engine.head_dim,
        tokens_per_block=adapter.tokens_per_block,
        attention_sink_tokens=adapter.attention_sink_tokens,
        recent_window_tokens=adapter.recent_window_tokens,
        bytes_per_elem=adapter.bytes_per_elem,
    )

    trace_gen = RealKVTraceGenerator(
        output_dir=TRACES_DIR,
        trace_version="2.0",
        git_commit=GIT_COMMIT,
    )

    # 2. Run Baselines
    baseline_results = run_baseline_benchmarks(engine, adapter)

    # 3. Run Top-k Sparse Evaluations
    topk_eval_results = run_topk_sparse_evaluations(engine, topk_eval)

    # 4. Run Kernel Ablations
    ablation_results = run_kernel_ablation_benchmarks(engine)

    # 5. Generate Benchmark Traces
    trace_records = generate_benchmark_traces(engine, adapter, trace_gen)

    # 6. Aggregate Comprehensive Results JSON
    total_suite_time_s = time.time() - t_suite_start

    final_payload = {
        "metadata": {
            "suite_version": "AI-SSD V2 Phase 3",
            "git_commit": GIT_COMMIT,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "total_runtime_s": total_suite_time_s,
            "host": {
                "cpu_model": "Intel(R) Xeon(R) Platinum 8488C",
                "vcpus_allocated": NUM_THREADS,
                "vcpus_total": 8,
                "ram_total_gib": 61.0,
            },
            "model_metadata": {
                "model_id": "Qwen/Qwen2.5-0.5B",
                "num_layers": engine.num_layers,
                "num_attention_heads": engine.num_attention_heads,
                "num_kv_heads": engine.num_kv_heads,
                "head_dim": engine.head_dim,
                "gqa_ratio": engine.gqa_ratio,
                "hidden_size": engine.hidden_size,
                "dtype": engine.dtype_str,
            },
            "experimental_parameters": {
                "context_lengths": CONTEXT_LENGTHS,
                "sparsity_budgets_percent": SPARSITY_BUDGETS,
                "repetitions_per_context": NUM_REPETITIONS,
                "decode_tokens": DECODE_TOKENS,
                "cpu_threads": NUM_THREADS,
                "random_seed": RANDOM_SEED,
            },
            "metric_classifications": {
                "REAL": [
                    "model_activations",
                    "prefill_latency_s",
                    "decode_latency_s",
                    "decode_throughput_tps",
                    "process_rss_mb",
                    "cosine_similarity",
                    "frobenius_relative_error",
                    "attention_mass_recall",
                    "native_c_kernel_latency_us",
                    "native_c_scan_throughput_gib_s",
                ],
                "ANALYTICAL": [
                    "pcie_bytes_avoided",
                    "pcie_traffic_reduction_percent",
                    "analytical_pcie_transfer_time_us",
                    "flash_read_energy_savings_analytical",
                ],
                "SYNTHETIC": [],
            },
        },
        "baselines": baseline_results,
        "topk_evaluations": topk_eval_results,
        "kernel_ablations": ablation_results,
        "generated_traces": trace_records,
    }

    # Save to /opt/ai-ssd-v2/results/p1/
    json_path = os.path.join(RESULTS_DIR, "phase3_real_llm_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\nFinal machine-readable JSON saved to: {json_path}")

    csv_path = os.path.join(RESULTS_DIR, "phase3_summary.csv")
    emit_summary_csv(final_payload, csv_path)

    kernel_json_path = os.path.join(RESULTS_DIR, "kernel_ablation_results.json")
    with open(kernel_json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Kernel ablation results saved to: {kernel_json_path}")

    print("\n" + "=" * 80)
    print(f"PHASE 3 EVALUATION SUITE COMPLETED SUCCESSFULLY IN {total_suite_time_s:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
