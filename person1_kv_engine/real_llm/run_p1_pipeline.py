"""Main End-to-End Real LLM + KV Trace + Sparse Evaluation Pipeline for AI-SSD V2 (P1).

Executes:
1. Real LLM inference on CPU (Qwen/Qwen2.5-0.5B).
2. Extraction of real layer-by-layer KV cache tensors and query vectors.
3. Accurate KV blockization separating physical 4 KiB K-pages and 4 KiB V-pages (8 KiB logical block).
4. Generation of real JSONL access traces, manifest metadata, and SHA-256 hash.
5. Empirical Top-k sparse attention evaluation against dense reference (1%, 5%, 10%, 20%, 50%).
6. Benchmarking of native Linux C kernel against NumPy reference.
"""

import sys
import os
import json
import time

# Ensure project root is in sys.path
PROJECT_ROOT = "/home/ubuntu/ai-ssd-p1"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.block_adapter import KVBlockAdapter
from person1_kv_engine.real_llm.trace_generator import RealKVTraceGenerator
from person1_kv_engine.real_llm.topk_evaluator import TopKEvaluator
from person1_kv_engine.real_llm.native_benchmark import run_kernel_benchmark


def main():
    print("============================================================")
    print("AI-SSD V2: PERSON 1 (P1) REAL LLM & KV CACHE PIPELINE")
    print("============================================================")

    # 1. Initialize Real LLM Engine on CPU
    print("\n[Step 1] Initializing Real LLM Engine on CPU (Qwen/Qwen2.5-0.5B)...")
    engine = RealLLMEngine(
        model_name="Qwen/Qwen2.5-0.5B",
        device="cpu",
        dtype="FP32",
        num_threads=4,
    )
    print(f"Model loaded: {engine.model_name}")
    print(f"Architecture: {engine.num_layers} layers, {engine.num_attention_heads} Q heads, {engine.num_kv_heads} KV heads (GQA {engine.gqa_ratio}:1)")
    print(f"Head dimension: {engine.head_dim}, Vocab size: {engine.vocab_size}")

    # 2. Run Deterministic Inference Workload
    prompt = (
        "Computational storage devices with in-storage compute enable high-throughput "
        "sparse attention retrieval by filtering key-value cache blocks directly on flash controllers."
    )
    max_new_tokens = 32
    print(f"\n[Step 2] Executing inference workload ({max_new_tokens} tokens)...")
    print(f"Prompt: '{prompt}'")
    inf_res = engine.run_inference(prompt=prompt, max_new_tokens=max_new_tokens, seed=42)

    print(f"Generated text: '{inf_res['generated_text']}'")
    print(f"Prefill time: {inf_res['timings']['prefill_time_s']:.3f}s")
    print(f"Decode time:  {inf_res['timings']['decode_time_s']:.3f}s ({inf_res['timings']['tokens_per_second']:.2f} tokens/s)")
    print(f"Total sequence length: {inf_res['total_tokens']} tokens")

    # 3. Blockize Real KV Cache with Physical Page Separation
    print("\n[Step 3] Slicing real KV activations into physical flash pages (4 KiB K, 4 KiB V, 8 KiB block)...")
    # For Qwen2.5 (head_dim=64, FP32=4 bytes):
    # 16 tokens * 1 head * 64 dim * 4 bytes = 4096 bytes (4 KiB) for Key
    # 16 tokens * 1 head * 64 dim * 4 bytes = 4096 bytes (4 KiB) for Value
    # Combined logical block = 8192 bytes (8 KiB)
    adapter = KVBlockAdapter(
        tokens_per_block=16,
        kv_heads_per_block=1,
        head_dim=engine.head_dim,
        dtype="FP32",
        attention_sink_tokens=4,
        recent_window_tokens=16,
    )
    print(f"Block Geometry: {adapter.tokens_per_block} tokens, {adapter.kv_heads_per_block} KV head, {adapter.head_dim} dim")
    print(f"Physical Key Page:   {adapter.key_size_bytes} bytes ({adapter.key_flash_pages} flash page)")
    print(f"Physical Value Page: {adapter.value_size_bytes} bytes ({adapter.value_flash_pages} flash page)")
    print(f"Combined Block:      {adapter.logical_block_bytes} bytes ({adapter.total_flash_pages} flash pages)")

    layer_blocks = adapter.blockize_all_layers(inf_res["layer_kv"])
    total_blocks = sum(len(b) for b in layer_blocks.values())
    print(f"Total KV blocks across all {engine.num_layers} layers: {total_blocks} blocks ({total_blocks * adapter.logical_block_bytes / (1024*1024):.2f} MB)")

    # 4. Generate Real Access Trace for P2 and P3
    print("\n[Step 4] Generating real access trace in /opt/ai-ssd-v2/traces/real_llm/...")
    trace_gen = RealKVTraceGenerator(
        output_dir="/opt/ai-ssd-v2/traces/real_llm",
        trace_version="2.0",
        git_commit="db7e0f8",
    )
    trace_path, manifest_path, sha_digest = trace_gen.generate_trace_from_run(
        inference_result=inf_res,
        layer_blocks=layer_blocks,
        workload_name="qwen2.5_0.5b_eval",
        top_k_percent=10.0,
    )
    trace_size_bytes = os.path.getsize(trace_path)
    print(f"Trace written:    {trace_path} ({trace_size_bytes / 1024:.2f} KiB)")
    print(f"Manifest written: {manifest_path}")
    print(f"SHA-256 Digest:   {sha_digest}")

    # 5. Evaluate In-Storage Top-k Attention vs Dense Reference
    print("\n[Step 5] Evaluating Dense Reference Attention vs Top-k Sparse Attention...")
    evaluator = TopKEvaluator(
        head_dim=engine.head_dim,
        tokens_per_block=adapter.tokens_per_block,
        attention_sink_tokens=adapter.attention_sink_tokens,
        recent_window_tokens=adapter.recent_window_tokens,
        bytes_per_elem=adapter.bytes_per_elem,
    )

    # Evaluate across mid-layer and final layer
    sample_layer = 12
    sample_q = inf_res["step_queries"][-1][sample_layer]
    sample_k = inf_res["layer_kv"][sample_layer]["k"]
    sample_v = inf_res["layer_kv"][sample_layer]["v"]

    sparsity_levels = [1.0, 5.0, 10.0, 20.0, 50.0]
    topk_results = evaluator.evaluate_query(
        query=sample_q,
        k_seq=sample_k,
        v_seq=sample_v,
        sparsity_levels=sparsity_levels,
    )

    print("\n--- MEASURED TOP-K EVALUATION RESULTS ---")
    print(f"{'Sparsity':<10} | {'Active/Total Blks':<18} | {'Token Ratio':<12} | {'Traffic Reduc':<14} | {'Cos Sim':<10} | {'Rel Error':<10} | {'Mass Recall':<12}")
    print("-" * 92)
    for sp_str, m in topk_results["sparsity_evaluations"].items():
        print(
            f"{sp_str:<10} | "
            f"{m['total_active_blocks']}/{topk_results['dense_reference']['total_blocks']:<16} | "
            f"{m['token_selection_ratio']*100.0:>6.1f}%     | "
            f"{m['pcie_traffic_reduction_percent']:>10.1f}%   | "
            f"{m['mean_cosine_similarity']:>8.4f} | "
            f"{m['mean_relative_error']:>8.4f} | "
            f"{m['mean_attention_mass_recall']*100.0:>8.2f}%"
        )

    # 6. Benchmark Native Linux C Kernel
    print("\n[Step 6] Benchmarking Native Linux C Kernel (instorage_attention.so with AVX2)...")
    kernel_bench = run_kernel_benchmark(
        num_blocks=128,
        tokens=16,
        heads=8,
        head_dim=engine.head_dim,
        top_k=16,
        num_iterations=50,
        results_dir="/opt/ai-ssd-v2/results",
    )
    print(f"Compiler:       {kernel_bench['compiler']}")
    print(f"Flags:          {kernel_bench['flags']}")
    print(f"C Kernel Lat:   {kernel_bench['performance']['native_c_latency_us']:.2f} us")
    print(f"NumPy Ref Lat:  {kernel_bench['performance']['numpy_reference_latency_us']:.2f} us")
    print(f"Speedup Factor: {kernel_bench['performance']['speedup_factor']:.2f}x")
    print(f"Throughput:     {kernel_bench['performance']['throughput_gib_per_second']:.2f} GiB/s")
    print(f"Numerical Err:  max {kernel_bench['numerical_validation']['max_topk_score_error']:.6e} (validation {'PASSED' if kernel_bench['numerical_validation']['passed'] else 'FAILED'})")

    # 7. Aggregate Full Experiment Summary
    summary = {
        "workload": "p1_real_llm_inference",
        "model": inf_res["model_metadata"],
        "prompt": prompt,
        "generated_tokens": max_new_tokens,
        "tokens_per_second": inf_res["timings"]["tokens_per_second"],
        "kv_geometry": {
            "tokens_per_block": adapter.tokens_per_block,
            "kv_heads_per_block": adapter.kv_heads_per_block,
            "head_dim": adapter.head_dim,
            "key_page_bytes": adapter.key_size_bytes,
            "value_page_bytes": adapter.value_size_bytes,
            "logical_block_bytes": adapter.logical_block_bytes,
            "total_blocks": total_blocks,
        },
        "trace": {
            "trace_path": trace_path,
            "manifest_path": manifest_path,
            "sha256": sha_digest,
            "size_bytes": trace_size_bytes,
            "total_events": os.path.getsize(trace_path),
        },
        "topk_evaluation": topk_results,
        "native_kernel": kernel_bench,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    results_path = "/opt/ai-ssd-v2/results/p1_real_llm_experiment.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Complete] Full experiment results saved to: {results_path}")
    print("============================================================")


if __name__ == "__main__":
    main()
