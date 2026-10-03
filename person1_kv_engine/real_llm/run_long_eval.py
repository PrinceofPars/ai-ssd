"""Run long-context (512+ tokens) real LLM evaluation to test real Top-k sparsity tradeoffs."""

import sys
import os
import json
import time
import numpy as np

PROJECT_ROOT = "/home/ubuntu/ai-ssd-p1"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.block_adapter import KVBlockAdapter
from person1_kv_engine.real_llm.trace_generator import RealKVTraceGenerator
from person1_kv_engine.real_llm.topk_evaluator import TopKEvaluator
from person1_kv_engine.real_llm.native_benchmark import run_kernel_benchmark

def main():
    print("=== RUNNING LONG CONTEXT REAL LLM WORKLOAD (512 TOKENS) ===")
    engine = RealLLMEngine(
        model_name="Qwen/Qwen2.5-0.5B",
        device="cpu",
        dtype="FP32",
        num_threads=4,
    )

    # 512-token prompt by repeating realistic architectural specification paragraphs
    base_paragraph = (
        "AI-SSD V2 computational storage architecture disaggregates Key and Value tensors into 4 KiB flash pages. "
        "The host processor issues top-k search requests to the solid state drive controller over the PCIe NVMe bus. "
        "The controller embedded processing unit reads Key pages directly from flash memory into on-chip SRAM buffers. "
        "The hardware accelerated attention scoring engine computes dot products between the query vector and candidate keys. "
        "Only the highest scoring key and value blocks are transferred back across the host interface. "
    )
    # Tokenize and scale to ~500 tokens
    prompt = base_paragraph * 7  # ~500 tokens
    max_new_tokens = 16

    print(f"Running inference with long context prompt (~500 tokens)...")
    t0 = time.time()
    inf_res = engine.run_inference(prompt=prompt, max_new_tokens=max_new_tokens, seed=42)
    print(f"Prompt tokens: {inf_res['prompt_tokens']}, Generated tokens: {inf_res['generated_tokens']}, Total: {inf_res['total_tokens']}")
    print(f"Prefill time: {inf_res['timings']['prefill_time_s']:.3f}s, Decode time: {inf_res['timings']['decode_time_s']:.3f}s")

    # Blockize
    adapter = KVBlockAdapter(
        tokens_per_block=16,
        kv_heads_per_block=1,
        head_dim=engine.head_dim,
        dtype="FP32",
        attention_sink_tokens=4,
        recent_window_tokens=16,
    )
    layer_blocks = adapter.blockize_all_layers(inf_res["layer_kv"])
    total_blocks = sum(len(b) for b in layer_blocks.values())
    print(f"Total KV blocks across all 24 layers: {total_blocks} blocks ({total_blocks * adapter.logical_block_bytes / (1024*1024):.2f} MB)")

    # Trace generation
    trace_gen = RealKVTraceGenerator(
        output_dir="/opt/ai-ssd-v2/traces/real_llm",
        trace_version="2.0",
        git_commit="db7e0f8",
    )
    trace_path, manifest_path, sha_digest = trace_gen.generate_trace_from_run(
        inference_result=inf_res,
        layer_blocks=layer_blocks,
        workload_name="qwen2.5_0.5b_context512",
        top_k_percent=10.0,
    )
    print(f"Real trace written to: {trace_path} ({os.path.getsize(trace_path)/1024:.2f} KiB)")
    print(f"Manifest written to:   {manifest_path}")
    print(f"SHA-256 Digest:        {sha_digest}")

    # Top-k Evaluation on long context
    evaluator = TopKEvaluator(
        head_dim=engine.head_dim,
        tokens_per_block=adapter.tokens_per_block,
        attention_sink_tokens=adapter.attention_sink_tokens,
        recent_window_tokens=adapter.recent_window_tokens,
        bytes_per_elem=adapter.bytes_per_elem,
    )

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

    print("\n=========================================================================================")
    print("REAL MEASURED TOP-K EVALUATION (512-TOKEN CONTEXT, LAYER 12)")
    print("=========================================================================================")
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
    print("=========================================================================================")

    # Run native C kernel benchmark
    kernel_bench = run_kernel_benchmark(
        num_blocks=total_blocks // engine.num_layers,
        tokens=16,
        heads=engine.num_attention_heads,
        head_dim=engine.head_dim,
        top_k=8,
        num_iterations=50,
        results_dir="/opt/ai-ssd-v2/results",
    )

    # Save full results
    summary = {
        "workload": "qwen2.5_0.5b_context512",
        "model": inf_res["model_metadata"],
        "prompt_length_tokens": inf_res["prompt_tokens"],
        "generated_tokens": max_new_tokens,
        "total_tokens": inf_res["total_tokens"],
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
            "size_bytes": os.path.getsize(trace_path),
        },
        "topk_evaluation": topk_results,
        "native_kernel": kernel_bench,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    results_path = "/opt/ai-ssd-v2/results/p1_real_llm_experiment.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
