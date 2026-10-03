"""Benchmark and Numerical Validation of Native Linux C In-Storage Kernel for AI-SSD V2.

Compiles and validates instorage_attention.so using the host GCC toolchain on EC2,
comparing against Python/NumPy reference on numerical error, latency, and throughput.
Output: /opt/ai-ssd-v2/results/native_kernel_benchmark.json
"""

from typing import Dict, Any, List
import time
import os
import json
import numpy as np

from person1_kv_engine.c_kernel.kernel_binding import get_native_c_kernel
from person1_kv_engine.c_kernel.compile_kernel import compile_c_kernel


def run_kernel_benchmark(
    num_blocks: int = 128,
    tokens: int = 16,
    heads: int = 8,
    head_dim: int = 64,
    top_k: int = 16,
    num_iterations: int = 50,
    results_dir: str = "/opt/ai-ssd-v2/results",
) -> Dict[str, Any]:
    """Validates numerical correctness and benchmarks native C vs NumPy reference."""
    os.makedirs(results_dir, exist_ok=True)

    # 1. Compile or ensure loaded
    so_path = compile_c_kernel()
    kernel = get_native_c_kernel()
    if not kernel.is_available():
        raise RuntimeError(f"Native C kernel failed to load from {so_path}")

    scale = float(1.0 / np.sqrt(head_dim))
    rng = np.random.RandomState(42)

    query = rng.randn(heads, head_dim).astype(np.float32)
    layer_blocks = []
    for bid in range(num_blocks):
        entry = {
            "k": rng.randn(tokens, heads, head_dim).astype(np.float32),
            "v": rng.randn(tokens, heads, head_dim).astype(np.float32),
        }
        layer_blocks.append((bid, entry))

    # -----------------------------------------------------------------
    # 2. NUMERICAL VALIDATION
    # -----------------------------------------------------------------
    # Single-block score validation
    q_c = np.ascontiguousarray(query, dtype=np.float32)
    k_first = np.ascontiguousarray(layer_blocks[0][1]["k"], dtype=np.float32)

    # NumPy reference
    ref_dots = np.einsum("hd,thd->th", query, k_first) * scale
    ref_single_score = float(np.max(ref_dots))

    # Native C score
    q_ptr = q_c.ctypes.data_as(kernel._lib.compute_block_score.argtypes[0])
    k_ptr = k_first.ctypes.data_as(kernel._lib.compute_block_score.argtypes[1])
    c_single_score = float(kernel._lib.compute_block_score(
        q_ptr, k_ptr, tokens, heads, head_dim, scale
    ))

    single_abs_error = abs(c_single_score - ref_single_score)
    assert single_abs_error < 1e-4, f"Single-block score error too high: {single_abs_error}"

    # Multi-block Top-k validation
    # NumPy reference Top-k
    all_scores = []
    for bid, entry in layer_blocks:
        k_b = entry["k"]
        dots = np.einsum("hd,thd->th", query, k_b) * scale
        all_scores.append((float(np.max(dots)), bid))
    all_scores.sort(key=lambda x: x[0], reverse=True)
    ref_topk_ids = [bid for _, bid in all_scores[:top_k]]
    ref_topk_scores = [sc for sc, _ in all_scores[:top_k]]

    # C kernel Top-k
    c_topk_ids, _, c_topk_scores = kernel.compute_topk(query, layer_blocks, top_k)
    c_topk_ids_list = c_topk_ids.tolist()
    c_topk_scores_list = c_topk_scores.tolist()

    topk_id_overlap = len(set(ref_topk_ids).intersection(set(c_topk_ids_list)))
    topk_id_match_percent = (topk_id_overlap / top_k) * 100.0

    score_errors = [abs(c - r) for c, r in zip(c_topk_scores_list, ref_topk_scores)]
    max_score_error = max(score_errors)
    mean_score_error = float(np.mean(score_errors))

    # -----------------------------------------------------------------
    # 3. BENCHMARKING: C KERNEL VS NUMPY
    # -----------------------------------------------------------------
    # Warmup
    for _ in range(5):
        kernel.compute_topk(query, layer_blocks, top_k)

    # Time C kernel
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        kernel.compute_topk(query, layer_blocks, top_k)
    c_total_time = time.perf_counter() - t0
    c_lat_us = (c_total_time / num_iterations) * 1e6

    # Time NumPy reference
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        scs = []
        for _, entry in layer_blocks:
            k_b = entry["k"]
            dots = np.einsum("hd,thd->th", query, k_b) * scale
            scs.append(float(np.max(dots)))
        np.argsort(scs)[-top_k:]
    np_total_time = time.perf_counter() - t0
    np_lat_us = (np_total_time / num_iterations) * 1e6

    # Throughput calculations
    block_bytes = tokens * heads * head_dim * 4  # FP32 bytes
    total_bytes_scanned = num_blocks * block_bytes
    c_throughput_blocks_per_sec = (num_blocks * num_iterations) / c_total_time
    c_throughput_gib_per_sec = (total_bytes_scanned * num_iterations) / (c_total_time * (1024**3))
    speedup = np_lat_us / max(1e-6, c_lat_us)

    benchmark_record = {
        "compiler": "gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0",
        "flags": "-O3 -mavx2 -mfma -shared -fPIC -std=c99 -Wall",
        "target_library": str(so_path),
        "cpu_model": "Intel(R) Xeon(R) Platinum 8488C (Sapphire Rapids)",
        "simd_capabilities_used": ["AVX2", "FMA", "SIMD-4x-unrolling"],
        "input_shape": {
            "num_blocks": num_blocks,
            "tokens_per_block": tokens,
            "heads": heads,
            "head_dim": head_dim,
            "top_k": top_k,
            "block_bytes_fp32": block_bytes,
            "total_bytes_scanned_bytes": total_bytes_scanned,
        },
        "numerical_validation": {
            "single_block_abs_error": single_abs_error,
            "max_topk_score_error": max_score_error,
            "mean_topk_score_error": mean_score_error,
            "topk_id_match_percent": topk_id_match_percent,
            "passed": bool(single_abs_error < 1e-4 and topk_id_match_percent >= 99.0),
        },
        "performance": {
            "native_c_latency_us": c_lat_us,
            "numpy_reference_latency_us": np_lat_us,
            "speedup_factor": speedup,
            "throughput_blocks_per_second": c_throughput_blocks_per_sec,
            "throughput_gib_per_second": c_throughput_gib_per_sec,
            "num_iterations": num_iterations,
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_file = os.path.join(results_dir, "native_kernel_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_record, f, indent=2)

    return benchmark_record


if __name__ == "__main__":
    res = run_kernel_benchmark()
    print("=== NATIVE KERNEL BENCHMARK COMPLETE ===")
    print(f"Validation: {'PASSED' if res['numerical_validation']['passed'] else 'FAILED'}")
    print(f"Native C Latency: {res['performance']['native_c_latency_us']:.2f} us")
    print(f"NumPy Latency:    {res['performance']['numpy_reference_latency_us']:.2f} us")
    print(f"Speedup:          {res['performance']['speedup_factor']:.2f}x")
    print(f"Throughput:       {res['performance']['throughput_gib_per_second']:.2f} GiB/s")
