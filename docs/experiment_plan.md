# Experiment Plan: AI-SSD Evaluation Matrix

This document outlines the benchmark suites, baseline comparisons, and metrics to collect during the hackathon.

---

## 1. Experimental Dimensions

| Parameter | Values Evaluated | Baseline Default |
| :--- | :--- | :--- |
| **Context Length** | 4096, 8192, 16384, 32768 tokens | 32768 |
| **Offload Ratio** | 0%, 20%, 40%, 60%, 80%, 90% | 80% |
| **Top-k Ratio** | 1%, 5%, 10%, 20%, 100% (Dense) | 10% |
| **FTL Scheme** | Conventional (Sequential) vs Tensor-Aware (Striped) | Tensor-Aware |
| **Prefetching** | Disabled vs Speculative Next-Layer | Enabled |
| **Precision** | FP16 (2B/elem), FP8 (1B/elem) | FP16 |

---

## 2. Benchmark Scripts & Objectives

1. **`benchmarks/run_baseline.py`**:
   - Measures pure GPU/DRAM baseline memory consumption and execution latency across context lengths.
2. **`benchmarks/run_offload.py`**:
   - Evaluates memory footprint reduction under different offload percentages.
3. **`benchmarks/run_topk.py`**:
   - Evaluates sparsity ratio vs. attention recall and I/O traffic savings.
4. **`benchmarks/run_ftl.py`**:
   - Evaluates read latency of Conventional vs Tensor-Aware FTL under parallel attention batch reads.
5. **`benchmarks/run_prefetch.py`**:
   - Evaluates prefetch accuracy, cache hit rates, and DRAM staging overhead.
6. **`benchmarks/run_full_system.py`**:
   - Runs full end-to-end inference pass and outputs unified `metrics.json`.
