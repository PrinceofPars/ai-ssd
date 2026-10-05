# AI-SSD V2 — Optimization 3: Canonical Llama 3 8B FP16 Baseline Report

## Executive Summary

This report establishes the canonical baseline for **Optimization-3**: evaluating **Llama 3 8B FP16** on the AI-SSD V2 computational storage architecture against both dense in-DRAM reference execution and the validated **Qwen3-4B FP32 release candidate** (`cc3972e`).

### Key Measured Findings

1. **End-to-End Speedup over Qwen3-4B FP32**:
   - **Llama 3 8B FP16 AI-SSD**: **12.873 ± 0.270 s** (**1.243 ± 0.026 tok/s**, peak **1.276 tok/s**)
   - **Qwen3-4B FP32 Release (`cc3972e`)**: **16.658 ± 0.402 s** (**0.961 ± 0.024 tok/s**, peak **1.000 tok/s**)
   - **Measured Throughput Improvement**: **$+29.4\%$** ($1.294\times$ speedup)
2. **DRAM Footprint Reduction**:
   - **Dense In-Memory KV Cache (DRAM)**: **514.00 MB**
   - **AI-SSD Active KV in Host DRAM**: **54.50 MB**
   - **DRAM Footprint Reduction**: **$89.4\%$ KV offload to NVMe**
   - **Peak Process RSS**: 16,452.6 MB vs Dense 17,677.7 MB ($-1,225.1\text{ MB}$ net host RAM reduction)
3. **Strict Invariant Verification**:
   - **Candidate Key Bytes to Host**: **0 bytes (100% In-Storage Filtering)**
   - **Token Correctness**: **16/16 exact bit-for-bit token match (100.0%) across all 5 repetitions** against dense reference `[8668, 3160, 11, 35208, 22781, 315, 23401, 72229, 315, 3552, 14644, 1428, 323, 94577, 1113, 91790]`
   - **Host-RAM Resident Offload Payload**: **0.0 MB** (`p2_resident_payload_mb = 0.0`, `p3_resident_payload_mb = 0.0`)

---

## 1. Experimental Configuration

| Parameter | Specification | Classification |
|---|---|---|
| Model | `NousResearch/Meta-Llama-3-8B` (un-gated reference) | `[REAL]` |
| Model Parameters | 8,030,261,248 (100.00% FP16) | `[REAL]` |
| Model Weight Footprint | 14.96 GB (safetensors) | `[REAL]` |
| Prompt Length (Context) | 4,096 tokens | `[REAL]` |
| Generation Length (Decode) | 16 tokens | `[REAL]` |
| CPU Execution Threads | 4 threads | `[REAL]` |
| Host Hardware | AWS EC2 Sapphire Rapids (4 physical cores / 8 vCPUs, 61.8 GB RAM) | `[REAL]` |
| Storage Target | QEMU/KVM Virtual NVMe Controller (`/dev/nvme0n1`) | `[VIRTUAL-DEVICE]` |
| In-Storage Filtering | Enabled (10% Top-K Computational Selection) | `[VIRTUAL-DEVICE]` |
| Storage Architecture | 8 channels, 64 KiB combined block retrieval | `[VIRTUAL-DEVICE]` |
| Pipeline Mode | Asynchronous Storage Pipeline (Contiguous Block Fetch) | `[VIRTUAL-DEVICE]` |
| Speculative Prefetch | OFF (Disabled for strict canonical baseline) | `[VIRTUAL-DEVICE]` |
| Repetitions | 5 independent executions (Seeds 42..46) | `[REAL]` |

---

## 2. Five-Repetition Canonical Measurement Data

Source data: `benchmarks/live_inference/results/optimization3/llama8b_fp16_baseline.json`

| Repetition | Seed | Wall Time (s) | Throughput (tok/s) | Peak RSS (MB) | Active KV (MB) | Candidate K → Host | Token Match |
|---|---|---|---|---|---|---|---|
| Rep 1 | 42 | 12.538 | 1.276 | 16,551.3 | 54.50 | 0 B | 16/16 PASS |
| Rep 2 | 43 | 13.237 | 1.209 | 15,927.9 | 54.50 | 0 B | 16/16 PASS |
| Rep 3 | 44 | 13.122 | 1.219 | 16,591.4 | 54.50 | 0 B | 16/16 PASS |
| Rep 4 | 45 | 12.835 | 1.247 | 16,680.9 | 54.50 | 0 B | 16/16 PASS |
| Rep 5 | 46 | 12.636 | 1.266 | 16,511.6 | 54.50 | 0 B | 16/16 PASS |
| **Mean ± Std** | — | **12.873 ± 0.270** | **1.243 ± 0.026** | **16,452.6 ± 268.3** | **54.50 ± 0.0** | **0 B ± 0.0** | **100% PASS** |

---

## 3. Comparative Architecture Analysis

| Metric | Dense In-Memory Llama 3 8B FP16 | AI-SSD Qwen3-4B FP32 (`cc3972e`) | AI-SSD Llama 3 8B FP16 (Optimization 3) | Llama vs Qwen AI-SSD Delta |
|---|---|---|---|---|
| Model Parameters | 8.03 B | 4.02 B | 8.03 B | $+100.0\%$ params |
| Precision | FP16 | FP32 | FP16 | Halved byte width |
| Wall Time (s) | 7.973 s | 16.658 ± 0.402 s | **12.873 ± 0.270 s** | **$-22.7\%$ latency** |
| Throughput (tok/s) | 2.007 tok/s | 0.961 ± 0.024 tok/s | **1.243 ± 0.026 tok/s** | **$+29.4\%$ throughput** |
| Peak Throughput | 2.007 tok/s | 1.000 tok/s | **1.276 tok/s** | **$+27.6\%$ peak tok/s** |
| Host DRAM KV Footprint | 514.00 MB | 122.60 MB | **54.50 MB** | **$-55.5\%$ DRAM KV** |
| KV DRAM Reduction vs Dense | 0.0% | 76.1% | **89.4%** | $+13.3\%$ offload |
| Peak Process RSS (MB) | 17,677.7 MB | ~17,600 MB | **16,452.6 MB** | **$-1,147.4\text{ MB}$ RSS** |
| Candidate K → Host (B) | N/A | 0 B | **0 B** | Zero-bus invariant verified |
| Token Correctness | 16/16 Reference | 16/16 Reference | **16/16 Reference** | 100% Bit-exact |

---

## 4. Component Timing Breakdown & Critical Path

From the measured component timers across the 16 decode steps:

```text
Total Wall Time: 12.538 s (100.0%)
├── Visible Compute: 8.919 s (71.1%)
│   ├── MLP & LayerNorm:          5.624 s (44.9%)
│   ├── QKV Projection:           0.772 s  (6.2%)
│   ├── Output Projection (O):    0.503 s  (4.0%)
│   ├── Attention Matmul (SDPA):  0.217 s  (1.7%)
│   ├── Active KV Reconstruction: 0.118 s  (0.9%)
│   ├── Active KV Concatenation:  0.103 s  (0.8%)
│   ├── RoPE Embedding:           0.046 s  (0.4%)
│   └── Bookkeeping:              0.012 s  (0.1%)
└── Visible Storage: 3.527 s (28.1%)
    ├── Storage Read (64 KiB blks): 1.764 s (14.1%)
    └── Top-K In-Storage Scoring:   1.523 s (12.1%)
Reconciliation Error: 0.73% (< 2.0% threshold)
```

### Insights
1. **GEMV Speedup from FP16**: Even though Llama 3 8B contains $2\times$ the weights, MLP latency is only 5.62s vs Qwen3-4B FP32's 4.55s, while generating tokens at 1.24 tok/s vs 0.96 tok/s.
2. **Top-K Scoring Efficiency**: The in-storage 128-dim AVX2/FMA/F16C kernel scanned 4,010,803,200 bytes (4.01 GB) of internal Key pages in just 1.52s (equivalent to $>2.63\text{ GB/s}$ internal scan rate), returning winning blocks with 0 bytes transferred over the bus.
3. **Storage Latency Overlap**: Virtual NVMe read latency averaged 35.2 µs per 4 KiB sector, delivering 969.86 MB/s aggregate throughput across the 8 simulated channels.
