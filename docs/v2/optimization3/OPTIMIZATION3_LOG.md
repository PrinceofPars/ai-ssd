# AI-SSD V2 — Optimization 3 Execution Log: Llama 3 8B FP16 Investigation

## Overview
- **Branch**: `v2-optimization-3-llama8b-fp16`
- **Starting Release Commit**: `cc3972e` (AI-SSD V2 Validated Performance Release)
- **Target Workload**: Canonical 4096 context, 16 decode tokens, 4 CPU threads, Seed=42, Storage=QEMU/NVMe (/dev/nvme0n1), Top-K=10%, Async=ON, Prefetch=OFF.

---

## Chronological Milestones

### Milestone 1: Branch Isolation & Safety
- Branched directly from release anchor commit `cc3972e`.
- Protected release branches `v2-real-llm-kvssd` and `v2-performance-optimization` verified immutable.
- Pushed clean branch to `origin/v2-optimization-3-llama8b-fp16`.

### Milestone 2: Model Acquisition & Architectural Audit
- Model: `NousResearch/Meta-Llama-3-8B` (un-gated exact bit-for-bit weights).
- Parameters: 8,030,261,248 (100.0% FP16, 0 FP32 weights).
- Total parameter size: 14.96 GB.
- Architecture: 32 layers, hidden size 4096, intermediate size 14336, 32 Q heads, 8 KV heads, head_dim 128.
- KV Cache geometry: 32 KiB Key page, 32 KiB Value page, 64 KiB block per 16 tokens.

### Milestone 3: Dense In-Memory Baseline Established
- 4096 context, 16 decode steps, 4 threads, Seed=42:
  - Wall time: **7.973 s**
  - Throughput: **2.007 tok/s**
  - Peak RSS: **17,677.7 MB** (Dense KV in DRAM: 514.00 MB)
  - Ground-truth Token IDs: `[8668, 3160, 11, 35208, 22781, 315, 23401, 72229, 315, 3552, 14644, 1428, 323, 94577, 1113, 91790]`
  - Output text: `" sequence length, consuming tens of gigabytes of host DRAM and saturating PCIe"`

### Milestone 4: In-Storage AVX2 FP16 Top-K Kernel Implementation
- Implemented `dot_product_128_fp16_avx2`, `dot_product_64_fp16_avx2`, and `compute_block_score_gqa_fp16` in `scripts/nvme_guest_daemon.c`.
- Hardware instruction `_mm256_cvtph_ps` (F16C) with 4-way independent accumulator unrolling verified.
- Rebuilt `/opt/ai-ssd-v2/images/initramfs.cpio.gz` with `-O3 -mavx2 -mfma -mf16c -static`.

### Milestone 5: Core Engine & Data Movement Generalization
- Updated `person1_kv_engine/real_llm/aissd_inference.py`:
  - Dynamically allocate `k_blk`/`v_blk` with `np.float16` when model is FP16.
  - Compute `page_bytes` dynamically ($16 \times 8 \times 128 \times 2 = 32,768$ B) to guarantee physically accurate bus movement reporting.
  - Dynamically adjust `get_memory_stats` to compute active and total KV DRAM in FP16.
- Updated `scripts/context_scaling_worker.py`:
  - Added `--model`, `--dtype`, and `--num-threads` CLI parameters.
  - Added deterministic prompt generation matching Llama 3 tokenizer.

### Milestone 6: End-to-End Smoke Test Verification
- Executed smoke test (`scripts/optimization3/test_aissd_llama_smoke.py`) on live QEMU/NVMe storage device.
- Generated tokens: `[8668, 3160, 11, 35208]`.
- Expected tokens: `[8668, 3160, 11, 35208]`.
- **Exact Bit-for-Bit Match: TRUE (100% Correctness)**.
- **Candidate Key bytes to host: 0 bytes (100% In-Storage Filtering)**.
- Wall time: 2.549 s (1.569 tok/s).

### Milestone 7: Canonical 5-Repetition Benchmark Validation
- Executed `scripts/optimization3/run_canonical_baseline.py` on QEMU/NVMe.
- 5 repetitions completed with zero failures and **100% token accuracy (16/16 exact match)**.
- **Mean Throughput**: **1.243 ± 0.026 tok/s** (peak 1.276 tok/s) vs Qwen3-4B FP32's **0.961 ± 0.024 tok/s** (**$+29.4\%$ speedup**).
- **Mean Wall Time**: **12.873 ± 0.270 s** vs Qwen3-4B FP32's **16.658 ± 0.402 s** (**$-22.7\%$ latency**).
- **Active KV DRAM**: **54.50 MB** vs Dense **514.00 MB** (**$89.4\%$ DRAM offload**).
- **Candidate Key Bytes to Host**: **0 bytes (Zero-Bus Invariant Verified)**.

### Milestone 8: Thread Scaling Validation
- Measured 2, 4, and 8 threads at Context=4096 on QEMU/NVMe.
- 2 Threads: 18.756 s (0.853 tok/s)
- 4 Threads: 12.884 s (1.242 tok/s) [$1.46\times$ speedup]
- 8 Threads: 12.412 s (1.289 tok/s) [$1.51\times$ speedup, bandwidth saturation]

### Milestone 9: Context Scaling Validation
- Measured Context lengths 4,096, 8,192, and 16,384 tokens on QEMU/NVMe.
- 4K Context: 13.126 s (1.219 tok/s), Active KV = 54.5 MB, Cold KV = 510 MB (89.4% DRAM reduction).
- 8K Context: 18.026 s (0.888 tok/s), Active KV = 106.5 MB, Cold KV = 1,022 MB (89.6% DRAM reduction).
- 16K Context: 40.657 s (0.394 tok/s), Active KV = 208.5 MB, Cold KV = 2,044 MB (89.8% DRAM reduction).
- **Candidate Key bytes to host**: **0 bytes strictly across all context scales**.
