# P1 — Agent Status

Role: REAL LLM + REAL KV CACHE ENGINE  
Worktree: `/home/ubuntu/ai-ssd-p1`  
Branch: `v2/p1-real-llm-kv`  
Baseline Commit: `db7e0f8`  
Assigned Tmux Session: `p1`  

---

## Current Session

Session ID:
SESSION-P1-PHASE1-DELIVERY

Started:
2026-10-02T22:15:00+05:30

Last Updated:
2026-10-03T00:26:00+05:30

---

## Current Milestone

M1 (Phase 1 Execution Complete — Real LLM Engine, KV Blockization, Trace Generation & Top-k Sparse Evaluation)

---

## Current Task

Phase 1 Complete. Handoff package prepared for P2 (FTL / FEMU) and P3 (Prefetch / System Integration).

---

## Completed

1. **Reconnaissance & Contract Auditing**:
   - Verified active tmux session (`p1`), non-empty `$TMUX`, and worktree isolation (`/home/ubuntu/ai-ssd-p1` on `v2/p1-real-llm-kv` at commit `db7e0f8`).
   - Audited existing V1 synthetic components (`WorkloadGenerator`, `mock_ssd.py`) and identified the need for real CPU causal LLM inference.
   - Discovered and addressed physical NAND flash page geometry discrepancy in V1 (dividing 16-token FP16 blocks into 2048 bytes K and 2048 bytes V). Formulated `docs/v2/proposals/PROP-001-KV-BLOCK-PAGE-GEOMETRY.md` establishing 4 KiB Key page, 4 KiB Value page, and 8 KiB combined logical block.
   - Formulated `docs/v2/proposals/PROP-002-KV-ACCESS-TRACE-SCHEMA.md` specifying standard `KVTrace` JSONL format.

2. **Real Model Selection & Resource Budgeting (Phase 1B)**:
   - Evaluated candidate CPU models under machine constraints (8 vCPUs, 61 GiB RAM).
   - Selected `Qwen/Qwen2.5-0.5B` (494M parameters, 24 layers, 14 query heads, 2 KV heads with GQA 7:1, `head_dim=64`, RoPE).
   - Executed in FP32 on CPU within target 4 vCPU core budget (`torch.set_num_threads(4)`), consuming only ~1.95 GiB RAM (3.1% of system memory) and achieving **20.87 tokens/second** generation speed.

3. **Real KV Extraction Engine (Phase 1C)**:
   - Implemented `RealLLMEngine` (`person1_kv_engine/real_llm/engine.py`).
   - Extracts genuine layer-by-layer, head-by-head, position-by-position $(K, V)$ activations from HuggingFace `DynamicCache` and attention weight distributions without synthetic data fabrication.

4. **Accurate KV Blockization Adapter (Phase 1D)**:
   - Implemented `KVBlockAdapter` (`person1_kv_engine/real_llm/block_adapter.py`).
   - Enforces physical NAND page boundary:
     * Key page: 16 tokens $\times$ 1 head $\times$ 64 dim $\times$ 4 bytes (FP32) = 4096 bytes (4 KiB = 1 flash page).
     * Value page: 16 tokens $\times$ 1 head $\times$ 64 dim $\times$ 4 bytes (FP32) = 4096 bytes (4 KiB = 1 flash page).
     * Combined logical KV block: 8192 bytes (8 KiB = 2 flash pages).
   - Validates memory buffer sizes programmatically against physical byte geometry.
   - Categorizes blocks into DRAM (initial attention sinks tokens 0..3, recent window 16 tokens) and SSD cold candidates with salience decay.

5. **Real Storage Access Trace Generation (Phase 1E)**:
   - Implemented `RealKVTraceGenerator` (`person1_kv_engine/real_llm/trace_generator.py`).
   - Generated real storage access traces from actual model execution in `/opt/ai-ssd-v2/traces/real_llm/`:
     * Long-context (703 tokens, 2112 blocks) trace: `trace_qwen2.5_0.5b_context512.jsonl` (4.14 MiB, 7872 events).
     * Manifest file: `trace_qwen2.5_0.5b_context512.manifest.json`.
     * SHA-256 Checksum: `8e58da7ba45ffc4a9fa84571c5c9a96250cd58488aa17be01205f282b3b6cab9`.
   - Records query ID, layer, head, block ID, token range, operation (`PREFILL_WRITE`, `DECODE_READ`, `TOPK_FILTER`, `TOPK_FETCH`), sub-page distinction (`KEY` vs `VALUE`), byte size, tier, and timestamps.

6. **Empirical Top-k Sparse Attention Evaluation (Phase 1F)**:
   - Implemented `TopKEvaluator` (`person1_kv_engine/real_llm/topk_evaluator.py`).
   - Evaluated Dense Reference Attention vs In-Storage Top-k Sparse Attention across 1%, 5%, 10%, 20%, 50% sparsity budgets on real 512+ token context.
   - Measured actual empirical tradeoff frontier:
     * **1.0% Sparsity**: 90.9% PCIe traffic reduction, 0.7127 cosine similarity, 14.02% attention mass recall.
     * **5.0% Sparsity**: 86.3% PCIe traffic reduction, 0.8153 cosine similarity, 21.77% attention mass recall.
     * **10.0% Sparsity**: 81.8% PCIe traffic reduction, 0.8696 cosine similarity, 27.97% attention mass recall.
     * **20.0% Sparsity**: 72.6% PCIe traffic reduction, 0.9136 cosine similarity, 35.93% attention mass recall.
     * **50.0% Sparsity**: 45.3% PCIe traffic reduction, 0.9695 cosine similarity, 64.18% attention mass recall.
   - Proved that 10% is not automatically "zero loss": 20% budget provides the sweet spot (>0.91 similarity, 72.6% PCIe bandwidth reduction).

7. **Native Linux C Kernel Build & Benchmarking (Phase 1G)**:
   - Resolved Linux shared library loading issue (V1 had committed a Windows PE `.dll` that failed with `invalid ELF header`).
   - Compiled freestanding `instorage_attention.c` into Linux ELF 64-bit shared object `instorage_attention.so` via GCC 11.4 with `-O3 -mavx2 -mfma -shared -fPIC -std=c99`.
   - Updated `kernel_binding.py` and `compile_kernel.py` for cross-platform Linux `.so` and Windows `.dll` support.
   - Benchmarked native kernel vs NumPy reference on EC2 (`native_benchmark.py`):
     * Numerical error: $\max |C - \text{ref}| = 4.77 \times 10^{-7}$ (Validation PASSED).
     * Throughput: **4.37 GiB/s** scanning Key blocks.
     * Results saved to `/opt/ai-ssd-v2/results/native_kernel_benchmark.json`.

8. **Comprehensive Unit Testing (Phase 1H)**:
   - Implemented `tests/test_p1_real_llm.py` covering byte-size calculation, canonical FP16 geometry, blockization, deterministic inference, trace serialization/deserialization, Top-k vs dense evaluation, and native C kernel numerical accuracy.
   - All 34 tests in `person1_kv_engine/tests/` passed (100% pass rate).

---

## Working On

Phase 1 handoff documentation and coordination with P2 and P3.

---

## Next

1. Coordinate with P2 for replaying `/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl` through FEMU/FTL channel models.
2. Coordinate with P3 for integrating real KV block loading into unified system storage interface.

---

## Blockers

None.

---

## Dependencies

- P2: FTL mapping & channel latency timing model for end-to-end replay.
- P3: System integration / prefetching orchestrator consuming real trace.

---

## Artifacts Created

- Core Modules:
  * `person1_kv_engine/real_llm/engine.py`
  * `person1_kv_engine/real_llm/block_adapter.py`
  * `person1_kv_engine/real_llm/trace_generator.py`
  * `person1_kv_engine/real_llm/topk_evaluator.py`
  * `person1_kv_engine/real_llm/native_benchmark.py`
  * `person1_kv_engine/real_llm/run_p1_pipeline.py`
  * `person1_kv_engine/real_llm/run_long_eval.py`
  * `person1_kv_engine/c_kernel/instorage_attention.so`
- Tests:
  * `person1_kv_engine/tests/test_p1_real_llm.py` (all 34 tests passing)
- Proposals & Research:
  * `docs/v2/research/REAL_LLM_KV_RESEARCH.md`
  * `docs/v2/proposals/PROP-001-KV-BLOCK-PAGE-GEOMETRY.md`
  * `docs/v2/proposals/PROP-002-KV-ACCESS-TRACE-SCHEMA.md`
- Shared Traces & Results:
  * `/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl`
  * `/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.manifest.json`
  * `/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.sha256`
  * `/opt/ai-ssd-v2/results/p1_real_llm_experiment.json`
  * `/opt/ai-ssd-v2/results/native_kernel_benchmark.json`

---

## Last Commit

db7e0f8 (aligned with shared V2 baseline). Ready for focused commit.
