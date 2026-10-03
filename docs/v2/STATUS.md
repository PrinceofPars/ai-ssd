# AI-SSD V2 — GLOBAL STATUS

## Project

Real LLM + Real KV Cache + NVMe + FEMU + Tensor-Aware FTL

## Current Phase

PHASE 8 — ASYNCHRONOUS STORAGE, DMA/PIPELINING & PREFETCH OPTIMIZATION (COMPLETE)

## Current Milestone

M8 — Overlapped Asynchronous KV Retrieval and Contiguous 8 KiB Combined Block Fetch on Virtual NVMe Device (/dev/nvme0n1)

## Last Global Update

Completed Phase 8: Asynchronous Storage, DMA/Pipelining & Prefetch Optimization:
- Implemented contiguous 8 KiB KV block retrieval (`read_block_batch`), reducing storage request transactions by 50% (from 1,080 down to 540 batches) and increasing QEMU/NVMe decode throughput to 0.791 tok/s (20.23s wall time).
- Implemented pipelined asynchronous speculative prefetching in `RealInferencePrefetchAdapter` using non-blocking background thread workers with reentrant lock synchronization in `QemuNvmeClient`.
- Successfully overlapped and hid 9.9575s of raw storage retrieval latency behind host computation (68.8% of raw storage time hidden), achieving 529 zero-wait DRAM hits.
- Verified critical-path reconciliation across all benchmark configurations with errors between 0.59% and 1.24% (< 2% target).
- Identified the dominant remaining bottleneck shift: Host CPU Compute (61.6% of wall time) now exceeds Storage Retrieval (37.7% of wall time).
- Maintained 100% exact token ID match (16/16 tokens) across all 6 benchmark runs.
- Preserved true host-RAM offload: P2 and P3 resident payload remains 0.0 MB, candidate Key bytes transferred to host remains 0 B.
- Deliverables: docs/v2/notes/PHASE8_ASYNC_STORAGE_RESULTS.md, benchmarks/live_inference/results/phase8_async_storage_results.json, scripts/run_phase8_benchmarks.py, tests/test_async_storage.py.

---

# Agent State

| Agent | Worktree | Branch | State |
|---|---|---|---|
| P1 | /home/ubuntu/ai-ssd | v2-real-llm-kvssd | PHASE 5 COMPLETE (QEMU/NVMe Live Storage + FTL Verified) |
| P2 | ../ai-ssd-p2 | v2/p2-femu-ftl | PHASE 3 COMPLETE |
| P3 | ../ai-ssd-p3 | v2/p3-system-integration | PHASE 3 COMPLETE |

---

# Dependency State

- P1 multi-context traces delivered to /opt/ai-ssd-v2/traces/real_llm/ (contexts: 128, 512, 1024, 2048, 4096; manifests and SHA-256 verified).
- P1 evaluation results published to /opt/ai-ssd-v2/results/p1/ (phase3_real_llm_results.json, phase3_summary.csv, real_inference_baseline.json, real_inference_ai_ssd.json).
- CLI Benchmark tool available at scripts/real_inference_benchmark.py.
- Context Scaling Suite available at scripts/run_scaling_suite.py and scripts/context_scaling_worker.py.
- Benchmark outputs: benchmarks/live_inference/results/phase5_qemu_nvme_results.json and optimization_c_results.json.

---

# Integration State

- Phase 2 Contract & Trace Replay Compatibility: PASS (100% verified across 7,872 events).
- Phase 3 P1 Real LLM & In-Storage Attention Evaluation: PASS (41/41 unit tests passing).
- Phase 5A Real Qwen Inference + AI-SSD Integration: PASS (44/44 unit tests passing).
- Phase 5C True Host-RAM Offload: PASS (127/127 pytest, 24/24 integration tests).
- Phase 5D 4K-32K Scaling Validation: PASS (100% token match across 4K, 8K, 16K, 32K).
- Phase 5 QEMU/NVMe Live Storage Integration: PASS (100% token match, 159/159 tests passing).

---

# Important

This file is shared state.

Agents must not overwrite another agent's status.

Each agent owns:

docs/v2/agents/P1_STATUS.md
docs/v2/agents/P2_STATUS.md
docs/v2/agents/P3_STATUS.md

---

# Next Step

Proceed to Phase 6: In-storage tensor filtering + real asynchronous prefetch pipeline tuning.