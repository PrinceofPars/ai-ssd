# AI-SSD V2 — GLOBAL STATUS

## Project

Real LLM + Real KV Cache + NVMe + FEMU + Tensor-Aware FTL

## Current Phase

PHASE 7 — COMPUTATIONAL STORAGE: IN-STORAGE TOP-K CANDIDATE FILTERING (COMPLETE)

## Current Milestone

M7 — Level B In-Storage Hardware AVX2/FMA Top-K Filtering on Virtual NVMe Device (/dev/nvme0n1)

## Last Global Update

Completed Phase 7: Computational Storage in QEMU/NVMe environment:
- Moved Key dot-product scoring and Top-K candidate selection inside the storage controller (`scripts/nvme_guest_daemon.c` with hardware AVX2/FMA intrinsics reading directly from `/dev/nvme0n1`).
- Completely eliminated candidate Key streaming to host: candidate Key bytes transferred to host = 0 bytes (100% reduction of non-winning candidate Keys across PCIe).
- Total storage bus data movement reduced by 81.47% (from 621.5 MB down to 115.2 MB).
- QEMU/NVMe decode time collapsed from 52.70s to 21.48s (2.45x measured speedup).
- Candidate Key read time collapsed from 32.63s to 4.08s (8.0x speedup).
- 100% exact token ID match (16/16 tokens) maintained across all evaluated modes.
- True host-RAM offload preserved: P2 and P3 resident payload remains 0.0 MB.
- Reports: docs/v2/notes/PHASE7_COMPUTATIONAL_STORAGE_RESULTS.md, benchmarks/live_inference/results/phase7_computational_storage_results.json.

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