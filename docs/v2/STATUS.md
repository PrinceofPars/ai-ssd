# AI-SSD V2 — GLOBAL STATUS

## Project

Real LLM + Real KV Cache + NVMe + FEMU + Tensor-Aware FTL

## Current Phase

PHASE 6 — CONTROLLED ABLATIONS, QEMU/NVME BOTTLENECK ISOLATION & END-TO-END EVIDENCE AUDIT (COMPLETE)

## Current Milestone

M6 — Bottleneck Isolation Proving Host-Side Candidate Key Streaming as the Sole NVMe Bottleneck

## Last Global Update

Completed Phase 6: Controlled Ablation Study on live Qwen3-4B-Instruct-2507 workload isolating all storage, FTL, and compute components:
- **Core Research Question Answered:** Isolated why live QEMU/NVMe takes ~51s vs ~10s file-backed: Host-side Top-k candidate scoring requires streaming 8.49 GB of candidate Key blocks across the virtual storage bus on every decode step. In file-backed mode, the OS cache handles this in 2.03s; over NVMe (bandwidth ~253 MB/s), it takes 31.71s (62.4% of decode wall time).
- **NVMe Protocol Overhead:** Detailed telemetry isolates binary command packing (0.038s), TCP send (0.047s), and guest roundtrip wait (0.924s) to just 1.01s total (<2.6% of storage time), proving protocol overhead is negligible; 97.4% is raw payload streaming.
- **FTL Multi-Channel Load Balancing:** Tensor-Aware FTL distributes 145,336 read requests across 8 channels with 1.08% load imbalance and 1.01 contention ratio vs Conventional FTL's 700.0% imbalance and 8.00 contention ratio (100% on Channel 0).
- **Prefetching Tradeoff:** Without parallel DMA hardware, speculative prefetching on a single-link serialized bus incurs a net 2.27s penalty (48.71s no-prefetch vs 50.98s with prefetch) due to bus contention with demand reads.
- **Top-K Sparse vs Dense:** Top-10% sparse attention reduces host active KV RAM by 89.4% (from 1,153 MB to 122.6 MB) and attention matmul time by 6.1x (from 4.16s to 0.69s).
- **Mathematical Foundation for Phase 7:** Proved that pushing dot-product scoring into the SSD controller (computational storage) will eliminate the 31.71s Key transfer bottleneck, projecting QEMU/NVMe decode time from 50.8s down to ~15.1s (1.06 tok/s).
- **Full Verification:** 151/151 unit tests passed, 24/24 contract tests passed, 16/16 exact token match maintained across all 6 ablation runs.
- **Reports:** docs/v2/notes/PHASE6_CONTROLLED_ABLATIONS.md, benchmarks/live_inference/results/phase6_ablation_results.json.

---

# Agent State

| Agent | Worktree | Branch | State |
|---|---|---|---|
| P1 | /home/ubuntu/ai-ssd | v2-real-llm-kvssd | PHASE 6 COMPLETE (Controlled Ablations & Bottlenecks Isolated) |
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