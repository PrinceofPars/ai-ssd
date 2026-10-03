# AI-SSD V2 — GLOBAL STATUS

## Project

Real LLM + Real KV Cache + NVMe + FEMU + Tensor-Aware FTL

## Current Phase

PHASE 5D — LARGE-CONTEXT SCALING (4K–32K) EMPIRICAL VALIDATION (COMPLETE)

## Current Milestone

M5D — 4K-to-32K True Host-RAM Offload Scaling Verified

## Last Global Update

Completed empirical context-length scaling benchmark from 4,096 to 32,768 tokens on Qwen/Qwen3-4B-Instruct-2507 (FP32, 4 threads, decode 16 tokens):
- Physical OS process RSS during decode remains virtually flat for AI-SSD (~16.0–16.5 GB) while Baseline scales from 20.1 GB to 44.1 GB.
- Net physical RAM savings scale from 3.58 GB at 4K to 27.71 GB at 32K (62.9% reduction in total process RSS).
- Storage backing file scales to 9.59 GB directly on physical disk with 0.00 MB resident storage payload in host RAM.
- Output accuracy: 100.0% exact token match with Baseline across all context lengths (4096, 8192, 16384, 32768).
- Subsystem test suite: 127/127 tests PASS (100%), scripts/run_tests.py: 24/24 PASS (100%).
Full report: docs/v2/notes/CONTEXT_SCALING_4K_32K_RESULTS.md.

---

# Agent State

| Agent | Worktree | Branch | State |
|---|---|---|---|
| P1 | /home/ubuntu/ai-ssd | v2-real-llm-kvssd | PHASE 5D COMPLETE (4K-32K Scaling Verified) |
| P2 | ../ai-ssd-p2 | v2/p2-femu-ftl | PHASE 3 COMPLETE |
| P3 | ../ai-ssd-p3 | v2/p3-system-integration | PHASE 3 COMPLETE |

---

# Dependency State

- P1 multi-context traces delivered to /opt/ai-ssd-v2/traces/real_llm/ (contexts: 128, 512, 1024, 2048, 4096; manifests and SHA-256 verified).
- P1 evaluation results published to /opt/ai-ssd-v2/results/p1/ (phase3_real_llm_results.json, phase3_summary.csv, eal_inference_baseline.json, eal_inference_ai_ssd.json).
- CLI Benchmark tool available at scripts/real_inference_benchmark.py.
- Context Scaling Suite available at scripts/run_scaling_suite.py and scripts/context_scaling_worker.py.
- Benchmark outputs: enchmarks/live_inference/results/context_scaling_results.csv and .json.

---

# Integration State

- Phase 2 Contract & Trace Replay Compatibility: PASS (100% verified across 7,872 events).
- Phase 3 P1 Real LLM & In-Storage Attention Evaluation: PASS (41/41 unit tests passing).
- Phase 5A Real Qwen Inference + AI-SSD Integration: PASS (44/44 unit tests passing).
- Phase 5C True Host-RAM Offload: PASS (127/127 pytest, 24/24 integration tests).
- Phase 5D 4K-32K Scaling Validation: PASS (100% token match across 4K, 8K, 16K, 32K).

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

Proceed to Phase 5E / Phase 6 (Ablation study, QEMU/FEMU live driver integration, or presentation/evidence reporting).