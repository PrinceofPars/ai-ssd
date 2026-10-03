# AI-SSD V2 — GLOBAL STATUS

## Project

Real LLM + Real KV Cache + NVMe + FEMU + Tensor-Aware FTL

## Current Phase

PHASE 5C ? TRUE HOST-RAM OFFLOAD & PHYSICAL KV RESIDENCY ELIMINATION (COMPLETE)

## Current Milestone

M5C ? True Host-RAM Offload Verified

## Last Global Update

Completed True Host-RAM Offload across P1, P2, and P3 to eliminate cold KV retention in RAM.
At Qwen3-4B Context=4096:
- Physical OS process RSS during decode drops from Baseline 19,405 MB down to AI-SSD 15,951 MB (-3,454 MB net physical reduction).
- Storage RAM payload drops from 4,590 MB to 0.00 MB via direct-access backing file storage with POSIX_FADV_DONTNEED.
- Output accuracy: 100.0% token match (16/16 tokens).
- Subsystem test suite: 127/127 tests PASS (100%), scripts/run_tests.py: 24/24 PASS (100%).
Full report: docs/v2/notes/TRUE_HOST_RAM_OFFLOAD_RESULTS.md.

---

# Agent State

| Agent | Worktree | Branch | State |
|---|---|---|---|
| P1 | ../ai-ssd-p1 | v2/p1-real-llm-kv | PHASE 5A COMPLETE |
| P2 | ../ai-ssd-p2 | v2/p2-femu-ftl | PHASE 3 COMPLETE |
| P3 | ../ai-ssd-p3 | v2/p3-system-integration | PHASE 3 COMPLETE |

---

# Dependency State

- P1 multi-context traces delivered to `/opt/ai-ssd-v2/traces/real_llm/` (contexts: 128, 512, 1024, 2048, 4096; manifests and SHA-256 verified).
- P1 evaluation results published to `/opt/ai-ssd-v2/results/p1/` (`phase3_real_llm_results.json`, `phase3_summary.csv`, `real_inference_baseline.json`, `real_inference_ai_ssd.json`).
- CLI Benchmark tool available at `scripts/real_inference_benchmark.py`.

---

# Integration State

- Phase 2 Contract & Trace Replay Compatibility: PASS (100% verified across 7,872 events).
- Phase 3 P1 Real LLM & In-Storage Attention Evaluation: PASS (41/41 unit tests passing).
- Phase 5A Real Qwen Inference + AI-SSD Integration: PASS (44/44 unit tests passing).

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

System integration of Phase 5A executable AI-SSD inference path with P2 virtual NVMe / FTL device layer.
