# AI-SSD V2 — GLOBAL STATUS

## Project

Real LLM + Real KV Cache + NVMe + FEMU + Tensor-Aware FTL

## Current Phase

PHASE 3 — REAL LLM / KV-CACHE EVALUATION (P1 COMPLETE)

## Current Milestone

M3

## Last Global Update

P1 real LLM and in-storage KV evaluation complete. Multi-context empirical benchmarks (128..4096 tokens), 5 sparsity budgets, AVX2 C kernel ablations, and real canonical traces generated and verified in /opt/ai-ssd-v2/results/p1/ and /opt/ai-ssd-v2/traces/real_llm/. Full evaluation report published at `docs/v2/P1_PHASE3_RESULTS.md`.

---

# Agent State

| Agent | Worktree | Branch | State |
|---|---|---|---|
| P1 | ../ai-ssd-p1 | v2/p1-real-llm-kv | PHASE 3 COMPLETE |
| P2 | ../ai-ssd-p2 | v2/p2-femu-ftl | NOT STARTED |
| P3 | ../ai-ssd-p3 | v2/p3-system-integration | NOT STARTED |

---

# Dependency State

- P1 multi-context traces delivered to `/opt/ai-ssd-v2/traces/real_llm/` (contexts: 128, 512, 1024, 2048, 4096; manifests and SHA-256 verified).
- P1 evaluation results published to `/opt/ai-ssd-v2/results/p1/` (`phase3_real_llm_results.json`, `phase3_summary.csv`, `kernel_ablation_results.json`).
- P2 / P3 can directly consume any real trace via `common/schemas/trace.py::CanonicalTraceRecord`.

---

# Integration State

- Phase 2 Contract & Trace Replay Compatibility: PASS (100% verified across 7,872 events).
- Phase 3 P1 Real LLM & In-Storage Attention Evaluation: PASS (41/41 unit tests passing).

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

P2 (FTL / FEMU) and P3 (Storage Integration) execute Phase 3 evaluations using the validated real traces and performance metrics.
