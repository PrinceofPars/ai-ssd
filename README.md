# AI-SSD: Co-Designed KV Cache & Storage Simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An architectural simulator for co-designing LLM Key-Value (KV) cache offloading with tensor-aware Solid-State Drives (SSDs). Built for rapid exploration of flash tiering, sparse attention (Top-k) retrieval, and speculative prefetching in long-context generative AI inference.

---

## Architecture Overview

```
                          LLM Workload (32K+ Context)
                                       │
                         Unified AI-SSD API (Person 3)
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
    KV Cache Engine (Person 1)                            SSD & FTL (Person 2)
    • Paged KV Block Manager                              • Flash Translation Layer (FTL)
    • Hot / Cold Classification                           • Conventional vs Tensor-Aware
    • Attention Scoring & Top-k                           • NAND Page/Block/Die/Channel Model
    • Tiered Eviction                                     • Latency & Contention Physics
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                       Speculative Prefetcher (Person 3)
                       • Layer-to-layer transition prediction
                       • Asynchronous host DRAM staging buffer
                                       │
                                       ▼
                      Evaluation & Streamlit Dashboard
```

---

## Parallel Development Rules & Ownership

Each module is owned by a single teammate. **Nobody should casually edit another person's main implementation directory.**

| Person | Domain | Directory | Owned Benchmarks |
| :--- | :--- | :--- | :--- |
| **Person 1** | AI & KV Cache | `person1_kv_engine/` | `benchmarks/run_baseline.py`, `run_offload.py`, `run_topk.py` |
| **Person 2** | SSD Hardware & FTL | `person2_ssd/` | `benchmarks/run_ftl.py` |
| **Person 3** | System, API, Prefetch & UI | `person3_system/` | `benchmarks/run_prefetch.py`, `run_full_system.py`, `demo/` |
| **Shared** | Schemas, Contracts, Config | `common/`, `config/`, `docs/` | Shared agreement before implementation |

### Mock Isolation Principle
To work in parallel without blocking one another:
- **Person 1** tests against [`person1_kv_engine.mock_ssd.MockSSD`](file:///person1_kv_engine/mock_ssd.py).
- **Person 2** tests against [`person2_ssd.mock_kv_engine.MockKVEngine`](file:///person2_ssd/mock_kv_engine.py).
- **Person 3** tests the unified pipeline using both mocks before plugging in the real implementations.

---

## Directory Structure

```text
ai-ssd/
├── docs/                      # Architectural docs, API specs, and experiment plans
├── config/                    # Workload, hardware, and experiment YAMLs
├── common/                    # Common schemas (KVBlock, KVRequest, KVResponse), constants, utilities
├── person1_kv_engine/         # KV cache, attention scoring, hot/cold classification, top-k
├── person2_ssd/               # NAND flash model, channels, conventional & tensor-aware FTL
├── person3_system/            # Unified API, speculative prefetcher, orchestrator, Streamlit dashboard
├── benchmarks/                # Individual and end-to-end benchmark runners
├── data/                      # Raw and generated simulation traces
├── results/                   # Benchmark CSVs, summary tables, and generated figures
├── demo/                      # Interactive demonstration scripts
└── scripts/                   # Setup, execution, and cleanup scripts
```

---

## Quickstart

### 1. Setup Environment
```bash
# Using pip
pip install -r requirements.txt
pip install -e .
```
On Windows:
```powershell
.\scripts\setup.ps1
```
On Linux/macOS:
```bash
chmod +x scripts/*.sh
./scripts/setup.sh
```

### 2. Run Subsystem Tests (Parallel Validation)
```bash
# Test shared contracts
pytest common/tests/ -v

# Test Person 1 with MockSSD
pytest person1_kv_engine/tests/ -v

# Test Person 2 with MockKVEngine
pytest person2_ssd/tests/ -v

# Test Person 3 with Mock Pipeline
pytest person3_system/tests/ -v
```

### 3. Launch Interactive Dashboard
```bash
streamlit run person3_system/dashboard/dashboard.py
```
