# AI-SSD: Co-Designed KV Cache & Storage Simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 63 Passed](https://img.shields.io/badge/tests-63%20passed-brightgreen.svg)]()

An architectural simulator for co-designing LLM Key-Value (KV) cache offloading with tensor-aware Solid-State Drives (SSDs). Built for breaking the KV cache memory wall in long-context (32K–128K) generative AI inference via flash tiering, sparse attention (Top-k) retrieval, and speculative prefetching.

---

## Verified Scorecard (32K Context Length)

| Architectural Metric | Baseline Target | Measured Value | Verification Status |
| :--- | :---: | :---: | :---: |
| **Host RAM Footprint Reduction** | $\ge 80.0\%$ | **80.0%** (16.4 GB $\rightarrow$ 3.28 GB) | Verified |
| **PCIe I/O Bus Traffic Saved** | $\ge 80.0\%$ | **90.0%** (214.7 MB $\rightarrow$ 21.4 MB) | Verified |
| **Multi-Channel FTL Read Speedup** | $\ge 7.00\times$ | **7.66×** (4,900 $\mu$s $\rightarrow$ 640 $\mu$s) | Verified |
| **Speculative Prefetch Cache Hit Rate** | $\ge 80.0\%$ | **97.0%** | Verified |
| **End-to-End Latency Overhead** | $\le 18.0\%$ | **+0.5%** (116.38 ms $\rightarrow$ 116.96 ms) | Verified |

---

## In-Depth Documentation & Guides

Comprehensive technical deep-dives explaining every component from scratch:
- **[Deep Dive Architecture & Execution Guide](docs/DEEP_DIVE_EXPLANATION.md)**: Explains the KV cache memory wall, why naive SSD offloading fails, mathematical formulas, physical NAND channel contention, online streaming softmax, and empirical results.
- **[Real LLM & VLM Deployment Guide](docs/REAL_LLM_VLM_DEPLOYMENT_GUIDE.md)**: Explains which optimizations can be applied directly to real LLMs (Llama-3, Mistral) and VLMs (Qwen2-VL), what is possible today on commodity hardware, what requires custom Computational Storage / ZNS SSDs, and what is physically impossible.
- **[Team Contributions & Subsystem Breakdown](docs/TEAM_CONTRIBUTIONS_AND_ROLES.md)**: Exhaustive breakdown of what Person 1, Person 2, and Person 3 engineered from scratch.
- **[Problem Statement & Math](docs/problem_statement.md)**: Mathematical formulation of the memory explosion.
- **[Experimental Scorecard](docs/results.md)**: Verified benchmark scorecards and scaling tables.

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

## Team Ownership & Structure

| Person | Domain | Directory Owned | Core Modules & Deliverables |
| :--- | :--- | :--- | :--- |
| **Person 1** | AI & KV Cache | `person1_kv_engine/` | Paged KV Pool, Hot/Cold Classifier, In-Storage Pruner, Native C SIMD Kernel (`instorage_attention.dll`), Online Softmax Merger |
| **Person 2** | SSD Hardware & FTL | `person2_ssd/` | 8-Channel / 4-Die NAND Model, Channel Contention Physics, Conventional FTL, Tensor-Aware Striping FTL |
| **Person 3** | System, API & UI | `person3_system/` | Physical Storage Adapter, Unified API Gateway, Speculative Prefetcher, Multi-Layer Pipeline, Streamlit Dashboard |
| **Shared** | Contracts & Config | `common/`, `config/` | Frozen schemas (`KVBlock`, `KVRequest`, `KVResponse`), constants, workload YAMLs |

---

## Directory Structure

```text
ai-ssd/
├── docs/                      # Deep dive docs, deployment guides, team contributions
├── config/                    # Workload, hardware, and experiment YAMLs
├── common/                    # Common schemas (KVBlock, KVRequest, KVResponse), constants, utilities
├── person1_kv_engine/         # KV cache, attention scoring, hot/cold classification, top-k
├── person2_ssd/               # NAND flash model, channels, conventional & tensor-aware FTL
├── person3_system/            # Unified API, speculative prefetcher, orchestrator, Streamlit dashboard
├── benchmarks/                # Individual and end-to-end benchmark runners
├── data/                      # Raw and generated simulation traces
├── results/                   # Benchmark CSVs, summary tables, and generated figures
├── demo/                      # Interactive demonstration scripts
└── scripts/                   # Setup, execution, test runners, and cleanup scripts
```

---

## Quickstart

### 1. Setup Environment
```powershell
# On Windows
.\scripts\setup.ps1
```
Or with pip:
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite (63 Unit & Integration Tests)
```powershell
.\.venv\Scripts\pytest
```

### 3. Run Benchmarks
```powershell
# Person 1 Benchmarks
python benchmarks/run_baseline.py
python benchmarks/run_offload.py
python benchmarks/run_topk.py

# Person 2 Benchmark (Conventional vs Tensor-Aware FTL Speedup)
python benchmarks/run_ftl.py

# Person 3 & Full System End-to-End Simulation
python benchmarks/run_prefetch.py
python benchmarks/run_full_system.py
```

### 4. Run Demonstration CLI
```powershell
python demo/demo.py
```

### 5. Launch Interactive Streamlit Dashboard
```powershell
streamlit run person3_system/dashboard/dashboard.py
```
