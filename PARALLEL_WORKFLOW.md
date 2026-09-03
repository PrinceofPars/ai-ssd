# AI-SSD Parallel Development & Team Workflow Guide

Welcome to the AI-SSD project! This repository is engineered so that **Person 1**, **Person 2**, and **Person 3** can work simultaneously with **zero waiting and zero cross-blocking**.

---

## 1. Team Ownership & Branch Map

```
                          main (Protected, always runnable)
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
feature/p1-kv-cache   feature/p2-ftl-storage   feature/p3-system-pipeline
    (Person 1)            (Person 2)               (Person 3)
```

| Role | Engineer | Directory Owned | Owned Benchmarks | Isolated Mock Interface |
| :--- | :--- | :--- | :--- | :--- |
| **Person 1** | AI & KV Cache | `person1_kv_engine/` | `benchmarks/run_baseline.py`<br>`benchmarks/run_offload.py`<br>`benchmarks/run_topk.py` | `person1_kv_engine/mock_ssd.py` (`MockSSD`) |
| **Person 2** | SSD Hardware & FTL | `person2_ssd/` | `benchmarks/run_ftl.py` | `person2_ssd/mock_kv_engine.py` (`MockKVEngine`) |
| **Person 3** | System, API, Prefetch & UI | `person3_system/` | `benchmarks/run_prefetch.py`<br>`benchmarks/run_full_system.py`<br>`demo/demo.py` | Connects `MockKVEngine` + `MockSSD` |

---

## 2. The Golden Rules of Parallel Work

> [!IMPORTANT]
> **Rule 1: Strict Directory Boundaries**
> - You have full freedom inside your owned folder (`person1_kv_engine/`, `person2_ssd/`, or `person3_system/`).
> - **Never directly edit another person's directory.**

> [!IMPORTANT]
> **Rule 2: Frozen Common Contracts**
> - The files in `common/schemas/` (`kv_block.py`, `request.py`, `result.py`, `metrics.py`) and `config/` are shared contracts.
> - **Nobody edits `common/` alone.** If an interface change is needed, the team must agree first.

> [!IMPORTANT]
> **Rule 3: Test Against Your Mock**
> - Person 1 must NEVER say *"I cannot test my KV eviction until Person 2 finishes the SSD."* Use `MockSSD`!
> - Person 2 must NEVER say *"I cannot test FTL striping until Person 1 finishes the attention scorer."* Use `MockKVEngine`!
> - Person 3 connects the mocks first to verify API routing and UI before real components arrive.

---

## 3. Quickstart per Person

### Person 1: AI / KV Cache Quickstart
```powershell
# 1. Create your branch
git checkout -b feature/p1-kv-cache

# 2. Run your independent tests
python scripts/run_tests.py

# 3. Run your benchmarks
python benchmarks/run_baseline.py
python benchmarks/run_offload.py
python benchmarks/run_topk.py
```
**Your Key Files**:
- `person1_kv_engine/cache_manager/kv_cache.py`: Manage paged block allocation and hot/cold offloading.
- `person1_kv_engine/topk/selector.py`: Filter candidate blocks using attention scores.
- `person1_kv_engine/mock_ssd.py`: Your simulated storage backend.

---

### Person 2: SSD / FTL Quickstart
```powershell
# 1. Create your branch
git checkout -b feature/p2-ftl-storage

# 2. Run your independent tests
python scripts/run_tests.py

# 3. Run your benchmark (Conventional vs Tensor-Aware FTL speedup)
python benchmarks/run_ftl.py
```
**Your Key Files**:
- `person2_ssd/ftl/tensor_aware.py`: Striping KV blocks across independent flash channels & dies.
- `person2_ssd/storage_model/latency.py`: Calculating channel contention and NAND read latency.
- `person2_ssd/mock_kv_engine.py`: Feeds synthetic KV traces to your storage engine.

---

### Person 3: System API, Prefetch & UI Quickstart
```powershell
# 1. Create your branch
git checkout -b feature/p3-system-pipeline

# 2. Run your independent tests
python scripts/run_tests.py

# 3. Run full system simulation & demo
python demo/demo.py
python benchmarks/run_full_system.py

# 4. Launch interactive dashboard
streamlit run person3_system/dashboard/dashboard.py
```
**Your Key Files**:
- `person3_system/api/ai_ssd.py`: Unified API gateway.
- `person3_system/prefetch/prefetcher.py`: Speculative DRAM prefetch buffer.
- `person3_system/dashboard/dashboard.py`: Streamlit demonstration app.

---

## 4. Integration Phase (Plugging Real Modules Together)

When Milestone 1 is completed:
1. **Person 1** replaces `MockSSD` with `person2_ssd.storage_model.io_model.StorageSimulator`.
2. **Person 3** replaces `MockKVEngine` with `person1_kv_engine.cache_manager.kv_cache.PagedKVCache`.
3. Run `python demo/demo.py` and `python benchmarks/run_full_system.py` to produce final competition graphs!
