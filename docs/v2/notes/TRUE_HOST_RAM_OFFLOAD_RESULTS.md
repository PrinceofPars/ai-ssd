# True Host-RAM Offload & Elimination of Redundant KV Residency
## Results & Verification Report

**Date:** October 3, 2026  
**Owner:** Person 1 (Real LLM Inference Optimization Owner)  
**Branch:** `v2-real-llm-kvssd`  
**Classification:** True Physical Storage Offload (`[REAL-STORAGE-FILE]`)

---

## 1. Executive Summary

During the large-context scaling audit of Qwen3-4B at Context = 4096, an architectural discrepancy was detected: AI-SSD consumed **~24.2 GB process RSS** compared to **~19.4 GB for Baseline** (+4.6 GB discrepancy), despite reporting an 89.4% active KV-cache reduction.

A rigorous read-only memory audit identified four primary causes:
1. **P1 PyTorch KV Retention:** The original prefill output (`prefill_out`) and unpruned KV cache (`past_key_values`) remained referenced in local Python scope (~1,152 MB).
2. **P3 Prefetch Adapter Duplication:** `_block_payloads` maintained an unbounded duplicate in-memory dictionary of NumPy arrays and raw bytes (~2,295 MB) that was never actually read during inference decode.
3. **P2 Storage Backend Duplication:** `RealInferenceStorageBackend._storage` stored quadruplicate copies (`k.copy()`, `v.copy()`, `k_bytes`, `v_bytes`) in anonymous RAM (~2,295 MB).
4. **Unbounded Telemetry Accumulation:** `_access_log` recorded every block access as an unpruned list of dictionaries, accumulating over 150,000 entries.

By implementing **True Host-RAM Offload** across P1, P2, and P3, physical OS process memory now drops **strictly below Baseline**:
- **Baseline Decode RSS (Qwen3-4B @ 4096):** `19,405.3 MB`
- **AI-SSD Decode RSS (Qwen3-4B @ 4096):** `15,951.0 MB`
- **Net Physical Host-RAM Reduction:** **3,454.3 MB (~3.45 GB)**
- **Token Accuracy:** **100.0% Exact Match** (16/16 tokens)

---

## 2. Architectural Implementation

### Phase B: P1 Prefill KV Release & Positional Decoding
- In `person1_kv_engine/real_llm/aissd_inference.py` (`run_aissd_decode`):
  - Added immediate release: `del prefill_out, pkv_prefill; gc.collect()` right after `kv_mgr.init_from_prefill()`.
  - In autoregressive decode loop, passed explicit `position_ids = torch.tensor([[cur_seq_len + step - 1]], device=input_ids.device)` with `use_cache=False`.
  - Eliminates the original ~1.15 GB unpruned PyTorch prefill KV from process heap while maintaining RoPE alignment.

### Phase C: P3 Prefetch Adapter Zero-Copy Optimization
- In `person3_system/prefetch/inference_adapter.py` (`RealInferencePrefetchAdapter`):
  - Bypassed redundant storage in `_block_payloads` when a real storage backend is attached (`has_real_backend`).
  - Removed duplicate `raw_bytes` copy from `StagedInferenceBlock(data=None)`.
  - Retained metadata and bounded LRU staging buffer (`_staging_buffer`) strictly for active/prefetched blocks.

### Phase D: P2 True Storage Backing Store
- In `person2_ssd/inference_backend.py` (`RealInferenceStorageBackend`):
  - Integrated high-throughput direct-access backing file (`/tmp/aissd_p2_{pid}_{id}.bin`) using POSIX `os.pwrite` and `os.pread`.
  - Called `os.posix_fadvise(..., POSIX_FADV_DONTNEED)` immediately after writing and reading to signal the OS kernel to release page cache pages.
  - Converted `_storage[(layer_idx, block_id)]` into a pure metadata directory (file offset, size, shape, dtype, channel mapping) storing **zero** payload bytes in anonymous RAM.
  - Added `close()` and `__del__()` hooks to unlink temporary backing files cleanly.
  - Capped `_access_log` to a 200-entry ring buffer.

---

## 3. Measured OS Telemetry (Linux `/proc/self/status`)

### A. Qwen3-4B (Context = 4096, Decode = 16, 4 CPU Threads, FP32)

| Lifecycle Stage | Metric | Baseline | AI-SSD (Before) | AI-SSD (After Offload) |
|---|---|---|---|---|
| **Model Weights Loaded** | VmRSS | 15,798 MB | 15,798 MB | 15,796 MB |
| **After Prefill (Ctx=4096)** | VmRSS | 19,428 MB | 19,424 MB | 19,424 MB |
| **P3 Resident Storage** | Payload MB | N/A | 2,295 MB | **0.00 MB** |
| **P2 Resident Storage** | Payload MB | N/A | 2,295 MB | **0.00 MB** |
| **Total Storage RAM** | Payload MB | N/A | 4,590 MB | **0.00 MB** |
| **After Prefill KV Release** | VmRSS | N/A | N/A | **15,842 MB** (-3,584 MB) |
| **Decode Phase (RSS min)** | VmRSS | 19,405 MB | 24,112 MB | **15,938 MB** |
| **Decode Phase (RSS avg)** | VmRSS | 19,405 MB | 24,181 MB | **15,948 MB** |
| **Decode Phase (RSS peak)**| VmRSS | 19,405 MB | 24,206 MB | **15,951 MB** |
| **Physical RSS Savings** | Net vs Base | 0 MB | **+4,801 MB (Deficit)** | **-3,454 MB (Savings)** |
| **Active KV Cache DRAM** | Active KV | 1,152 MB | 122 MB | **122 MB (-89.4%)** |
| **Output Token Accuracy** | Exact Match | 100% | 100% | **100.0% (16/16)** |

### B. Canonical Benchmark: Qwen2.5-0.5B (Context = 512, Decode = 16, 3 Reps)

```
================================================================
                  PROCESS RAM TELEMETRY (MB)
================================================================
| Metric                 | Baseline           | AI-SSD             |
|------------------------|--------------------|--------------------|
| Min RSS                |  2842.3 ? 0.5    MB |  2553.8 ? 5.7    MB |
| Avg RSS                |  2842.4 ? 0.3    MB |  2554.1 ? 5.3    MB |
| Peak RSS               |  2842.4 ? 0.3    MB |  2554.2 ? 5.3    MB |
| KV memory (active)     |          12.38 MB |           1.97 MB |
| KV reduction           |              0.0% |             84.1% |
| Non-KV RAM (Peak-KV)   |         2830.0 MB |         2552.2 MB |
================================================================
```

---

## 4. Subsystem & Integration Test Verification

All test suites passed 100% cleanly:
- `person1_kv_engine/tests`: 5 passed
- `person2_ssd/tests`: 45 passed
- `person3_system/tests`: 77 passed
- **Total Pytest Suite:** **127 / 127 PASS** (100%)
- **System Integration Runner (`scripts/run_tests.py`):** **24 / 24 PASS** (100%)
