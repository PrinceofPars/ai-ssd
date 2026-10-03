# TRUE HOST-RAM OFFLOAD & REDUNDANT KV ELIMINATION PLAN

**Target Branch:** `v2-real-llm-kvssd`  
**Owner:** Person 1 (Real LLM Inference Path)  
**Assigned Tmux Session:** `p1`  
**Execution Context:** AWS EC2 `c7i.2xlarge` / Intel Xeon Platinum 8488C  
**Date:** October 3, 2026  

---

## 1. Problem Statement & Audit Findings

At `Qwen/Qwen3-4B-Instruct-2507` with Context = 4096 tokens, our read-only memory audit discovered that while AI-SSD reported an active KV DRAM reduction of 89.4% (1,152 MB -> 122.6 MB), the actual OS process Resident Set Size (`VmRSS`) **increased** by ~4.5 GB:

- **Baseline Final RSS:** 19,405.0 MB (Peak: 25,287.3 MB)
- **AI-SSD Final RSS:** 24,204.6 MB (Peak: 25,946.1 MB)
- **Net RSS Deficit:** **+4,799.6 MB (+24.7%)**

### Exact Breakdown of the +4.6 GB Memory Inflation:
1. **Original PyTorch Prefill KV (1,152.0 MB):** In `person1_kv_engine/real_llm/aissd_inference.py`, `prefill_out` and `pkv_prefill` remained in local scope in `run_aissd_decode` and were never deleted or garbage collected.
2. **P3 Adapter Unbounded Payloads (2,295.0 MB):** In `person3_system/prefetch/inference_adapter.py`, `write_block()` inserted `k_arr.copy()`, `v_arr.copy()`, and raw concatenated `bytes` into `self._block_payloads[(layer, block)]` for all 9,180 blocks. Because `self.storage_backend` has `read_key_page`, this dictionary was never even read during decode.
3. **P2 Backend Redundant Tables (2,295.0 MB):** In `person2_ssd/inference_backend.py`, `write_block()` stored `k_arr.copy()`, `v_arr.copy()`, `k_bytes`, and `v_bytes` in `self._storage[(layer, block)]`, keeping 4 copies in anonymous RAM.
4. **P2 Access Log (150,000+ dicts):** Appended dictionaries on every read and write, fragmenting Python heap.

---

## 2. Target Architectural Lifetime of a KV Block

```
     [Prefill Forward (Context = 4096)]
                    ?
                    ?
     Full KV Tensor Created in PyTorch (1,152 MB)
                    ?
                    ?
     [P1: Blockization & Storage Write]
       - Extract Sinks (4 tokens) -> retain in P1 DRAM
       - Extract Recent Window (16 tokens) -> retain in P1 DRAM
       - Extract Historical Blocks (4076 tokens -> 254 blocks/layer)
       - Write to Storage Backend via P3 Adapter -> P2 FTL
                    ?
                    ?
     [RELEASE STEP: True Memory Reclamation]
       - del prefill_out, pkv_prefill
       - gc.collect()
       - Original 1,152 MB PyTorch KV cache completely freed from heap!
                    ?
                    ?
     [P3: Bounded Staging Cache Only]
       - Eliminate `_block_payloads` full replica dictionary
       - Bounded LRU staging buffer (`_staging_buffer`) capacity = 512 blocks (~64 MB)
       - Evicts oldest blocks when capacity exceeded
                    ?
                    ?
     [P2: Storage-Backed Representation]
       - Backed by persistent block store / file storage (raw I/O / pread)
       - Eliminate redundant duplicate numpy + bytes in anonymous process heap
       - Retain FTL channel striping, coordinates, and latency telemetry
                    ?
                    ?
     [Autoregressive Decode Step]
       - Top-k candidate keys scored via AVX2 SIMD kernel directly from storage
       - Only winning Value blocks (10% = 26 blocks) fetched into active attention
       - Attention computes on working slice (122.6 MB)
```

---

## 3. Systematic Implementation Phases

### Phase B: Release Original PyTorch KV
- In `run_aissd_decode()`, after `kv_mgr.init_from_prefill(pkv_prefill)` completes and all blocks are persisted:
  ```python
  del prefill_out
  del pkv_prefill
  gc.collect()
  ```
- Verify that `kv_mgr` retained cloned sinks and recent window so decode executes with exact numerical validity.

### Phase C: Eliminate P3 Unbounded Duplication
- In `RealInferencePrefetchAdapter`:
  - When `storage_backend` is attached (production path), `_block_payloads` is not populated.
  - P3 maintains only metadata `_block_meta` and bounded LRU `_staging_buffer`.
  - Demand reads transparently pass through to `storage_backend.read_key_page()` and `read_value_page()`.

### Phase D & E: Storage-Backed P2 Representation
- In `RealInferenceStorageBackend`:
  - Eliminate the quadruplicate RAM storage (`k`, `v`, `k_bytes`, `v_bytes`).
  - Persist block payloads to a raw direct-access backing file (`/tmp/aissd_p2_storage.bin` or configurable directory) using `os.pwrite` and `os.pread`.
  - Drop OS page-cache pages for cold blocks using `posix_fadvise(POSIX_FADV_DONTNEED)` so data does not occupy kernel buffer cache or process anonymous RAM.
  - Preserve all FTL channel telemetry, contention statistics, and tensor-aware striping.

### Phase F & G: Controlled 4096-Context Validation
- Measure before/after RSS using `/proc/self/status` across all checkpoints.
- Verify Baseline Peak RSS vs. AI-SSD Peak RSS.
- Acceptance condition: AI-SSD Peak RSS must be **strictly lower** than Baseline Peak RSS.

