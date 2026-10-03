# V1 Codebase Audit & Gap Analysis (P3 Reconnaissance)

**Date**: 2026-10-02
**Agent**: P3 (System Integration / Storage API / Prefetch / Experiments)
**Baseline Commit**: `db7e0f8`

## 1. Reusable Components
- `common/schemas/kv_block.py`: `KVBlock` dataclass and enum structures (`StorageTier`, `DType`). Well-defined fields for block ID, layer ID, token start/count, head start/count, head dimension, byte sizes.
- `common/schemas/request.py`: `KVRequest` envelope and `KVOperation` enum.
- `common/schemas/result.py`: `KVResponse` structure and `OperationStatus`.
- `common/constants.py`: Hardware timing parameters ($t_R$, $t_{PROG}$, bus transfer, PCIe overhead).
- `person2_ssd/storage_model/io_model.py`: Multi-channel flash simulator (`StorageSimulator`) providing analytical conventional vs tensor-aware channel conflict simulation.

## 2. Mock Code & Synthetic Assumptions in V1
1. **Prefetch Hit Rate Claim (~97%)**:
   - In `person3_system/integration/pipeline.py` (lines 100-113), the simulation looped over layers passing the exact same `sample_bids` to every layer.
   - `NextLayerPredictor.predict_next_layer_blocks()` predicted `target_bid = bid + stride` (stride=0), prefetching the identical block IDs for layer $L+1$.
   - In `SpeculativePrefetcher.is_staged()`, `is_hit = (hit_ratio >= 0.80)` counted anything $\ge 80\%$ as a hit, and prediction accuracy was hardcoded to `0.90` in output metrics.
   - **Verdict**: The 97% hit rate was an artifact of synthetic identical-block access patterns across layers. In real LLM inference, layer-to-layer attention patterns are dynamic and must be measured empirically.

2. **Unified API Gateway (`person3_system/api/ai_ssd.py`)**:
   - `_handle_topk` implemented top-k selection as `candidate_blocks[:k]`, completely bypassing real attention scoring.
   - Latencies were hardcoded: `latency_us = 1.0` for hits and `len(selected) * 25.0` for misses.
   - Byte calculations assumed uniform 4096 bytes per block regardless of actual data type or token count.

3. **Benchmarks (`benchmarks/run_baseline.py`, `run_full_system.py`)**:
   - Time-to-first-token (`ttft_ms`) and token throughput (`tokens_per_sec = 45.0`) were computed with hardcoded linear formulas rather than actual execution measurements.

## 3. Missing Interfaces & Integration Gaps for V2
1. **Storage Backend Abstraction (`StorageBackend`)**:
   - No uniform abstract interface exists to plug in Mock storage, in-memory/file storage, analytical FTL (`person2_ssd`), or real/FEMU NVMe block devices.
2. **Real Trace Consumer (`TraceReader`)**:
   - Missing schema-validated trace reader for P1's real LLM trace (`/opt/ai-ssd-v2/traces/real_llm/`).
   - Must validate schema versions, model metadata, block ID boundaries, head/layer ranges, monotonically increasing sequence numbers, and reject malformed traces cleanly.
3. **Rigorous Prefetch Accounting**:
   - Needs separate accounting for useful prefetches, useless prefetches (cache pollution), late prefetches, hit rate, demand misses, memory consumption, and extra I/O bandwidth overhead charged to the storage subsystem.
4. **Reproducible Experiment Runner**:
   - Needs full provenance logging (commit, CPU, RAM, kernel, OS, Python/library versions, random seed, backend, FTL mode, prefetch mode) exporting to JSON/JSONL/CSV.
5. **System Baselines & Ablations**:
   - Standardized runner for all 5 required system configurations: Dense DRAM, KV Offload (Conventional), Conventional FTL + Top-k, Tensor-Aware FTL + Top-k, and Tensor-Aware FTL + Prefetch.
