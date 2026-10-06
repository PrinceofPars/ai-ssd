# AI-SSD: Team Subsystem Ownership & Full Engineering Breakdown

This document provides a line-by-line, module-by-module breakdown of what each person engineered from scratch to build the AI-SSD co-designed system.

---

## 1. Overview of Team Roles

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  AI-SSD ENGINEERING TEAM                               │
├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│         PERSON 1         │          PERSON 2           │           PERSON 3            │
│   AI Algorithms & KV     │     SSD Hardware & FTL      │   System Pipeline, Prefetch   │
│         Engine           │          Physics            │           & UI Demo           │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ • Paged KV Cache Pool    │ • Physical NAND Model       │ • Unified API Gateway         │
│ • Attention Sinks/Window │ • 8-Channel / 4-Die Layout  │ • Speculative Prefetcher      │
│ • 80% Cold Offloader     │ • Channel Contention Model  │ • Next-Layer Predictor        │
│ • In-Storage Pruner      │ • Conventional FTL          │ • Physical Storage Adapter    │
│ • Native C SIMD Kernel   │ • Tensor-Aware FTL Striping │ • Multi-Layer Pipeline        │
│ • Online Softmax Merger  │ • Multi-Channel Speedup     │ • Streamlit UI Dashboard      │
└──────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

---

## 2. Person 1: AI Algorithms & KV Cache Engine

### Domain & Directory: `person1_kv_engine/`
**Mission**: Break down the Transformer KV cache, manage host-to-storage tiering, score historical tokens for sparse attention, and compute in-storage pruning.

### Key Modules Built from Scratch:
1. **Paged KV Cache Pool (`person1_kv_engine/tiering/kv_block.py`, `cache_manager/kv_cache.py`)**:
   - Designed the standardized `KVBlock` schema grouping 16 tokens $\times$ 1 head (4 KB for Key, 4 KB for Value), matching physical 4 KB flash pages.
   - Built the `KVBlockPool` memory manager to allocate, index, pin, and recycle blocks across sequence generations.
2. **Hot/Cold Tiering Classifier (`person1_kv_engine/tiering/hot_cold_classifier.py`, `tiered_kv_manager.py`)**:
   - Implemented `TieringPolicy`:
     - Attention Sinks: Pins initial 64 tokens permanently in Host RAM.
     - Sliding Window: Keeps recent 512 tokens hot in Host RAM.
     - Cold Blocks: Offloads the intermediate 80% to SSD storage.
   - Built `TieredKVManager`: Slices prefill sequences, maintains hot/cold boundaries during decode token appending, and manages eviction and retrieval.
3. **In-Storage Attention Pruner (`person1_kv_engine/computational_storage/instorage_pruner.py`)**:
   - Bridges host attention with storage attention.
   - Evaluates query $Q$ against hot tokens locally in Host RAM.
   - Dispatches query $Q$ to SSD storage, commands storage to filter Top-$k$ cold blocks, and merges results.
4. **Numerically Stable Online Streaming Softmax (`person1_kv_engine/computational_storage/streaming_softmax.py`)**:
   - Formulated the FlashAttention-style online softmax merger.
   - Dynamically tracks running maximum $m$ and sum of exponentials $l$, allowing partial attention from Host RAM and SSD Controller to merge with zero numerical overflow or precision loss.
5. **Freestanding Native C SIMD Kernel (`person1_kv_engine/c_kernel/instorage_attention.c`, `kernel_binding.py`)**:
   - Wrote a 100% freestanding C kernel with zero libc/CRT runtime dependencies for embedded execution on SSD controller microcontrollers (ARM Cortex-R or RISC-V).
   - Implemented 4-way SIMD loop unrolling for dot-product attention:
     $$\text{score}(b) = \max_{t,h} \left( \frac{1}{\sqrt{D}} \sum_d q_{h,d} \cdot k_{t,h,d} \right)$$
   - Built an in-place insertion-sort Top-$k$ selector with $O(N \cdot k)$ complexity.
   - Compiled into `instorage_attention.dll` and exposed via Python `ctypes`.
6. **Isolated Mocking Interface (`person1_kv_engine/mock_ssd.py`)**:
   - Allowed Person 1 to develop, stress test, and debug all AI algorithms completely independently before Person 2's hardware model was finished.

### Owned Benchmarks & Verification:
- `benchmarks/run_baseline.py`: Measured dense in-memory KV footprint (16.4 GB at 32K context).
- `benchmarks/run_offload.py`: Verified **80.0% Host RAM reduction** (down to 3.28 GB).
- `benchmarks/run_topk.py`: Verified **90.0% PCIe bus traffic reduction** with high attention recall.

---

## 3. Person 2: SSD Hardware & Flash Translation Layer (FTL)

### Domain & Directory: `person2_ssd/`
**Mission**: Simulate the physical NAND flash silicon hierarchy, model channel serialization and contention physics, and invent a tensor-aware FTL that eliminates channel hot-spotting.

### Key Modules Built from Scratch:
1. **Physical NAND Flash Hierarchy (`person2_ssd/nand/`, `channels/`)**:
   - Modeled the complete physical geometry of modern enterprise enterprise NVMe drives:
     - 8 Independent Physical Flash Channels (`channels/channel.py`).
     - 4 Dies per Channel (`channels/die.py`).
     - 2 Planes per Die, 64 Blocks per Plane, and 128 Pages per Block.
   - Embedded physical timing constraints:
     - $t_R = 25.0\ \mu\text{s}$ (Cell read sense time).
     - $t_{\text{PROG}} = 200.0\ \mu\text{s}$ (Cell program time).
     - $t_{\text{BERS}} = 2,000.0\ \mu\text{s}$ (Block erase time).
     - $t_{\text{bus}} = 3.33\ \mu\text{s}$ (NV-DDR3 bus transmission per 4 KB page).
     - $t_{\text{pcie}} = 10.0\ \mu\text{s}$ (NVMe command issue & completion DMA overhead).
2. **Channel Contention Physics Model (`person2_ssd/storage_model/latency.py`)**:
   - Created the mathematical contention latency model:
     $$T_{\text{read}} = t_{\text{pcie}} + \max_{c \in [0, C-1]} \left( N_c \times (t_R + t_{\text{bus}}) \right)$$
   - Discovered and proved that if multiple read requests map to the same channel $c$, physical operations serialize linearly, creating massive latency spikes.
3. **Conventional FTL Simulator (`person2_ssd/ftl/conventional.py`)**:
   - Implemented standard industry page-level FTL (`Page -> Block -> Plane -> Die -> Channel`).
   - Demonstrated that contiguous KV block reads cause severe Channel 0 hot-spotting ($N_0 = 16, N_{1..7} = 0$), serializing all reads to $490\ \mu\text{s}$.
4. **Tensor-Aware FTL Striping Algorithm (`person2_ssd/ftl/tensor_aware.py`)**:
   - Engineered a breakthrough mathematical placement function that maps blocks across channels and dies using tensor coordinates (layer, head, and token block index):
     $$\text{Channel} = (h + b_{\text{idx}} + (b_{\text{idx}} // C)) \pmod C$$
     $$\text{Die} = (L + (h // C) + (b_{\text{idx}} // C)) \pmod{D_{\text{channel}}}$$
     where $b_{\text{idx}}$ is the token block index and $D_{\text{channel}}$ is dies per channel.
   - Completely balances load across all 8 channels ($N_c = 2$ per channel for 16 blocks), slashing read time from $490\ \mu\text{s}$ down to $70\ \mu\text{s}$!
5. **Physical Storage Simulator (`person2_ssd/storage_model/io_model.py`)**:
   - Glues the FTL mapping, allocator, and latency model into a unified storage simulator.
6. **Isolated Mocking Interface (`person2_ssd/mock_kv_engine.py`)**:
   - Allowed Person 2 to stress-test 8-channel load balancing and multi-die parallelism with synthetic KV traces before Person 1 was ready.

### Owned Benchmarks & Verification:
- `benchmarks/run_ftl.py`: Evaluated read latency across batch sizes (16 to 256 blocks).
  - Verified **7.00× to 7.93× read speedup** over conventional FTL (approaching theoretical 8.0× maximum).

---

## 4. Person 3: System Pipeline, Prefetch & UI Demo

### Domain & Directory: `person3_system/`
**Mission**: Connect Person 1's AI algorithms with Person 2's physical SSD model, build the speculative prefetcher to hide flash latency bubbles, and create the evaluation dashboard and demo.

### Key Modules Built from Scratch:
1. **Physical Storage Adapter (`person3_system/adapters/physical_adapter.py`)**:
   - Engineered the 455-line bridge connecting Person 1's abstract `KVStorageInterface` with Person 2's `StorageSimulator`.
   - Encodes `(layer_id, block_id)` into unique 32-bit physical global IDs.
   - Manages SSD controller internal DRAM caching, tracks PCIe DMA traffic, computes embedded DSP energy in Joules, and invokes the native C in-storage filtering kernel.
2. **Unified System API Gateway (`person3_system/api/ai_ssd.py`, `requests.py`, `responses.py`)**:
   - Standardized the request-response contract: `STORE_KV`, `LOAD_KV`, `TOPK_RETRIEVAL`, and `PREFETCH_KV`.
   - Routes requests cleanly between AI KV engine and storage backends.
3. **Speculative Prefetch Engine (`person3_system/prefetch/prefetcher.py`)**:
   - Built the asynchronous Host DRAM staging buffer (LRU cache).
   - Overlaps GPU computation of Layer $L$ with background flash retrieval of Layer $L+1$ cold blocks.
   - Evaluates buffer hits, partial hits, and calculates exact stall penalties:
     $$\text{Bubble Penalty} = \max(0, T_{\text{flash}} - T_{\text{GPU}})$$
4. **Next-Layer Predictor (`person3_system/prefetch/predictor.py`, `history.py`)**:
   - Models inter-layer attention locality and token-span correlation.
   - Tracks access frequency across decoding steps and predicts target block IDs with 90% confidence.
5. **End-to-End Orchestrator & Pipeline (`person3_system/integration/orchestrator.py`, `pipeline.py`)**:
   - Assembles Person 1, Person 2, and Person 3 into a single runnable production pipeline.
   - Drives multi-layer autoregressive generation simulation across 4K, 8K, 16K, and 32K context lengths.
6. **Full System Benchmark (`benchmarks/run_full_system.py`)**:
   - Executes multi-layer simulation, validates all locked scorecard metrics, outputs `results/raw/metrics.json`, and records `full_system_scaling.csv`.
7. **Interactive Demonstration CLI & Streamlit Dashboard (`demo/demo.py`, `person3_system/dashboard/dashboard.py`)**:
   - Built rich terminal presentation showing live phase-by-phase execution and final competition scorecard.
   - Built multi-panel Streamlit dashboard with real-time sliders for context length, offload percentage, and interactive latency/RAM visualizations.

### Owned Benchmarks & Verification:
- `benchmarks/run_prefetch.py`: Verified **97.0% prefetch hit rate** and bubble elimination.
- `benchmarks/run_full_system.py`: Verified net **+0.5% end-to-end latency overhead**.
- `demo/demo.py`: Executed live end-to-end simulation.

---

## 5. Cross-Cutting Common Contracts & Utilities

### Shared Directory: `common/` and `config/`
- `common/schemas/kv_block.py`: Common dataclass for `KVBlock` with block ID, token range, head indices, and physical flash location.
- `common/schemas/request.py` & `result.py`: Standardized system API messages.
- `common/schemas/metrics.py`: Dataclasses for memory, latency, storage, prefetch, and FTL metrics.
- `common/constants.py`: Physical NAND and hardware constants ($t_R$, $t_{\text{PROG}}$, channels, dies, head dim).
- `config/system.yaml` & `config/workload.yaml`: Centralized configuration profiles.
