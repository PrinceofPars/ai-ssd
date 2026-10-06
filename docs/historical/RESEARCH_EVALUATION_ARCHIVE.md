# AI-SSD Research & Architecture Evaluation Archive (Consolidated)

This document consolidates historical V2 research evaluations, I/O engine benchmarks, and V1 audit findings into a single reference archive.

---

## Source: docs/v2/research/README.md

# V2 Research

Research performed by agents should be recorded when it affects
an implementation decision.

Suggested format:

Question:

Source:

Finding:

Impact:

Decision:

Agent:


---

## Source: docs/v2/research/V1_AUDIT_FINDINGS.md

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


---

## Source: docs/v2/research/REAL_LLM_KV_RESEARCH.md

# Real LLM KV Cache Research & Architecture Findings

**Author**: Person 1 (P1 - Real LLM + Real KV Cache Engine)  
**Date**: October 2026  
**Target Environment**: AWS EC2 (8 vCPU Intel Xeon Platinum 8488C Sapphire Rapids, 61 GiB RAM, AVX-512 / AVX2 / AMX)

---

## 1. Executive Summary

In AI-SSD V1, key-value (KV) cache behavior was modeled synthetically using random normal arrays (`np.random.randn`) with artificial clustering. For V2, our mandate is to establish a rigorous, reproducible pipeline starting from a **real causal language model on CPU**, extracting its genuine multi-head / grouped-query KV activations, blockizing them according to physical flash page constraints, generating deterministic I/O access traces for P2 (FTL / FEMU) and P3 (Prefetch / Integration), and empirically evaluating in-storage Top-$k$ attention pruning against a full dense attention reference.

---

## 2. Model Selection: Qwen/Qwen2.5-0.5B

### 2.1 Feasibility & Resource Budget Analysis
We evaluated candidate CPU models under the strict EC2 resource constraints (8 vCPU, ~64 GiB RAM, CPU only):
- **Model**: `Qwen/Qwen2.5-0.5B` (Qwen 2.5 series, released late 2024).
- **Parameters**: 494M parameters.
- **Precision**: Float32 inference on CPU (memory footprint: ~1.95 GiB weights), consuming only **3.1% of the 61 GiB machine RAM**, leaving >55 GiB free for OS, FEMU/QEMU, P2, and P3.
- **CPU Execution Speed**: Measured at **~20.0 tokens/second** on 4 vCPUs (`Intel Xeon Platinum 8488C`).
- **Architectural Suitability**:
  - `num_hidden_layers`: 24
  - `num_attention_heads` ($Q$): 14
  - `num_key_value_heads` ($KV$): 2 (Grouped Query Attention - GQA with ratio 7:1)
  - `head_dim`: 64
  - `hidden_size`: 896
  - `vocab_size`: 151,936
  - Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU activation.
  - Fully ungated (no private Hugging Face token required; downloads immediately and reliably).

### 2.2 Why GQA (Grouped Query Attention) Matters for AI-SSD
Modern production LLMs (Llama 3, Qwen 2.5, Mistral) utilize GQA rather than Multi-Head Attention (MHA). In GQA, multiple query heads share a single KV head (here, 7 query heads per KV head). This means:
- The KV cache is shared across query head groups.
- The storage footprint of the KV cache is reduced by $7\times$ compared to MHA, but access patterns from different query heads in the same group contend for the same physical KV blocks in storage.
- Modeling real GQA is critical for realistic FTL and prefetching evaluation in P2 and P3.

---

## 3. Real KV Cache Extraction Mechanism

HuggingFace `transformers` (v5.x / v4.36+) uses `DynamicCache` during generation.
For each transformer layer $l \in [0, 23]$:
- Key tensor: shape `[batch_size, num_kv_heads, seq_len, head_dim]` (e.g. `[1, 2, seq_len, 64]`)
- Value tensor: shape `[batch_size, num_kv_heads, seq_len, head_dim]` (e.g. `[1, 2, seq_len, 64]`)

During autoregressive decode step $t$, the query vector for head $h$ is $Q_{t, h} \in \mathbb{R}^{1 \times D}$.
The attention logits before softmax are:
$$S_{t, h, j} = \frac{Q_{t, h} \cdot K_{j, \text{group}(h)}^\top}{\sqrt{D}}$$
The attention distribution is:
$$A_{t, h} = \text{softmax}(S_{t, h, :}) \in \mathbb{R}^{t}$$

---

## 4. Physical KV Blockization & Page Separation

### 4.1 Physical vs Logical Constraints
A standard NAND flash memory page is physically **4096 bytes (4 KiB)**.
Flash controllers read and program at page granularity.
- **Key Page**: In attention scoring ($Q \cdot K^\top$), only the Key tensor is inspected during the selection / pruning phase. Value tensors are NOT touched during Top-$k$ block selection!
- **Value Page**: Only after candidate blocks are selected does the attention engine retrieve the corresponding Value tensors to compute $\sum A_j V_j$.

Therefore, Key and Value must be stored and addressed as **distinct physical pages**:
- For 16 tokens, 1 head, `head_dim = 128`, `FP16` (2 bytes/elem):
  $$\text{Key Page} = 16 \times 1 \times 128 \times 2 = 4096 \text{ bytes (4 KiB)}$$
  $$\text{Value Page} = 16 \times 1 \times 128 \times 2 = 4096 \text{ bytes (4 KiB)}$$
  $$\text{Logical KV Block} = 4096 + 4096 = 8192 \text{ bytes (8 KiB)}$$
- For `Qwen2.5-0.5B` (`head_dim = 64`, `FP32`, 4 bytes/elem):
  $$\text{Key Page (16 tokens)} = 16 \times 1 \times 64 \times 4 = 4096 \text{ bytes (4 KiB)}$$
  $$\text{Value Page (16 tokens)} = 16 \times 1 \times 64 \times 4 = 4096 \text{ bytes (4 KiB)}$$
  $$\text{Logical KV Block} = 8192 \text{ bytes (8 KiB)}$$
- For `Qwen2.5-0.5B` (`head_dim = 64`, `FP16`, 2 bytes/elem, 32 tokens):
  $$\text{Key Page (32 tokens)} = 32 \times 1 \times 64 \times 2 = 4096 \text{ bytes (4 KiB)}$$
  $$\text{Value Page (32 tokens)} = 32 \times 1 \times 64 \times 2 = 4096 \text{ bytes (4 KiB)}$$
  $$\text{Logical KV Block} = 8192 \text{ bytes (8 KiB)}$$

---

## 5. In-Storage Top-$k$ Attention Filtering

### 5.1 Principle of Offloaded Top-$k$
In standard host-driven LLM decoding with tiered storage:
1. Host requests ALL cached KV blocks over PCIe (e.g. 512 KiB - 64 MiB per token per layer).
2. Host computes attention scores.
3. PCIe bus saturates, stalling token generation.

In AI-SSD in-storage computing:
1. Host issues a lightweight `KV_TOPK` request containing only Query vector $Q_{t, h}$ (256 bytes) and candidate block IDs.
2. The SSD controller's embedded core (simulated via C kernel) reads ONLY the Key pages from local NAND buffers into controller SRAM/DRAM.
3. The controller computes dot-product block salience:
   $$\text{score}(B) = \max_{t \in B} \frac{Q \cdot K_t^\top}{\sqrt{D}}$$
4. The controller identifies the top $k$ highest-scoring blocks.
5. Only the corresponding selected $V$ (and $K$) pages are transferred across the PCIe bus to host DRAM.
6. Bandwidth reduction: from $100\%$ down to $1\% - 20\%$ depending on sparsity budget!

### 5.2 Attention Sink Phenonemon
Real LLM attention distributions (Xiao et al., StreamingLLM; Zhang et al., H2O) exhibit two non-uniform properties:
1. **Initial Attention Sinks**: The first 4 tokens (prompt initializers) receive persistent high attention mass regardless of context length. These MUST always be kept in fast Host DRAM / GPU HBM.
2. **Recent Local Window**: The most recent tokens (e.g. last 32-64 tokens) capture local syntactic dependencies and should remain host-resident.
3. **Middle Context**: The bulk of long-context KV history resides in SSD flash, where Top-$k$ pruning achieves massive bandwidth savings with minimal perplexity degradation.


---

## Source: docs/v2/research/IO_ENGINES.md

# Practical Linux I/O Mechanisms for AI-SSD V2 (P3 Research)

**Date**: 2026-10-02
**Agent**: P3 (System Integration)

## 1. Overview of Evaluated I/O Mechanisms

To connect KV offloading with real storage / NVMe / FEMU, the upper storage layer requires low overhead, high concurrency, and fine-grained latency profiling without OS buffering artifacts.

| Mechanism | Kernel Req | Pros | Cons | Verdict for V2 |
|---|---|---|---|---|
| **Synchronous POSIX (`read`/`pread`)** | Any Linux | Deterministic, simple, universal, zero dependency | Blocks calling thread, cannot overlap I/O easily without threads | Baseline engine for deterministic analytical timing |
| **POSIX Direct I/O (`O_DIRECT`)** | Any Linux | Bypasses Linux page cache, forces actual storage I/O, realistic latency | Alignment requirements (typically 512B or 4096B boundaries) | **Mandatory** for real NVMe/file bench to prevent page-cache masking |
| **POSIX AIO (`libaio`)** | Linux 2.6+ | Kernel-level async for block devices | Only truly async with `O_DIRECT`, awkward signal/eventfd interface, legacy | Usable, but surpassed by io_uring |
| **Threadpool Asynchronous (`concurrent.futures`)** | Any Python | Portable, works across all backends, allows clean task pipelining | Thread context switch overhead (~2-5 us) | **Primary user-space async** for portable mock & analytical backends |
| **`io_uring` (Linux 5.1+)** | Linux 5.1+ | Zero-copy submission/completion queues, lowest system call overhead, polling mode (`IORING_SETUP_SQPOLL`) | Requires external C-binding or Python wrapper (`liburing`), kernel version sensitivity | Ideal for future high-speed FEMU NVMe raw character/block device |

## 2. Selection Rationale for V2 Implementation

1. **Analytical / Mock Mode**:
   - Uses `StorageBackend` interface with simulated hardware physics (NAND channel contention, bus transfer delay, queue wait).
   - Python async wrapper via `concurrent.futures.ThreadPoolExecutor` provides non-blocking futures with deterministic simulated time stamps.

2. **File / RAM-Disk / Block Storage Mode**:
   - `FileStorageBackend` implements both buffered and `O_DIRECT` modes.
   - Using `O_DIRECT` ensures that read/write requests measure true disk I/O rather than RAM cache hits.

3. **FEMU / NVMe Device Mode**:
   - Direct I/O against `/dev/nvmeXnY` or file-backed image.
   - Decoupled via `StorageBackend` so switching to `io_uring` requires only a drop-in backend implementation without touching prefetch or experiment runner logic.


---

## Source: docs/v2/research/FEMU_QEMU_NVME_EVALUATION.md

# FEMU vs. QEMU Native Virtual NVMe: Architecture, Feasibility & Evaluation

**Author**: Person 2 (Storage & Systems Engineer)  
**Date**: 2026-10-02  
**Context**: AI-SSD V2 Virtual Storage Evaluation  

---

## 1. Background & Objectives

The primary mandate of Person 2 in AI-SSD V2 is to transition from the pure V1 Python analytical simulator into a verified 3-level storage evaluation methodology:

1. **Level 1**: Analytical / Cycle-Style Simulator (NAND geometry, contention physics, bus transfer).
2. **Level 2**: Trace-Driven Storage Evaluation (Replaying LLM KV access traces against FTL models, tracking queue depth, channel contention, and die utilization).
3. **Level 3**: Executable Virtual NVMe Storage Path (Operating an emulated NVMe controller under the Linux kernel NVMe driver, exercised via `fio` and `nvme-cli`).

This research document evaluates the practical implementation options for Level 3: **FEMU** vs. **QEMU Native Virtual NVMe**.

---

## 2. FEMU Investigation & Feasibility Analysis

### 2.1 What is FEMU?
FEMU (Fast, Exact, Flash Emulator) is an open-source NVMe SSD emulator developed by Huaicheng Li et al. (published in USENIX FAST 2018). It is implemented as a specialized patchset on top of QEMU (historically QEMU 2.9, 5.0, or 7.0).

FEMU implements an internal C-based flash translation layer (FTL) and delay emulation engine:
- **Black-box SSD (`bbssd`)**: Emulates page-level FTL with channel/chip/die delay queuing.
- **ZNS SSD (`zns`)**: Emulates Zoned Namespaces.
- **OC-SSD (`ocssd`)**: Emulates Open-Channel SSD 2.0 interface.
- **NoSSD (`nossd`)**: Pure NVMe pass-through with minimal delay emulation.

### 2.2 Feasibility & Constraints on the Current Host Machine
The evaluation environment is an AWS EC2 instance with:
- **8 vCPUs** (Shared across P1, P2, P3, and OS).
- **61 GiB RAM**.
- Target resource allocation for P2: **~2 vCPUs**.

#### Critical Limitations of In-Tree FEMU Compilation:
1. **Compilation Footprint & Host Disruption**:
   - Building FEMU requires cloning the full QEMU repository (~1.2 GB), configuring with Meson/Ninja, and compiling ~4,000 C/C++ source files.
   - A full QEMU compilation consumes 6–8 cores at 100% CPU for 20–35 minutes, generating 8–15 GB of build artifacts.
   - On a shared 8-vCPU instance, this creates severe resource contention, risking timeouts or crashes in P1 (LLM training/inference) and P3 (orchestration).
2. **Custom FTL Logic Inflexibility**:
   - FEMU's internal FTL is written in low-level C (`femu/bbssd/ftl.c`) and implements fixed sequential/greedy striping.
   - Injecting AI-SSD's tensor-aware placement algorithm into FEMU requires modifying QEMU C internals, risking emulator instability and kernel memory corruption inside the guest.
3. **Reproducibility**:
   - Compiling out-of-tree QEMU forks creates brittle toolchain dependencies across different environments.

---

## 3. The QEMU Native Virtual NVMe Path (Adopted Level 3 Path)

### 3.1 Architecture
The host environment already provides:
- **QEMU 6.2.0** (`/usr/bin/qemu-system-x86_64`) pre-installed.
- **KVM Acceleration** (`/dev/kvm` accessible with hardware virtualization).
- **In-Kernel Linux NVMe Driver** (`CONFIG_BLK_DEV_NVME=y` built into host kernel `6.5.0-1020-aws`).
- **Standard Storage Tooling**: `nvme-cli` (v1.16) and `fio` (v3.28).

### 3.2 Implemented Level 3 Storage Pipeline
To achieve a completely reproducible, zero-overhead virtual NVMe execution path:
1. **Virtual Controller**: QEMU PCI NVMe 1.4 controller (`-device nvme`).
   - Configured with `num_queues=8` (or `max_ioqpairs=8`) to model 8 independent host I/O queue pairs matching the 8 NAND channels.
   - Formatted with 4096-byte logical and physical block size (`logical_block_size=4096, physical_block_size=4096`), matching the 4 KB KV block / flash page unit.
   - Serial number: `v2-ai-ssd-001`.
2. **Storage Backend**: A sparse raw disk image (`/opt/ai-ssd-v2/images/v2_nvme.raw`, 1.0 GiB logical, 0 bytes initial physical).
3. **Lightweight Micro-Kernel Boot Environment**:
   - Direct kernel boot (`-kernel /opt/ai-ssd-v2/images/vmlinuz`) with hardware KVM acceleration.
   - Tailored micro-initramfs (`/opt/ai-ssd-v2/images/initramfs.cpio.gz`, 18 MB) containing static BusyBox, `nvme-cli`, and `fio` with dynamic libraries resolved.
   - Boots in **< 1.5 seconds**, executes automated device discovery, controller validation, namespace query, direct I/O read/write tests, and multi-depth `fio` benchmarks, then powers off cleanly.

### 3.3 Verification & Smoke Test Results (Measured 2026-10-02)
- **Controller Identified**: `QEMU NVMe Ctrl`, Serial: `v2-ai-ssd-001`, Firmware: `1.0`.
- **Namespace Verified**: `/dev/nvme0n1`, Size: `0x40000` blocks (1.0 GiB), `lbads: 12` (4096 B per LBA).
- **Direct I/O Sequential Write (4KB blocks)**: Passed (100 blocks written cleanly).
- **FIO Random Read (4KB, QD=8, libaio)**:
  - Throughput: **92.1 MiB/s** (96.6 MB/s).
  - IOPS: **23.6k IOPS**.
  - Average Latency: **338.57 $\mu$s** (min: 73 $\mu$s, 99th percentile: 379 $\mu$s).
- **FIO Random Write (4KB, QD=8, libaio)**:
  - Throughput: **98.0 MiB/s** (103 MB/s).
  - IOPS: **25.1k IOPS**.
  - Average Latency: **317.94 $\mu$s** (min: 64 $\mu$s, 99th percentile: 379 $\mu$s).

---

## 4. Key Limitations & Distinction Between Simulation Levels

As mandated by the V2 experimental protocol:
- **Level 3 (Virtual NVMe / QEMU)** proves full operating system and driver compatibility, DMA queue dispatch, and block-level I/O through the Linux kernel storage stack. However, the host backing file stores data in host OS page cache/file system blocks; it does *not* emulate internal NAND channel contention or $t_R$/$t_{\text{PROG}}$ physics.
- **Level 1 (Analytical Simulator)** and **Level 2 (Trace Replayer)** provide the cycle-accurate contention model for 8 channels, 4 dies/channel, and physical bus serialization.
- Level 3 results must **never** be conflated with Level 1/2 results. Both paths are maintained and reported with clear separation.
