# AI-SSD V2 — Final System Architecture

## 1. Overview

The AI-SSD V2 platform implements a hardware/software co-designed computational-storage subsystem for Large Language Model (LLM) KV cache offload. Rather than treating the solid-state drive as a passive block device requiring high-bandwidth PCIe streaming of all candidate Key vectors into host DRAM, AI-SSD V2 integrates in-storage tensor-aware filtering, contiguous 8 KiB Key-Value block packaging, asynchronous pipelining, and multi-channel flash translation.

```text
                                  Qwen3-4B
                                     │
                                     ▼
                              P1 KV Engine
                                     │
                              ┌──────┴──────┐
                              │             │
                            Query          KV
                              │             │
                              ▼             ▼
                        P3 Prefetch    Blockization
                              │             │
                              └──────┬──────┘
                                     ▼
                            P2 Storage Backend
                                     │
                                     ▼
                           QEMU/NVMe Controller  [VIRTUAL-DEVICE]
                                     │
                            ┌────────┴────────┐
                            │                 │
                     In-storage Top-K    FTL Mapping
                     [VIRTUAL-DEVICE]  [VIRTUAL-DEVICE]
                            │                 │
                            └────────┬────────┘
                                     ▼
                               Winning KV
                                     │
                                     ▼
                              Qwen Attention
                                     │
                                     ▼
                                Next Token
```

---

## 2. Architectural Components & Boundaries

### 2.1 Person 1: KV Engine & Attention Execution (`[REAL]`)
- **Model Execution**: Genuine Hugging Face PyTorch implementation of `Qwen/Qwen3-4B-Instruct-2507` running on host CPU (4 dedicated OpenMP/BLAS threads, FP32 precision).
- **KV Partitioning**:
  - **Attention Sinks**: 4 tokens (resident in host DRAM).
  - **Recent Window**: 16 tokens (resident in host DRAM).
  - **Historical Cache**: Offloaded to storage in fixed 16-token blocks ($16 \times 8 \times 128 \times 4\text{ B} = 65,536\text{ B}$ for K, $65,536\text{ B}$ for V = $131,072\text{ B}$ per combined KV block).
- **Active Selection**: $10\%$ Top-$K$ candidate selection executed per transformer layer per decode step.

### 2.2 Person 3: Prefetch Adapter & Asynchronous Pipeline (`[REAL]`)
- **Asynchronous DMA Engine**: Non-blocking retrieval orchestration with reentrant lock synchronization (`threading.RLock`) in `QemuNvmeClient`.
- **Contiguous 8 KiB Retrieval**: Combines separate Key and Value transfers into contiguous single batch reads, cutting storage request transactions by 50% (from 1,080 down to 540 batches).
- **Speculative Inter-Layer Prefetching**: Overlaps raw storage retrieval latency behind host attention matmul and MLP execution. In Phase 8/9 ablation (Run F), 9.94 s of raw storage latency (68.3%) was hidden.

### 2.3 Person 2: Multi-Channel Flash FTL & NVMe Controller (`[VIRTUAL-DEVICE]`)
- **Direct NVMe Integration**: Communicates directly through the Linux kernel NVMe driver (`/dev/nvme0n1`) inside a hardware-accelerated QEMU/KVM virtual machine.
- **8-Channel Tensor-Aware FTL**:
  - Strips GQA heads and block coordinates across 8 parallel flash channels.
  - Achieves near-perfect channel balance (**0.86% load imbalance** vs **700.0% in conventional sequential mapping**).
  - Resolves flash channel contention from a contention ratio of 83.26 down to 10.50.
- **In-Storage Computational Top-K**:
  - The host sends **only** the Query vector $Q$ and candidate metadata block descriptors ($0.33\text{ MB}$ total).
  - The NVMe controller executes GQA dot-products directly against NVMe storage blocks without moving candidate Key pages across the PCIe bus.
  - **Strict Invariant**: Candidate Key bytes transferred to host $= 0\text{ bytes}$.
  - PCIe bus data movement is reduced from $592.7\text{ MB}$ down to $109.8\text{ MB}$ (**81.5% reduction**).

---

## 3. Dataflow & Latency Accounting

$$\text{Critical Path} = T_{\text{visible storage}} + T_{\text{visible compute}}$$
$$\text{Reconciliation Error (\%)} = \frac{|T_{\text{wall}} - T_{\text{critical path}}|}{T_{\text{wall}}} \times 100\% < 2.0\%$$

In the canonical AI-SSD configuration (Run E, Context=4096, Decode=16):
- **Wall Time**: $20.92\text{ s}$ ($0.765\text{ tok/s}$)
- **Visible Storage Time**: $8.15\text{ s}$ ($39.0\%$)
- **Visible Compute Time**: $12.64\text{ s}$ ($60.4\%$)
- **Dominant Bottleneck**: Host CPU Compute ($60.4\%$) exceeds Storage Retrieval ($39.0\%$).
