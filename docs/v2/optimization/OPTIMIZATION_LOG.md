# AI-SSD V2 — Performance Optimization Ledger

**Branch**: `v2-performance-optimization`  
**Starting Baseline Commit**: `2a6d5537a11ee4a336873307bda488d76c489388`  
**Hardware Platform**: AWS EC2 Dedicated Sapphire Rapids Node (4 physical cores / 8 vCPUs, 61.8 GB RAM)  
**Canonical Benchmark Target**: Qwen/Qwen3-4B-Instruct-2507, Context=4096, Decode=16, FP32, 4 CPU threads, Seed=42, Virtual NVMe (/dev/nvme0n1), Computational Top-K ON, Async ON, Prefetch OFF.

---

## Immutable Baseline Measurements (5 Repetitions)

Measured on October 4, 2026 before any code changes on branch `v2-performance-optimization`:

- **Wall Time**: **20.33 s ± 0.36 s** (Min: 19.67 s, Max: 20.69 s)
- **Throughput**: **0.787 tok/s ± 0.014 tok/s** (Min: 0.773, Max: 0.814)
- **Peak RSS**: **17,028.4 MB ± 635.8 MB** (Active KV DRAM: 122.6 MB)
- **Candidate Key Bytes to Host**: **0 bytes** (Strict In-Storage Invariant)
- **Winning KV Bytes to Host**: **109.68 MB** (54.84 MB Key + 54.84 MB Value)
- **Total PCIe Bus Movement**: **109.84 MB**
- **Critical-Path Reconciliation Error**: **0.60% ± 0.03%** (< 2.0%)
- **Token Output**: **16/16 Exact Match** (`[11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13]`)

### Baseline Component Breakdown (16 Decode Steps)

| Component | Baseline Mean Time (s) | % of Wall Time | Domain | Optimization Candidate |
|---|:---:|:---:|:---:|:---:|
| **Top-K Scoring / In-Storage Filter** | 5.40 s | 26.6% | Storage / Guest VM | YES (High potential) |
| **Visible Storage (Winning KV Retrieval)** | 7.76 s | 38.2% | Storage I/O & Socket IPC | YES (High potential) |
| **MLP & LayerNorm (`mlp_and_norm_s`)** | 4.66 s | 22.9% | PyTorch CPU Linear (GEMV) | YES (High potential) |
| **QKV Projection (`qkv_proj_s`)** | 0.95 s | 4.7% | PyTorch CPU Linear | YES (Moderate potential) |
| **Output Projection (`out_proj_s`)** | 0.60 s | 3.0% | PyTorch CPU Linear | NO (< 1s) |
| **Attention Matmul (`attn_matmul_s`)** | 0.45 s | 2.2% | PyTorch CPU Attention | YES (GQA SDPA fusion) |
| **Active Tensor Concat (`active_concat_s`)** | 0.20 s | 1.0% | PyTorch Tensor Alloc | YES (Buffer reuse) |
| **Tensor Reconstruction (`tensor_recon_s`)** | 0.13 s | 0.6% | Data conversion | NO (< 0.2s) |
| **RoPE Embedding (`rope_s`)** | 0.05 s | 0.2% | Rotary Math | NO (< 0.1s) |
| **Bookkeeping / Prefetch Dispatch** | 0.01 s | < 0.1% | Python Orchestration | NO (< 0.01s) |

---

## Hardware Profiling & Thread-Scaling Analysis

### Multi-Level Profiling Summary
1. **Level 1 (Application Timeline)**: Visible Host Compute accounts for 61.2% ($12.45\text{ s}$), Visible Storage accounts for 38.2% ($7.76\text{ s}$).
2. **Level 2 (cProfile)**: Python function call overhead itself is negligible ($< 0.1\text{ s}$). Primary time is spent waiting on socket I/O in `_recv_exact` / `recv` ($21.1\text{ s}$ total including prefill write) and `torch._C._nn.linear` ($60.9\text{ s}$ accumulated CPU time across threads).
3. **Level 3 (PyTorch Profiler)**: `aten::linear` / `aten::mm` dominates PyTorch CPU operator execution (77.2% of operator CPU time across 4,048 calls).
4. **Level 4 (Hardware Counters via `perf stat`)**:
   - IPC: **2.15 – 2.17** insn/cycle (high compute efficiency).
   - CPU Utilization: **2.98 – 3.13 CPUs** on 4 threads.

### Thread-Scaling Experiment (1, 2, 4, 6, 8 Threads)

| Threads | Throughput (tok/s) | Wall Time (s) | Visible Storage (s) | Visible Compute (s) | IPC | CPU Utilization |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.854 | 18.72 | 6.32 | 12.28 | 2.09 | 2.11 CPUs |
| 2 | 0.838 | 19.08 | 6.18 | 12.79 | 2.15 | 3.13 CPUs |
| **4 (Canonical)** | **0.878** | **18.22** | **6.16** | **11.95** | **2.17** | **2.98 CPUs** |
| 6 | 0.842 | 19.00 | 6.19 | 12.70 | 2.16 | 3.10 CPUs |
| 8 | 0.841 | 19.02 | 6.17 | 12.74 | 2.15 | 3.00 CPUs |

*Finding*: Performance peaks at 4 CPU threads. The physical host has 4 physical cores (8 vCPUs via SMT). Spanning across hyperthreads (6 and 8 threads) introduces thread synchronization contention and cache thrashing without adding hardware execution units. 4 threads remains the optimal CPU parallelism configuration.

---

## Optimization Experiments Ledger

*(Entries will be added sequentially as optimizations are evaluated)*

### Experiment OPT-001: Native Grouped-Query Attention (SDPA) Kernel Fusion

- **Target Component**: Attention Layer (ttn_matmul_s)
- **Root Cause**: Manual tensor expansion via epeat_interleave(gqa=7) followed by manual 	orch.matmul(q, k.T), softmax(), and 	orch.matmul(weights, v) incurred redundant memory copies and uncoalesced DRAM accesses across 36 transformer layers.
- **Implementation**: Replaced manual expansion and multi-step matmul/softmax with PyTorch's native 	orch.nn.functional.scaled_dot_product_attention(q, act_k, act_v, scale=scaling, enable_gqa=True).
- **Invariants Checked**:
  - Exact Token Match: **16/16 (100% bit-exact)**: [11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13]
  - Candidate Key bytes to host: **0 bytes** (Invariant held)
  - P2/P3 resident payload: **0.0 MB** (Invariant held)
- **Measured Results (5 Repetitions)**:
  - ttn_matmul_s: **0.183 s ± 0.002 s** (vs Baseline: 0.450 s, **-59.3%**)
  - Wall Time: **19.44 s ± 0.30 s** (vs Baseline: 20.33 s ± 0.36 s, **-4.38%**, saving 0.89s)
  - Throughput: **0.823 ± 0.012 tok/s** (vs Baseline: 0.787 ± 0.014 tok/s, **+4.57%**)
  - Peak RSS: **16,830.5 MB** (vs Baseline: 17,028.4 MB)
- **Decision**: **ACCEPTED & COMMITTED**.

### Experiment OPT-002: Consolidated Socket Response Buffer in NVMe Guest Daemon

- **Target Component**: Visible Storage Retrieval / Batch Read (winning_v_reads_s / IPC socket transport)
- **Hypothesis**: Consolidating fragmented item header and payload socket writes in OP_BATCH_READ into a single contiguous transmission will reduce TCP packet fragmentation and host ecv() syscall count.
- **Implementation**: In scripts/nvme_guest_daemon.c, buffered all winning item headers and payloads into a single contiguous memory buffer sent via one write_all() call; updated person2_ssd/nvme_client.py to receive the full batch in a single _recv_exact() call.
- **Invariants Checked**:
  - Exact Token Match: **16/16 (100% bit-exact)** ([11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13])
  - Candidate Key bytes to host: **0 bytes** (Invariant held)
  - P2/P3 resident payload: **0.0 MB** (Invariant held)
- **Measured Results (5 Repetitions)**:
  - Wall Time: **20.20 s ± 0.34 s** (vs OPT-001: 19.44 s ± 0.30 s, +3.9% regression)
  - Throughput: **0.792 ± 0.013 tok/s** (vs OPT-001: 0.823 ± 0.012 tok/s)
- **Root Cause & Architectural Insight**: The Phase 8 asynchronous pipeline already overlaps winning KV block retrieval in the background with host CPU execution (MLP/attention). Therefore, socket transport latency is off the critical path. Dynamic heap allocations (malloc/ree) inside the 2GB guest VM added minor overhead.
- **Decision**: **REJECTED & REVERTED** (Implementation reverted to clean commit ddad54).

### Experiment OPT-003: 128-Dimensional AVX2/FMA SIMD Attention Vectorization in NVMe Daemon

- **Target Component**: In-Storage Top-K Scoring (	opk_scoring_s)
- **Hypothesis**: The guest daemon's compute_block_score_gqa contained a hand-tuned AVX2 kernel only for head_dim == 64, falling back to an unvectorized scalar loop for Qwen3-4B's head_dim == 128 (computing ~4.6 billion scalar operations across 16 decode steps). Implementing an 8-way unrolled 256-bit AVX2/FMA kernel (dot_product_128_avx2) with quad-register accumulation pipelining will drastically reduce in-storage compute latency.
- **Implementation**: In scripts/nvme_guest_daemon.c, created dot_product_128_avx2 utilizing _mm256_fmadd_ps across 4 independent accumulator registers (cc0, cc1, cc2, cc3), eliminating all scalar inner loops. Recompiled with -O3 -mavx2 -mfma -static into /opt/ai-ssd-v2/images/initramfs.cpio.gz.
- **Invariants Checked**:
  - Exact Token Match: **16/16 (100% bit-exact)** ([11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13])
  - Candidate Key bytes to host: **0 bytes** (Invariant held)
  - P2/P3 resident payload: **0.0 MB** (Invariant held)
- **Measured Results (5 Repetitions)**:
  - 	opk_scoring_s: **2.32 s ± 0.02 s** (vs Baseline: 5.40 s ± 0.03 s, **-57.0%**, saving 3.08 seconds)
  - Wall Time: **16.66 s ± 0.40 s** (vs Baseline: 20.33 s ± 0.36 s, **-18.06%**, saving 3.67 seconds; Min: **15.99 s**)
  - Throughput: **0.961 ± 0.024 tok/s** (vs Baseline: 0.787 ± 0.014 tok/s, **+22.1%**; Max: **1.000 tok/s**)
  - Peak RSS: **17,357.0 MB** (vs Baseline: 17,028.4 MB)
- **Decision**: **ACCEPTED & COMMITTED**.
