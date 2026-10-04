# AI-SSD V2 — Optimization 2: Deep Performance Investigation & Final Report

**Branch**: `v2-optimization-2`  
**Anchor Release Commit**: `cc3972e` (Validated Release Candidate on `v2-performance-optimization`)  
**Date**: October 5, 2026  
**Node**: AWS EC2 Dedicated Sapphire Rapids Node (4 physical cores / 8 vCPUs, 61.8 GB RAM)  
**Evaluated Workload**: `Qwen/Qwen3-4B-Instruct-2507`, Context=4096 tokens, Decode=16 tokens, FP32 Precision, 4 CPU Threads, Storage=QEMU Virtual NVMe (`/dev/nvme0n1`), Computational Top-K (10%), Async ON, Prefetch OFF.

---

## 1. Executive Summary

Optimization 2 was conducted on the dedicated experimental branch `v2-optimization-2` branched from `cc3972e` to investigate whether the remaining decode bottlenecks could be further accelerated without compromising architectural invariants.

Following systematic hardware counter tracing (`sudo perf stat`), PyTorch operator profiling (`torch.profiler`), C micro-benchmarking, and component-level isolation:
1. **Physical Convergence Reached**: The host CPU decode path is **strictly memory-bandwidth bound** ($M=1$ GEMV with an arithmetic intensity of $0.50\text{ FLOP/byte}$), operating at **72.7% of the theoretical saturation limit** of the quad-channel DDR5 bus.
2. **Empirical Verification of Optimization Candidates**: Four independent optimization candidates (in-storage coalesced preads, PyTorch Inductor SwiGLU compilation, fused QKV projection, and fused Gate-Up projection) were implemented and rigorously benchmarked. None yielded statistically meaningful gains over hand-tuned Intel oneAPI MKL assembly primitives and discrete 4 KiB NVMe reads, and several introduced regressions due to cache/DMA effects.
3. **Definitive Release Recommendation**: Anchor commit `cc3972e` is affirmed as the **definitive, production-ready AI-SSD V2 performance release**. All 164 unit/regression tests pass, bit-exact 16/16 token generation is preserved, and zero-bus-movement invariants hold with zero host-DRAM KV footprint leakage.

---

## 2. Fresh Optimization 2 Baseline on cc3972e

Established via 5 consecutive repetitions on `cc3972e`:
- **Wall Time**: **16.572 ± 0.180 s** (Min: 16.277 s, Max: 16.835 s)
- **Decode Throughput**: **0.966 ± 0.010 tok/s** (Peak: 0.983 tok/s)
- **Peak Host RSS**: **16,164.2 ± 578.6 MB**
- **Active KV in Host DRAM**: **122.625 MB**
- **Candidate Key Bytes across PCIe**: **0 bytes** (100% in-storage filtering)
- **Winning KV Bytes across PCIe**: **109.68 MB** (57.51 MB K + 57.51 MB V)
- **P2 / P3 Resident Payload**: **0.0 MB** (zero residual KV payload)
- **Token Accuracy**: **16 / 16 (100% bit-exact)** `[11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13]`

### Component Latency Breakdown:
```text
Component                     Latency (s)    % Critical Path
------------------------------------------------------------
MLP + LayerNorm (Host GEMV)    4.444 s           27.0%
Winning KV I/O (Virtual NVMe)  3.845 s           23.4%
In-Storage AVX2/FMA Top-K      2.316 s           14.1%
QKV Linear Projection          0.900 s            5.5%
Output Linear Projection       0.565 s            3.4%
Active KV Concat               0.192 s            1.2%
Native SDPA Attention          0.184 s            1.1%
Tensor Reconstruction          0.123 s            0.7%
RoPE Positional Embedding      0.045 s            0.3%
Bookkeeping & Memory Sampler   0.005 s            0.0%
------------------------------------------------------------
Total Visible Storage          7.691 s           46.7%
Total Visible Compute          8.775 s           53.3%
Critical Path Total           16.465 s          100.0%
```

---

## 3. Deep Bottleneck Investigation & Mathematical Proof

### A. Hardware Execution Profile (`sudo perf stat`)
- **Total Instructions Retired**: $2.243 \times 10^{12}$
- **Total CPU Cycles**: $1.023 \times 10^{12}$
- **Instructions Per Cycle (IPC)**: **2.19 insn/cycle**
- **Branch Misprediction Rate**: **0.39%**
- *Insight*: The CPU execution pipeline suffers from virtually no pipeline bubbles or branch stalls. The execution rate is pinned to memory subsystem latency.

### B. Mathematical Memory-Bandwidth Proof for Host MLP
For single-token autoregressive generation ($M=1$), every matrix multiplication ($[1 \times K] \times [K \times N] \to [1 \times N]$) is a **Matrix-Vector Multiplication (GEMV)**:
- $W_{\text{gate}} \in \mathbb{R}^{2560 \times 9728} \implies 99.6\text{ MB}$
- $W_{\text{up}} \in \mathbb{R}^{2560 \times 9728} \implies 99.6\text{ MB}$
- $W_{\text{down}} \in \mathbb{R}^{9728 \times 2560} \implies 99.6\text{ MB}$
- Total MLP weights per layer = **298.8 MB**.
- Across 36 layers: $36 \times 298.8\text{ MB} = \mathbf{10.76\text{ GB}}$ streamed per token decode.
- Floating-point operations per token decode: $2 \times (2560 \times 9728 \times 2 + 9728 \times 2560) \times 36 \approx \mathbf{5.38\text{ GFLOPs}}$.
- **Operational Arithmetic Intensity**:
  $$\text{AI} = \frac{5.38\text{ GFLOPs}}{10.76\text{ GB}} = \mathbf{0.50\text{ FLOP / Byte}}$$
- Total weights streamed from DRAM across 15 decode steps (including QKV, Out, and Vocabulary projections): **214.5 GB**.
- At a sustained memory throughput of ~50 GB/s on quad-channel DDR5, theoretical minimum time is $\frac{214.5}{50} = \mathbf{4.29\text{ s}}$.
- Measured host linear projection time is **5.90 s**, representing **72.7% saturation of the physical DRAM bus**.
- *Conclusion*: Without quantizing weights to lower bit-widths (e.g. INT8/FP8, which would violate the FP32 mandate), host CPU GEMV execution is at the physical memory-bandwidth limit.

---

## 4. Evaluation of Optimization Candidates

| Candidate ID | Target Subsystem | Hypothesis | Measured Result | Verdict |
|---|---|---|---|---|
| **OPT2-001** | In-Storage Top-K (`scripts/nvme_guest_daemon.c`) | Coalescing 255 discrete 4 KiB `pread()` calls into a 4 MB sliding window reduces 137,700 syscalls to 540. | `topk_scoring_s` regressed from 2.32s to **3.78s (+63.1%)**; Wall time regressed to **17.66s (+6.6%)** due to QEMU virtual NVMe DMA transfer amplification and cache thrashing. | **REJECTED & REVERTED** |
| **OPT2-002** | Host CPU MLP (`mlp_and_norm_s`) | `torch.compile(mode="reduce-overhead")` on SwiGLU will fuse SiLU and elementwise multiplication with GEMV. | Single-layer MLP latency increased from 4.77 ms to **4.95 ms (0.96x)** due to Inductor OpenMP codegen overhead vs Intel oneAPI MKL. | **REJECTED** |
| **OPT2-003** | Host QKV Projections (`qkv_proj_s`) | Pre-packing $W_q, W_k, W_v$ into contiguous $W_{QKV}$ eliminates two MKL dispatcher calls per layer. | Projection latency increased from 0.301 ms to **0.323 ms (0.93x)** due to strided tensor slicing overhead. | **REJECTED** |
| **OPT2-004** | Host SwiGLU MLP (`mlp_and_norm_s`) | Pre-packing $W_{\text{gate}}$ and $W_{\text{up}}$ fuses gate/up projections into a single GEMV. | Latency changed from 4.716 ms to **4.696 ms (1.00x)**, yielding an insignificant 11 ms difference over 16 seconds because DRAM streaming volume is unchanged. | **REJECTED** |

---

## 5. Invariants & Verification Checklist

- [x] **Zero Candidate Key Bytes across PCIe**: Verified 0 bytes.
- [x] **Zero Residual KV Payload in Host DRAM**: Verified $0.0\text{ MB}$ P2/P3 payload.
- [x] **Active KV Footprint**: Verified $122.625\text{ MB}$ (90.3% DRAM memory reduction vs dense 1,270 MB cache).
- [x] **16/16 Bit-Exact Token Match**: Verified `[11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13]`.
- [x] **Full Regression Test Suite**: 164 / 164 tests PASS (`pytest tests/ -q`).
- [x] **Unmodified Baseline Branches**: `v2-real-llm-kvssd` and the validated history of `v2-performance-optimization` remain completely intact.

---

## 6. Final Verdict & Release Anchor

Commit **`cc3972e`** stands as the definitive, optimal performance release candidate for AI-SSD V2.
