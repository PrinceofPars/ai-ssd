# AI-SSD V2 — Optimization 2: Deep Performance Investigation & Research

**Branch**: `v2-optimization-2`  
**Anchor Commit**: `cc3972e` (Validated Release Candidate)  
**Date**: October 5, 2026  
**Platform**: AWS EC2 Dedicated Sapphire Rapids Node (4 physical cores, 8 vCPUs, 61.8 GB RAM, QEMU Virtual NVMe)  

---

## 1. Executive Research Mandate

Optimization 2 represents an open-ended, scientifically disciplined performance investigation targeting the remaining critical-path latency components in AI-SSD V2. 

Following the successful completion of Optimization 1 (which achieved an 18.04% wall time reduction via 128-dim AVX2/FMA Top-K vectorization and fused native GQA SDPA), the remaining decode execution timeline is distributed across two primary domains:

1. **Host-Side CPU Execution (60–65% of decode wall time)**:
   - **MLP & LayerNorm (`mlp_and_norm_s` ≈ 4.55 s / 27.5%)**
   - **QKV & Output Linear Projections (`qkv_proj_s` + `out_proj_s` ≈ 1.51 s / 9.1%)**
   - **Active Tensor Concat & Reconstruction (`active_concat_s` + `tensor_recon_s` ≈ 0.31 s / 1.8%)**
   - **Attention Matmul (`attn_matmul_s` ≈ 0.18 s / 1.1%)**
2. **Storage Subsystem & Virtual Device Latency (35–40% of decode wall time)**:
   - **Visible Winning KV Retrieval (`winning_v_reads_s` ≈ 3.81 s / 23.0%)**
   - **In-Storage Computational Top-K Scoring (`topk_scoring_s` ≈ 2.32 s / 14.0%)**

---

## 2. Deep Dive: Host-Side CPU MLP (`mlp_and_norm_s`)

### A. Mathematical Formulation & Computational Characteristics
During autoregressive generation, tokens are decoded one at a time ($M=1$). In `Qwen/Qwen3-4B-Instruct-2507`:
- Hidden dimension $D = 2560$.
- Intermediate (MLP) dimension $F = 9728$.
- Architecture uses **SwiGLU** activation:
  $$\text{gate} = \text{SiLU}(X \cdot W_{\text{gate}}), \quad \text{up} = X \cdot W_{\text{up}}$$
  $$\text{hidden} = \text{gate} \odot \text{up}$$
  $$\text{output} = \text{hidden} \cdot W_{\text{down}}$$
- **Weight Tensor Shapes**:
  - $W_{\text{gate}} \in \mathbb{R}^{2560 \times 9728}$ (FP32: $2560 \times 9728 \times 4\text{ B} = 99,614,720\text{ bytes} \approx 99.6\text{ MB}$)
  - $W_{\text{up}} \in \mathbb{R}^{2560 \times 9728}$ (FP32: $99.6\text{ MB}$)
  - $W_{\text{down}} \in \mathbb{R}^{9728 \times 2560}$ (FP32: $99.6\text{ MB}$)
- **Total MLP Weights per Layer**: $298.8\text{ MB}$.
- **Across 36 Transformer Layers**: $36 \times 298.8\text{ MB} = \mathbf{10,758\text{ MB}} \approx \mathbf{10.76\text{ GB}}$.

### B. Arithmetic Intensity & Hardware Bottleneck Classification
For batch size 1 ($M=1$), every matrix multiplication ($[1 \times K] \times [K \times N] \to [1 \times N]$) is fundamentally a **Matrix-Vector Multiplication (GEMV)**, NOT a compute-intensive Matrix-Matrix Multiplication (GEMM).
- **Floating Point Operations per Token Decode**:
  $$2 \times (2560 \times 9728 + 2560 \times 9728 + 9728 \times 2560) \times 36 = 2 \times 74,711,040 \times 36 \approx \mathbf{5.38\text{ GFLOPs}}$$
- **Memory Bytes Read from DRAM per Token Decode**:
  $$\mathbf{10.76\text{ GB}}$$
- **Operational Arithmetic Intensity**:
  $$\text{Arithmetic Intensity} = \frac{5.38\text{ GFLOPs}}{10.76\text{ GB}} \approx \mathbf{0.50\text{ FLOP / Byte}}$$

On the Intel Xeon Platinum 8488C CPU:
- Peak Memory Bandwidth (DDR5-4800 quad-channel): $\approx 150\text{ GB/s}$ theoretical; $\approx 45–60\text{ GB/s}$ sustained across 4 threads.
- Theoretical minimum time to stream 10.76 GB of weights from DRAM at 50 GB/s:
  $$t_{\text{stream}} = \frac{10.76\text{ GB}}{50\text{ GB/s}} \approx \mathbf{0.215\text{ s per token}} \times 16\text{ tokens} = \mathbf{3.44\text{ s}}$$
- Measured baseline `mlp_and_norm_s`: **4.55 s** (75.6% of theoretical memory bandwidth limit!).
- *Scientific Conclusion*: The CPU MLP execution is **strictly memory-bandwidth bound**. Optimizations cannot reduce the memory volume if weights remain uncompressed FP32, but can minimize cache thrashing, eliminate redundant allocations, avoid view copies, and leverage AVX-512 weight packing or PyTorch compile / oneDNN fused primitives.

---

## 3. Deep Dive: Visible Winning-V Retrieval (`winning_v_reads_s`)

### A. System Architecture & Latency Components
In AI-SSD V2, winning blocks (top 10% scored candidate blocks) are retrieved across the virtual NVMe subsystem:
- 25 blocks per layer $\times$ 36 layers = 900 winning blocks per token decode.
- Over 16 decode steps = 14,400 winning blocks.
- With contiguous 8 KiB retrieval, each request fetches Key and Value payloads together.
- In `winning_v_reads_s` (3.81 s total):
  $$\frac{3.81\text{ s}}{16\text{ steps} \times 36\text{ layers}} \approx 6.6\text{ ms per layer} \implies \frac{6.6\text{ ms}}{25\text{ blocks}} \approx \mathbf{264\,\mu\text{s per block batch}}$$

### B. Investigation Vectors:
1. **Asynchronous Pipelined Overlap**: While layer $L$ is executing MLP GEMV on the host CPU, can layer $L+1$'s winning blocks be retrieved with zero CPU wait?
2. **Zero-Copy Re-Assembly**: Eliminating numpy array conversions and contiguous clone copies when stitching retrieved buffers into PyTorch activation tensors.

---

## 4. Deep Dive: In-Storage Top-K Microarchitectural Headroom

In Optimization 1, replacing the unvectorized scalar loop with `dot_product_128_avx2` dropped `topk_scoring_s` from 5.40s to 2.32s (-57%).
Can further microarchitectural tuning within the guest C daemon yield additional gains?
- **AVX2 Cache Pre-fetching**: Using `_mm_prefetch((const char*)(k_vec + 64), _MM_HINT_T0)` to overlap memory bus fetch latency.
- **AVX-512 Vectorization**: If supported by the Sapphire Rapids host and exposed via QEMU `-cpu host`, expanding from 256-bit AVX2 to 512-bit AVX-512 (`_mm512_fmadd_ps`) computes 16 floats per cycle instead of 8.
- **Top-K Heap Pruning**: Maintaining a min-heap or early-pruning candidate blocks whose first few dimensions cannot exceed the current $K$-th threshold.

---

## 5. References & Literature

1. **FlashAttention & Kernel Fusion**: Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022.
2. **LLM CPU Inference Optimization**: Intel Extension for PyTorch (IPEX) & oneDNN Architecture Whitepapers (GEMV memory streaming optimizations).
3. **Computational Storage Architectures**: SNIA Computational Storage Architecture and Programming Model, Version 1.0 (2022).
4. **Grouped-Query Attention**: Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023.
