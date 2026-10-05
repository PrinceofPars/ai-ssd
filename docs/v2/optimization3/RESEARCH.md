# AI-SSD V2 — Optimization 3: Llama 3 8B FP16 Architectural Research & Investigation

## Research Question

> **Can an 8B-class Llama 3 model running in FP16 provide better end-to-end AI-SSD inference performance than the current Qwen3-4B FP32 implementation, primarily because FP16 halves model-weight memory bandwidth?**

---

## 1. Architectural & Memory Bandwidth Hypothesis

In CPU-based autoregressive Large Language Model (LLM) decode generation, arithmetic intensity is extremely low:
$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}} \approx \frac{2 \times P}{P \times B_{\text{weight}}} = \frac{2}{B_{\text{weight}}} \text{ FLOP/byte}$$

Where:
- $P$ is parameter count.
- $B_{\text{weight}}$ is bytes per parameter.

### Comparison Table: Qwen3-4B FP32 vs Llama 3 8B FP16

| Architectural Attribute | Qwen3-4B (FP32 Release) | Llama 3 8B (Optimization 3) | Ratio / Impact |
|---|---|---|---|
| Total Parameters ($P$) | 4,020,000,000 | 8,030,261,248 | $2.00\times$ parameters |
| Weight Precision ($B_{\text{weight}}$) | FP32 (4 bytes) | FP16 (2 bytes) | $0.50\times$ byte width |
| Total Weight Footprint | 14.30 GB | 14.96 GB | **$+4.6\%$ total weight bytes** |
| Layers ($L$) | 36 | 32 | $-11.1\%$ layers |
| Hidden Dimension ($D$) | 2560 | 4096 | $+60\%$ hidden size |
| Intermediate Dimension ($F$) | 6912 | 14336 | $+107\%$ intermediate size |
| Query Heads ($H_Q$) | 20 | 32 | $+60\%$ query heads |
| KV Heads ($H_{KV}$) | 8 | 8 | $1.0\times$ (Identical GQA grouping) |
| Head Dimension ($d_k$) | 64 | 128 | $2.0\times$ head dimension |
| KV Page Geometry | $16 \text{ tok} \times 8 \times 64 \times 4\text{B} = 4\text{ KiB}$ | $16 \text{ tok} \times 8 \times 128 \times 2\text{B} = 32\text{ KiB}$ | $8\times$ bytes per page |
| Combined Block Size | 8 KiB (4 KiB K + 4 KiB V) | 64 KiB (32 KiB K + 32 KiB V) | $8\times$ bytes per block |
| Arithmetic Intensity | 0.5 FLOP/byte | 1.0 FLOP/byte | **$2.0\times$ compute efficiency** |

---

## 2. In-Storage Computational Engine Adaptation

### Native 128-dim FP16 AVX2/FMA/F16C Kernel
To support in-storage Top-K candidate filtering directly within the QEMU/NVMe virtual controller environment without host-RAM transfer:
1. **F16C Hardware Decompression**: Utilizing `_mm256_cvtph_ps` to unpack 16-bit half-precision floating-point Keys directly into 256-bit AVX registers.
2. **Quad Accumulator Pipelines**: 4 independent accumulator registers (`acc0..acc3`) hide FMA instruction latency and maximize execution throughput.
3. **Zero-Bus Offload**: Only the query vector $Q$ is transferred across the storage interface; candidate Key pages are scanned directly from virtual NVMe storage (`/dev/nvme0n1`), guaranteeing `candidate_k_bytes_to_host == 0`.

---

## 3. Evidence Classification Taxonomy

Following the project's strict measurement standards:
- `[REAL]`: Process RSS telemetry (`/proc/self/status`), bit-for-bit token verification, PyTorch CPU execution.
- `[VIRTUAL-DEVICE]`: QEMU/KVM virtual NVMe device `/dev/nvme0n1` block I/O, guest kernel NVMe driver, in-guest computational storage daemon.
- `[ANALYTICAL]`: NAND flash timing models (t_r, t_prog, t_xfer).
- `[PROJECTED]`: Extrapolated performance scaling curves beyond measured ranges.
