# AI-SSD V2 — Release Summary

**Release Tag**: `v2.0.0-rc`  
**Commit**: `cc3972e`  
**Branch**: `v2-performance-optimization`  

---

### What is AI-SSD V2?
AI-SSD V2 is a computational-storage architecture for Large Language Model (LLM) autoregressive inference that offloads historical Key-Value (KV) cache tensors to an NVMe solid-state storage subsystem, performing candidate Key selection directly inside storage to overcome host DRAM capacity bottlenecks.

---

### What problem does it solve?
As context lengths scale to 32K+ tokens, the KV cache footprint expands dramatically (exceeding tens of gigabytes per concurrent sequence), exhausting high-speed host DRAM. Conventional storage offload saturates the PCIe bus by streaming gigabytes of unpruned candidate Keys back to the host CPU for attention scoring. AI-SSD V2 executes attention scoring directly within the storage device controller, transferring only the top-scoring winning blocks across PCIe.

---

### How does the architecture work?
1. **Real Model Inference**: Executes real `Qwen/Qwen3-4B-Instruct-2507` (FP32 precision) using PyTorch.
2. **KV Blockization & Host Offload**: Divides prefill KV cache into discrete token blocks (16 tokens/block), pin-protecting attention sinks (64 tokens) and local window (128 tokens) in host DRAM while offloading all historical blocks to virtual NVMe storage (`/dev/nvme0n1`).
3. **In-Storage Computational Top-K (AVX2/FMA)**: For each layer, the host transmits only the current token's query vector ($Q_t$, 512 bytes). A native C guest daemon scans candidate Key blocks in storage and computes GQA dot-product scores using a custom 128-dimensional AVX2/FMA kernel (`dot_product_128_avx2`).
4. **Asynchronous Pipelined Retrieval**: Winning Key and Value blocks are retrieved across virtual NVMe concurrently with host CPU MLP and layer normalization execution.
5. **Real Attention & Generation**: The host reconstructs the active KV cache tensor from sinks, winning blocks, and local window, executing PyTorch native scaled dot-product attention (`F.scaled_dot_product_attention(enable_gqa=True)`) to generate the next token.

---

### What was actually measured?
All metrics were gathered from real OS execution on a dedicated AWS EC2 Sapphire Rapids instance (4 physical cores, 61.8 GB RAM) running Linux KVM and QEMU virtual NVMe:
- Real wall-clock time and decode throughput across 5 repetitions.
- Real Linux kernel process RSS via `/proc/self/status` high-resolution sampling.
- Real hardware performance counters via `perf stat` (IPC, cycles, instructions, CPU utilization).
- Real virtual NVMe block operations, read/write byte volumes, and latencies.
- Bit-exact token-ID matching against dense PyTorch all-DRAM execution.

---

### What are the strongest numbers?
- **End-to-End Decode Latency**: **16.66 s ± 0.40 s** (16 decode steps, context=4096), an **18.04% reduction** over the Phase 9 baseline (20.33 s).
- **Decode Throughput**: **0.961 ± 0.024 tok/s** (**+22.05% speedup** over Phase 9's 0.787 tok/s), peaking at **1.000 tok/s**.
- **In-Storage Scoring Latency**: Reduced from **5.40 s** to **2.32 s** (**-57.0% reduction**, saving 3.08 seconds per token generation).
- **Attention Matmul Latency**: Reduced from **0.448 s** to **0.184 s** (**-58.9% reduction** via native GQA SDPA).
- **KV DRAM Reduction**: **90.0% reduction** in active KV cache memory footprint (122.6 MB vs 1,226 MB).
- **PCIe Bus Traffic Elimination**: **0 bytes** of candidate Keys transferred to host (100% reduction in candidate Key traffic).
- **Mathematical Correctness**: **16/16 (100% bit-exact)** token-ID equality against dense PyTorch.

---

### What remains unproven?
- **Physical Enterprise SSD NAND Latency**: The emulated QEMU/NVMe virtual controller operates over Linux loopback/POSIX disk image; physical NAND flash cell wear, page program/erase cycles, and physical controller bus contention are not captured.
- **Physical ASIC/FPGA Power & Energy**: Power draw of physical in-controller compute engines is projected rather than measured with hardware power meters.
