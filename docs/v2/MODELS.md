# AI-SSD V2 — Model Registry & Supported Architectures

## Executive Overview

AI-SSD V2 operates not as a general black-box language model runner, but as an **in-storage near-data computing system** for Transformer KV caching. The storage subsystem (comprising Person 1 KV extraction, Person 2 multi-channel FTL/QEMU NVMe computational controller, and Person 3 speculative DRAM prefetch adapter) interacts directly with sub-layer Key and Value tensors.

Consequently, model support is governed by strict geometric, computational, and layout invariants across the host inference engine, the C acceleration kernels, the physical flash mapping layout, and the computational storage firmware.

---

## Model Classification Summary

Every evaluated model falls into one of four rigor classifications:

1. **FULLY SUPPORTED**: Can execute both dense reference baseline and end-to-end AI-SSD computational storage inference (`P1 -> P3 -> P2 -> QEMU NVMe`) with real PyTorch execution, hardware Top-$k$ filtering, exact token generation, and DRAM offload.
2. **PARTIALLY SUPPORTED**: Can execute in-memory KV extraction, trace generation, and analytical storage simulation, but requires updates to NVMe guest daemon kernels or page geometry contracts for live computational storage execution.
3. **DENSE ONLY**: Can be instantiated and executed via standard HuggingFace `model(..., use_cache=True)` on CPU, but cannot execute AI-SSD KV offloading or attention interception due to missing projection hooks, incompatible head dimensions, or unsupported hybrid attention mechanisms.
4. **UNSUPPORTED**: Fails at initialization or forward pass; cannot execute dense baseline or AI-SSD inference without major architectural refactoring.

---

## Comprehensive Model Compatibility Table

| Model | Dense | AI-SSD | Precision | Status | Reason |
|---|---|---|---|---|---|
| **Qwen/Qwen3-4B-Instruct-2507** | YES | YES | FP32 | **FULLY SUPPORTED** | Canonical primary evaluation model; 36 layers, 32 Q heads, 2 KV heads, `head_dim=64`, exact 4 KiB/page flash alignment, full AVX2 C-kernel & QEMU NVMe integration. |
| **Qwen/Qwen3-8B** | YES | YES | FP16 | **FULLY SUPPORTED** | Optimization 3 target model; 36 layers, 32 Q heads, 8 KV heads, `head_dim=128`, validated with 128-dim FP16 AVX2/F16C kernel and NVMe guest daemon. |
| **Qwen/Qwen2.5-0.5B** | YES | YES | FP32 | **FULLY SUPPORTED** | Canonical Phase 3 evaluation model; 24 layers, 14 Q heads, 2 KV heads, `head_dim=64`, exact 4 KiB/page flash alignment, full live trace and P1-P3-P2 validation. |
| **NousResearch/Meta-Llama-3-8B** | YES | YES | FP16 | **FULLY SUPPORTED** | Llama architecture verified in Opt 3; 32 layers, 32 Q heads, 8 KV heads, `head_dim=128`, GQA ratio 4, standard RoPE and RMSNorm, matches 128-dim FP16 AVX2 kernel. |
| **meta-llama/Llama-2-7b-hf** | YES | NO | FP32 / FP16 | **DENSE ONLY** | MHA (32 Q heads, 32 KV heads, `head_dim=128`). Fails flash page alignment: 16 tokens x 32 KV heads x 128 dim x 2 bytes = 128 KiB/page (32x physical 4 KiB flash page limit). |
| **Qwen/Qwen3.5-0.8B** | YES | NO | BF16 / FP32 | **UNSUPPORTED** | Hybrid linear/full attention (`Qwen3_5ForConditionalGeneration`). 18 out of 24 layers are linear attention (DeltaNet/Mamba state-space recurrence, not standard KV cache); missing standard `self_attn.k_proj`/`v_proj`; uses interleaved M-RoPE; `head_dim=256` exceeds all C-kernels. |
| **Qwen/Qwen3.5-4B** | YES | NO | BF16 / FP32 | **UNSUPPORTED** | Hybrid linear/full attention (`Qwen3_5ForConditionalGeneration`). 24 out of 32 layers are linear attention; `head_dim=256`; multi-dimensional M-RoPE; `attn_output_gate` unhandled by attention interceptor. |
| **DeepSeek-V2 / DeepSeek-V3** | YES | NO | FP16 / BF16 | **UNSUPPORTED** | Multi-Head Latent Attention (MLA). Decouples KV into compressed latent vector $c_t^{KV}$ (512 dim) and decoupled RoPE key $k_t^R$ (64 dim); no explicit K/V head separation; non-dot-product attention. |
| **Mistral-7B-v0.1 / v0.3** | YES | NO | FP16 | **DENSE ONLY** | GQA (32 Q heads, 8 KV heads, `head_dim=128`). Dense execution succeeds, but sliding window attention (SWA, 4096 window) conflicts with static historical block offloading unless window-aware eviction is wired. |

---

## Detailed Model Analysis

### 1. Qwen3-4B (`Qwen/Qwen3-4B-Instruct-2507`)
- **Status**: **FULLY SUPPORTED**
- **Architecture**: `Qwen3ForCausalLM`
- **Layers**: 36
- **Attention Geometry**:
  - Query heads: 32 (or 14 active in instruct config variant)
  - Key/Value heads: 2
  - Head dimension: 64
  - GQA ratio: 16:1 (or 7:1)
  - Scaling factor: $\frac{1}{\sqrt{64}} = 0.125$
- **Precision / Dtype**: FP32 (`torch.float32`, 4 bytes/element)
- **KV Page Layout**:
  - Tokens per block: 16
  - Key page size: $16 \text{ tokens} \times 2 \text{ heads} \times 64 \text{ dim} \times 4 \text{ bytes} = 4,096 \text{ bytes}$ (Exact 4 KiB flash page match)
  - Value page size: $16 \text{ tokens} \times 2 \text{ heads} \times 64 \text{ dim} \times 4 \text{ bytes} = 4,096 \text{ bytes}$ (Exact 4 KiB flash page match)
  - Logical block size: 8,192 bytes (8 KiB, 1 Key page + 1 Value page)
- **Generation & Attention Path**:
  - Prefill: Standard PyTorch forward pass initializes `DynamicCache`. Historical blocks ($[4 : T-16]$) are blockized into 8 KiB chunks and transferred to storage.
  - Interception: Per-layer hook wraps `self_attn.forward`. Queries and Keys pass through standard Q/K RMSNorm and `apply_rotary_pos_emb`.
  - Storage Dispatch: `select_and_fetch_active_kv` dispatches Q to in-storage AVX2 kernel (`instorage_topk_filter_gqa_avx2`) or QEMU NVMe guest daemon via `OP_COMPUTE_TOPK`.
  - Reconstruction: Active working set ($4 \text{ sink tokens} + \text{top-}k \text{ historical} + 16 \text{ recent tokens}$) concatenated and processed through `torch.nn.functional.scaled_dot_product_attention(enable_gqa=True)`.

### 2. Qwen3-8B (`Qwen/Qwen3-8B`)
- **Status**: **FULLY SUPPORTED**
- **Architecture**: `Qwen3ForCausalLM`
- **Layers**: 36
- **Attention Geometry**:
  - Query heads: 32
  - Key/Value heads: 8
  - Head dimension: 128
  - GQA ratio: 4:1
  - Scaling factor: $\frac{1}{\sqrt{128}} \approx 0.088388$
- **Precision / Dtype**: FP16 (`torch.float16`, 2 bytes/element)
- **KV Page Layout**:
  - Tokens per block: 16
  - Key page size: $16 \text{ tokens} \times 8 \text{ heads} \times 128 \text{ dim} \times 2 \text{ bytes} = 32,768 \text{ bytes}$ (32 KiB, or $8 \times 4 \text{ KiB}$ sub-pages)
  - Value page size: $16 \text{ tokens} \times 8 \text{ heads} \times 128 \text{ dim} \times 2 \text{ bytes} = 32,768 \text{ bytes}$ (32 KiB)
  - Logical block size: 65,536 bytes (64 KiB)
- **Modifications Required from Qwen3-4B**:
  - Head Dimension: Expanded from 64 to 128.
  - Precision: Transitioned from FP32 (4 bytes) to FP16 (2 bytes).
  - Acceleration Kernel: Replaced 64-dim FP32 dot-product with 128-dim AVX2/F16C hardware conversion kernel (`dot_product_128_fp16_avx2` utilizing `_mm256_cvtph_ps` and FMA).
  - Storage Backing: Updated `nvme_guest_daemon.c` and `QemuNvmeClient` with flag `is_fp16 = 1` for half-precision in-storage Top-$k$ scoring.
  - Multi-Head Interleaving: Expanded `cand_bids` memory indexing to handle 8 KV heads striped across 8 NAND channels.

### 3. Qwen2.5-0.5B (`Qwen/Qwen2.5-0.5B`)
- **Status**: **FULLY SUPPORTED**
- **Architecture**: `Qwen2ForCausalLM`
- **Layers**: 24
- **Attention Geometry**:
  - Query heads: 14
  - Key/Value heads: 2
  - Head dimension: 64
  - GQA ratio: 7:1
  - Scaling factor: $\frac{1}{\sqrt{64}} = 0.125$
- **Precision / Dtype**: FP32 (`torch.float32`)
- **KV Page Layout**:
  - Key page size: $16 \times 2 \times 64 \times 4 = 8,192 \text{ bytes}$ (or $16 \times 1 \times 64 \times 4 = 4,096 \text{ bytes}$ per head)
  - Matches canonical V2 page geometry. Fully benchmarked across Phase 1, Phase 2, and Phase 3.

### 4. Qwen3.5 (`Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B`)
- **Status**: **UNSUPPORTED**
- **Architecture**: `Qwen3_5ForConditionalGeneration` (hybrid Linear Attention + Full Attention)
- **Detailed Root Cause Failure Analysis**:
  1. **Model Class & Forward API Failure**:
     - `aissd_inference.py` assumes `model.model.layers[i].self_attn`. In Qwen3.5, the model is structured under `model.model.layers`, but layer types are heterogeneous: `linear_attention` (18 layers) and `full_attention` (6 layers).
     - Hooking fails because linear attention layers do NOT expose `self_attn`, `q_proj`, `k_proj`, or `v_proj`. They expose recurrent state projections (`in_proj`, `conv1d`, `x_proj`, `dt_proj`, `out_proj`).
  2. **KV Cache Representation Mismatch**:
     - Standard LLMs maintain token-indexed KV caches of shape `[batch, kv_heads, seq_len, head_dim]`.
     - Qwen3.5 linear attention layers do NOT maintain token-indexed KV caches. They maintain fixed-size recurrent hidden states ($\text{SSM state} \in \mathbb{R}^{d_{state} \times d_{conv}}$). There are no historical token keys/values to blockize, store in flash, or retrieve via Top-$k$.
  3. **Rotary Position Embedding (RoPE) Incompatibility**:
     - Qwen3.5 uses Multi-dimensional Rotary Position Embeddings (M-RoPE) with interleaved temporal/spatial/channel sections (`[11, 11, 10]`) and `partial_rotary_factor = 0.25`.
     - The AI-SSD hook imports `from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb`, which expects standard 1D RoPE tensors `(cos, sin)` matching `head_dim`. Passing M-RoPE tensors causes immediate shape broadcasting exceptions: `RuntimeError: The size of tensor a (256) must match the size of tensor b (64) at non-singleton dimension 3`.
  4. **Head Dimension & C-Kernel Overflow**:
     - For full attention layers in Qwen3.5, `head_dim = 256`.
     - AI-SSD native C-kernels (`instorage_attention.c`) and QEMU guest daemon (`nvme_guest_daemon.c`) strictly support `head_dim == 64` and `head_dim == 128`. Passing `head_dim == 256` falls back to scalar loops or produces segmentation faults / incorrect dot-product scaling.
  5. **Gated Output Projection**:
     - Full attention layers in Qwen3.5 incorporate output gating (`attn_output_gate = True`), where attention output is elementwise multiplied by a sigmoid gate projection: $Y = O \odot \sigma(G)$.
     - `aissd_inference.py` computes $Y = \text{SDPA}(Q, K, V) \cdot W_O$, completely omitting gate activation and corrupting output logits.

---

## Architectural Compatibility Requirements Matrix

| Subsystem Component | Qwen2.5-0.5B | Qwen3-4B | Qwen3-8B | Meta-Llama-3-8B | Qwen3.5 |
|---|---|---|---|---|---|
| **Model Class** | `Qwen2ForCausalLM` | `Qwen3ForCausalLM` | `Qwen3ForCausalLM` | `LlamaForCausalLM` | `Qwen3_5ForCondGen` (Incompatible) |
| **Attention Type** | Standard GQA | Standard GQA | Standard GQA | Standard GQA | Hybrid Linear/Full (Incompatible) |
| **KV Cache Type** | `DynamicCache` | `DynamicCache` | `DynamicCache` | `DynamicCache` | Hybrid Recurrent/Cache |
| **Head Dim** | 64 | 64 | 128 | 128 | 256 (Kernel Incompatible) |
| **RoPE Type** | 1D RoPE | 1D RoPE | 1D RoPE | 1D RoPE | M-RoPE (Incompatible) |
| **Q/K Normalization** | No | Yes (`q_norm`, `k_norm`) | Yes (`q_norm`, `k_norm`) | No | LayerNorm on Q/K |
| **Flash Page Mapping** | 4 KiB / 8 KiB | 4 KiB / 8 KiB | 32 KiB (Multi-page) | 32 KiB (Multi-page) | N/A (State-space recurrent) |
| **In-Storage Kernel** | 64-dim FP32 | 64-dim FP32 | 128-dim FP16 AVX2 | 128-dim FP16 AVX2 | Unsupported (Requires 256-dim) |
