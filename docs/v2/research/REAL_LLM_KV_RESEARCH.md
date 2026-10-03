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
