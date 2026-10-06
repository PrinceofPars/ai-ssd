# Architecture-Adaptive Model State Support in AI-SSD V2

## Overview

AI-SSD V2 decouples model architectures from underlying computational storage mechanisms. Instead of hard-coding assumptions that every language model uses standard Transformer Key-Value tensors structured as:

$$\text{layers} \times \text{KV heads} \times \text{sequence} \times \text{head\_dim}$$

the runtime provides an **Architecture-Adaptive State Management Layer** composed of two primary abstractions:

1. **`ModelAdapter`**: Encapsulates model-specific internal wiring, forward-pass hooks, rotary positional embeddings (RoPE), and layer wrapping.
2. **`StateProvider`**: Encapsulates serialization, blockification, storage offload, Top-$K$ retrieval, and working-set reconstruction for reusable sequence states.

---

## Architectural Flow

```text
       Hugging Face / PyTorch Model
                    │
                    ▼
            ModelAdapter
   (QwenAdapter, MistralAdapter, HybridJambaAdapter)
                    │
                    ▼
              StateProvider
   (TransformerKV, SlidingWindowKV, HybridState)
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
    Host DRAM Tier     AI-SSD NVMe Storage Tier
   • Attention Sinks   • 8 KiB Flash Blocks (4K K + 4K V)
   • Recent Window     • In-Storage SIMD Top-K Selection
   • SSM / Recurrent   • Zero Candidate Keys to Host PCIe
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
          State Reconstruction
                    │
                    ▼
         Layer Attention / SDPA
```

---

## Supported Architectures and Classification

AI-SSD classifies model architectures into three strict compatibility tiers:

1. **`FULL`**: The model's reusable state is primarily attention Key-Value activations. Historical blocks are offloaded to AI-SSD storage, and Top-$K$ selection executes directly on the storage controller (`candidate_k_bytes_to_host = 0 B`).
2. **`PARTIAL`**: Hybrid architectures (e.g., Attention + Mamba/SSM). Attention Key-Value tensors are offloaded to AI-SSD storage, while fixed-size recurrent/SSM hidden states remain resident in Host DRAM.
3. **`UNSUPPORTED`**: Pure state-space or RNN architectures (e.g., Mamba, RWKV). These models maintain compact $O(1)$ recurrent hidden states rather than growing sequence-indexed Key-Value caches, offering no separable block structure for Top-$K$ attention offload.

---

## Registered Models Matrix

| Model Identifier | Hugging Face ID | Architecture Family | Precision | Attention Type | State Provider | Compatibility Level |
|---|---|---|:---:|:---:|---|:---:|
| `qwen3-4b` | `Qwen/Qwen3-4B-Instruct-2507` | Transformer (Qwen2) | FP32 | GQA (16:4) | `TransformerKVStateProvider` | **`FULL`** |
| `qwen3-8b` | `Qwen/Qwen3-8B` | Transformer (Qwen2) | FP16 | GQA (32:8) | `TransformerKVStateProvider` | **`FULL`** |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B` | Transformer (Qwen2) | FP32 | GQA (14:2) | `TransformerKVStateProvider` | **`FULL`** |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B-Instruct` | Transformer (Qwen2) | FP32 | GQA (16:4) | `TransformerKVStateProvider` | **`FULL`** |
| `tiny-mistral` | `openaccess-ai-collective/tiny-mistral` | Transformer (Mistral) | FP32 / FP16 | GQA (16:4) + Sliding | `SlidingWindowKVStateProvider` | **`FULL`** |
| `mistral-7b` | `mistralai/Mistral-7B-v0.1` | Transformer (Mistral) | FP16 | GQA (32:8) + Sliding | `SlidingWindowKVStateProvider` | **`FULL`** |
| `jamba` | `ai21labs/AI21-Jamba-1.5-Mini` | Hybrid (Transformer + SSM) | FP16 | GQA (32:8) + Mamba | `HybridStateProvider` | **`PARTIAL`** |
| `mamba` | `state-spaces/mamba-130m-hf` | Pure State-Space (Mamba) | FP32 | None (SSM) | N/A | **`UNSUPPORTED`** |

---

## State Provider Abstractions

### 1. `TransformerKVStateProvider`
- Handles standard multi-head attention (MHA), grouped-query attention (GQA), and multi-query attention (MQA).
- Partitions sequence history into:
  - **Attention Sinks**: First 4 tokens permanently resident in Host DRAM to anchor softmax distribution.
  - **Recent Window**: Last 16 tokens resident in Host DRAM for local syntax continuity.
  - **Offloaded Historical Blocks**: Segmented into 16-token chunks stored as 8 KiB logical blocks on AI-SSD (4 KiB Key page + 4 KiB Value page).
- Top-$K$ query filtering executes inside the storage device.

### 2. `SlidingWindowKVStateProvider`
- Extends `TransformerKVStateProvider` for models with localized attention spans (e.g. Mistral's 4,096-token sliding window).
- Computes effective window boundaries:

$$\text{Active Window} = [\max(0, \text{total\_tokens} - W), \text{total\_tokens}]$$

- Blocks strictly older than the active window are skipped during retrieval, eliminating redundant I/O traffic.

### 3. `HybridStateProvider`
- Manages hybrid models containing both Transformer self-attention layers and Mamba/SSM recurrent layers (e.g., Jamba).
- **Attention Layers**: Key and Value activations are blockified and offloaded to AI-SSD flash storage.
- **SSM Layers**: Recurrent state vectors are retained in Host DRAM because their size is fixed across all sequence lengths and must be updated every token.

---

## CLI Usage

### List Available Models & Compatibility
```bash
python scripts/demo_inference.py --list-models
```

### Run Validated Model Benchmarks
```bash
# Canonical Qwen3-4B FP32 benchmark (4 CPU threads)
python scripts/demo_inference.py --model qwen3-4b --context 4096 --decode-tokens 16 --threads 4

# Canonical Qwen3-8B FP16 benchmark (4 CPU threads)
python scripts/demo_inference.py --model qwen3-8b --context 4096 --decode-tokens 16 --threads 4

# Tiny-Mistral GQA / Sliding-Window benchmark
python scripts/demo_inference.py --model tiny-mistral --context 4096 --decode-tokens 16 --threads 4
```

### Inspecting Unsupported Models
If a user requests a pure state-space or unsupported model:
```bash
python scripts/demo_inference.py --model mamba
```
Output:
```text
[UNSUPPORTED ARCHITECTURE] Model 'mamba' cannot be executed through AI-SSD.
Reason: Pure state-space recurrent model lacks separable KV cache for attention Top-K offload
AI-SSD requires architectures with separable Attention Key-Value caches for Top-K offload.
```
