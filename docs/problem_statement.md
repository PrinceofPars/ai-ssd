# Problem Statement: The KV Cache Memory Wall

## 1. Context & Motivation
Large Language Model (LLM) inference consists of two phases:
1. **Prefill Phase**: Processing input prompt tokens in parallel.
2. **Decode Phase**: Generating one token at a time autoregressively.

During autoregressive generation, keys and values of previous tokens are cached to avoid quadratic recomputation. As context lengths scale from 4K to 32K, 64K, or 128K tokens, the memory footprint of the KV cache explodes:

$$\text{KV Cache Size} = 2 \times \text{layers} \times \text{heads} \times \text{head\_dim} \times \text{context\_len} \times \text{bytes\_per\_element}$$

For a 32-layer, 32-head, `head_dim=128` model at 32K context with FP16:
$$\text{Size} = 2 \times 32 \times 32 \times 128 \times 32768 \times 2 \approx 17.18 \text{ GB}$$

At 128K context, this reaches $\approx 68.7 \text{ GB}$ per concurrent request—surpassing the memory capacity of single GPUs and making serving economically prohibitive.

---

## 2. Bottlenecks in Naive SSD Offloading
Offloading cold KV blocks to conventional SSDs introduces severe latency bottlenecks:
1. **Unstructured Access**: Sparse attention access patterns lead to random 4KB reads.
2. **Channel Contention**: Conventional FTL blindly places sequential LBAs onto the same flash die/channel, causing channel serialization delays.
3. **I/O Overhead**: Fetching entire uncompressed, unpruned KV caches overwhelms the PCIe/NVMe bus.

---

## 3. The AI-SSD Solution
AI-SSD solves this through end-to-end co-design:
1. **Sparse Attention (Top-k) Pruning**: Identifies the 10-20% most critical cold blocks, cutting SSD I/O traffic by 80-90%.
2. **Tensor-Aware FTL**: Stripes correlated KV blocks across independent flash channels and dies, unlocking multi-channel flash parallelism.
3. **Speculative Prefetching**: Predicts next-layer attention needs and pre-stages data in host DRAM, hiding flash read latency.
