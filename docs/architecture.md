# System Architecture: AI-SSD

The AI-SSD project bridges long-context LLM inference with flash storage architecture. Instead of treating the storage layer as a dumb block device, AI-SSD co-designs the KV cache management algorithm and the Flash Translation Layer (FTL).

---

## 1. High-Level Dataflow

```
   [Inference Engine]
           │
           │ (1) Attention Query & Context Tokens
           ▼
┌──────────────────────────────────────────────────────────┐
│                   Unified AI-SSD API                     │
│               (person3_system/api/ai_ssd.py)             │
└───────────────┬──────────────────────────┬───────────────┘
                │                          │
        (2) KV Requests            (3) Storage Operations
                ▼                          ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│       KV Cache Engine        │   │         SSD Engine           │
│   (person1_kv_engine/)       │   │       (person2_ssd/)         │
│                              │   │                              │
│ • Paged KV Blocks            │   │ • Page-level FTL             │
│ • Attention Sinks & Decay    │   │ • Tensor-Aware Striping      │
│ • Hot/Cold Classifier        │   │ • Multi-channel NAND Bus     │
│ • Top-k Sparse Selector      │   │ • Die/Plane Parallelism      │
└───────────────┬──────────────┘   └──────────────┬───────────────┘
                │                                 │
                │        (4) Block Prefetch       │
                └───────────────►◄────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │     Speculative Prefetcher    │
                 │   (person3_system/prefetch/)  │
                 │                               │
                 │ • Next-layer Predictor        │
                 │ • DRAM Staging Buffer         │
                 └───────────────────────────────┘
```

---

## 2. Subsystem Division of Responsibilities

### Subsystem 1: KV Cache Engine (`person1_kv_engine/`)
- Breaks KV tensors into standardized `KVBlock` units (16 tokens $\times$ 1 head, FP16 = 4 KB).
- Distinguishes hot blocks (recent generation window and initial attention sink tokens) from cold blocks.
- Manages memory tiering: GPU VRAM $\rightarrow$ Host DRAM $\rightarrow$ NVMe SSD.
- Computes attention scores and performs Top-$k$ selection to fetch only the essential $k\%$ of cold KV blocks.

### Subsystem 2: SSD / FTL Simulator (`person2_ssd/`)
- Simulates physical NAND Flash hierarchy: Pages, Blocks, Planes, Dies, and Channels.
- Models flash physical timing constraints:
  - $t_R$: Page read latency ($\sim 25\mu s$)
  - $t_{PROG}$: Page program latency ($\sim 200\mu s$)
  - $t_{BERS}$: Block erase latency ($\sim 2ms$)
  - Channel bus transfer serialization & PCIe transfer overhead.
- Implements both:
  1. **Conventional FTL**: Standard greedy page-level mapping.
  2. **Tensor-Aware FTL**: Intelligently stripes co-accessed KV blocks across different channels and dies to maximize parallel read throughput.

### Subsystem 3: System API & Prefetch Orchestrator (`person3_system/`)
- Provides the single point of entry (`ai_ssd.py`) with clean request/response contracts.
- Implements speculative prefetching: predicts and stages layer $L+1$ KV blocks in DRAM while layer $L$ computes attention.
- Serves the interactive evaluation dashboard using Streamlit.
