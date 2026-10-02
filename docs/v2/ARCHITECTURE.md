# AI-SSD V2 Architecture

## Target

Build a progressively realistic AI-SSD system around:

Real LLM
    ↓
Real KV Cache
    ↓
KV Management / Paging
    ↓
Storage Backend
    ↓
NVMe / FEMU
    ↓
FTL
    ↓
Tensor-Aware Placement
    ↓
Prefetching
    ↓
End-to-End Evaluation

## Main Workstreams

### P1

Real LLM
Real KV cache
Attention
Top-k KV retrieval
KV trace generation

### P2

FEMU
Virtual NVMe
FTL
NAND/channel model
Tensor-aware mapping

### P3

Storage abstraction
I/O
Prefetch
Experiment framework
Integration
Benchmarking

## Important

This document establishes the high-level architecture only.

The detailed execution schedule will be defined separately.
