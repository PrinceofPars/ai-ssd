# Proposal 002: Standardized Real KV Access Trace Schema and JSONL Storage Format

**Author**: Person 1 (P1 - Real LLM + Real KV Cache Engine)  
**Date**: October 2026  
**Status**: PROPOSED  
**Target File**: `common/schemas/trace.py`  
**Shared Trace Path**: `/opt/ai-ssd-v2/traces/real_llm/`  
**Affected Agents**: P1 (Trace Producer), P2 (FTL / FEMU Replay Consumer), P3 (Prefetch / Trace Replay Orchestrator)

---

## 1. Problem Statement

In V1, memory access traces were synthetically generated in-memory using `WorkloadGenerator`. In V2, real LLM execution on CPU produces genuine token-by-token attention access patterns. P2 (FTL / FEMU) and P3 (Prefetch / System Integration) need a standardized, deterministic, and serializable trace format to replay flash I/O operations without depending on running PyTorch or language model weights directly.

Currently, `docs/v2/CONTRACTS.md` lists `KVTrace` as a planned contract, but no formal dataclass or JSONL specification has been defined in `common/schemas/`.

---

## 2. Proposed Contract: `common/schemas/trace.py`

### 2.1 Trace Entry / Event Dataclass
```python
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any

class TraceOperation(str, Enum):
    PREFILL_WRITE = "PREFILL_WRITE"  # Appending prompt KV blocks
    DECODE_READ = "DECODE_READ"      # Reading KV blocks for attention
    TOPK_FILTER = "TOPK_FILTER"      # In-storage candidate scoring (Key read only)
    TOPK_FETCH = "TOPK_FETCH"        # Host retrieval of selected blocks (Key + Value read)
    EVICT_SSD = "EVICT_SSD"          # Evicting cold blocks from Host to SSD
    PREFETCH_READ = "PREFETCH_READ"  # Prefetching predicted blocks

@dataclass
class KVTraceEntry:
    """Represents a single atomic or batched KV I/O event during real model execution."""
    event_id: int
    query_id: int
    step: int
    layer_id: int
    head_id: int
    operation: str  # TraceOperation string value
    
    # Block identification
    block_id: int
    token_start: int
    token_end: int
    
    # Page and transfer properties
    sub_page: str  # "KEY", "VALUE", or "BOTH"
    byte_size: int
    
    # Salience & Tiering metadata
    is_attention_sink: bool = False
    is_recent_window: bool = False
    hotness: float = 1.0
    
    # Top-k pruning context
    candidate_blocks: Optional[List[int]] = None
    selected_blocks: Optional[List[int]] = None
    topk_k: Optional[int] = None
    
    # Timing
    timestamp_ns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KVTraceEntry:
        return cls(**data)
```

### 2.2 Trace Header / Manifest Dataclass
Every trace file (`.jsonl`) is accompanied by a manifest metadata file (`.json`):
```python
@dataclass
class KVTraceManifest:
    model_name: str
    model_architecture: str
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    tokens_per_block: int
    key_page_bytes: int
    value_page_bytes: int
    logical_block_bytes: int
    
    prompt: str
    prompt_tokens: int
    generated_tokens: int
    total_steps: int
    total_events: int
    
    trace_format_version: str = "2.0"
    git_commit: str = "db7e0f8"
    timestamp_utc: str = ""
    sha256_checksum: str = ""
```

---

## 3. Storage Format

- **Trace Data File**: `/opt/ai-ssd-v2/traces/real_llm/trace_<model>_<workload>.jsonl`
  Each line is a valid JSON object serialization of `KVTraceEntry`.
- **Manifest File**: `/opt/ai-ssd-v2/traces/real_llm/trace_<model>_<workload>.manifest.json`
  Single JSON object containing `KVTraceManifest`.
- **Checksum**: SHA-256 hash verified against the `.jsonl` trace file to guarantee reproducibility.

---

## 4. Reason for Change

1. **Replay Decoupling**: P2 (FEMU/NVMe) and P3 (Prefetch) can replay the exact trace without installing PyTorch or downloading weights on their VM/subsystem.
2. **Key/Value Sub-Page Disaggregation**: The `sub_page` field allows FTL to simulate reading 4 KiB Key pages for in-storage filtering, and only transferring 4 KiB Value pages for selected blocks.
3. **Reproducibility**: Clear metadata and checksums guarantee all team members benchmark the identical workload.

---

## 5. Compatibility Impact

- New additive contract (`common/schemas/trace.py`).
- Existing `common/schemas/request.py`, `result.py`, and `metrics.py` remain intact.
