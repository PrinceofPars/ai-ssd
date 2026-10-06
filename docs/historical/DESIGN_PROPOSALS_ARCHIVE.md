# AI-SSD Design Proposals Archive (Consolidated)

This document consolidates historical V2 design proposals into a single reference archive.

---

## Source: docs/v2/proposals/README.md

# V2 Contract Proposals

Do not directly modify shared contracts when the change affects
another agent.

Create a proposal containing:

- Current contract
- Proposed contract
- Reason
- Affected agents
- Migration requirements
- Compatibility considerations


---

## Source: docs/v2/proposals/PROP-001-KV-BLOCK-PAGE-GEOMETRY.md

# Proposal 001: Explicit Separation of Physical Key/Value Flash Pages and 8 KiB Logical KV Block Geometry

**Author**: Person 1 (P1 - Real LLM + Real KV Cache Engine)  
**Date**: October 2026  
**Status**: PROPOSED  
**Target File**: `common/schemas/kv_block.py` and `common/constants.py`  
**Affected Agents**: P1 (KV Engine), P2 (FTL / FEMU), P3 (Prefetch / System Integration)

---

## 1. Problem Statement

In the initial V1 baseline implementation (`common/constants.py` and `common/schemas/kv_block.py`), the default block size is defined as:
```python
# V1 constants.py
DEFAULT_KEY_SIZE_BYTES = (DEFAULT_BLOCK_TOKENS * DEFAULT_KV_HEADS_PER_BLOCK * DEFAULT_HEAD_DIM * BYTES_PER_FP16) // 2
DEFAULT_VALUE_SIZE_BYTES = DEFAULT_KEY_SIZE_BYTES
DEFAULT_BLOCK_SIZE_BYTES = DEFAULT_KEY_SIZE_BYTES + DEFAULT_VALUE_SIZE_BYTES  # 4096 bytes (4 KB)
```
And in `kv_block.py::create_default`:
```python
bytes_per_elem = 2 if dtype.upper() == "FP16" else 1
total_size = token_count * kv_head_count * head_dim * bytes_per_elem
half_size = total_size // 2
return cls(..., key_size_bytes=half_size, value_size_bytes=half_size, ...)
```

This computation exhibits a critical physical inconsistency:
- 16 tokens $\times$ 1 head $\times$ 128 head dimension $\times$ 2 bytes (FP16) = **4096 bytes (4 KiB) for the Key tensor alone**.
- 16 tokens $\times$ 1 head $\times$ 128 head dimension $\times$ 2 bytes (FP16) = **4096 bytes (4 KiB) for the Value tensor alone**.
- Dividing `total_size // 2` assigns 2048 bytes (2 KiB) to Key and 2048 bytes (2 KiB) to Value, which physically corresponds to only **8 tokens of Key and 8 tokens of Value**, while claiming `token_count = 16`.
- Furthermore, describing a combined K+V block as a single 4 KiB NAND page obscures the physical reality of NAND flash storage: a flash page is 4096 bytes. Therefore, a complete 16-token (dim=128, FP16) Key segment occupies **one complete 4 KiB flash page**, and the Value segment occupies **a separate 4 KiB flash page**, yielding an **8 KiB logical block**.
- In-storage Top-$k$ attention scoring reads **ONLY the Key page** during candidate scoring, leaving the Value page untouched in flash until post-selection. Conflating K and V into a single 4 KiB unit prevents modeling this read asymmetry in FTL (P2) and prefetchers (P3).

---

## 2. Current Contract

In `common/schemas/kv_block.py`:
```python
@dataclass
class KVBlock:
    block_id: int
    layer_id: int
    token_start: int
    token_count: int
    kv_head_start: int
    kv_head_count: int
    head_dim: int
    dtype: str
    key_size_bytes: int
    value_size_bytes: int
    storage_tier: str
    hotness: float = 1.0
    physical_location: Optional[str] = None
```

---

## 3. Proposed Contract Update

Maintain full backwards-compatibility with the existing dataclass attributes, while correcting the size calculation helper methods and providing explicit page properties:

```python
@dataclass
class KVBlock:
    block_id: int
    layer_id: int
    token_start: int
    token_count: int
    kv_head_start: int
    kv_head_count: int
    head_dim: int
    dtype: str
    key_size_bytes: int
    value_size_bytes: int
    storage_tier: str
    hotness: float = 1.0
    physical_location: Optional[str] = None

    @property
    def total_size_bytes(self) -> int:
        return self.key_size_bytes + self.value_size_bytes

    @property
    def key_page_count(self) -> int:
        """Number of 4 KiB NAND flash pages required for Key data."""
        return (self.key_size_bytes + 4095) // 4096

    @property
    def value_page_count(self) -> int:
        """Number of 4 KiB NAND flash pages required for Value data."""
        return (self.value_size_bytes + 4095) // 4096

    @classmethod
    def calculate_sizes(
        cls,
        token_count: int,
        kv_head_count: int,
        head_dim: int,
        dtype: str = "FP16"
    ) -> Tuple[int, int]:
        """Calculates accurate (key_size_bytes, value_size_bytes)."""
        bytes_per_elem = 4 if dtype.upper() in ("FP32", "FLOAT32") else (2 if dtype.upper() in ("FP16", "FLOAT16", "BF16") else 1)
        k_size = token_count * kv_head_count * head_dim * bytes_per_elem
        v_size = k_size
        return k_size, v_size

    @classmethod
    def create_default(
        cls,
        block_id: int,
        layer_id: int,
        token_start: int,
        token_count: int = 16,
        kv_head_start: int = 0,
        kv_head_count: int = 1,
        head_dim: int = 128,
        dtype: str = "FP16",
        storage_tier: str = "GPU",
        hotness: float = 1.0,
    ) -> KVBlock:
        k_bytes, v_bytes = cls.calculate_sizes(token_count, kv_head_count, head_dim, dtype)
        return cls(
            block_id=block_id,
            layer_id=layer_id,
            token_start=token_start,
            token_count=token_count,
            kv_head_start=kv_head_start,
            kv_head_count=kv_head_count,
            head_dim=head_dim,
            dtype=dtype,
            key_size_bytes=k_bytes,
            value_size_bytes=v_bytes,
            storage_tier=storage_tier,
            hotness=hotness,
        )
```

And update `common/constants.py`:
```python
DEFAULT_KEY_SIZE_BYTES = DEFAULT_BLOCK_TOKENS * DEFAULT_KV_HEADS_PER_BLOCK * DEFAULT_HEAD_DIM * BYTES_PER_FP16  # 4096 bytes (4 KiB)
DEFAULT_VALUE_SIZE_BYTES = DEFAULT_KEY_SIZE_BYTES  # 4096 bytes (4 KiB)
DEFAULT_BLOCK_SIZE_BYTES = DEFAULT_KEY_SIZE_BYTES + DEFAULT_VALUE_SIZE_BYTES  # 8192 bytes (8 KiB)
```

---

## 4. Reason for Change

1. **Physical Accuracy**: 16 tokens $\times$ 1 head $\times$ 128 dim $\times$ 2 bytes = 4096 bytes = exactly one physical 4 KiB flash page for Key. Value is an identical 4096 bytes = one 4 KiB flash page. Total logical block = 8 KiB.
2. **I/O Asymmetry in Top-$k$**: Enables P2 (FTL) to accurately model 4 KiB read traffic when fetching only Key pages for candidate evaluation, rather than an artificial 2 KiB slice.
3. **Multi-Architecture Support**: Supports varying `head_dim` (e.g. 64 for Qwen/TinyLlama, 128 for LLaMA-2/3) and dtypes (FP32, FP16, FP8) with exact byte formulas.

---

## 5. Compatibility Impact

- **Zero Breaking Schema Changes**: All existing fields (`block_id`, `layer_id`, `key_size_bytes`, `value_size_bytes`, etc.) remain identical.
- **P2 / FEMU Impact**: FTL will now map 4 KiB Key pages and 4 KiB Value pages to separate flash LBAs (or contiguous LBA pairs), correctly capturing the 4 KiB physical page granularity of real NAND flash.
- **P3 Impact**: Storage backend and prefetcher will allocate 8 KiB buffers per logical block (4 KiB Key + 4 KiB Value).


---

## Source: docs/v2/proposals/PROP-002-KV-ACCESS-TRACE-SCHEMA.md

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


---

## Source: docs/v2/proposals/P2_STORAGE_BACKEND_INTERFACE.md

# Proposal: P2 Storage Backend Interface Specification for P3 Integration

**Proposal ID**: PROPOSAL-P2-001  
**Author**: Person 2 (Storage & Systems Engineer)  
**Target Consumer**: Person 3 (System Pipeline & Orchestrator)  
**Status**: SUBMITTED FOR REVIEW  

---

## 1. Overview & Objective

To decouple Person 3's pipeline orchestration from low-level storage mechanics, this document defines the standardized `StorageBackend` interface provided by Person 2. 

P3 can consume either the **Analytical FTL Backend** (Level 1/2 for cycle-accurate hardware simulation) or the **Virtual NVMe Backend** (Level 3 for executable Linux block I/O) through an identical API contract.

---

## 2. Shared Data Structures & Contract Types

### 2.1 Request Format (`StorageIORequest`)
```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

class StorageOp(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    ERASE = "ERASE"
    QUERY = "QUERY"

@dataclass
class StorageIORequest:
    """
    Standard I/O descriptor submitted by P3 to P2 storage backends.
    """
    request_id: str
    operation: StorageOp
    
    # Logical addressing
    block_id: int                     # Global 32-bit KV block ID
    lba_start: Optional[int] = None   # Logical Block Address (4096-byte sectors)
    byte_offset: int = 0              # Byte offset within the block/sector
    byte_length: int = 4096           # Transfer length (default: 4 KB KV block)
    
    # Tensor context (used for deterministic tensor-aware striping)
    layer_id: int = 0
    head_id: int = 0
    token_start: int = 0
    token_count: int = 16
    
    # Optional payload for writes
    data: Optional[bytes] = None
    
    # Metadata & priority
    priority: int = 0                 # 0 = Normal, 1 = High (demand fetch vs prefetch)
    timestamp_ns: Optional[int] = None
```

### 2.2 Completion / Result Format (`StorageIOResult`)
```python
class StorageStatus(str, Enum):
    SUCCESS = "SUCCESS"
    QUEUED = "QUEUED"
    ERR_DEVICE_BUSY = "ERR_DEVICE_BUSY"
    ERR_CAPACITY_EXCEEDED = "ERR_CAPACITY_EXCEEDED"
    ERR_INVALID_ADDRESS = "ERR_INVALID_ADDRESS"

@dataclass
class StorageIOResult:
    """
    Completion descriptor returned to P3 upon operation completion.
    """
    request_id: str
    status: StorageStatus
    
    # I/O Telemetry
    bytes_transferred: int
    latency_us: float                 # Simulated or measured device latency in microseconds
    queue_wait_us: float              # Time spent waiting in submission queue
    device_service_us: float          # Time spent executing on flash / controller
    
    # Channel & Die Telemetry (Level 1/2)
    assigned_channel: Optional[int] = None
    assigned_die: Optional[int] = None
    channel_contention_index: float = 1.0  # Ratio of channel load to average
    
    # Physical/Logical Mapping Info
    physical_location: Optional[str] = None # e.g., "ch3_die1_pl0_blk4_pg12"
    lba_address: Optional[int] = None
    
    # Data payload (for READ operations)
    data: Optional[bytes] = None
    error_message: Optional[str] = None
```

---

## 3. Storage Backend Base Class (`BaseStorageBackend`)

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class BaseStorageBackend(ABC):
    """
    Abstract interface exposed by Person 2 to Person 3.
    """
    
    @abstractmethod
    def read_block(self, block_id: int, layer_id: int, head_id: int, token_idx: int) -> StorageIOResult:
        """Synchronously reads a single 4 KB KV block."""
        pass
        
    @abstractmethod
    def write_block(self, block_id: int, layer_id: int, head_id: int, token_idx: int, data: Optional[bytes] = None) -> StorageIOResult:
        """Synchronously writes a single 4 KB KV block."""
        pass
        
    @abstractmethod
    def read_batch(self, requests: List[StorageIORequest]) -> List[StorageIOResult]:
        """
        Executes a batch of parallel read requests.
        Evaluates multi-channel parallelism and calculates batch completion latency.
        """
        pass
        
    @abstractmethod
    def get_telemetry(self) -> Dict[str, Any]:
        """Returns cumulative I/O statistics, channel utilization, and queue metrics."""
        pass
        
    @abstractmethod
    def reset(self) -> None:
        """Resets mapping tables and internal state between benchmark runs."""
        pass
```

---

## 4. Backend Implementations Provided by P2

1. **`AnalyticalStorageBackend` (Level 1/2)**:
   - Configurable modes: `"conventional"` vs. `"tensor_aware"`.
   - Accurate 8-channel, 4-die NAND contention calculation.
   - Zero physical disk dependence; rapid in-memory execution for large multi-layer simulation runs.
2. **`VirtualNVMeStorageBackend` (Level 3)**:
   - Dispatches real block I/O requests to `/dev/nvme0n1` or the raw virtual storage image `/opt/ai-ssd-v2/images/v2_nvme.raw`.
   - Uses direct I/O (`O_DIRECT`) to bypass OS cache.
   - Measures real OS kernel syscall, NVMe driver, and DMA transfer timings.

---

## 5. Error Behavior & Assumptions

- **Capacity Limit**: The storage backend provides a capacity of 1.0 GiB (262,144 blocks of 4 KB). Exceeding capacity returns `ERR_CAPACITY_EXCEEDED`.
- **Idempotency**: Writing to an existing `block_id` updates the mapping table and writes to a new page (standard out-of-place flash update).
- **Asynchronous Semantics**: Batches submitted via `read_batch` simulate concurrent submission across independent queues (matching NVMe submission queues and independent flash channels).
