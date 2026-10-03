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