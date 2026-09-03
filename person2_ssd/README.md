# Person 2: SSD / FTL Simulator

**Owner**: Person 2 (Storage & Systems Engineer)  
**Owned Benchmark Scripts**:
- `benchmarks/run_ftl.py`

---

## Directory Organization

```text
person2_ssd/
├── nand/
│   ├── page.py                 # Flash page state (FREE, VALID, INVALID)
│   ├── block.py                # Flash erase block & wear-count tracker
│   └── nand.py                 # Die & plane hardware model
├── channels/
│   ├── channel.py              # Flash channel bus controller & serialization
│   └── die.py                  # Independent die execution model
├── ftl/
│   ├── conventional.py         # Standard page-level FTL mapping & GC
│   ├── tensor_aware.py         # Channel-striped co-designed KV allocator
│   └── mapping.py              # LPN -> PPN mapping table
├── kv_allocator/
│   ├── allocator.py            # High-level block placement API
│   └── placement.py            # Striping strategies (channel/die/plane)
├── storage_model/
│   ├── latency.py              # Analytical latency calculation (tR, tPROG, tBERS)
│   ├── bandwidth.py            # Bus transfer & channel contention model
│   └── io_model.py             # Combined I/O request simulator
├── mock_kv_engine.py           # Standalone MockKVEngine for zero-dependency local testing
└── tests/
    └── test_p2_mock.py         # Unit tests proving Person 2 module works independently
```

---

## Independent Parallel Development Guide

Person 2 can develop and benchmark both Conventional and Tensor-Aware FTL **without waiting for Person 1's real LLM code**.
Import and use `MockKVEngine`:

```python
from person2_ssd.mock_kv_engine import MockKVEngine

kv_engine = MockKVEngine()
blocks = kv_engine.generate_kv_blocks(num_blocks=64)
# Pass blocks to Person 2's FTL and measure channel striping speedup!
```
