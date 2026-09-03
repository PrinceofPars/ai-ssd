# Person 1: AI / KV Cache Engine

**Owner**: Person 1 (AI / Systems Engineer)  
**Owned Benchmark Scripts**:
- `benchmarks/run_baseline.py`
- `benchmarks/run_offload.py`
- `benchmarks/run_topk.py`

---

## Directory Organization

```text
person1_kv_engine/
├── baseline/
│   └── baseline_kv.py          # Ground-truth uncompressed KV cache
├── cache_manager/
│   ├── kv_cache.py             # Top-level multi-tier cache controller
│   ├── hot_cold.py             # Attention sink & sliding window classification
│   ├── eviction.py             # GPU -> DRAM -> SSD tier migration policy
│   └── block_manager.py        # Token-to-KVBlock allocation and indexing
├── attention/
│   ├── attention.py            # Layer attention computation abstraction
│   └── scoring.py              # Query-Key dot product importance estimation
├── topk/
│   ├── selector.py             # Top-k block filtering
│   └── evaluator.py            # Precision/recall evaluator against dense attention
├── workloads/
│   └── generator.py            # Synthetic & realistic token generation traces
├── mock_ssd.py                 # Standalone MockSSD for zero-dependency local testing
└── tests/
    └── test_p1_mock.py         # Unit tests proving Person 1 module works independently
```

---

## Independent Parallel Development Guide

Person 1 can develop and test the entire KV cache and Top-k algorithm **without waiting for Person 2's SSD**.
Import and use `MockSSD`:

```python
from person1_kv_engine.mock_ssd import MockSSD

ssd = MockSSD()
ssd.store_block(block)
loaded = ssd.load_block(block.block_id)
```
