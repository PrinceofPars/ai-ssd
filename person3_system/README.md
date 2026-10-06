# Person 3: System, API, Prefetch & Dashboard

**Owner**: Person 3 (System Architect & Integration Lead)  
**Owned Benchmark Scripts & Demos**:
- `benchmarks/run_prefetch.py`
- `benchmarks/run_full_system.py`
- `demo/demo.py`
- `person3_system/dashboard/dashboard.py` (Interactive Streamlit Dashboard)

---

## Directory Organization

```text
person3_system/
├── api/
│   ├── ai_ssd.py               # Main Unified API facade
│   ├── requests.py             # Request validator and factory
│   └── responses.py            # Standardized response generator
├── prefetch/
│   ├── predictor.py            # Next-layer KV block predictor
│   ├── history.py              # Sequence access pattern history
│   └── prefetcher.py           # Asynchronous DRAM staging buffer
├── integration/
│   ├── orchestrator.py         # Connects KV engine and SSD engine
│   └── pipeline.py             # Full inference simulation loop
├── dashboard/
│   └── dashboard.py            # Interactive Streamlit application
└── tests/
    └── test_p3_mock_pipeline.py # Integration test using mock components
```

---

## Testing & Architecture Reference

- **Subsystem Tests**: `pytest person3_system/tests/ tests/test_nvme_client.py`
- **Canonical Architecture**: See [`docs/v2/ARCHITECTURE.md`](../docs/v2/ARCHITECTURE.md)
- **Benchmarking Protocol**: See [`docs/historical/EXPERIMENT_PROTOCOL.md`](../docs/historical/EXPERIMENT_PROTOCOL.md)

---

## Independent Development

Person 3 can build and verify the API gateway, speculative prefetch pipeline, and Streamlit dashboard independently using `MockKVEngine` and `MockSSD`:

```bash
streamlit run person3_system/dashboard/dashboard.py
```
