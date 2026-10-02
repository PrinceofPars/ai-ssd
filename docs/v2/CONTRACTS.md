# AI-SSD V2 Contracts

This document will contain the shared interfaces between P1, P2
and P3.

The contracts should be frozen before major integration work.

Potential shared objects include:

- KVBlock
- KVRequest
- KVResponse
- KVTrace
- StorageBackend
- Metrics
- ExperimentConfig
- ExperimentResult

No agent should silently change a shared contract.

Contract changes should be documented in:

docs/v2/proposals/
