# AI-SSD V2 Decisions

This file records important project-wide technical decisions.

## DECISION-001: Abstract StorageBackend Architecture
Date: 2026-10-02
Decision: Introduce `StorageBackend` abstract base class with synchronous, asynchronous, and batch submission interfaces.
Reason: Decouples upper-level LLM orchestrator and speculative prefetcher from specific physical/simulated storage mechanisms (Mock, File Direct I/O, Analytical FTL, FEMU NVMe).
Alternatives considered: Direct coupling to FEMU C emulator or in-memory arrays.
Affected agents: P1, P2, P3
Status: ACCEPTED

## DECISION-002: Direct I/O and User-Space Threadpool Asynchrony
Date: 2026-10-02
Decision: Adopt POSIX Direct I/O (`O_DIRECT`) and user-space threadpool executor for storage simulation and real disk access.
Reason: Prevents Linux kernel page cache from masking flash read/write latencies; eliminates kernel dependencies associated with io_uring while delivering predictable, reproducible latency profiling.
Alternatives considered: Synchronous buffered I/O, libaio, io_uring.
Affected agents: P3
Status: ACCEPTED

## DECISION-003: Canonical Shared Trace Schema & KV Physical Sizing Contract
Date: 2026-10-03
Decision: Define `common/schemas/trace.py` and `common/schemas/kv_block.py` as canonical contracts unifying P1 real LLM traces and P3 trace reader. Enforce explicit physical dimensions: Key page = 4,096 bytes, Value page = 4,096 bytes, combined logical block = 8,192 bytes.
Reason: Resolves Phase 1 integration blocker where P3 TraceReader expected synthetic header and P1 emitted real traces with external manifest and distinct operation naming. Prevents analytical backend from hardcoding 4096 bytes on combined 8192-byte writes.
Alternatives considered: Re-generating P1 trace files to match synthetic P3 reader format (rejected: real production traces must be preserved).
Affected agents: P1, P2, P3
Status: ACCEPTED

## DECISION-004: Direct Integration with P2 Deterministic Tensor Mapping
Date: 2026-10-03
Decision: Standardize P3's `AnalyticalFTLBackend` on P2's canonical `DeterministicTensorMapper` (`person2_ssd/kv_allocator/tensor_mapping.py`) instead of maintaining a divergent analytical mapping.
Reason: Guarantees bit-exact physical NAND channel, die, plane, block, and page resolution across 8 flash channels between P2 FTL and P3 system telemetry.
Alternatives considered: Maintaining separate analytical striping logic in P3 (rejected: creates cross-agent divergence).
Affected agents: P2, P3
Status: ACCEPTED

## DECISION-005: Real Prefetching Methodology & Metric Discipline
Date: 2026-10-03
Decision: Deprecate the synthetic 99.2% prefetch hit rate claim and replace it with empirical evaluation on the real Qwen2.5-0.5B KV trace across Conservative (28.5%), Normal (65.3%), and Aggressive (96.0%) policies with rigorous accounting of useless prefetches (cache pollution) and wasted bytes.
Reason: Prevents unscientific claims based on artificial sequential synthetic traces and reflects real transformer inter-layer attention locality.
Alternatives considered: Retaining static hit-rate parameters.
Affected agents: P3
Status: ACCEPTED

## DECISION-006: Separation of Real, Analytical, and Virtual-Device Measurements
Date: 2026-10-03
Decision: Strictly categorize all reported metrics in documentation and JSON exports as `REAL`, `ANALYTICAL`, or `VIRTUAL-DEVICE`. Prohibit fabricating combined end-to-end latency without explicit classification.
Reason: Maintains scientific credibility and clarity between physical CPU inference, simulated flash FTL timing models, and guest virtual NVMe hardware baselines.
Alternatives considered: Conflating all numbers into a single synthetic latency figure.
Affected agents: P1, P2, P3
Status: ACCEPTED
