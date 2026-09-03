# Original User Request

## Initial Request — 2026-09-03T08:38:01Z

Requested team: Multi-agent team with specialized roles across NAND physics, FTL algorithms, and benchmarking

Build a comprehensive, high-fidelity analytical Python simulator for modern multi-channel Solid-State Drives (SSDs) and Flash Translation Layers (FTL). The system must model flash timing physics and demonstrate the throughput benefits of tensor-aware striped KV block placement over conventional FTL during sparse LLM KV cache retrieval.

Working directory: c:\Users\Admin\Documents\projects\ai-ssd
Integrity mode: development

## Requirements

### R1. NAND Flash Hierarchy & Contention Timing Model
Implement an analytical cycle-level physical simulation of NAND flash memory (Pages, Blocks, Planes, Dies, and Channels). The timing model must account for NAND read (t_R), program (t_PROG), block erase (t_BERS), bus transfer latency, and channel contention/serialization when multiple read requests target the same channel concurrently.

### R2. Conventional vs. Tensor-Aware FTL Allocation
Implement two distinct FTL allocation strategies adhering to the existing common/schemas/kv_block.py contract:
1. Conventional FTL: Standard sequential page-level mapping table that allocates blocks without attention layout awareness, causing channel hot-spotting under concurrent attention head access.
2. Tensor-Aware FTL: Intelligent multi-channel and multi-die striping that distributes co-accessed attention KV blocks across distinct channels and dies round-robin to maximize read parallelism.

### R3. Standalone Verification & Benchmark Suite
Provide an isolated test suite in person2_ssd/tests/ that verifies FTL placement and NAND physics using the standalone MockKVEngine, requiring zero dependencies on other teammates' modules. Implement benchmarks/run_ftl.py to compare conventional vs. tensor-aware read latency across batch sizes (16, 32, 64, 128, 256) and export results to results/raw/ftl_results.csv.

### R4. Architectural Boundaries
All simulator implementation must reside within person2_ssd/ and benchmarks/run_ftl.py. Do not modify files in person1_kv_engine/, person3_system/, or the frozen contracts in common/schemas/.

## Acceptance Criteria

### Physical Timing & Contention Modeling
- [ ] Concurrently reading N blocks mapped to the same channel takes strictly longer than reading N blocks striped across N distinct channels.
- [ ] Latency model accounts for t_R, bus transfer time per page, and channel queue serialization.

### FTL Performance & Speedup
- [ ] Tensor-aware FTL achieves at least a 2.5x speedup in estimated read latency compared to conventional FTL when reading 64 or more parallel KV blocks across 8 channels.
- [ ] Mapping table accurately records physical locations in "ch<C>_die<D>_pl<P>_blk<B>_pg<G>" format and successfully translates logical block IDs to physical locations.

### Test & Benchmark Verification
- [ ] Running python scripts/run_tests.py passes all tests with zero failures.
- [ ] Running python benchmarks/run_ftl.py executes without errors and outputs results/raw/ftl_results.csv containing columns experiment, batch_size, conventional_latency_us, tensor_aware_latency_us, and speedup_x.
- [ ] Zero external package requirements beyond the Python standard library to run tests and benchmarks.
