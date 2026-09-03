# TEST_INFRA: Opaque-Box E2E Testing Track Infrastructure

## 1. Overview & Architectural Boundary
The **Dual-Track Testing Strategy** in AI-SSD separates unit/mock-level developer tests (Track 1) from opaque-box End-to-End (E2E) requirement-driven acceptance tests (Track 2).

This document establishes the architecture, test methodology, tier taxonomy, execution runner, and quality gates for the **E2E Testing Track (Track 2)**.

### Architectural Boundaries & Scope
- **SUT (System Under Test)**: The Person 2 SSD Subsystem, consisting of the NAND physical hierarchy model, cycle-level timing/contention physics model, Flash Translation Layer (FTL) strategies (`ConventionalFTL` and `TensorAwareFTL`), and the `StorageSimulator` orchestration engine.
- **Opaque-Box Principle**: Tests interact with the subsystem strictly via documented public APIs:
  - `person2_ssd.storage_model.io_model.StorageSimulator`
  - `person2_ssd.mock_kv_engine.MockKVEngine`
  - `common.schemas.kv_block.KVBlock`
  - `person2_ssd.storage_model.latency.LatencyModel`
  - `person2_ssd.ftl.conventional.ConventionalFTL`
  - `person2_ssd.ftl.tensor_aware.TensorAwareFTL`
  - `person2_ssd.nand.page.FlashPage`, `person2_ssd.nand.block.FlashBlock`
- **Zero External Dependencies**: The test suite and runner execute entirely on Python 3 standard library modules (`unittest`, `math`, `time`, `sys`, `pathlib`, `re`, `csv`, `typing`).

---

## 2. Four-Tier Testing Methodology

The E2E test suite (`tests/e2e/test_ssd_ftl_e2e.py`) is organized into four rigorous test tiers, systematically applying established software testing methodologies:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        E2E TEST SUITE TIERS                            │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Category-Partition Feature Coverage (R1, R2, R3)               │
│         - NAND Physics (5 tests)        - Conventional FTL (5 tests)   │
│         - Tensor-Aware FTL (5 tests)    - Address Format (5 tests)     │
│         - Benchmark & Speedup (5 tests)                                │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Boundary Value Analysis (BVA) & Corner Cases                   │
│         - Batch Size Limits (5 tests)   - Block Exhaustion (5 tests)   │
│         - Empty/Unmapped (5 tests)      - Irregular Heads (5 tests)    │
│         - Long Sequences (5 tests)                                     │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Pairwise Combinatorial Cross-Feature Interactions (6 tests)    │
│         - Striping + Contention        - Multi-Die + Serialization    │
│         - Re-allocation Parity         - Multi-Layer Top-K Balancing  │
│         - Simulator Mode Switching     - Polymorphic Reads Breakdown   │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 4: Real-World Application Workloads (5 tests)                     │
│         - 32-Layer Sparse Decode       - Streaming Sink+Recent Top-K   │
│         - Prefill + Multi-Turn Decode  - Grouped-Query Attention (GQA)│
│         - Effective Throughput & IOPS  (>7x Speedup Verification)      │
└────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Category-Partition Feature Coverage
Applies the Category-Partition method to exhaustively partition functional input domains and verify core feature contracts against explicit mathematical and physical models:
- **Feature 1 (R1 NAND Physics)**: Verification of exact physical timing formulas ($t_R = 25.0\,\mu s$, $t_{bus} = 5.0\,\mu s$, $t_{PROG} = 200.0\,\mu s$, $t_{BERS} = 2000.0\,\mu s$, $t_{pcie} = 10.0\,\mu s$), NAND page lifecycle states (`FREE` -> `VALID` -> `INVALID` -> `FREE`), and channel queue serialization.
- **Feature 2 (R2 Conventional FTL Hot-Spotting)**: Verification of sequential page-level allocation concentrating all parallel attention requests on Channel 0, forming linear serialization bottlenecks ($T = t_{pcie} + N \times (t_R + t_{bus})$).
- **Feature 3 (R2 Tensor-Aware Striping)**: Verification of multi-channel round-robin striping ($T = t_{pcie} + \lceil N/8 \rceil \times (t_R + t_{bus})$), elimination of odd-channel parity starvation, and multi-die distribution across all 4 dies.
- **Feature 4 (R2 Address Formatting & Translation)**: Verification of canonical physical address format `ch<C>_die<D>_pl<P>_blk<B>_pg<G>`, coordinate bounds, bidirectional lookup parity (`translate` $\leftrightarrow$ `reverse_translate`), and stale mapping eviction.
- **Feature 5 (R3 Benchmark Execution & Acceptance)**: Verification of benchmark batch scaling across $[16, 32, 64, 128, 256]$, acceptance threshold $\ge 2.5\times$ speedup at batch size 64+, output CSV schema compliance, and monotonic latency scaling.

### Tier 2: Boundary Value Analysis (BVA) & Corner Cases
Probes extreme limits, edge conditions, invalid inputs, and resource boundaries:
- **Feature 1 (Batch Size Boundaries)**: $N=0$ (zero latency, empty dictionary), $N=1$ ($40.0\,\mu s$ base latency), prime batch sizes ($N=3, 7, 13, 31$, verifying load difference $\le 1$), sub-channel counts ($N \in [2..7]$), and large-scale batches ($N=1024$).
- **Feature 2 (Limits & Capacity Exhaustion)**: Exact drive capacity exhaustion raising `RuntimeError`, flash block page limit overflow returning `None`, double-programming rejection raising `ValueError`, erase wear endurance limits (`is_bad_block`), and post-reset capacity recovery.
- **Feature 3 (Empty & Unmapped Requests)**: Empty query lists, unmapped logical IDs (`-1`, `999999`) returning `None`, unmapped physical strings returning `None`, batch reads with non-existent IDs returning `0.0`, and malformed location strings defaulting to channel 0.
- **Feature 4 (Irregular Head Counts & Layout Geometries)**: Single-head MQA ($H=1$), non-power-of-two architectures ($H=24, 40$), `head_major` layouts, missing/None block attributes, and $token\_count=0$ handling without `ZeroDivisionError`.
- **Feature 5 (Large Sequence Lengths)**: Context windows up to 131,072 tokens ($8,192$ blocks), validating all physical coordinates remain strictly within hardware bounds with zero translation collisions.

### Tier 3: Pairwise Combinatorial Cross-Feature Interactions
Validates non-trivial interactions between distinct subsystem components:
- Allocation strategy combined with analytical latency model contention.
- Multi-die interleaving combined with flash page state progression and bus queue scheduling.
- Bijective reverse translation during live block re-allocation / migration across channels.
- Multi-layer synthetic KV cache generation combined with top-k sparse retrieval.
- StorageSimulator lifecycle resets combined with dynamic FTL mode transitions.
- Polymorphic read interfaces (`KVBlock` objects, integer IDs, mixed lists) cross-checked against latency breakdown metrics.

### Tier 4: Real-World Application Workloads
Simulates end-to-end production AI inference access patterns:
- **32-Layer Sparse Decode Phase**: Models 32-layer, 32-head transformer sparse attention retrieval (128-256 blocks), demonstrating $\ge 2.5\times$ (and observed $>7.0\times$) speedup.
- **Attention Sink + Recent Context Retrieval**: Models StreamingLLM sink cache (prompt anchors) combined with sliding recent context cache across flash channels.
- **Prefill + Multi-Turn Decode Lifecycle**: End-to-end cache ingestion (1,024 blocks across 32 layers) followed by 10 consecutive decode turns, verifying physical page read disturb metrics (`read_count`).
- **Grouped-Query Attention (GQA)**: 8 KV groups mapped across 8 flash channels, proving zero contention for GQA decode requests.
- **Throughput & IOPS Scaling**: Quantitative evaluation of read throughput in GB/s and effective IOPS under batch size 256.

---

## 3. Test Runner & Execution Commands

### Standard Runner Command
Execute the standalone E2E test runner from the repository root:
```powershell
python tests/e2e/run_e2e.py
```

### Direct Unittest Execution
The test suite is fully compatible with the standard Python `unittest` module:
```powershell
python -m unittest tests/e2e/test_ssd_ftl_e2e.py -v
```

### Complete Project Verification Command
To verify both Track 1 (subsystem mock tests) and Track 2 (E2E testing suite):
```powershell
python scripts/run_tests.py
python tests/e2e/run_e2e.py
python benchmarks/run_ftl.py
```

---

## 4. Coverage Thresholds & Quality Gates

| Gate Metric | Requirement | Verification Method |
| :--- | :--- | :--- |
| **Pass Rate** | **100% Pass** (0 Failures, 0 Errors) | `tests/e2e/run_e2e.py` exit code 0 |
| **Tier 1 Coverage** | $\ge 5$ tests per feature (25 tests min) | Tier 1 suite count verification |
| **Tier 2 Coverage** | $\ge 5$ tests per feature (25 tests min) | Tier 2 suite count verification |
| **Tier 3 Coverage** | $\ge 5$ pairwise combinatorial tests | Tier 3 suite count verification |
| **Tier 4 Coverage** | $\ge 5$ real-world application tests | Tier 4 suite count verification |
| **Total Test Count** | $\ge 60$ comprehensive test cases | Test runner summary output |
| **Acceptance Speedup** | Speedup $\ge 2.5\times$ for $N \ge 64$ | `test_t1_benchmark_speedup_acceptance_threshold` |
| **Contention Inequality** | $T_{\text{conv}} > T_{\text{ta}}$ for all $N \ge 2$ | `test_t1_nand_channel_contention_serialization` |
| **Format Conformance** | 100% canonical `ch<C>_die<D>_pl<P>_blk<B>_pg<G>` | `test_t1_address_canonical_regex_conformance` |
| **Odd-Channel Starvation** | Zero odd-channel load deficit | `test_t1_ta_ftl_zero_odd_channel_starvation` |
| **Dependency Free** | Pure Python standard library only | Standard Python 3 interpreter execution |
