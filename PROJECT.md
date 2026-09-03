# Project: AI-SSD & FTL Simulator (Person 2 Subsystem)

## Architecture
- **Boundary & Scope**: All simulator code resides exclusively within `person2_ssd/` and `benchmarks/run_ftl.py`. No modifications allowed to `person1_kv_engine/`, `person3_system/`, or `common/schemas/`.
- **NAND Flash Hierarchy**: 5-level physical modeling (Channel -> Die -> Plane -> Block -> Page).
  - 8 Channels, 4 Dies/Channel, 2 Planes/Die, 512 Blocks/Plane, 128 Pages/Block, 4096 Bytes/Page.
- **Timing Physics Model**:
  - $t_R = 25.0\,\mu s$ (NAND page read sensing)
  - $t_{bus} = 5.0\,\mu s$ per 4 KB page (channel bus transfer)
  - $t_{PROG} = 200.0\,\mu s$ (NAND page program)
  - $t_{BERS} = 2000.0\,\mu s$ (NAND block erase)
  - $t_{pcie} = 10.0\,\mu s$ (PCIe/NVMe command dispatch)
  - Channel contention: Concurrent page transfers on the same channel serialize linearly.
- **FTL Allocation Strategies**:
  - `ConventionalFTL`: Sequential page-level allocation ignoring tensor structure, concentrating parallel requests on single channels (hot-spotting).
  - `TensorAwareFTL`: Tensor-aware multi-channel and multi-die striping distributing co-accessed KV blocks across distinct channels and dies round-robin to maximize read parallelism.
- **Address Format**: Canonical format `ch<C>_die<D>_pl<P>_blk<B>_pg<G>` with bidirectional mapping table translation.
- **Mock & Testing**: Pure Python 3 standard library standalone test suite in `person2_ssd/tests/test_p2_mock.py` discovered by `scripts/run_tests.py`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | NAND Physical Hierarchy | 5-level modeling: Channel, Die, Plane, Block, Page with state management | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Timing Physics & Latency Model | Cycle-level analytical model accounting for t_R, t_bus, t_PROG, t_BERS, and PCIe overhead | M1 | ORIGINAL_REQUEST §R1 |
| 3 | Channel Contention & Serialization | Analytical serialization queue when multiple concurrent reads target the same channel | M1 | ORIGINAL_REQUEST §R1 |
| 4 | Conventional FTL Strategy | Sequential page-level allocation without attention layout awareness | M2 | ORIGINAL_REQUEST §R2 |
| 5 | Tensor-Aware Striped FTL Strategy | Multi-channel & multi-die round-robin striping for co-accessed attention KV blocks | M2 | ORIGINAL_REQUEST §R2 |
| 6 | 8-Channel Striping Parity Fix | Fix channel distribution formula to eliminate odd-channel starvation under MockKVEngine | M2 | Survey findings |
| 7 | Physical Address Translation | Strict canonical format `ch<C>_die<D>_pl<P>_blk<B>_pg<G>` and translation logic | M2 | ORIGINAL_REQUEST §R2 |
| 8 | Standalone Test Suite in person2_ssd/tests/ | Zero external dependencies, pure stdlib, isolated MockKVEngine verification | M3 | ORIGINAL_REQUEST §R3 |
| 9 | Contention Latency Verification Test | Verify reading N blocks on same channel takes strictly longer than across N channels | M3 | ORIGINAL_REQUEST §Acceptance |
| 10 | Speedup Acceptance Verification (>=2.5x) | Verify Tensor-Aware achieves >= 2.5x speedup for batch sizes >= 64 across 8 channels | M3 | ORIGINAL_REQUEST §Acceptance |
| 11 | FTL Benchmark Runner | Execute across batch sizes [16, 32, 64, 128, 256] and export results/raw/ftl_results.csv | M3 | ORIGINAL_REQUEST §R3 |
| 12 | Opaque-Box E2E Testing Suite (Tiers 1-4) | Comprehensive requirement-driven test harness with TEST_INFRA.md and TEST_READY.md | M4 | Orchestrator Dual-Track |
| 13 | Adversarial Coverage Hardening (Tier 5) | White-box adversarial testing and forensic audit integrity verification | M5 | Orchestrator Final Gate |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | NAND Flash Hierarchy & Contention Timing Model | `person2_ssd/nand/`, `person2_ssd/channels/`, `person2_ssd/storage_model/` | None | DONE |
| M2 | Conventional vs Tensor-Aware FTL Allocation | `person2_ssd/ftl/`, `person2_ssd/mock_kv_engine.py` | M1 | DONE |
| M3 | Standalone Test Suite & Benchmark Runner | `person2_ssd/tests/test_p2_mock.py`, `benchmarks/run_ftl.py` | M1, M2 | DONE |
| M4 | E2E Testing Track Suite (Tiers 1-4) | E2E test infra, `TEST_INFRA.md`, `TEST_READY.md` | None (Parallel) | DONE |
| M5 | Final Milestone: 100% E2E Pass & Tier 5 Hardening | Full integration, Tier 5 adversarial tests, forensic audit | M1, M2, M3, M4 | DONE |

## Interface Contracts
### `FlashBlock` & `FlashPage` (`person2_ssd/nand/`)
- `FlashPage(page_id: int)`: `program(block_id: int)`, `invalidate()`, `erase()`, `state: PageState`
- `FlashBlock(block_id: int, pages_count: int = 128)`: `allocate_page(logical_block_id: int) -> Optional[FlashPage]`, `erase()`

### `LatencyModel` (`person2_ssd/storage_model/latency.py`)
- `calculate_batch_read_latency(physical_locations: List[str]) -> float`: Returns latency in microseconds ($\mu s$).
- Formula: $T = t_{pcie} + \max_{c \in [0..7]} (N_c \times (t_R + t_{bus}))$.
- Address parsing: Extracts channel index $c$ from location string matching `"ch<C>_die<D>_pl<P>_blk<B>_pg<G>"`.

### `FTL` Base & Subclasses (`person2_ssd/ftl/`)
- `allocate(block: KVBlock) -> str`: Allocates physical location string adhering to `"ch<C>_die<D>_pl<P>_blk<B>_pg<G>"`.
- `translate(block_id: int) -> Optional[str]`: Bidirectional lookup of physical location.
- `get_mapping_table() -> Dict[int, str]`: Mapping table snapshot.

### `StorageSimulator` (`person2_ssd/storage_model/io_model.py`)
- `store_block(block: KVBlock) -> str`: Calls FTL allocator and returns physical location.
- `read_blocks(blocks: List[KVBlock]) -> float`: Resolves physical locations and evaluates batch read latency.

## Code Layout
- `person2_ssd/nand/`: NAND physical classes (`page.py`, `block.py`, `nand.py`)
- `person2_ssd/channels/`: Channel contention classes (`channel.py`)
- `person2_ssd/ftl/`: FTL strategies (`base.py`, `conventional.py`, `tensor_aware.py`)
- `person2_ssd/storage_model/`: Latency and simulator engines (`latency.py`, `io_model.py`)
- `person2_ssd/mock_kv_engine.py`: Standalone mock generator
- `person2_ssd/tests/`: Standalone test suite (`test_p2_mock.py`)
- `benchmarks/run_ftl.py`: Benchmark execution script
- `results/raw/ftl_results.csv`: Benchmark output data
