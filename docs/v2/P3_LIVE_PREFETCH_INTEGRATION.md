# P3 Live Prefetch Integration Specification & Verification

**Document Version**: 1.0.0  
**Phase**: Live Inference Integration — P3 Prefetch Adapter  
**Author**: Person 3 (System Integration / Storage API / Prefetch / Experiments)  
**Worktree**: `/home/ubuntu/ai-ssd-p3`  
**Branch**: `v2/p3-system-integration`  
**Status**: COMPLETE & VERIFIED (All 38 P3 Tests Passing)

---

## 1. Executive Summary & Purpose

Following the Live Inference Audit, Person 3 (P3) has prepared and verified `RealInferencePrefetchAdapter` as an active wrapper around Person 2's (P2) `RealInferenceStorageBackend`. 

This wrapper sits directly between **P1's Live Qwen Inference Engine** (`AISSDKVManager`) and **P2's Storage Backend** (`RealInferenceStorageBackend` / tensor-aware FTL), enabling:
1. **Actual KV tensor data transfer** (NumPy float32 tensors with exact shape `[16, 2, 64]` or `[1, 16, 64]` — strictly NOT metadata-only).
2. **Speculative DRAM staging** of upcoming KV blocks before the attention kernel demands them.
3. **Transparent pass-through** for non-prefetched demand reads, write operations, and block eviction.
4. **Empirical runtime telemetry** tracking demand hits, misses, prefetch requests, useful bytes, wasted bytes, and DRAM staging memory.
5. **Zero artificial latency injection** (`time.sleep` is strictly prohibited; all operations run at native hardware speed).

---

## 2. P3 Prefetch Adapter Architecture & Data-Flow

`RealInferencePrefetchAdapter` serves as a transparent, high-performance proxy wrapping the P2 storage backend:

```
       +-----------------------------------------------------------+
       |           P1: Real Qwen Inference Engine                   |
       |  - AISSDKVManager                                         |
       |  - Top-k Attention Kernel                                 |
       +-----------------------------------------------------------+
                   |                             |
      (Prefetch: Upcoming Blocks)       (Demand Reads: Needed Blocks)
                   |                             |
                   v                             v
       +-----------------------------------------------------------+
       |           P3: RealInferencePrefetchAdapter                 |
       |                                                           |
       |  +-----------------------------------------------------+  |
       |  |          Host DRAM Staging Buffer (LRU)             |  |
       |  |  - Real NumPy Tensors (Key & Value pages)           |  |
       |  |  - StagedInferenceBlock metadata                    |  |
       |  +-----------------------------------------------------+  |
       |                                |                          |
       |               Cache HIT? ------+                          |
       |               |             |                             |
       |             (YES)          (NO)                           |
       |               |             |                             |
       |      Return from DRAM   Demand Fetch                      |
       |      (Zero storage I/O) (Sync backend read)               |
       |               |             |                             |
       |               |             +---------------+             |
       |               |                             |             |
       |  +---------------------------------------+  |             |
       |  | Telemetry Accounting Engine           |  |             |
       |  | - hits, misses, useful/wasted bytes   |  |             |
       |  | - staging memory tracking             |  |             |
       |  +---------------------------------------+  |             |
       +---------------------------------------------+             |
                                                     |             |
                                                     v             v
       +-----------------------------------------------------------+
       |           P2: RealInferenceStorageBackend                 |
       |  - write_block() / read_key_page() / read_value_page()     |
       |  - Tensor-Aware Physical FTL / NVMe Block Device          |
       +-----------------------------------------------------------+
```

---

## 3. Exact Adapter API Exposed to P1

The adapter exposes both P1-native dual-method calls (`read_key_page`, `read_value_page`, `write_block`) and unified storage calls (`read`, `prefetch`).

### Method Signatures

```python
class RealInferencePrefetchAdapter:
    def __init__(
        self,
        storage_backend: Optional[Any] = None,
        staging_capacity_blocks: int = 256,
        enable_prefetch: bool = True,
        head_dim: int = 64,
        tokens_per_block: int = 16,
        num_heads: int = 2,
        dtype: np.dtype = np.float32,
    ): ...

    # --- P1 AISSDKVManager Direct Compatibility Methods ---

    def write_block(
        self,
        layer_idx: int,
        block_id: int,
        k_block: np.ndarray,
        v_block: np.ndarray,
        head_id: int = 0,
        token_start: int = 0,
    ) -> bool:
        """Write KV block tensors to P2 storage backend and invalidate staging."""

    def read_key_page(
        self,
        layer_idx: int,
        block_id: int,
        head_id: int = 0,
        token_start: int = 0,
    ) -> np.ndarray:
        """Read 4KB Key page for in-storage / host Top-k scoring."""

    def read_value_page(
        self,
        layer_idx: int,
        block_id: int,
        head_id: int = 0,
        token_start: int = 0,
    ) -> np.ndarray:
        """Read 4KB Value page for attention winner computation."""

    def read_block(
        self,
        layer_idx: int,
        block_id: int,
        head_id: int = 0,
        token_start: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read both Key and Value pages (8KB logical block) in one operation."""

    # --- P3 Prefetch & Staging Methods ---

    def prefetch(
        self,
        block_ids: List[int],
        layer_id: int = 0,
        sub_page: str = "BOTH",
        timeout_ms: float = 100.0,
    ) -> int:
        """
        Speculatively fetch upcoming blocks into host DRAM staging buffer.
        Extracts and stores ACTUAL NumPy tensor arrays. Returns count of blocks staged.
        """

    def prefetch_blocks(
        self,
        layer_idx: int,
        block_ids: List[int],
        page_type: str = "BOTH",
        timeout_ms: float = 100.0,
    ) -> int:
        """Alias for prefetch with P1/P2 argument order."""

    def read(
        self,
        layer_idx: int,
        block_id: int,
        page_type: str = "BOTH",
        head_id: int = 0,
        token_start: int = 0,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Unified read: checks DRAM staging first; falls back to backend on miss."""

    def record_hit(self, layer_idx: int, block_id: int, page_type: str = "BOTH") -> None:
        """Record an explicit cache hit event."""

    def record_miss(self, layer_idx: int, block_id: int, page_type: str = "BOTH") -> None:
        """Record an explicit cache miss event."""

    def predict_and_prefetch(
        self,
        current_layer: int,
        current_topk_blocks: List[int],
        lookahead_layers: int = 1,
        timeout_ms: float = 100.0,
    ) -> List[int]:
        """Layer-pipelined speculative prefetch: prefetches layer L + lookahead blocks."""

    # --- Telemetry & Management ---

    def get_telemetry(self) -> Dict[str, Any]:
        """Return empirical runtime counters, hit rates, and memory footprint."""

    def reset_stats(self) -> None:
        """Reset all metric counters to zero."""
```

### P1 I/O Counter Property Passthroughs
The adapter exposes properties expected by P1's `AISSDKVManager`:
- `bytes_read`: Total bytes transferred from backend storage.
- `bytes_written`: Total bytes written to backend storage.
- `blocks_read`: Total block fetch operations.
- `blocks_written`: Total block write operations.
- `requests`: Combined request count.

---

## 4. Exact Interface Expected from P2 Backend

The adapter wraps P2's `RealInferenceStorageBackend` (located in `person2_ssd/inference_backend.py`). It expects the following interface:

| Method / Attribute | Signature / Type | Description |
| :--- | :--- | :--- |
| `write_block` | `(layer_idx, block_id, k_block, v_block, head_id=0, token_start=0) -> bool` | Persists Key & Value tensors to FTL. |
| `read_key_page` | `(layer_idx, block_id, head_id=0, token_start=0) -> np.ndarray` | Returns float32 Key array (`[16, 2, 64]` or `[1, 16, 64]`). |
| `read_value_page`| `(layer_idx, block_id, head_id=0, token_start=0) -> np.ndarray` | Returns float32 Value array (`[16, 2, 64]` or `[1, 16, 64]`). |
| `read_block` | `(layer_idx, block_id, head_id=0, token_start=0) -> (np.ndarray, np.ndarray)`| Returns tuple of (k_block, v_block). |
| `contains_block` | `(layer_idx, block_id) -> bool` | Returns True if block exists in storage. |
| `evict_block` | `(layer_idx, block_id) -> bool` | Removes block from storage. |
| `get_telemetry` | `() -> Dict[str, Any]` | Returns backend FTL / I/O stats. |
| `reset_stats` | `() -> None` | Resets backend statistics. |

*Fallback Support*: The adapter also gracefully accepts legacy byte-oriented backends exposing `store_kv(block_id, layer_id, ...)`, `load_key_page(block_id, layer_id)`, `load_value_page(block_id, layer_id)`, and `load_kv(block_id, layer_id)`.

---

## 5. Verification & Test Suite

The integration test suite `tests/test_inference_prefetch_adapter.py` thoroughly validates all aspects of the wrapper:

```bash
pytest tests/test_inference_prefetch_adapter.py -v
```

### Test Results (11/11 Passed in 0.11s)
1. `test_adapter_initialization`: Confirms default parameters, clean state, and 0 bytes baseline.
2. `test_p1_compatible_write_and_read_key_page`: Verifies write of `[16, 2, 64]` float32 array, demand read, and bitwise data equality.
3. `test_p1_compatible_read_value_page`: Verifies write of Value page, demand read, and bitwise data equality.
4. `test_p1_compatible_read_block`: Verifies unified reading of both Key and Value pages.
5. `test_unified_read_interface`: Tests unified `read()` with `page_type="KEY"`, `"VALUE"`, and `"BOTH"`.
6. `test_live_data_prefetch_roundtrip_staging`: **Confirms prefetch stages ACTUAL tensor data (not metadata-only)** and returns correct arrays upon demand hit.
7. `test_prefetch_hit_vs_demand_miss_telemetry`: Verifies telemetry counting for hits, misses, and hit rate calculation.
8. `test_useless_prefetch_and_eviction`: Verifies LRU capacity enforcement, eviction of unconsumed blocks, and accounting of `wasted_bytes` and `useless_prefetches`.
9. `test_staging_memory_tracking`: Verifies dynamic host DRAM staging buffer memory accounting.
10. `test_no_artificial_latency`: Confirms 50 read operations complete in under 0.1 seconds (< 2 ms/op), ensuring zero artificial sleep.
11. `test_p2_real_inference_storage_backend_wrapper`: **Direct integration test with P2's `RealInferenceStorageBackend`**, verifying tensor roundtrip, prefetch hit, and telemetry synchronization.

Full P3 Test Suite: **38/38 passing (100%)**.

---

## 6. Staging Telemetry Definitions & Formulas

The adapter produces real runtime telemetry through `get_telemetry()`:

| Metric | Formula / Source | Definition |
| :--- | :--- | :--- |
| `demand_requests` | Total `read`, `read_key_page`, `read_value_page`, `read_block` calls | Total demand reads requested by LLM inference. |
| `demand_hits` | Reads satisfied from DRAM staging | Demand requests satisfied without storage access. |
| `demand_misses` | Reads requiring synchronous storage I/O | Demand requests that were not prefetched in time. |
| `prefetch_requests`| Speculative block requests issued | Total blocks requested for speculative staging. |
| `useful_prefetches`| Staged blocks consumed by demand read | Prefetched blocks that successfully served inference. |
| `useless_prefetches`| Staged blocks evicted before consumption | Prefetched blocks that wasted transfer bandwidth. |
| `useful_bytes` | `sum(useful_prefetches * block_size)` | Bytes transferred that were actually used. |
| `wasted_bytes` | `sum(useless_prefetches * block_size)` | Bytes transferred that were discarded unused. |
| `staging_memory_bytes`| `staged_blocks * block_size` | Current DRAM footprint occupied by staged tensors. |
| `prefetch_hit_rate`| `demand_hits / (demand_hits + demand_misses)` | Ratio of demand reads served from prefetch cache. |
| `prefetch_accuracy`| `useful_prefetches / prefetch_requests` | Ratio of prefetch requests that proved useful. |

---

## 7. Prediction Policy Assessment: Live Inference vs. Trace-Only Limitations

### Direct Question
*Can the prediction policy genuinely be used during live inference, or is it limited to trace-only / analytical evaluation?*

### Direct Answer
**The prediction policy can genuinely be used during live inference, but its mechanism is fundamentally different from offline trace replay.**

#### What is Limited to Trace-Only / Analytical Evaluation:
- **Offline Future-Token Oracle Lookahead**: In the Phase 3 trace replay evaluation, the benchmark replayed an already-generated 512-token sequence. In that setting, the prefetcher could inspect future trace events (tokens $T+1, T+2$) to know with 100% certainty which KV blocks would be selected by Top-$k$.
- **The Phase 3 96.0% Hit Rate**: That figure was measured on offline trace replay with pre-recorded token trajectories. **It MUST NOT be cited or claimed as the expected hit rate of live autoregressive inference.** In live generation, future tokens do not yet exist, making future-token oracle lookahead mathematically impossible.

#### What is Genuinely Usable in Live Inference:
- **Inter-Layer Speculative Pipelining ($L \to L+1$)**:
  During autoregressive decode of token $T$, when Layer $L$ evaluates Top-$k$, Layer $L+1$ block IDs can be predicted and prefetched asynchronously from storage into host DRAM while GPU/CPU compute executes Layer $L$'s attention matrix multiplications and FFN forward pass.
- **Cross-Layer Cluster Correlation**:
  Empirical transformer attention studies show that tokens attended to in Layer $L$ have high correlation ($r \approx 0.72 - 0.88$) with tokens attended to in Layer $L+1$. Prefetching Layer $L+1$ blocks corresponding to Layer $L$'s Top-$k$ winners yields genuine prefetch hits without knowing future tokens.
- **Temporal Locality across Decode Steps**:
  KV blocks representing active prompt context and recent tokens exhibit high reuse probability across adjacent decode steps ($T \to T+1$). Retaining these in DRAM staging provides predictable hit rates.

---

## 8. Remaining Blockers Before P1 Can Use This

**Zero blockers exist on the P3 side.**

To enable live inference with prefetching, Person 1 (P1) only needs to instantiate `RealInferencePrefetchAdapter` in `aissd_inference.py`:

```python
# In P1's aissd_inference.py:
from person2_ssd.inference_backend import RealInferenceStorageBackend
from person3_system.prefetch.inference_adapter import RealInferencePrefetchAdapter

# 1. Instantiate P2 backend
p2_backend = RealInferenceStorageBackend()

# 2. Wrap with P3 prefetch adapter
prefetch_adapter = RealInferencePrefetchAdapter(
    storage_backend=p2_backend,
    staging_capacity_blocks=256,
    enable_prefetch=True
)

# 3. Pass prefetch adapter to AISSDKVManager
kv_manager = AISSDKVManager(storage_backend=prefetch_adapter)
```

The prefetch adapter is fully operational, verified, and ready for end-to-end integration.
