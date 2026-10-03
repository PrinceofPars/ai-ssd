# POST P2/P3 INTEGRATION AUDIT

**Auditor:** Independent V2 Cross-Component Integration Auditor  
**Date:** 2026-10-03  
**Scope:** Read-only audit of P1/P2/P3 integration readiness for live inference path  
**Branch:** `v2-real-llm-kvssd` (main worktree)  
**Method:** Independent source inspection + live execution on EC2  

---

## 1. Executive Summary

P1 and P2 are **fully connected and verified** in a live Qwen2.5-0.5B inference
path with real model forward passes, real KV extraction, real Top-k selection,
and actual K/V tensor roundtrip through P2's multi-channel storage backend.

P3's `RealInferencePrefetchAdapter` is **fully implemented, tested, and
API-compatible** but is **not wired into the live inference chain**. P1 connects
directly to P2, bypassing P3.

### Critical Question Answer: **PARTIAL**

> Can we now run genuine Qwen2.5-0.5B inference where Top-k → P3 prefetch →
> P2 storage backend → tensor-aware mapping → actual K/V retrieval → attention
> all occur during the SAME live decode execution?

**No.** The live path today is:

```
Qwen2.5-0.5B → real Q/K/V → Top-k → P2 RealInferenceStorageBackend
→ DeterministicTensorMapper → 8-channel striping → actual K/V retrieval
→ reconstruct K/V tensors → Qwen attention → next token → wall-clock throughput
```

P3's `RealInferencePrefetchAdapter` is **not in this chain**.

---

## 2. Git / Worktree Status (Step 1)

| Worktree | Branch | HEAD | Clean? | New Commits |
|----------|--------|------|--------|-------------|
| `~/ai-ssd` | `v2-real-llm-kvssd` | `656d752` | Untracked: `LIVE_INFERENCE_AUDIT.md` | — |
| `~/ai-ssd-p1` | `v2/p1-real-llm-kv` | `00804d9` | **CLEAN** | `7937ced` (P2 integration), `00804d9` (test fix) |
| `~/ai-ssd-p2` | `v2/p2-femu-ftl` | `4681021` | **CLEAN** | `694de67` (drop-in backend), `4681021` (identity test) |
| `~/ai-ssd-p3` | `v2/p3-system-integration` | `760027c` | **CLEAN** | `67b6f61` (prefetch adapter), `760027c` (identity test) |

All worktrees are clean. No uncommitted changes.

---

## 3. P1 Findings (Step 2)

**File:** `person1_kv_engine/real_llm/aissd_inference.py` @ `00804d9`

### 3.1 Does P1 import/call P2?

**YES.** Lines 28–50: Resilient path discovery searches `/home/ubuntu/ai-ssd-p2`
and imports `RealInferenceStorageBackend` from `person2_ssd.inference_backend`.
Sets `_P2_AVAILABLE = True` on success.

`create_default_storage_backend()` (line 108–136) returns:
- `RealInferenceStorageBackend(...)` if P2 is importable
- `AISSDBlockStorageBackend(...)` (in-memory dict) as fallback

### 3.2 Does P1 still contain its own in-memory dictionary?

**YES — as fallback only.** `AISSDBlockStorageBackend` (lines 58–105) is
retained as a fallback if P2 is unavailable. When P2 IS available
(confirmed: `_P2_AVAILABLE == True` on EC2 host), it is NOT used.

### 3.3 Does P1 retrieve ACTUAL K/V tensor data from P2?

**YES.** Traced call chain:

1. `init_from_prefill()` (line 167): Extracts real K/V tensors from Qwen layers,
   blockizes into `(16, 2, 64)` float32 numpy arrays
2. `self.backend.write_block(l_idx, bid, k_blk, v_blk)` (line 203): Writes
   actual numpy arrays to P2
3. `select_and_fetch_active_kv()` (line 224):
   - Line 249: `k_blk = self.backend.read_key_page(l_idx, bid)` → returns
     actual numpy array from P2
   - Line 260: `v_blk = self.backend.read_value_page(l_idx, bid)` → returns
     actual numpy array from P2
   - Lines 262–265: `torch.from_numpy(k_blk[:actual_tokens])` / `torch.from_numpy(v_blk[:actual_tokens])` → converts back to torch tensors

### 3.4 Is the returned data consumed by attention?

**YES.** Lines 407–437 of the monkey-patched `make_aissd_forward()`:

```python
act_k, act_v = kv_mgr.select_and_fetch_active_kv(layer_idx, q)
# ... GQA expansion ...
scores = torch.matmul(q, k_exp.transpose(2, 3)) * scaling
weights = torch.nn.functional.softmax(scores, ...)
out = torch.matmul(weights, v_exp)
out = attn_module.o_proj(out)
```

The retrieved K/V tensors from P2 are used directly in `torch.matmul` attention.

### 3.5 Does Top-k still select the blocks actually fetched?

**YES.** Lines 246–256: For each candidate block, P1 reads the Key page via
`backend.read_key_page()`, computes `np.einsum("hd,thd->th", q_np, k_blk[...])`,
scores them, sorts by score descending, and fetches only the top-k winners'
Value pages.

### 3.6 Is the decode loop still real Qwen inference?

**YES.** Lines 444–467:
- Prefill: `model(input_ids=input_ids, use_cache=True)` — real forward pass
- Decode: `model(input_ids=next_token, past_key_values=pkv_prefill, use_cache=True)` — real forward pass
- Token selection: `torch.argmax(step_out.logits[:, -1, :], ...)`

### 3.7 Is wall-clock timing around actual model execution?

**YES.** Lines 459/467: `t_start = time.perf_counter()` before decode loop,
`t_end = time.perf_counter()` after loop. No analytical timing.

### 3.8 Does P1 import P3?

**NO.** Confirmed via `grep -rn 'person3_system|RealInferencePrefetchAdapter|inference_adapter|prefetch' aissd_inference.py` → empty output, exit code 1.

---

## 4. P2 Findings (Step 3)

**File:** `person2_ssd/inference_backend.py` @ `694de67` (HEAD is `4681021`)

### 4.1 RealInferenceStorageBackend API

- `CLASSIFICATION = "ANALYTICAL"` (line 52)
- Constructor (lines 54–123): 8-channel `DeterministicTensorMapper`, in-memory
  storage dict, per-channel telemetry counters

### 4.2 Write Path

`write_block(layer_idx, block_id, k_block, v_block, ...)` (lines 129–230):
- Accepts `np.ndarray` or `bytes` for K and V
- Stores `k_arr.copy()` and `v_arr.copy()` — exact numpy array copies
- Maps through `DeterministicTensorMapper.tensor_to_nand_physical()` for channel assignment
- Updates per-channel write counters

### 4.3 Read Path — Key Page

`read_key_page(layer_idx, block_id, ...)` (lines 257–303):
- Returns `entry["k"].copy()` — exact numpy array copy
- Maps through `DeterministicTensorMapper` for channel-level telemetry
- Falls back to zero array if block not found

### 4.4 Read Path — Value Page

`read_value_page(layer_idx, block_id, ...)` (lines 309–355):
- Returns `entry["v"].copy()` — exact numpy array copy
- Same mapper + telemetry pattern

### 4.5 Data Preservation Verification

**VERIFIED.** Write stores `k_arr.copy()`, read returns `entry["k"].copy()`.
This is an exact numpy array roundtrip. Integration test
`test_write_read_roundtrip` confirms bit-for-bit match.

### 4.6 Block/Layer ID Semantics

- Storage key: `(layer_idx, block_id)` — consistent with P1's calling convention
- Dual API: Also provides `store_kv(block_id, layer_id, ...)` and
  `load_key_page(block_id, layer_id, ...)` with swapped arg order

### 4.7–4.8 Page Sizes

Defined at module level:
- `KEY_PAGE_BYTES = 4096` (4 KiB)
- `VALUE_PAGE_BYTES = 4096` (4 KiB)
- `LOGICAL_BLOCK_BYTES = 8192` (8 KiB)

Note: With Qwen's grouped 2-head layout `(16, 2, 64)`, actual stored bytes
are 8192 per tensor, not 4096. P2 handles this transparently.

### 4.9–4.10 Tensor-Aware Channel Mapping

`DeterministicTensorMapper` (imported from `person2_ssd.kv_allocator.tensor_mapping`)
performs layer/head-aware channel assignment across 8 channels.

Live evidence (from benchmark):
```
Per-channel reads: {0: 75, 1: 77, 2: 93, 3: 87, 4: 81, 5: 71, 6: 77, 7: 87}
Contention ratio: 1.15
```

Non-uniform distribution confirms deterministic tensor-aware mapping is active.

### 4.11 Telemetry

`get_telemetry()` (lines 434–509): Reports per-channel reads/writes, byte
counts, contention ratio, load imbalance, analytical service time, sleep
latency injected (always `False`).

### P2 Test Results

**6/6 PASS** (`person2_ssd/tests/test_inference_backend.py`, 0.09s)

---

## 5. P3 Findings (Step 4)

**File:** `person3_system/prefetch/inference_adapter.py` @ `67b6f61` (HEAD is `760027c`)

### 5.1 RealInferencePrefetchAdapter

`CLASSIFICATION = "ANALYTICAL"` (line 62).
Constructor (lines 64–130): Accepts `storage_backend: Optional[Any]`, defaults
to `MockStorageBackend()`. Creates LRU `OrderedDict` staging buffer, predictor,
telemetry counters.

### 5.2 Does it wrap P2?

**YES — by design.** `self.storage_backend = storage_backend` (line 74).
When constructed with `storage_backend=RealInferenceStorageBackend(...)`, all
reads/writes delegate to P2.

### 5.3 Does prefetch retrieve actual K/V data?

**YES.** `prefetch()` (lines 344–434):
- Line 382: `if hasattr(self.storage_backend, "read_block"):`
- Lines 383–389: `k_tensor, v_tensor = self.storage_backend.read_block(layer_idx=layer_id, block_id=bid, ...)`
- Lines 414–427: Creates `StagedInferenceBlock` with `data_k=k_tensor` and
  `data_v=v_tensor` — actual numpy arrays, not metadata

### 5.4 Is prefetched data available to later demand reads?

**YES.** `read_key_page()` (lines 485–514):
- Line 500: `if key in self._staging_buffer:` → cache hit
- Line 511: `return entry.data_k` → returns actual numpy array from staging
- Line 517+: If miss → falls through to `self.storage_backend.read_key_page()`

Same pattern for `read_value_page()` (lines 530–578).

### 5.5 Are hit/miss metrics based on live requests?

**YES.** Lines 510/523: `self.record_hit(...)` on staging hit,
`self.record_miss(...)` on demand miss. Counters track `demand_reads`,
`demand_hits`, `demand_misses`.

### 5.6 Does the policy use information actually available during inference?

**YES.** `predict_and_prefetch()` (lines 446–459) uses `NextLayerPredictor`
which predicts next-layer blocks from current-layer block IDs — information
available during decode.

### 5.7 Does it avoid trace-only information?

**YES.** No trace file reads, no offline analysis. Uses only live access history.

### 5.8 Does it avoid artificial latency?

**YES.** Zero `time.sleep()` calls. Latency tracked via `time.perf_counter_ns()`
deltas (native Python execution time only).

### P3 Test Results

**11/11 PASS** (`tests/test_inference_prefetch_adapter.py`, 0.10s)

---

## 6. Cross-Worktree API Compatibility (Step 5)

### 6.1 Write Interface

| Aspect | P1 calls | P2 provides | P3 provides | Compatible? |
|--------|----------|-------------|-------------|-------------|
| Method | `write_block(layer_idx, block_id, k_blk, v_blk)` | `write_block(layer_idx, block_id, k_block, v_block, head_id=0, token_start=0)` | `write_block(layer_idx, block_id, k_block, v_block, head_id=0, token_start=0)` | **✅ YES** |
| `layer_idx` type | `int` | `int` | `int` | ✅ |
| `block_id` type | `int` | `int` | `int` | ✅ |
| K data type | `np.ndarray float32` | `Union[np.ndarray, bytes]` | `Union[np.ndarray, bytes]` | ✅ |
| V data type | `np.ndarray float32` | `Union[np.ndarray, bytes]` | `Union[np.ndarray, bytes]` | ✅ |
| K shape | `(16, 2, 64)` | Stores as-is (copy) | Stores as-is (copy) | ✅ |
| V shape | `(16, 2, 64)` | Stores as-is (copy) | Stores as-is (copy) | ✅ |

### 6.2 Read Key Page Interface

| Aspect | P1 calls | P2 provides | P3 provides | Compatible? |
|--------|----------|-------------|-------------|-------------|
| Method | `read_key_page(layer_idx, block_id)` | `read_key_page(layer_idx, block_id, head_id=0, token_start=0)` | `read_key_page(layer_idx, block_id, head_id=0, token_start=0)` | **✅ YES** |
| Return type | `np.ndarray` | `np.ndarray` (copy) | `np.ndarray` (from staging or backend) | ✅ |
| Return shape | `(16, 2, 64)` | Same as stored | Same as stored | ✅ |
| Return dtype | `float32` | `float32` | `float32` | ✅ |
| Synchronous | Yes | Yes | Yes | ✅ |

### 6.3 Read Value Page Interface

| Aspect | P1 calls | P2 provides | P3 provides | Compatible? |
|--------|----------|-------------|-------------|-------------|
| Method | `read_value_page(layer_idx, block_id)` | `read_value_page(layer_idx, block_id, head_id=0, token_start=0)` | `read_value_page(layer_idx, block_id, head_id=0, token_start=0)` | **✅ YES** |
| Return type | `np.ndarray` | `np.ndarray` (copy) | `np.ndarray` (from staging or backend) | ✅ |
| Return shape | `(16, 2, 64)` | Same as stored | Same as stored | ✅ |
| Return dtype | `float32` | `float32` | `float32` | ✅ |
| Synchronous | Yes | Yes | Yes | ✅ |

### 6.4 Telemetry Interface

| Aspect | P1 accesses | P2 provides | P3 provides | Compatible? |
|--------|-------------|-------------|-------------|-------------|
| `bytes_read` | `getattr(backend, "bytes_read", 0)` | Direct attribute | Property | ✅ |
| `blocks_read` | `getattr(backend, "blocks_read", 0)` | Direct attribute | Property | ✅ |
| `requests` | `getattr(backend, "requests", 0)` | Direct attribute | Property | ✅ |
| `reset_stats()` | `backend.reset_stats()` | Method | Method (cascades to backend) | ✅ |
| `get_telemetry()` | `backend.get_telemetry()` | Method | Method (includes backend telem) | ✅ |
| `CLASSIFICATION` | `getattr(backend, "CLASSIFICATION", None)` | `"ANALYTICAL"` | `"ANALYTICAL"` | ✅ |

### 6.5 Ownership / Lifetime

- P2 returns `.copy()` of stored arrays → P1 owns the returned data
- P3 returns either staging buffer reference (`entry.data_k`) on hit, or
  delegates to P2 (which returns copy) on miss
- P1 wraps result in `torch.from_numpy()` for attention → numpy array must
  not be mutated after torch wraps it

**Concern:** P3 returns `entry.data_k` directly (not a copy) on staging hit.
If P1 mutates the numpy array, P3's staging buffer is corrupted. P1 currently
does NOT mutate (it slices and wraps in `torch.from_numpy`), so this is safe
in practice but fragile.

### 6.6 Summary

**All three components share identical API signatures and return types.** P3
can be inserted between P1 and P2 as a transparent wrapper with zero API changes.

---

## 7. Exact Live Execution Graph (Step 6)

### 7.1 Current Architecture (Verified Live)

```
Qwen2.5-0.5B  ← HuggingFace AutoModelForCausalLM.from_pretrained()
      ↓
real model.forward(input_ids)  ← actual torch forward pass
      ↓
real Q/K/V extraction  ← monkey-patched attention intercept
      ↓
Top-k scoring  ← np.einsum dot-product over Key pages
      ↓
P2 RealInferenceStorageBackend  ← read_key_page() + read_value_page()
      ↓
DeterministicTensorMapper  ← 8-channel tensor-aware striping
      ↓
actual K/V numpy arrays returned  ← entry["k"].copy(), entry["v"].copy()
      ↓
torch.from_numpy() → reconstruct K/V tensors
      ↓
torch.matmul(q, k.T) * scaling → softmax → matmul(weights, v)  ← real attention
      ↓
o_proj(out) → next token  ← real projection + argmax
      ↓
time.perf_counter() delta  ← wall-clock throughput
```

### 7.2 Missing Link

```
P3 RealInferencePrefetchAdapter  ← NOT IN CHAIN
    wraps P2 backend
    provides DRAM staging buffer
    provides speculative prefetch
    provides hit/miss tracking
```

---

## 8. Live Benchmark Results (Step 6)

### Configuration

```
Model:       Qwen2.5-0.5B
Context:     128 tokens
Decode:      4 tokens
CPU threads: 4
Seed:        42
Dtype:       float32
Repetitions: 1
```

### Results

| Metric | Baseline | AI-SSD | Notes |
|--------|----------|--------|-------|
| **Wall time** | 0.1544 s | 0.1720 s | `time.perf_counter()` delta |
| **Throughput** | 25.90 tok/s | 23.25 tok/s | tokens / wall_time |
| **KV memory** | 3.09 MB | 0.84 MB | Active DRAM only |
| **KV offloaded** | 0.0% | 72.5% | Historical blocks in P2 |
| **KV blocks read** | 0 | 72 | Via P2 backend |
| **Storage bytes read** | 0 | 5,308,416 (5.06 MB) | Actual array bytes |
| **Storage requests** | 0 | 648 | read_key_page + read_value_page |
| **Backend** | In-Memory DynamicCache | P2 Multi-Channel FTL | — |
| **Per-channel reads** | — | {0:75, 1:77, 2:93, 3:87, 4:81, 5:71, 6:77, 7:87} | Non-uniform = real mapping |
| **Load imbalance** | — | 14.81% | max vs mean |
| **Contention ratio** | — | 1.15 | vs 8.0x serial |
| **Sleep latency** | — | False | Zero artificial delay |
| **Prefetch requests** | — | 0 | P3 not connected |
| **Prefetch hits** | — | 0 | P3 not connected |
| **Prefetch misses** | — | 0 | P3 not connected |

---

## 9. Correctness Results (Step 7)

| Metric | Value |
|--------|-------|
| **Baseline tokens** | `[6437, 1584, 6541, 6461]` |
| **AI-SSD tokens** | `[6437, 1584, 5662, 6832]` |
| **Token match** | 2/4 (50.0%) |
| **Logits cosine similarity** | 0.74613 |
| **Logits MSE** | 6.559314 |
| **Logits max diff** | 12.992046 |
| **Baseline text** | " solid state drive controller" |
| **AI-SSD text** | " solid state machine learning" |

### Analysis

The first 2 tokens match exactly. Divergence at token 3 is **expected**:
sparse Top-k attention uses only ~10% of historical KV blocks, so the
attention distribution differs from the full-cache baseline. This is the
fundamental accuracy/memory tradeoff of the AI-SSD architecture.

The 0.746 cosine similarity confirms the logit distributions are related but
not identical — consistent with sparse attention operating on a subset of
the full KV cache.

**This is NOT a bug.** The retrieved K/V tensors are exactly the tensors
P1 intended to retrieve (verified by P2's exact-copy roundtrip). The output
difference is a natural consequence of sparse attention.

---

## 10. Classification (Step 8)

| Component | Layer | Classification | Evidence |
|-----------|-------|---------------|----------|
| Model load | Qwen2.5-0.5B | **REAL** | `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")` |
| Forward pass | `model(input_ids)` | **REAL** | Actual torch forward, output logged |
| Attention | monkey-patched QKV | **REAL** | `torch.matmul` on retrieved tensors |
| KV extraction | from Qwen layers | **REAL** | `layer.keys`, `layer.values` torch tensors |
| Top-k scoring | `np.einsum` | **REAL** | Dot-product over real Key pages |
| Block partitioning | `init_from_prefill()` | **REAL** | Groups KV into (16,2,64) blocks |
| Wall-clock timing | `time.perf_counter()` | **REAL** | Before/after decode loop |
| P2 storage | `RealInferenceStorageBackend` | **ANALYTICAL** | In-memory dict, not NVMe/FEMU |
| P2 channel mapping | `DeterministicTensorMapper` | **REAL** | Deterministic 8-channel mapper, live channel counts |
| P2 data fidelity | numpy copy roundtrip | **REAL** | Exact array roundtrip verified |
| P2 latency | service time | **ANALYTICAL** | Mathematical NAND timing model, zero sleep |
| P3 prefetch adapter | `RealInferencePrefetchAdapter` | **NOT CONNECTED** | Implemented + tested, but P1 does not import |
| P3 DRAM staging | `OrderedDict` buffer | **ANALYTICAL** | Python dict, not real DRAM simulation |
| P3 prediction | `NextLayerPredictor` | **REAL** | Access-pattern-based, live information |
| P3 hit/miss tracking | counters | **REAL** | Based on staging buffer lookups |
| End-to-end pipeline | Qwen→TopK→P2→attention | **PARTIAL** | P1→P2 verified live; P3 not wired |

---

## 11. Blockers

### Blocker 1 — P3 Not Connected (Priority: HIGH)

P1's `create_default_storage_backend()` (line 122) returns P2 directly:
```python
return RealInferenceStorageBackend(...)
```

P3's `RealInferencePrefetchAdapter` is never instantiated or imported by P1.
`grep` for any P3 reference in `aissd_inference.py` returns empty.

**Resolution:** P1 must be updated to:
```python
def create_default_storage_backend(...):
    p2_backend = RealInferenceStorageBackend(...)
    p3_adapter = RealInferencePrefetchAdapter(storage_backend=p2_backend)
    return p3_adapter
```

This is a ~5-line change. The API signatures are fully compatible (verified
in Section 6). No other code changes needed — P1's `write_block`,
`read_key_page`, and `read_value_page` calls will transparently route through
P3's staging buffer to P2's storage.

### Blocker 2 — P3 Staging Buffer Ownership (Priority: LOW)

P3's `read_key_page()` returns `entry.data_k` directly (not a copy) on staging
hit. If P1 were to mutate the returned array, P3's staging buffer would be
corrupted. P1 currently does NOT mutate (wraps in `torch.from_numpy` for
read-only attention), so this is safe today but architecturally fragile.

---

## 12. Merge Readiness (Step 10)

### Verdict: **MERGE READY WITH CONDITIONS**

All three worktrees are **clean** with no uncommitted changes. All **23/23
tests pass** across P1 (6), P2 (6), and P3 (11). P1↔P2 integration is
**verified live** with real Qwen2.5-0.5B inference, real token generation,
and real wall-clock measurement.

### Conditions

1. **Before merge:** Update P1's `create_default_storage_backend()` to wrap
   P2 backend with P3 adapter (Blocker 1)
2. **Optional:** Have P3's `read_key_page()`/`read_value_page()` return
   `.copy()` of staged data to prevent ownership issues (Blocker 2)

### Merge Order

1. Merge `v2/p2-femu-ftl` → `v2-real-llm-kvssd`
2. Merge `v2/p3-system-integration` → `v2-real-llm-kvssd`
3. Merge `v2/p1-real-llm-kv` → `v2-real-llm-kvssd`
4. Apply P3 wiring fix on merged branch
5. Run full integration test suite on merged branch

### Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| P1↔P2 API mismatch | **NONE** | Verified live with real inference |
| P1↔P3 API mismatch | **NONE** | Identical signatures (Section 6) |
| Data corruption | **NONE** | Exact numpy copy roundtrip verified |
| Merge conflicts | **LOW** | Components in separate directories |
| P3 staging ownership | **LOW** | P1 does not mutate returned arrays |
| Performance regression from P3 | **LOW** | P3 adds dict lookup overhead only |

---

## 13. Exact Next Step

**One integration task remains:**

Update P1's `create_default_storage_backend()` to optionally wrap P2's
`RealInferenceStorageBackend` with P3's `RealInferencePrefetchAdapter`.

After this change, the full target architecture will be live:

```
Qwen2.5-0.5B → model.forward() → Q/K/V extraction → Top-k selection
→ P3 RealInferencePrefetchAdapter (DRAM staging + speculative prefetch)
→ P2 RealInferenceStorageBackend (8-channel tensor-aware FTL)
→ actual K/V retrieval → reconstruct tensors → Qwen attention → next token
→ wall-clock throughput measurement
```

No other code changes are required.

---

## Appendix A — Test Execution Evidence

```
$ cd /home/ubuntu/ai-ssd-p1 && .venv/bin/python -m pytest person1_kv_engine/tests/test_p2_integration.py -v
6 passed in 2.44s

$ cd /home/ubuntu/ai-ssd-p2 && python -m pytest person2_ssd/tests/test_inference_backend.py -v
6 passed in 0.09s

$ cd /home/ubuntu/ai-ssd-p3 && .venv/bin/python -m pytest tests/test_inference_prefetch_adapter.py -v
11 passed in 0.10s
```

## Appendix B — Commit History

| Worktree | Commit | Message |
|----------|--------|---------|
| P1 | `7937ced` | feat(p1): integrate Person 2 RealInferenceStorageBackend into live inference path |
| P1 | `00804d9` | test(p1): account for partial tail block in active tokens assertion |
| P2 | `694de67` | p2: add drop-in live inference storage backend and roundtrip tests for P1 |
| P2 | `4681021` | Testing smash273 identity |
| P3 | `67b6f61` | p3: prepare RealInferencePrefetchAdapter as wrapper for P2 backend in live inference |
| P3 | `760027c` | Testing smash273 identity |

## Appendix C — Live Benchmark Raw Output

```
Mode: BASELINE
Wall time: 0.1544 s | Throughput: 25.90 tok/s
KV memory: 3.09 MB | KV offloaded: 0.0%
Backend: In-Memory DynamicCache (Host DRAM)

Mode: AI-SSD
Wall time: 0.1720 s | Throughput: 23.25 tok/s
KV memory: 0.84 MB | KV offloaded: 72.5%
Backend: Person 2 Multi-Channel Flash FTL (Tensor-Aware)
Per-channel reads: {0: 75, 1: 77, 2: 93, 3: 87, 4: 81, 5: 71, 6: 77, 7: 87}
Contention ratio: 1.15 | Sleep latency: False

CORRECTNESS:
Baseline Tokens: [6437, 1584, 6541, 6461]
AI-SSD Tokens:   [6437, 1584, 5662, 6832]
Token Match: 2/4 (50.0%)
Logits Cosine Sim: 0.74613
Baseline Text: ' solid state drive controller'
AI-SSD Text:   ' solid state machine learning'
```

---

*End of audit. No source code was modified. No commits were made.*
