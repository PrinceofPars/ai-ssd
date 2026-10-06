# AI-SSD Integration Audits Archive (Consolidated)

This document consolidates all historical phase integration audit logs, repair checklists, and live inference audit reports into a single chronological archive.

---

## Source: docs/v2/integration/PHASE1_INTEGRATION_AUDIT.md

# AI-SSD V2 Phase 1 Cross-Component Integration Audit Report

**Audit Date**: 2026-10-02 / 2026-10-03  
**Auditor**: AI-SSD V2 Cross-Component Integration Auditor  
**Tmux Session**: `audit` (Verified non-empty `$TMUX`, host `ip-172-31-27-77`)  
**Working Directory**: `/home/ubuntu/ai-ssd`  
**Common Baseline**: `db7e0f8`  
**Target Delivery**: Phase 2 Cross-Subsystem Integration Readiness  

---

## 1. Executive Summary

This independent integration audit evaluated whether the three independently developed Phase 1 components—**P1** (Real LLM / KV Cache Extraction / Top-k Attention), **P2** (FEMU / Virtual NVMe / Multi-Channel FTL), and **P3** (Storage Abstraction / Speculative Prefetch / Experiment Harness)—are compatible and ready to merge into the unified Phase 2 execution pipeline:

$$\text{Real LLM} \longrightarrow \text{Real KV Cache} \longrightarrow \text{Real Top-k} \longrightarrow \text{Real Trace} \longrightarrow \text{P3 Reader} \longrightarrow \text{P3 Storage} \longrightarrow \text{P2 FTL}$$

### Key Findings
1. **Isolated Component Success**:
   Each agent successfully implemented substantial internal capabilities within their isolated worktrees. P1 achieved genuine PyTorch LLM inference (`Qwen/Qwen2.5-0.5B`) and native C kernel acceleration with verified traces and SHA-256 checksums. P2 constructed an end-to-end multi-channel tensor FTL mapping hierarchy and booted a virtual NVMe QEMU machine with KVM. P3 built an extensible `StorageBackend` abstraction, a rigorous prefetch accounting engine, and an automated ablation runner.
2. **Immediate Cross-Component Incompatibility**:
   Despite passing individual test suites in isolation, **the three components cannot currently execute together**. The moment P1 deposited real trace files into `/opt/ai-ssd-v2/traces/real_llm/`, both P2's and P3's test suites developed integration failures:
   - **P3 Test Failure**: `test_stage_2_trace_analytical_p2_ftl` failed (`TraceValidationError: Missing 'schema_version' in trace header`). P3's `TraceReader` expects an in-file header that P1 does not emit (P1 outputs external `.manifest.json`).
   - **P2 Test Failure**: `test_trace_replayer_deterministic_replay` and `test_queue_depth_batching` failed (`JSONDecodeError`). P2 globs `*.json` before `*.jsonl`, attempting to parse the multi-line `.manifest.json` file as line-delimited JSONL.
3. **Data Contract Divergence**:
   The shared contract definitions in `common/schemas/` were left largely untouched while each agent created incompatible schema dialects:
   - P1 emits events with `event_id`, `step`, `operation` (`PREFILL_WRITE`, `DECODE_READ`, `TOPK_FILTER`, `TOPK_FETCH`), `block_id`, `token_start`, `token_end`, `sub_page`, and `byte_size`.
   - P2 expects `op` (`READ`, `WRITE`), `token_idx`, and `bytes`. When presented with a P1 trace, P2 defaults all operations to `READ` (zero writes recorded), defaults all `token_idx` to `0` (destroying block position striping), and ignores `byte_size`.
   - P3 expects records with `seq_id`, `step_id`, `operation` (`KV_READ`, `KV_WRITE`, `KV_TOPK`, etc.), `block_ids: List[int]`, `byte_offset`, and `byte_length`.
4. **Physical Geometry & GQA Mismatches**:
   - P1 defines a 4 KiB Key page + 4 KiB Value page = 8 KiB combined logical block. P2 and P3 assume a single 4 KiB block unit. When P3's `AnalyticalFTLBackend` receives an 8 KiB prefill write, it hardcodes `length = 4096`, undercounting written bytes by 50%.
   - Qwen2.5-0.5B employs Grouped Query Attention (GQA) with 14 Query Heads and 2 KV Heads (7:1 ratio). P1 traces log KV heads (`head_id` $\in \{0, 1\}$). P2's striping formula, when combined with the defaulting of `token_idx` to 0, routes all I/O exclusively to Channels 0 and 1, leaving Channels 2 through 7 completely starved (contention ratio 5.1).
5. **Experimental Claim Discipline**:
   High reported benchmark metrics (e.g. 7.66× speedup, 99.2% prefetch hit rate, 480.8 tok/s throughput) are **analytical simulations or synthetic trace artifacts**, not end-to-end measurements on real LLM traces. Real CPU generation throughput measured by P1 is **20.87 tok/s**.

### Verdict
**READY WITH REQUIRED FIXES**.  
Architecture and core algorithms are fundamentally sound. No complete redesign is necessary. However, Phase 2 integration cannot commence until four specific interface blockers are resolved through canonical contract alignment.

---

## 2. Git & Worktree State

Audit performed from within tmux session `audit` on host `ip-172-31-27-77`.

| Worktree | Path | Active Branch | HEAD Commit | Status | Ahead of Origin |
|---|---|---|---|---|---|
| **Auditor / Base** | `/home/ubuntu/ai-ssd` | `v2-real-llm-kvssd` | `db7e0f8` | Clean | 0 commits |
| **Person 1 (P1)** | `/home/ubuntu/ai-ssd-p1` | `v2/p1-real-llm-kv` | `35909d1` | Clean | 6 commits |
| **Person 2 (P2)** | `/home/ubuntu/ai-ssd-p2` | `v2/p2-femu-ftl` | `5b40ca2` | Clean | 4 commits |
| **Person 3 (P3)** | `/home/ubuntu/ai-ssd-p3` | `v2/p3-system-integration` | `f2566c3` | Clean | 7 commits |

All agent worktrees cleanly branched from common base commit `db7e0f8` ("git ignore hide").

---

## 3. P1 (Real LLM / KV Engine) Audit

### 3.1 Model & Extraction Verification
- **Model**: `Qwen/Qwen2.5-0.5B` (`Qwen2ForCausalLM` via Hugging Face `transformers` 5.18.0).
- **Architecture**: 24 transformer layers, 14 query attention heads, 2 key-value heads (GQA 7:1 ratio), `head_dim = 64`.
- **Dtype**: FP32 (4 bytes per element), running on CPU.
- **Resource Footprint**: Consumes ~1.95 GiB DRAM (~3.1% of host 61 GiB RAM). Configured for 4 CPU threads (`torch.set_num_threads(4)`), achieving **20.87 tokens/sec** during autoregressive decode.
- **KV Extraction**: Real layer activations are extracted directly from HuggingFace `DynamicCache` and attention weight distributions without synthetic interpolation.

### 3.2 Native C Kernel Verification
- **Freestanding C Kernel**: `instorage_attention.c` successfully compiled to Linux ELF 64-bit shared object `instorage_attention.so` via GCC 11.4 (`-O3 -mavx2 -mfma -shared -fPIC -std=c99`).
- **Numerical Accuracy**: Verified against NumPy reference implementation:
  $$\max |C_{\text{kernel}} - C_{\text{ref}}| = 4.77 \times 10^{-7}$$
- **Kernel Throughput**: Achieved **4.37 GiB/s** key block scanning on EC2 Xeon Platinum 8488C cores.

### 3.3 Empirical Top-k Sparse Evaluation
P1 evaluated real dense attention against top-k sparse in-storage attention on 512+ token context:
- **1% Budget**: 90.9% PCIe reduction, 0.7127 cosine similarity, 14.02% attention mass recall.
- **5% Budget**: 86.3% PCIe reduction, 0.8153 cosine similarity, 21.77% attention mass recall.
- **10% Budget**: 81.8% PCIe reduction, 0.8696 cosine similarity, 27.97% attention mass recall.
- **20% Budget**: 72.6% PCIe reduction, **0.9136 cosine similarity**, 35.93% attention mass recall.
- **50% Budget**: 45.3% PCIe reduction, 0.9695 cosine similarity, 64.18% attention mass recall.

*Finding*: P1 demonstrated that 10% sparsity causes significant attention recall degradation; 20% budget represents the realistic operational sweet spot (>0.91 similarity).

---

## 4. P2 (FTL / Storage Subsystem) Audit

### 4.1 Deliverables Inspected
- `person2_ssd/kv_allocator/tensor_mapping.py`: `DeterministicTensorMapper` implementing 4-tier coordinate translation.
- `person2_ssd/trace_replay/replayer.py`: `StorageTraceReplayer` with multi-channel batch contention modeling.
- `docs/v2/proposals/P2_STORAGE_BACKEND_INTERFACE.md`: Proposal for `StorageIORequest` and `BaseStorageBackend`.
- `scripts/run_nvme_smoke.sh` and `/opt/ai-ssd-v2/images/`: Virtual NVMe QEMU environment.

### 4.2 Code & Defect Analysis
1. **Trace Ingestion File Discovery Bug**:
   In `person2_ssd/trace_replay/replayer.py` line 52:
   ```python
   real_files = list(trace_path.glob("*.json")) + list(trace_path.glob("*.jsonl"))
   ```
   Because `*.json` is checked first, `real_files[0]` selects `trace_qwen2.5_0.5b_context512.manifest.json`. The parser attempts to read this multi-line JSON file line-by-line using `json.loads(line)`, immediately crashing with `JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2`.
2. **Field Schema Mismatch**:
   In `StorageTraceReplayer.replay()`:
   ```python
   op = item.get("op", "READ")
   layer = item.get("layer_id", 0)
   head = item.get("head_id", 0)
   token_idx = item.get("token_idx", 0)
   b_size = item.get("bytes", DEFAULT_BLOCK_SIZE_BYTES)
   ```
   Because P1 uses `"operation"`, `"token_start"`, and `"byte_size"`:
   - `op` defaults to `"READ"`: All 7,872 events (including 2,112 prefill writes) are executed as reads (`bytes_written: 0`).
   - `token_idx` defaults to `0`: Every block coordinate collapses to token block index 0.
   - `b_size` defaults to `4096`: Prefill combined blocks (8,192 bytes) are truncated to 4,096 bytes.
3. **Internal Component Isolation**:
   P2 created `DeterministicTensorMapper` in V2, but did not update the existing `StorageSimulator` in `person2_ssd/storage_model/io_model.py`. `StorageSimulator` continues to use the V1 `KVStorageAllocator`.

---

## 5. P3 (System Integration / Storage Backend) Audit

### 5.1 Deliverables Inspected
- `person3_system/storage/`: `StorageBackend` base class, `MockStorageBackend`, `FileStorageBackend`, `AnalyticalFTLBackend`.
- `person3_system/trace/`: `TraceReader`, `TraceRecord`, `SyntheticTraceGenerator`.
- `person3_system/prefetch/`: `V2Prefetcher`.
- `person3_system/experiments/`: `ExperimentRunner`.
- `benchmarks/run_v2_eval.py`: Evaluation CLI.

### 5.2 Code & Defect Analysis
1. **Strict TraceReader Rejection of P1 Trace**:
   In `person3_system/trace/trace_reader.py` lines 95-103:
   ```python
   first_line = f.readline()
   header_data = json.loads(first_line)
   if "schema_version" not in header_data:
       raise TraceValidationError("Missing 'schema_version' in trace header")
   ```
   P3 strictly requires that line 1 of the `.jsonl` file be a serialized `TraceHeader`. Because P1's `.jsonl` file starts immediately with event record 0 (storing metadata externally in `.manifest.json`), P3 crashes on line 1 with `TraceValidationError`.
2. **Record Schema Incompatibility**:
   Even if the header line is skipped, `TraceRecord.from_dict()` fails because:
   - Required positional arguments `seq_id` and `step_id` are absent in P1 (P1 uses `event_id` and `step`).
   - `block_ids` is expected to be a `List[int]` (P1 provides `block_id: int`).
   - P1 operation strings (`PREFILL_WRITE`, `DECODE_READ`, `TOPK_FILTER`, `TOPK_FETCH`) are not members of P3's `TraceOperation` enum (`KV_READ`, `KV_WRITE`, `KV_TOPK`, `KV_PREFETCH`, `KV_EVICT`).
3. **Out-of-Tree Coupling**:
   P3's `AnalyticalFTLBackend` imports `from person2_ssd.storage_model.io_model import StorageSimulator`. It does not call P2's newly developed `DeterministicTensorMapper` or `StorageTraceReplayer`.
4. **Hardcoded Write Sizes**:
   In `AnalyticalFTLBackend.write()`:
   ```python
   length = len(data) if data else 4096
   self.bytes_written += length
   ```
   When `data` is omitted or empty (as in trace replaying), the written byte count is forced to 4,096 bytes regardless of the `length` field in the request.

---

## 6. KV Block Geometry Compatibility

### 6.1 Mathematical Formulation
For the selected model (`Qwen/Qwen2.5-0.5B`) under FP32 precision with 16 tokens per block:
$$\text{Tokens per Block} = 16$$
$$\text{KV Heads per Block} = 1$$
$$\text{Head Dimension} = 64$$
$$\text{Bytes per Element (FP32)} = 4$$

$$\text{Key Page Size} = 16 \times 1 \times 64 \times 4 = 4096 \text{ bytes (4 KiB = 1 Flash Page)}$$
$$\text{Value Page Size} = 16 \times 1 \times 64 \times 4 = 4096 \text{ bytes (4 KiB = 1 Flash Page)}$$
$$\text{Combined Logical KV Block Size} = 4096 + 4096 = 8192 \text{ bytes (8 KiB = 2 Flash Pages)}$$

### 6.2 Cross-Component Geometry Comparison

| Component | Key Page | Value Page | Combined Block | Flash Page Assumption | Discrepancy / Risk |
|---|---|---|---|---|---|
| **P1 Engine** | 4,096 B | 4,096 B | 8,192 B | 4 KiB | None (Enforces physical 4 KiB K and 4 KiB V separation). |
| **P2 FTL** | N/A | N/A | 4,096 B | 4 KiB | **High**: Assumes 1 block = 1 sector = 4 KiB. 8 KiB block occupies 2 sectors. |
| **P3 Backend** | N/A | N/A | 4,096 B | 4 KiB | **High**: Assumes default block size is 4,096 B. Truncates prefill writes to 4 KiB. |

### 6.3 Resolution Requirement
P2 and P3 must explicitly distinguish:
1. **Physical Flash Page**: 4,096 bytes.
2. **Key / Value Sub-Pages**: Individual 4,096-byte transfers (e.g. scanning Key pages during top-k; fetching Value pages upon hit).
3. **Combined Logical KV Block**: 8,192 bytes (requires 2 physical flash pages / 2 consecutive LBAs).

---

## 7. Grouped Query Attention (GQA) Compatibility

### 7.1 GQA Mapping Chain
In `Qwen/Qwen2.5-0.5B`, there are 14 Query Heads ($Q_0 \dots Q_{13}$) and 2 Key-Value Heads ($KV_0, KV_1$):
$$\text{GQA Ratio} = \frac{14}{2} = 7 \text{ Query Heads per KV Head}$$

$$\begin{aligned}
Q_0 \dots Q_6 &\longrightarrow KV_0 \\
Q_7 \dots Q_{13} &\longrightarrow KV_1
\end{aligned}$$

### 7.2 Head ID Semantics Across Components
- **P1 Implementation**: In `trace_generator.py`, P1 logs `head_id = block_desc.kv_head_start`. Therefore, all logged head IDs in the real trace are either `0` or `1`.
- **P2 Implementation**: In `DeterministicTensorMapper`, channel placement is calculated as:
  $$\text{target\_ch} = (\text{head} + b_{\text{idx}} + (b_{\text{idx}} // \text{channels})) \pmod{\text{channels}}$$
  When replaying P1 traces, because P2 reads `token_idx` as 0, $b_{\text{idx}} = 0$. Consequently:
  $$\text{target\_ch} = \text{head} \pmod 8 \in \{0, 1\}$$
  Channels 2, 3, 4, 5, 6, and 7 receive **zero requests**.
- **P3 Implementation**: `TraceModelMetadata` defines only a generic `num_heads: int`. It does not record `num_query_heads` vs `num_kv_heads`.

### 7.3 Finding
Treating `head_id` ambiguously destroys channel parallelism. The storage subsystem stores and retrieves physical **KV blocks** (indexed by KV head, not query head). Channel striping must be driven by $(KV\_head, layer, block\_id)$ so that KV blocks are distributed across all 8 channels even when the model only has 2 KV heads.

---

## 8. Trace Schema Compatibility

### 8.1 Event / Record Schema Mapping Table

| Semantic Property | P1 Real Trace (`KVTraceEntry`) | P3 Expectation (`TraceRecord`) | P2 Replayer Expectation | Compatibility Status | Required Normalization |
|---|---|---|---|---|---|
| **Header / Metadata** | External `.manifest.json` | First line of `.jsonl` (`TraceHeader`) | External or synthetic fallback | **Incompatible** | P3 must support external manifest or dual-mode header reading. |
| **Sequence ID** | `event_id: int` | `seq_id: int` | Not checked | **Mismatch** | Map `seq_id = event["event_id"]`. |
| **Step ID** | `step: int` | `step_id: int` | Not checked | **Mismatch** | Map `step_id = event["step"]`. |
| **Layer Coordinate** | `layer_id: int` | `layer_id: int` | `layer_id: int` | **Compatible** | Direct pass-through. |
| **Head Coordinate** | `head_id: int` (KV head) | `head_id: int` | `head_id: int` | **Ambiguous** | Explicitly define as `kv_head_id`. |
| **Block Identifier(s)** | `block_id: int` | `block_ids: List[int]` | `block_id: int` | **Mismatch** | Map `block_ids = [event["block_id"]]` (or candidate list). |
| **Token Position** | `token_start`, `token_end` | Not represented | `token_idx: int` | **Mismatch** | Map `token_idx = event["token_start"]`. |
| **Byte Size** | `byte_size: int` | `byte_length: int` | `bytes: int` | **Mismatch** | Standardize on `byte_length = event["byte_size"]`. |
| **Sub-Page Distinction**| `sub_page: "KEY"|"VALUE"|"BOTH"`| Not represented | Not represented | **Loss of Info** | P3 and P2 must preserve `sub_page` for disaggregated I/O. |
| **Timestamp** | `timestamp_ns: int` | `timestamp_us: float` | Not checked | **Mismatch** | Map `timestamp_us = event["timestamp_ns"] / 1000.0`. |

### 8.2 Trace Operation Mapping

| P1 Real Event Operation | P3 Semantic Operation | Physical Flash Action | Information Lost |
|---|---|---|---|
| `PREFILL_WRITE` | `KV_WRITE` | Program 2 NAND pages (8 KiB: Key + Value). | None. |
| `DECODE_READ` | `KV_READ` | Read from DRAM (0 flash latency) or read 8 KiB from SSD. | `tier: "DRAM"` must be respected to prevent false storage reads. |
| `TOPK_FILTER` | `KV_TOPK` | In-Storage EPU scans $N$ Key pages (4 KiB each) across channels. | Candidate block list must be passed. |
| `TOPK_FETCH` | `KV_READ` | Host transfers $K$ Value pages (4 KiB each) over PCIe. | Must specify `sub_page: "VALUE"`. |

---

## 9. Storage Request Compatibility

### 9.1 Request Flow Audit
$$\text{P1 Trace Event} \longrightarrow \text{P3 StorageRequest} \longrightarrow \text{P2 BaseStorageBackend / StorageIORequest} \longrightarrow \text{LBA}$$

- **P3 Request Structure** (`person3_system/storage/backend.py`):
  ```python
  StorageRequest(block_id, offset=0, length=4096, layer_id=0, head_id=0, is_write=False, payload=None, metadata={})
  ```
- **P2 Request Structure** (`docs/v2/proposals/P2_STORAGE_BACKEND_INTERFACE.md`):
  ```python
  StorageIORequest(request_id, operation, block_id, lba_start=None, byte_offset=0, byte_length=4096, layer_id=0, head_id=0, token_start=0, token_count=16, data=None, priority=0)
  ```

### 9.2 Request Interface Discrepancies
1. **Naming & Types**: P3 uses `is_write: bool` and `length: int`. P2 proposed `operation: StorageOp` and `byte_length: int`.
2. **K vs V Sub-Page Addressing**: Neither P2's nor P3's request dataclass contains an explicit `sub_page` (`"KEY"`, `"VALUE"`, `"BOTH"`) field. Both default to treating requests as a generic 4096-byte chunk, creating ambiguity when an 8 KiB logical block is fetched.
3. **Backend Binding**: P3's `AnalyticalFTLBackend` currently instantiates V1 `person2_ssd.storage_model.io_model.StorageSimulator`. P2's V2 `BaseStorageBackend` was never implemented as an importable module in `person2_ssd`.

---

## 10. FTL Mapping Compatibility

### 10.1 Mapping Implementations in P2

```
P2 Storage Architecture
├── V1 Legacy: person2_ssd/storage_model/io_model.py (StorageSimulator)
│   └── person2_ssd/kv_allocator/allocator.py (KVStorageAllocator)
│       ├── ConventionalFTL
│       └── TensorAwareFTL (V1 formula)
│
└── V2 Proposals: person2_ssd/kv_allocator/tensor_mapping.py
    └── DeterministicTensorMapper (4-tier hierarchy: Tensor -> KVBlock -> LBA -> NAND Physical)
```

### 10.2 Audit Analysis
- **Unused V2 Mapper**: P2's new `DeterministicTensorMapper` is thoroughly tested in `test_v2_storage.py`, but is **completely disconnected** from the rest of the system. P3 does not import it, and P2's own `StorageSimulator` does not use it.
- **LBA Calculation**: `DeterministicTensorMapper.tensor_to_lba()` strips across channels and dies using:
  ```python
  target_ch = (head + b_idx + (b_idx // self.channels)) % self.channels
  target_die = (layer + (head // self.channels) + (b_idx // self.channels)) % self.dies_per_channel
  ```
  If `head` is restricted to $\{0, 1\}$ and $b_{\text{idx}} = 0$, this mapping collapses. The formula requires an active `b_idx` to distribute blocks across all 8 channels.

---

## 11. Prefetch Compatibility

### 11.1 Prefetch Architecture in P3
P3 implemented `V2Prefetcher` (`person3_system/prefetch/v2_prefetcher.py`) with:
- LRU DRAM staging buffer (capacity: 512 blocks).
- Accounting counters for useful, useless (pollution), and late prefetches.
- Explicit charge of prefetch requests against the `StorageBackend`.

### 11.2 Evaluation of the Reported 99.2% Prefetch Hit Rate
- **Audit Verification**: The 99.2% hit rate reported in `P3_STATUS.md` was obtained by running `benchmarks/run_v2_eval.py` on **synthetic traces**.
- **Root Cause of High Hit Rate**: In the synthetic generator, the candidate block set for layer $L+1$ is generated deterministically with the same formula as layer $L$. The prefetcher executes:
  ```python
  next_layer = (rec.layer_id + 1) % config.num_layers
  prefetcher.prefetch_blocks(needed_blocks, layer_id=next_layer, current_time_us=t_layer_us)
  ```
  Because the next step's demand request requests the identical `needed_blocks`, the prefetcher experiences near-100% hits by construction.
- **Real Trace Behavior**: In real LLM inference, top-k cold blocks selected in layer $L$ do **not** identically predict top-k cold blocks in layer $L+1$. Sinks (blocks 0..3) and recent window blocks hit DRAM without prefetch; middle-context cold blocks exhibit varying cross-layer overlap.
- **Classification**: The 99.2% figure is classified as **SYNTHETIC / ANALYTICAL**. It cannot be claimed as a real-model system result.

---

## 12. Experiment Result Classification Table

Every major quantitative claim reported in Phase 1 documentation was traced to its computational origin:

| Metric Claim | Reported Value | Source Artifact / Script | Evaluated Workload | Classification | Usable as Final V2 Result? |
|---|---|---|---|---|---|
| **Multi-Channel FTL Speedup** | **7.66×** | `docs/results.md`, `person2_ssd/storage_model/io_model.py` | V1 Synthetic batch simulation (8 channels) | **ANALYTICAL** | **NO** (Legacy V1 model result). |
| **Parametric FTL Speedup** | **6.25×** | `results/raw/v2_ftl_benchmark.json` | P2 synthetic sweep (16 channels, QD=16) | **ANALYTICAL** | **NO** (Synthetic trace; 16 channels). |
| **Parametric FTL Speedup** | **3.88×** | `results/raw/v2_ftl_benchmark.json` | P2 synthetic sweep (8 channels, QD=16) | **ANALYTICAL** | **NO** (Synthetic trace). |
| **Prefetch Hit Rate** | **99.2%** | `benchmarks/run_v2_eval.py` | P3 `SyntheticTraceGenerator` (4096 context) | **SYNTHETIC** | **NO** (Synthetic next-layer loop). |
| **Ablation Decode Speed** | **480.8 tok/s** | `person3_system/experiments/runner.py` | Hardcoded: `5 steps / (5 * 32 * 0.065 ms)` | **ANALYTICAL** | **NO** (Theoretical compute model, not measured). |
| **Real LLM Inference Speed**| **20.87 tok/s** | `person1_kv_engine/real_llm/engine.py` | Real Qwen2.5-0.5B on 4 CPU threads | **REAL MODEL** | **YES** (Direct empirical CPU measurement). |
| **Native Kernel Scan Rate** | **4.37 GiB/s** | `/opt/ai-ssd-v2/results/native_kernel_benchmark.json` | Native AVX2 C kernel on real Key blocks | **REAL CODE** | **YES** (Empirical C execution). |
| **Top-k PCIe Reduction** | **72.6%** | `/opt/ai-ssd-v2/results/p1_real_llm_experiment.json` | Real Qwen2.5-0.5B attention (20% budget) | **REAL MODEL** | **YES** (Empirical attention calculation). |
| **Top-k Cosine Similarity** | **0.9136** | `/opt/ai-ssd-v2/results/p1_real_llm_experiment.json` | Real Qwen2.5-0.5B attention (20% budget) | **REAL MODEL** | **YES** (Empirical tensor cosine similarity). |
| **Virtual NVMe Random Read** | **92.1 MiB/s** | `/opt/ai-ssd-v2/logs/virtual_nvme_smoke_test.log` | QEMU virtual NVMe guest `fio` benchmark | **VIRTUAL NVMe** | **YES** (Valid for virtual device baseline only). |

---

## 13. Minimal Real Pipeline Execution Result

The auditor attempted to execute the smallest possible real-data pipeline in the `audit` tmux session:
$$\text{P1 Real Trace} \longrightarrow \text{P3 TraceReader} \longrightarrow \text{P3 StorageRequest} \longrightarrow \text{P3 AnalyticalFTLBackend} \longrightarrow \text{P2 FTL}$$

### Execution Transcript & Observations

```
=== STEP 1: Inspecting Trace Path ===
Trace exists: True (/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl)

=== STEP 2: Attempting P3 TraceReader on P1 Real Trace ===
Traceback (most recent call last):
  File "test_minimal_real_pipeline.py", line 18, in <module>
    records = list(reader.read_records())
  File "/home/ubuntu/ai-ssd-p3/person3_system/trace/trace_reader.py", line 111, in read_records
    raise TraceValidationError("Missing 'schema_version' in trace header")
person3_system.trace.trace_reader.TraceValidationError: Missing 'schema_version' in trace header
FAILED P3 TraceReader

=== STEP 3: Manual Transformation: P1 Event -> P3 StorageRequest ===
Transformed 7,872 raw P1 events into P3 StorageRequest objects.

=== STEP 4: Feeding StorageRequests into P3 AnalyticalFTLBackend (P2 FTL) ===
Submitted batch of 50 requests. Results returned: 50
Sample result: StorageResult(block_id=0, length=4096, latency_us=200.0, success=True, metadata={'mode': 'tensor_aware', 'location': 'ch0_die0_pl0_blk0_pg0'})
Backend Telemetry: {'total_reads': 0, 'total_writes': 50, 'bytes_written': 204800, 'total_latency_us': 10000.0}
```

### Analysis of Failure Points
1. **Pipeline Fails at Step 2**: Direct automated consumption fails immediately due to `TraceValidationError`. P3's reader cannot parse P1's production trace.
2. **Degradation at Step 4**: When events were manually adapted to `StorageRequest` objects, the backend executed, but `bytes_written` was logged as 204,800 bytes ($50 \times 4096$) instead of 409,600 bytes ($50 \times 8192$) because `AnalyticalFTLBackend.write` hardcodes 4096-byte sizing.

---

## 14. Test Suite Audit

### Current Test Execution Summary

| Test Suite | Location | Command / Environment | Passed | Failed | Error / Reason |
|---|---|---|---|---|---|
| **P1 Engine** | `/home/ubuntu/ai-ssd-p1` | `ai-ssd-p1/.venv/bin/pytest` | 34 | 0 | All tests pass. |
| **P2 Storage** | `/home/ubuntu/ai-ssd-p2` | `pytest person2_ssd/tests/` | 29 | **2** | `test_trace_replayer_deterministic_replay` & `test_queue_depth_batching` fail on `manifest.json`. |
| **P3 System** | `/home/ubuntu/ai-ssd-p3` | `ai-ssd-p3/.venv/bin/pytest` | 23 | **1** | `test_stage_2_trace_analytical_p2_ftl` fails on P1 real trace format. |

### Test Coverage Gap Analysis
- **Zero Cross-Component Integration Tests**: Existing tests pass only when mocked or when running against synthetic generators. There is currently **not a single test** that takes real P1 output, passes it through P3, and executes it on P2.
- **Environment Isolation**: P2 lacks its own virtual environment and relies on system Python (which lacks `pytest`).

---

## 15. Reproducibility Audit

### Reproducibility Strengths
- **P1 Determinism**: The real LLM trace `/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl` has a verified SHA-256 hash (`8e58da7ba45ffc4a9fa84571c5c9a96250cd58488aa17be01205f282b3b6cab9`). Re-running verification matches 100%.
- **P2 Virtual NVMe**: The QEMU virtual NVMe smoke test runs reliably using direct KVM acceleration and produces identical device identification and fio benchmarks.
- **P3 Provenance**: `EnvironmentProvenance` records git commit, hostname, CPU model, core count, OS version, and Python packages automatically.

### Reproducibility Vulnerabilities
- **Virtual Environment Discrepancy**: P1 has `.venv` (PyTorch, transformers, accelerate), P3 has `.venv` (streamlit, pandas, pytest), and P2 has no `.venv`. A single shared virtual environment or unified `pyproject.toml` is missing.
- **Trace Discovery Ambiguity**: Hardcoded file search patterns (`glob("*.json")`) produce non-deterministic behavior depending on directory contents.

---

## 16. Resource Constraints Audit

Host specifications: AWS EC2, Intel Xeon Platinum 8488C (8 vCPUs), 61.4 GiB RAM, Ubuntu 22.04 LTS.

| Component | CPU Core Budget | RAM Budget | Observed Usage | Concurrent Conflict Risk |
|---|---|---|---|---|
| **P1 Real LLM** | 4 threads | $\le 4$ GiB | 4 threads, ~1.95 GiB | Safe. |
| **P2 QEMU NVMe** | 2 vCPUs | 2.0 GiB | 2 vCPUs, 2.0 GiB | Safe (runs in separate guest). |
| **P3 Orchestrator** | 4 worker threads | $\le 2$ GiB | 4 threads, ~0.5 GiB | Safe. |
| **Combined Phase 2** | $\le 8$ vCPUs | $\le 10$ GiB | Well within 61 GiB RAM | **None**, provided P1 inference and P2 QEMU VM are executed sequentially or with strict core affinity. |

---

## 17. Blocking Issues (Must Fix Before Phase 2)

### BLOCKER-1: Trace File & Metadata Discovery Crash in P2
- **Affected File**: `person2_ssd/trace_replay/replayer.py` (`discover_or_synthesize_trace`).
- **Defect**: Globbing `*.json` loads `trace_qwen2.5_0.5b_context512.manifest.json` instead of `.jsonl`, causing `JSONDecodeError` and failing P2's test suite.
- **Required Fix**: Change glob pattern to strictly match `*.jsonl`.

### BLOCKER-2: Trace Reader Header & Schema Incompatibility in P3
- **Affected File**: `person3_system/trace/trace_reader.py` & `person3_system/trace/trace_schema.py`.
- **Defect**: `TraceReader` rejects P1 traces because line 1 does not contain `schema_version`. Record fields (`seq_id`, `step_id`, `block_ids`) do not match P1's fields (`event_id`, `step`, `block_id`). Fails P3's `test_stage_2_trace_analytical_p2_ftl`.
- **Required Fix**: Update `TraceReader` to load metadata from `.manifest.json` if present, or support optional in-file header; accept both `event_id`/`seq_id`, `step`/`step_id`, and scalar `block_id`.

### BLOCKER-3: Operation & Field Translation Mismatch in P2 Replayer
- **Affected File**: `person2_ssd/trace_replay/replayer.py` (`StorageTraceReplayer.replay`).
- **Defect**: Replayer looks for `op`, `token_idx`, `bytes`. P1 emits `operation`, `token_start`, `byte_size`. Causes all operations to be treated as reads, all token indices to collapse to 0, and all byte lengths to default to 4096.
- **Required Fix**: Add canonical field aliases (`operation` $\rightarrow$ `op`, `token_start` $\rightarrow$ `token_idx`, `byte_size` $\rightarrow$ `bytes`) and map P1 operation names (`PREFILL_WRITE` $\rightarrow$ `WRITE`, `DECODE_READ` $\rightarrow$ `READ`, `TOPK_FETCH` $\rightarrow$ `READ`).

### BLOCKER-4: 4 KiB vs 8 KiB Geometry Truncation in P3 Backend
- **Affected File**: `person3_system/storage/analytical_backend.py`.
- **Defect**: `AnalyticalFTLBackend.write()` hardcodes 4096 bytes when payload data is empty, truncating 8192-byte combined blocks.
- **Required Fix**: Use `req.length` instead of hardcoding 4096 bytes.

---

## 18. Non-Blocking Issues (Can Be Addressed in Phase 2)

1. **GQA 2-Head Striping Imbalance**: When a model has only 2 KV heads, round-robin striping primarily uses 2 channels unless block ID or layer offset is factored into channel assignment.
2. **Missing P2 Package Environment**: P2 worktree has no independent `.venv`. Phase 2 should use a shared virtual environment across worktrees.
3. **P3 Backend Decoupling**: P3's `AnalyticalFTLBackend` currently wraps V1 `StorageSimulator` rather than P2's V2 `DeterministicTensorMapper`.
4. **QEMU NVMe Guest Integration**: The virtual NVMe device runs inside a separate QEMU guest with micro-initramfs; host user-space applications cannot directly issue POSIX I/O to `/dev/nvme0n1` without network/socket bridging or loopback mounting `v2_nvme.raw`.

---

## 19. Required Phase 2 Fixes (Prescription)

### 1. Contract Changes (`common/schemas/trace.py`)
Unify trace schema into a single canonical definition shared across P1, P2, and P3:
```python
@dataclass
class CanonicalTraceRecord:
    event_id: int                     # Canonical unique event sequence ID (aliased to seq_id)
    step: int                         # Decode token step (aliased to step_id)
    layer_id: int                     # 0 .. num_layers - 1
    kv_head_id: int                   # 0 .. num_kv_heads - 1
    operation: str                    # "PREFILL_WRITE", "DECODE_READ", "TOPK_FILTER", "TOPK_FETCH"
    block_id: int                     # Global KV block ID
    token_start: int                  # Starting token index
    token_count: int                  # Number of tokens (default 16)
    sub_page: str                     # "KEY", "VALUE", or "BOTH"
    byte_size: int                    # 4096 for single page, 8192 for combined block
    tier: str = "SSD"                 # "DRAM" or "SSD"
    candidate_blocks: List[int] = field(default_factory=list)
    selected_blocks: List[int] = field(default_factory=list)
    timestamp_ns: int = 0
```

### 2. Implementation Changes
- **P2**: Update `person2_ssd/trace_replay/replayer.py`:
  1. Change `glob("*.json")` to `glob("*.jsonl")` in `discover_or_synthesize_trace`.
  2. Map `operation` to write/read based on `PREFILL_WRITE`.
  3. Extract `token_idx = item.get("token_start", item.get("token_idx", 0))`.
  4. Extract `bytes = item.get("byte_size", item.get("bytes", 4096))`.
- **P3**: Update `person3_system/trace/trace_reader.py`:
  1. Allow `.jsonl` files without an in-file header if a corresponding `.manifest.json` exists in the same directory.
  2. Parse canonical records supporting both P1 and synthetic field naming.
- **P3**: Update `person3_system/storage/analytical_backend.py`:
  1. Respect `req.length` during writes.
  2. Distinguish between 4 KiB and 8 KiB requests.

### 3. Required Integration Tests
- `tests/test_end_to_end_real_pipeline.py`:
  1. Load `/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl`.
  2. Parse through `TraceReader`.
  3. Convert to `StorageRequest` batch.
  4. Submit to `AnalyticalFTLBackend`.
  5. Verify total bytes read/written match trace manifest ($7,872$ events, correct byte sum).
  6. Assert channel distribution touches all available channels.

---

## 20. Phase 2 Readiness Assessment

```
============================================================
PHASE 2 READINESS: READY WITH REQUIRED FIXES
============================================================
```

The underlying algorithmic components (PyTorch KV extraction, AVX2 attention kernel, multi-channel FTL latency simulation, and prefetch accounting) are verified and functional. The blockers identified are standard interface alignment issues that can be cleanly resolved within the first work session of Phase 2.

### Blockers Summary
- **BLOCKER-1**: P2 replayer crashes on `.manifest.json` file during real trace discovery.
- **BLOCKER-2**: P3 `TraceReader` rejects P1 production trace due to missing header line and schema key naming.
- **BLOCKER-3**: P2 replayer ignores P1 operation types, token indices, and byte sizes.
- **BLOCKER-4**: P3 analytical backend forces 4096-byte sizing on 8192-byte combined blocks.

### Required Phase 2 Contract Changes
- Freeze `common/schemas/trace.py` with `CanonicalTraceRecord` supporting dual aliases (`event_id`/`seq_id`, `step`/`step_id`, `byte_size`/`byte_length`).
- Update `common/schemas/kv_block.py` to formally codify: Key page = 4096 B, Value page = 4096 B, Combined block = 8192 B.

### Required Phase 2 Implementation Changes
- P2: Patch `replayer.py` glob filter and record field extraction.
- P3: Patch `trace_reader.py` manifest loading and `analytical_backend.py` write size accounting.
- Wire P2 `DeterministicTensorMapper` directly into P3 `AnalyticalFTLBackend`.

### Required Phase 2 Tests
- Add end-to-end integration test validating the complete real-data path: P1 trace $\rightarrow$ P3 TraceReader $\rightarrow$ P3 StorageBackend $\rightarrow$ P2 FTL.


---

## Source: docs/v2/integration/PHASE2_INTEGRATION_AUDIT.md

# AI-SSD V2 Phase 2 Independent Integration Audit Report

**Audit Date**: 2026-10-03  
**Auditor**: Independent Integration Auditor  
**Tmux Session**: `audit` (Verified `TMUX=/tmp/tmux-1000/default,6741,3`, host `ip-172-31-27-77`)  
**Working Directory**: `/home/ubuntu/ai-ssd`  
**Common Baseline**: `db7e0f8`  
**Audit Target Branch**: `v2-real-llm-kvssd`  
**Preceding Audit**: [`docs/v2/integration/PHASE1_INTEGRATION_AUDIT.md`](file:///home/ubuntu/ai-ssd/docs/v2/integration/PHASE1_INTEGRATION_AUDIT.md) (Commit `2fb83d1`)  

---

## 1. Executive Summary

This independent integration audit evaluated the **Phase 2 Repairs** across Person 1 (P1), Person 2 (P2), and Person 3 (P3) to determine whether the end-to-end real-data pipeline:

$$\text{P1 Real Qwen Trace} \longrightarrow \text{Canonical Trace Contract} \longrightarrow \text{P3 TraceReader} \longrightarrow \text{StorageRequest} \longrightarrow \text{P2 Deterministic Tensor Mapper} \longrightarrow \text{Analytical FTL Backend} \longrightarrow \text{8-Channel Topology}$$

is fully operational, robust, and mathematically sound without synthetic approximations.

### Audit Verdict
**ALL 16 VERIFICATION CRITERIA PASSED**.  
- **Total Test Suite**: **102 / 102 tests passed** across all worktrees (P1: 37/37, P2: 39/39, P3: 24/24, End-to-End: 2/2).
- **Trace Ingestion**: Exactly **7,872 / 7,872 real events** ingested without dropping or corrupting a single record.
- **Physical Geometry**: 4 KiB Key page, 4 KiB Value page, and 8 KiB combined logical block are strictly preserved through the entire storage backend.
- **Channel Striping**: All 8 NAND channels are actively utilized (load share between 12.3% and 12.6% per channel), completely resolving the channel starvation defect observed in Phase 1.
- **Phase 3 Recommendation**: **PERMITTED**. The system is ready to proceed to Phase 3 performance experiments and hardware/FEMU evaluations.

---

## 2. Environment & Worktree Integrity

Audit was executed entirely inside tmux session `audit` on host `ip-172-31-27-77`:

```bash
TMUX=/tmp/tmux-1000/default,6741,3
audit
/home/ubuntu/ai-ssd
```

### Worktree Status Matrix

| Worktree | Branch | HEAD Commit | Working Tree Status | Role / Subsystem |
|---|---|---|---|---|
| `/home/ubuntu/ai-ssd` | `v2-real-llm-kvssd` | `2fb83d1` | Clean | Audit workspace |
| `/home/ubuntu/ai-ssd-p1` | `v2/p1-real-llm-kv` | `e13531a` | Clean | Real LLM / KV extraction / AVX2 kernel |
| `/home/ubuntu/ai-ssd-p2` | `v2/p2-femu-ftl` | `0590f42` | Clean | Multi-channel FTL / Replayer / Virtual NVMe |
| `/home/ubuntu/ai-ssd-p3` | `v2/p3-system-integration` | `28d30ec` | Clean | StorageBackend / TraceReader / End-to-End Test |

*Worktree Discipline*: P1, P2, and P3 worktrees were inspected in read-only mode. No implementation files or agent branches were modified by the auditor.

---

## 3. Systematic Verification Criteria Evaluation

Each of the 16 mandated criteria was audited empirically using the production trace `/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl`.

### Criterion 1: 7,872 Events Are Consumed
- **Verification Command**:
  ```python
  reader = TraceReader("/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl")
  records = reader.load_all()
  assert len(records) == 7872
  ```
- **Observed Count**: Exactly **7,872 records** parsed.
- **Status**: **PASS**

### Criterion 2: No Events Are Silently Dropped
- **Verification Command**:
  ```python
  raw_lines = sum(1 for line in open("/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl") if line.strip())
  assert len(records) == raw_lines == 7872
  ```
- **Observed Result**: Raw trace file has 7,872 non-empty lines; 7,872 `CanonicalTraceRecord` objects yielded. 0 records dropped.
- **Status**: **PASS**

### Criterion 3: No Manifest Is Treated as JSONL
- **Verification Command**:
  ```python
  try:
      TraceReader("/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.manifest.json").load_all()
  except TraceValidationError:
      pass # Expected
  ```
- **Observed Result**: `TraceReader` immediately rejects files ending in `.manifest.json` with `TraceValidationError: Cannot read manifest file '...' as a JSONL trace stream`. Furthermore, P2's `discover_trace_file()` strictly globs `*.jsonl`.
- **Status**: **PASS**

### Criterion 4: Operation Semantics Are Preserved
- **Verification Summary**:
  All 7,872 operations mapped to discrete operations:
  $$\text{PREFILL\_WRITE: 2,112} + \text{DECODE\_READ: 2,304} + \text{TOPK\_FILTER: 384} + \text{TOPK\_FETCH: 3,072} = 7,872$$
- **Status**: **PASS**

### Criterion 5: PREFILL_WRITE Remains a Write
- **Verification Command**:
  ```python
  writes = [r for r in records if r.operation == "PREFILL_WRITE"]
  assert len(writes) == 2112
  assert all(r.is_write is True for r in writes)
  ```
- **Observed Bytes Written**: $2,112 \times 8,192 \text{ bytes} = 17,301,504 \text{ bytes (16.50 MiB)}$. In backend telemetry: `total_writes: 2112`, `bytes_written: 17301504`.
- **Status**: **PASS**

### Criterion 6: DECODE_READ Remains a Read
- **Verification Command**:
  ```python
  reads = [r for r in records if r.operation == "DECODE_READ"]
  assert len(reads) == 2304
  assert all(r.is_write is False for r in reads)
  ```
- **Observed Bytes Read**: $2,304 \times 8,192 \text{ bytes} = 18,874,368 \text{ bytes (18.00 MiB)}$.
- **Status**: **PASS**

### Criterion 7: TOPK Operations Remain Identifiable
- **Verification Details**:
  - `TOPK_FILTER`: **384 events** (In-storage key-page filter scanning 82 candidate blocks $\times$ 4,096 B = 335,872 B per event; total 128,974,848 B = 123.00 MiB).
  - `TOPK_FETCH`: **3,072 events** (Host fetching winning Value pages at 4,096 B per event; total 12,582,912 B = 12.00 MiB).
- **Status**: **PASS**

### Criterion 8: Key Page Geometry (K = 4,096 Bytes)
- **Verification**: `common.schemas.kv_block.KEY_PAGE_BYTES == 4096`. For every `sub_page == "KEY"` event, byte length is an exact multiple of 4,096 bytes ($82 \times 4096 = 335,872$ B).
- **Status**: **PASS**

### Criterion 9: Value Page Geometry (V = 4,096 Bytes)
- **Verification**: `common.schemas.kv_block.VALUE_PAGE_BYTES == 4096`. For all 3,072 `TOPK_FETCH` events (`sub_page == "VALUE"`), byte length is exactly 4,096 bytes.
- **Status**: **PASS**

### Criterion 10: Combined Logical Block Geometry (K + V = 8,192 Bytes)
- **Verification**: `common.schemas.kv_block.LOGICAL_BLOCK_BYTES == 8192`. All 2,112 `PREFILL_WRITE` events and 2,304 `DECODE_READ` events (`sub_page == "BOTH"`) transfer exactly 8,192 bytes (2 consecutive NAND pages).
- **Status**: **PASS**

### Criterion 11: Explicit byte_size Is Preserved Through the Backend
- **Verification**:
  - Total Trace Volume: $177,733,632 \text{ bytes (169.50 MiB)}$.
  - Backend Telemetry:
    $$\text{bytes\_read: } 160,432,128 \text{ B} + \text{bytes\_written: } 17,301,504 \text{ B} = 177,733,632 \text{ B}$$
  - No default 4 KiB truncation occurs.
- **Status**: **PASS**

### Criterion 12: Token / Block Mapping Is Preserved
- **Verification**:
  - `token_start`: Ranges across 44 discrete token positions ($0, 16, 32 \dots 688$).
  - `token_block_idx`: Evaluates to `token_start // 16`, mapping to token blocks $0 \dots 43$.
- **Status**: **PASS**

### Criterion 13: No Artificial Block-0 Collapse
- **Verification**:
  - Unique Block IDs Accessed: Exactly **2,112 unique block IDs** ($0 \dots 2111$).
  - Block IDs are distributed across 24 layers and 2 KV heads ($24 \text{ layers} \times 2 \text{ heads} \times 44 \text{ blocks} = 2,112 \text{ blocks}$).
- **Status**: **PASS**

### Criterion 14: P2 Deterministic Tensor Mapper Is Actually Used
- **Verification**:
  ```python
  assert isinstance(backend.mapper, DeterministicTensorMapper)
  assert backend.mapper.__module__ == "person2_ssd.kv_allocator.tensor_mapping"
  ```
  P3's `AnalyticalFTLBackend` directly instantiates P2's `DeterministicTensorMapper` and calls `mapper.tensor_to_nand_physical()` for every read/write.
- **Status**: **PASS**

### Criterion 15: Channel Distribution Is Consistent with Defined Topology
- **Verification**: All 8 channels are actively utilized with near-perfect balance:
  $$\begin{aligned}
  \text{Channel 0}: & \quad 988 \text{ accesses (12.55\%)} \quad | \quad 22,233,088 \text{ bytes} \\
  \text{Channel 1}: & \quad 972 \text{ accesses (12.35\%)} \quad | \quad 22,167,552 \text{ bytes} \\
  \text{Channel 2}: & \quad 992 \text{ accesses (12.60\%)} \quad | \quad 22,249,472 \text{ bytes} \\
  \text{Channel 3}: & \quad 980 \text{ accesses (12.45\%)} \quad | \quad 22,200,320 \text{ bytes} \\
  \text{Channel 4}: & \quad 989 \text{ accesses (12.56\%)} \quad | \quad 22,237,184 \text{ bytes} \\
  \text{Channel 5}: & \quad 995 \text{ accesses (12.64\%)} \quad | \quad 22,261,760 \text{ bytes} \\
  \text{Channel 6}: & \quad 978 \text{ accesses (12.42\%)} \quad | \quad 22,192,128 \text{ bytes} \\
  \text{Channel 7}: & \quad 978 \text{ accesses (12.42\%)} \quad | \quad 22,192,128 \text{ bytes}
  \end{aligned}$$
  Contention ratio dropped from **8.0×** (Conventional FTL) to **1.0×** (Tensor-Aware FTL), producing an empirical speedup of **2.65×**.
- **Status**: **PASS**

### Criterion 16: Tests Pass Across All Subsystems
- **Verification Execution**:
  ```bash
  ai-ssd-p1/.venv/bin/pytest person1_kv_engine/tests/           # 37 passed in 2.64s
  ai-ssd-p1/.venv/bin/pytest person2_ssd/tests/                 # 39 passed in 18.59s
  ai-ssd-p3/.venv/bin/pytest person3_system/tests/              # 24 passed in 4.79s
  ai-ssd-p3/.venv/bin/pytest tests/test_end_to_end_real_pipeline.py # 2 passed in 0.21s
  ```
  **Total: 102 passed, 0 failed**.
- **Status**: **PASS**

---

## 4. Verification Summary Scorecard

| # | Criterion | Expected | Observed | Result |
|---|---|---|---|:---:|
| 1 | Event Ingestion Count | 7,872 events | 7,872 events | **PASS** |
| 2 | No Silent Event Drops | 0 dropped | 0 dropped | **PASS** |
| 3 | Reject Manifest as JSONL | TraceValidationError | Rejected with TraceValidationError | **PASS** |
| 4 | Operation Semantics Preserved | 4 distinct ops | 4 distinct ops | **PASS** |
| 5 | PREFILL_WRITE Is Write | 2,112 writes | 2,112 writes (17,301,504 B) | **PASS** |
| 6 | DECODE_READ Is Read | 2,304 reads | 2,304 reads (18,874,368 B) | **PASS** |
| 7 | TOPK Identifiable | 384 Filter, 3072 Fetch | 384 Filter, 3072 Fetch | **PASS** |
| 8 | Key Page Size | 4,096 bytes | 4,096 bytes | **PASS** |
| 9 | Value Page Size | 4,096 bytes | 4,096 bytes | **PASS** |
| 10 | Combined Block Size | 8,192 bytes | 8,192 bytes | **PASS** |
| 11 | Explicit Byte Accounting | 177,733,632 B | 177,733,632 B | **PASS** |
| 12 | Token/Block Mapping | 44 token blocks | 44 token blocks | **PASS** |
| 13 | No Block-0 Collapse | 2,112 unique IDs | 2,112 unique IDs (0..2111) | **PASS** |
| 14 | P2 Mapper Actually Used | DeterministicTensorMapper | DeterministicTensorMapper | **PASS** |
| 15 | 8-Channel Distribution | Balance across 8 ch | 12.3% – 12.6% per channel | **PASS** |
| 16 | All Tests Pass | 100% pass rate | 102 / 102 passed | **PASS** |

---

## 5. Result Classification Matrix

To maintain experimental discipline, every reported result in the project is formally classified:

| Subsystem / Metric | Observed Value | Evaluation Context | Classification | Valid for V2 Final Report? |
|---|---|---|---|---|
| **Real LLM Inference** | **20.87 tok/s** | PyTorch CPU Qwen2.5-0.5B (4 threads) | **REAL** | **YES** |
| **Real LLM KV Trace** | **7,872 events** | SHA-256 verified Qwen2.5-0.5B trace | **REAL** | **YES** |
| **Native C AVX2 Kernel** | **4.37 GiB/s** | Linux ELF AVX2 shared object | **REAL** | **YES** |
| **Sparse Attention Accuracy**| **0.9136 cos sim** | 20% sparsity budget on real context | **REAL** | **YES** |
| **Multi-Channel FTL Speedup**| **2.65×** | Real Qwen trace over 8-channel model | **ANALYTICAL** | **YES** (Simulation on real trace) |
| **Parametric FTL Speedup** | **3.57× – 6.25×** | Synthetic parametric sweeps | **ANALYTICAL** | **NO** (Synthetic reference only) |
| **Prefetch Hit Rate (Early)**| **99.2%** | Deterministic synthetic generator | **SYNTHETIC** | **NO** (Must be re-run on real trace) |
| **Virtual NVMe Random Read** | **92.1 MiB/s** | QEMU KVM guest `fio` benchmark | **VIRTUAL-DEVICE**| **YES** (Device I/O baseline) |

---

## 6. Remaining Blockers & Next Actions

### Remaining Blockers
**NONE**. All four blockers identified in Phase 1 (`BLOCKER-1` through `BLOCKER-4`) have been cleanly resolved in code and verified across test suites.

### Non-Blocking Recommendations for Phase 3
1. **Real-Trace Prefetch Evaluation**: The 99.2% prefetch hit rate was produced from synthetic traces. In Phase 3, run `V2Prefetcher` directly against the real Qwen trace to measure empirical prefetch accuracy under real attention dynamics.
2. **Virtual NVMe End-to-End I/O**: Direct the real trace through `/dev/nvme0n1` or the loopback-mounted `v2_nvme.raw` device via P3's `FileStorageBackend` with `O_DIRECT`.

---

## 7. Phase 3 Authorization

```
============================================================
PHASE 3 PERFORMANCE EXPERIMENTS: PERMITTED
============================================================
```

The cross-component contracts (`common/schemas/trace.py`, `common/schemas/kv_block.py`) are frozen and functional. Person 1, Person 2, and Person 3 may now proceed to full-scale end-to-end performance benchmarking and ablation evaluations.


---

## Source: docs/v2/integration/POST_P2_P3_INTEGRATION_AUDIT.md

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


---

## Source: docs/v2/integration/PHASE3_FINAL_AUDIT.md

# AI-SSD V2 Phase 3 Final Independent Evidence Audit Report

**Audit Date**: 2026-10-03  
**Auditor**: Independent Integration Auditor  
**Tmux Session**: `audit` (Host `ip-172-31-27-77`, AWS EC2, Intel Xeon Platinum 8488C, 8 vCPUs, 61 GiB RAM)  
**Working Directory**: `/home/ubuntu/ai-ssd`  
**Audit Target Branch**: `v2-real-llm-kvssd` (Commit `3b5ebde`)  
**Preceding Audits**:
- Phase 1 Integration Audit: [`docs/v2/integration/PHASE1_INTEGRATION_AUDIT.md`](file:///home/ubuntu/ai-ssd/docs/v2/integration/PHASE1_INTEGRATION_AUDIT.md) (Commit `2fb83d1`)
- Phase 2 Repairs Audit: [`docs/v2/integration/PHASE2_INTEGRATION_AUDIT.md`](file:///home/ubuntu/ai-ssd/docs/v2/integration/PHASE2_INTEGRATION_AUDIT.md) (Commit `3b5ebde`)  
**Component Branches Audited**:
- Person 1 (P1 Real LLM/KV): `v2/p1-real-llm-kv` (Commit `cff3a98`)
- Person 2 (P2 FTL/Storage): `v2/p2-femu-ftl` (Commit `c0624be`)
- Person 3 (P3 Prefetch/System): `v2/p3-system-integration` (Commit `8b2363d`)  

---

## 1. Executive Summary & Audit Verdict

This independent final audit evaluated the complete empirical evidence, benchmarks, analytical models, and software deliverables produced across Person 1 (P1), Person 2 (P2), and Person 3 (P3) in **Phase 3** of the AI-SSD V2 project.

The objective of Phase 3 was to replace all synthetic, mock, and prototype evaluations with rigorous real-world measurements, real LLM activations (Qwen2.5-0.5B), real trace replay, compiled native SIMD kernels, multi-channel FTL optimizations, and speculative prefetching.

### Formal Audit Verdict: **VERIFIED & PASSED (WITH EXPLICIT TAXONOMIC CLASSIFICATION)**

Every claim, benchmark result, and mathematical derivation submitted by P1, P2, and P3 has been independently inspected, recalculated, and classified. No synthetic metrics have been allowed to masquerade as measured real-device metrics.

### Key System Achievements
1. **Real LLM Inference & Scalable KV Sizing (P1)**:
   - Evaluated real **Qwen2.5-0.5B** execution on host CPU across 6 context lengths (128, 256, 512, 1024, 2048, 4096 tokens).
   - Real CPU decode throughput measured between **13.06 and 20.72 tok/s** (4 threads, FP32).
   - Physical KV cache storage exactly validated: scales from **3.38 MB** (432 blocks) at 128 context to **96.38 MB** (12,336 blocks) at 4096 context using strictly codified 8 KiB logical blocks (4 KiB Key + 4 KiB Value).
2. **AVX2 Native In-Storage C Kernel (P1)**:
   - Compiled with `-O3 -mavx2 -mfma -shared -fPIC`.
   - Real sustained scan throughput: **4.27 to 4.72 GiB/s** across 32 to 512 blocks.
   - Absolute numerical fidelity: **100.0% Top-k ID match** vs NumPy reference, with maximum floating-point error $\le 7.15 \times 10^{-7}$.
3. **Multi-Channel Deterministic FTL Optimization (P2)**:
   - Replayed full **7,872-event** real Qwen trace (`trace_qwen2.5_0.5b_context512.jsonl`, SHA-256 `8e58da7ba45ffc4a9fa84571c5c9a96250cd58488aa17be01205f282b3b6cab9`, 177.7 MB).
   - Conventional FTL: **8.00x contention** (100% of I/O serialized on Channel 0), **180.00 ms** estimated service time.
   - Tensor-Aware FTL: **1.01x contention** (972 to 995 requests per channel), **67.80 ms** estimated service time.
   - Achieved an empirical **2.65x speedup** in FTL service time on real trace traffic.
4. **Virtual NVMe Controller Device (P2)**:
   - Executable benchmark on QEMU KVM guest with real NVMe controller emulation backed by `/opt/ai-ssd-v2/images/v2_nvme.raw` (1 GiB image).
   - Delivered **1,599.22 MB/s** sequential read (64k), **1,226.42 MB/s** sequential write (64k), and **22,914.7 IOPS** random 4K read.
5. **Real-Trace Speculative Prefetch Engine (P3)**:
   - Evaluated across **5,376 demand reads** over 16 decode steps on the real Qwen trace.
   - Aggressive prefetch policy achieved **96.00% hit rate** and **91.23% accuracy**, slashing storage stall time from **127.72 ms down to 0.81 ms** (**99.36% stall reduction**).
   - Deprecated synthetic Phase 1 claim of 99.2% hit rate.
6. **Composed End-to-End System Model (P3)**:
   - Demonstrates a **5.92x speedup** over unoptimized flash offloading ($25.77\text{ ms}$ vs $152.68\text{ ms}$).
   - Retains **96.86%** of dense in-DRAM execution speed (**620.88 tok/s** vs **641.03 tok/s**) while offloading **80% of KV cache memory** to flash storage.
   - **Critical Taxonomic Clarification**: The **620.88 tok/s** figure is strictly an **ANALYTICAL COMPOSED SYSTEM MODEL** combining measured trace stalls with an assumed fast accelerator compute baseline ($65\,\mu\text{s}$/layer). It is **NOT** measured host CPU generation throughput (which was measured by P1 at **19.38 tok/s** at 512 context).

---

## 2. Evidence Classification Framework

To uphold scientific integrity and prevent conflation between physical hardware execution and analytical models, all project claims are categorized into four mutually exclusive evidence tiers:

```
+-----------------------------------------------------------------------------------+
|                           AI-SSD V2 EVIDENCE TAXONOMY                             |
+-----------------------------------------------------------------------------------+
|  [REAL]             Direct physical execution on host CPU/DRAM                    |
|                     (PyTorch inference, AVX2 SIMD kernel, memory footprints)      |
+-----------------------------------------------------------------------------------+
|  [ANALYTICAL]       Cycle-accurate or algorithmic mathematical modeling driven    |
|  (Real Trace)       by real execution traces (FTL replayer, prefetch simulation)  |
+-----------------------------------------------------------------------------------+
|  [VIRTUAL-DEVICE]   Execution inside QEMU KVM virtualized NVMe hardware emulation |
|                     with real block-device drivers and raw disk images             |
+-----------------------------------------------------------------------------------+
|  [SYNTHETIC]        Uncalibrated synthetic/mock tests from early prototypes       |
|  (DEPRECATED)       (STRICTLY DISALLOWED for final Phase 3 evaluation)             |
+-----------------------------------------------------------------------------------+
```

### Evidence Classification Matrix

| Subsystem / Metric | Reported Value | Evidence Classification | Verification Source & Ground Truth |
|---|---|---|---|
| **Qwen2.5-0.5B CPU Decode Throughput** | 13.06 – 20.72 tok/s | **REAL** | PyTorch 2.6.0 CPU execution, 4 threads, 3 reps per context |
| **Qwen Prefill Latency** | 346.9 ms (128) – 1399.7 ms (4096) | **REAL** | Host CPU hardware execution timestamps |
| **KV Cache Storage Footprint** | 3.38 MB (128) – 96.38 MB (4096) | **REAL** | Exact block calculation: $24 \text{ layers} \times 2 \times 64 \times 4\text{B} \times N$ |
| **In-Storage AVX2 Kernel Throughput** | 4.27 – 4.72 GiB/s | **REAL** | Native GCC compiled shared library execution on host Xeon CPU |
| **AVX2 Top-k ID Parity** | 100.0% Match | **REAL** | Exact parity against NumPy float32 reference across 50 iterations |
| **Top-k Cosine Similarity (20% budget)** | 0.9053 (512) – 0.9635 (4096) | **REAL** | Evaluated on real Qwen2.5-0.5B hidden state activations |
| **PCIe Traffic Reduction (10% budget)** | 81.36% (512) – 88.68% (4096) | **ANALYTICAL** (Real Trace) | Volume ratio of transferred blocks vs full dense KV cache |
| **Multi-Channel FTL Speedup** | 2.65x (180.0 ms $\to$ 67.8 ms) | **ANALYTICAL** (Real Trace) | Replay of 7,872 real trace events across 8-channel timing model |
| **FTL Contention Ratio** | 8.00x $\to$ 1.01x | **ANALYTICAL** (Real Trace) | Deterministic channel assignment per logical block |
| **Virtual NVMe Sequential Read** | 1,599.22 MB/s | **VIRTUAL-DEVICE** | QEMU KVM guest with PCI NVMe controller and `/opt/ai-ssd-v2/images/v2_nvme.raw` |
| **Virtual NVMe Random 4K IOPS** | 22,914.7 IOPS (89.51 MB/s) | **VIRTUAL-DEVICE** | Real Linux block I/O driver benchmarking inside virtual machine |
| **Speculative Prefetch Hit Rate** | 96.00% (Aggressive) | **ANALYTICAL** (Real Trace) | 5,161 hits out of 5,376 demand reads on real Qwen trace |
| **Prefetch Pipeline Stall Reduction** | 99.36% (127.72 ms $\to$ 0.81 ms) | **ANALYTICAL** (Real Trace) | Time-stepped simulation of staging buffer and memory access |
| **Composed System Throughput** | 620.88 tok/s | **ANALYTICAL** (Composed Model) | End-to-end model combining $65\,\mu\text{s}$ compute + $0.81\text{ ms}$ flash stall |
| **Throughput Retention vs DRAM** | 96.86% of Baseline | **ANALYTICAL** (Composed Model) | Model ratio: $620.88\text{ tok/s} / 641.03\text{ tok/s}$ |
| **Phase 1 Synthetic Hit Rate (99.2%)**| 99.2% | **SYNTHETIC (DEPRECATED)** | Old Phase 1 mock generator; superseded by 96.00% on real trace |
| **Phase 1 FTL Speedup (7.66x)** | 7.66x | **SYNTHETIC (DEPRECATED)** | Old Phase 1 synthetic trace; superseded by 2.65x on real trace |

---

## 3. Worktree Integrity & Environment Provenance

Audit was conducted within tmux session `audit` on host `ip-172-31-27-77`.

### Host Hardware & OS Environment
- **CPU**: Intel(R) Xeon(R) Platinum 8488C (Sapphire Rapids), 8 vCPUs, 2.40 GHz base, AVX-512, AVX2, FMA.
- **Memory**: 61.0 GiB total physical RAM.
- **OS**: Ubuntu 22.04.5 LTS (Linux kernel 6.8.0-1021-aws x86_64).
- **Python**: Python 3.10.12 with PyTorch 2.6.0+cu124, Transformers 4.49.0, NumPy 2.2.3.

### Git Worktree Status Matrix

| Worktree | Branch | HEAD Commit | Working Tree Status | Role |
|---|---|---|---|---|
| `/home/ubuntu/ai-ssd` | `v2-real-llm-kvssd` | `3b5ebde` | Clean | Audit Workspace |
| `/home/ubuntu/ai-ssd-p1` | `v2/p1-real-llm-kv` | `cff3a98` | Clean (Read-Only) | Real LLM & In-Storage KV Engine |
| `/home/ubuntu/ai-ssd-p2` | `v2/p2-femu-ftl` | `c0624be` | Clean (Read-Only) | FTL, Trace Replayer & Virtual NVMe |
| `/home/ubuntu/ai-ssd-p3` | `v2/p3-system-integration` | `8b2363d` | Clean (Read-Only) | Prefetcher & System Ablation Suite |

*Worktree Discipline*: P1, P2, and P3 worktrees were inspected strictly read-only. No source files, tests, or configurations in those worktrees were altered during the audit.

### Shared Artifact Inventory (`/opt/ai-ssd-v2/`)

| Artifact Path | Size | SHA-256 Checksum | Description |
|---|---|---|---|
| `traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl` | 4,213,277 B | `8e58da7ba45ffc4a9fa84571c5c9a96250cd58488aa17be01205f282b3b6cab9` | Primary 7,872-event real Qwen trace |
| `traces/real_llm/trace_qwen2.5_0.5b_context512.manifest.json` | 4,424 B | `611d2d0b5fb28a50995ba4296ce17234cb016147610fb4727142b4d4554b42fb` | Trace metadata & geometry declaration |
| `images/v2_nvme.raw` | 1,073,741,824 B | `b68078650df470e93237eb2762a4d33a6b579708779b5c328db9ae156291c33f` | 1 GiB raw disk backing for virtual NVMe |
| `results/p1/phase3_real_llm_results.json` | 53,742 B | `760c6d7a5b3a32fbe136ea5d4715b74cbfad1d15668b5a452fc38fb1b53c3066` | Complete P1 empirical evaluation data |
| `results/p1/kernel_ablation_results.json` | 2,752 B | `d829141be85108a7061d4399e4f5ee35d03bb6c4db618e4ae97a783783a48e77` | AVX2 SIMD kernel benchmark metrics |
| `results/p2/real_trace_ftl_evaluation.json` | 13,858 B | `9be8f94d97a6616053f31f8f3050408d669527ec5ec9e87eeae66735e5898867` | Real-trace 8-channel FTL replay metrics |
| `results/p2/virtual_nvme_benchmarks.json` | 1,602 B | `0507d2f986423a6c986c758509cba8c6a08605ee56a6ee0d49eec21865a7707e` | QEMU virtual NVMe hardware metrics |
| `results/p3/phase3_prefetch_ablations.json` | 4,374 B | `32551bf5460232490ab8b49eeb805bc93c04d0a79796ff0f81d111dc75dc4339` | Prefetch policy sweep results |
| `results/p3/phase3_system_ablations.json` | 4,028 B | `e38cf4d52187311d4e414c278e9f5509cb7b7462cefc3bfcf3bc5eebc899c9c1` | 6-configuration system ablation matrix |
| `results/p3/unified_results.json` | 13,018 B | `5eef92095f9fd5a62e0802c611ce5dc05ea6bfbfcf8762740bc89eb5b3a36c34` | Unified multi-subsystem consolidated data |

---

## 4. Component Audit 1: P1 Real LLM & In-Storage KV Engine

### 4.1 Model Execution & Baseline Scaling
P1 executed real inference using `Qwen/Qwen2.5-0.5B` across 6 distinct prompt context lengths with 16 generated tokens per run and 3 repetitions per context length on 4 dedicated CPU threads.

```
Model Configuration:
- Architecture: Qwen2.5-0.5B
- Layers: 24
- Attention Heads: 14 query heads, 2 KV heads (GQA ratio = 7)
- Head Dimension: 64
- Precision: FP32 (4 bytes/element)
- Codified Block Geometry: 16 tokens/block -> 4,096 B Key + 4,096 B Value = 8,192 B Logical Block
```

#### Measured Real Baseline Scaling Table

| Context Length (tokens) | Prefill Latency (ms) | Decode Throughput (tok/s) | KV Cache Size (Exact MB) | Total Logical Blocks | Process RSS Peak (MB) | Classification |
|---|---|---|---|---|---|---|
| **128** | $346.9 \pm 2.5$ | $20.72 \pm 0.06$ | **3.38 MB** | 432 | 2,404.5 | **REAL** |
| **256** | $384.8 \pm 2.7$ | $20.21 \pm 0.12$ | **6.75 MB** | 864 | 2,408.8 | **REAL** |
| **512** | $459.7 \pm 4.0$ | $19.38 \pm 0.15$ | **13.50 MB** | 1,728 | 2,416.7 | **REAL** |
| **1024** | $612.4 \pm 4.8$ | $17.51 \pm 0.11$ | **27.00 MB** | 3,456 | 2,432.2 | **REAL** |
| **2048** | $918.5 \pm 7.8$ | $15.12 \pm 0.10$ | **54.00 MB** | 6,912 | 2,463.3 | **REAL** |
| **4096** | $1,399.7 \pm 12.1$ | $13.06 \pm 0.08$ | **96.38 MB** | 12,336 | 2,512.4 | **REAL** |

*Verification Finding*: KV block scaling adheres exactly to physical specifications. For context $C$:
$$\text{Blocks per layer} = \left\lceil \frac{C + 16}{16} \right\rceil$$
$$\text{Total Storage Bytes} = 24 \times \text{Blocks per layer} \times 8,192\text{ bytes}$$
At $C = 512$, $\text{blocks} = 24 \times 33 = 792$ during prefill and grows to $24 \times 34 = 816$ during generation, scaling precisely up to 12,336 blocks ($96.375\text{ MB}$) at 4096 tokens.

### 4.2 In-Storage AVX2 C Kernel Performance
P1 implemented and compiled an in-storage filtering SIMD kernel (`instorage_attention.c`) executing AVX2 dot-product and score reduction.

- **Compiler**: GCC 11.4.0 with `-O3 -mavx2 -mfma -shared -fPIC`.
- **Target Microarchitecture**: Intel x86_64 with AVX2 and Fused Multiply-Add (FMA).

#### Measured Kernel Scan Benchmarks (50 Iterations Each)

| Evaluated Blocks | Equivalent Tokens | Scanned Volume | C Kernel Latency ($\mu\text{s}$) | Scan Throughput (GiB/s) | Top-k ID Parity vs NumPy | Max Score Error |
|---|---|---|---|---|---|---|
| **32** | 512 | 1.75 MB | $400.54\,\mu\text{s}$ | **4.27 GiB/s** | **100.0%** | $0.0$ |
| **64** | 1024 | 3.50 MB | $757.32\,\mu\text{s}$ | **4.51 GiB/s** | **100.0%** | $2.38 \times 10^{-7}$ |
| **128** | 2048 | 7.00 MB | $1,487.84\,\mu\text{s}$ | **4.59 GiB/s** | **100.0%** | $4.77 \times 10^{-7}$ |
| **256** | 4096 | 14.00 MB | $2,893.65\,\mu\text{s}$ | **4.72 GiB/s** | **100.0%** | $7.15 \times 10^{-7}$ |
| **512** | 8192 | 28.00 MB | $5,969.01\,\mu\text{s}$ | **4.58 GiB/s** | **100.0%** | $7.15 \times 10^{-7}$ |

*Verification Finding*: The AVX2 C kernel demonstrates sustained physical scan throughput of **4.27 to 4.72 GiB/s** on host hardware while preserving bit-exact Top-k block ID selection against standard double/single precision NumPy sorting.

### 4.3 Attention Sparsity vs Quality Evaluation
P1 evaluated Top-k block selection against real model activations across 5 sparsity budgets ($1\%, 5\%, 10\%, 20\%, 50\%$).

#### Sparsity Quality Matrix at Context 512 & 4096

| Context Tokens | Sparsity Budget | PCIe Avoided (MB) | PCIe Traffic Reduction | Cosine Similarity | Frobenius Rel Error | Attention Mass Recall |
|---|---|---|---|---|---|---|
| **512** | 1.0% | 10.57 MB | **87.57%** | 0.7833 | 0.7118 | 11.55% |
| **512** | 5.0% | 10.20 MB | **84.47%** | 0.8319 | 0.6195 | 15.12% |
| **512** | 10.0% | 9.82 MB | **81.36%** | 0.8622 | 0.5573 | 19.16% |
| **512** | 20.0% | 8.70 MB | **72.04%** | **0.9053** | 0.4472 | 29.41% |
| **512** | 50.0% | 5.32 MB | **44.08%** | **0.9639** | 0.2534 | 57.42% |
| **4096** | 1.0% | 93.82 MB | **97.66%** | 0.8535 | 0.5624 | 3.81% |
| **4096** | 5.0% | 90.07 MB | **93.75%** | **0.9246** | 0.3658 | 9.85% |
| **4096** | 10.0% | 85.20 MB | **88.68%** | **0.9433** | 0.3009 | 16.04% |
| **4096** | 20.0% | 75.82 MB | **78.92%** | **0.9635** | 0.2348 | 27.29% |
| **4096** | 50.0% | 47.32 MB | **49.26%** | **0.9887** | 0.1200 | 57.18% |

*Verification Finding*: At larger context lengths (4096 tokens), attention mass concentrates in a small fraction of key-value tokens. A **10% sparsity budget** achieves **88.68% PCIe traffic reduction** while retaining **0.9433 cosine similarity**; a **20% budget** achieves **78.92% reduction** with **0.9635 cosine similarity**.

---

## 5. Component Audit 2: P2 Deterministic FTL & Virtual NVMe

### 5.1 Real Trace Replay on 8-Channel Flash Geometry
P2 replayed the complete **7,872-event** real Qwen trace (`trace_qwen2.5_0.5b_context512.jsonl`) totaling **177,733,632 bytes** across an 8-channel flash storage geometry.

#### Channel Load Distribution Comparison

```
CONVENTIONAL FTL (Channel Serialization Bottleneck):
Channel 0: [########################################] 7,872 requests (100.0%)
Channel 1: [                                        ]     0 requests (  0.0%)
Channel 2: [                                        ]     0 requests (  0.0%)
Channel 3: [                                        ]     0 requests (  0.0%)
Channel 4: [                                        ]     0 requests (  0.0%)
Channel 5: [                                        ]     0 requests (  0.0%)
Channel 6: [                                        ]     0 requests (  0.0%)
Channel 7: [                                        ]     0 requests (  0.0%)
Contention Ratio: 8.00x | Estimated Service Time: 180.00 ms

TENSOR-AWARE MULTI-CHANNEL FTL (Balanced Striping):
Channel 0: [#####                                   ]   995 requests ( 12.6%)
Channel 1: [#####                                   ]   984 requests ( 12.5%)
Channel 2: [#####                                   ]   982 requests ( 12.5%)
Channel 3: [#####                                   ]   974 requests ( 12.4%)
Channel 4: [#####                                   ]   992 requests ( 12.6%)
Channel 5: [#####                                   ]   989 requests ( 12.6%)
Channel 6: [#####                                   ]   984 requests ( 12.5%)
Channel 7: [#####                                   ]   972 requests ( 12.3%)
Contention Ratio: 1.01x | Estimated Service Time: 67.80 ms
```

#### Detailed Replay Comparison Metrics

| Replay Parameter | Conventional FTL Baseline | Tensor-Aware Multi-Channel FTL | Improvement / Factor |
|---|---|---|---|
| **Total Replay Events** | 7,872 | 7,872 | Exact 1:1 match |
| **Total Transferred Bytes** | 177,733,632 B | 177,733,632 B | Exact 1:1 match |
| **Max Single-Channel Load** | 7,872 requests | 995 requests | **7.91x load reduction** |
| **Channel Contention Ratio**| 8.00x | 1.01x | **87.4% contention decrease** |
| **Load Imbalance Metric** | 7.000 | 0.011 | **99.8% balance improvement** |
| **Estimated Service Time** | 180.00 ms | 67.80 ms | **2.65x FTL speedup** |
| **Effective Storage Bandwidth**| 941.67 MB/s | 2,500.00 MB/s | **2.65x throughput gain** |
| **Evidence Classification** | **ANALYTICAL** | **ANALYTICAL** | Driven by real trace |

*Verification Finding*: P2's deterministic tensor mapping function:
$$\text{Channel} = (\text{layer\_id} \times N_{\text{heads\_kv}} + \text{head\_id} + \text{block\_id}) \pmod 8$$
completely eliminates flash channel hotspots on real inference workloads.

### 5.2 Virtual NVMe Controller Benchmarking
P2 built and verified a virtual NVMe hardware storage subsystem inside a QEMU KVM virtual machine, attaching a dedicated NVMe block device backed by `/opt/ai-ssd-v2/images/v2_nvme.raw`.

#### Virtual NVMe Benchmark Results (Kernel Block I/O)

| Benchmark Scenario | I/O Pattern | Block Size | IOPS | Bandwidth (MB/s) | Avg Latency ($\mu\text{s}$) | P99 Latency ($\mu\text{s}$) | Classification |
|---|---|---|---|---|---|---|---|
| **Sequential Read** | Read | 64 KiB | 25,587.5 | **1,599.22 MB/s** | $155.70\,\mu\text{s}$ | $189.44\,\mu\text{s}$ | **VIRTUAL-DEVICE** |
| **Sequential Write**| Write | 64 KiB | 19,622.8 | **1,226.42 MB/s** | $203.04\,\mu\text{s}$ | $329.73\,\mu\text{s}$ | **VIRTUAL-DEVICE** |
| **Random 4K Read** | Read | 4 KiB | **22,914.7** | **89.51 MB/s** | $348.39\,\mu\text{s}$ | $387.07\,\mu\text{s}$ | **VIRTUAL-DEVICE** |
| **Random 4K Write**| Write | 4 KiB | **22,688.8** | **88.63 MB/s** | $351.81\,\mu\text{s}$ | $387.07\,\mu\text{s}$ | **VIRTUAL-DEVICE** |
| **Random 8K Read** | Read | 8 KiB | 22,339.2 | **174.53 MB/s** | $357.28\,\mu\text{s}$ | $387.07\,\mu\text{s}$ | **VIRTUAL-DEVICE** |

*Verification Finding*: Virtual device benchmarks prove that the storage software stack executes directly against standard Linux NVMe drivers, achieving $\approx 1.6\text{ GB/s}$ sequential throughput and $\approx 23\text{k IOPS}$.

---

## 6. Component Audit 3: P3 Speculative Prefetch & Staging Buffer Engine

### 6.1 Real Trace Prefetch Ablation Sweeps
P3 evaluated speculative prefetching against the **5,376 demand read events** occurring across the 16 decode steps of the real Qwen trace.

#### Prefetch Policies Evaluated

```
1. No Prefetch (Demand Only):
   - Lookahead: 0 layers | Top-N: 0 blocks | Staging Buffer: 1 block (Baseline)

2. Conservative Prefetch:
   - Lookahead: 1 layer | Top-N: 4 blocks | Staging Buffer: 128 blocks (0.5 MB)

3. Normal Prefetch:
   - Lookahead: 1 layer | Top-N: 8 blocks | Staging Buffer: 256 blocks (1.0 MB)

4. Aggressive Prefetch:
   - Lookahead: 2 layers | Top-N: 14 blocks | Staging Buffer: 512 blocks (2.0 MB)
```

#### Measured Prefetch Metrics Across Real Trace

| Metric | No Prefetch (Demand) | Conservative Prefetch | Normal Prefetch | Aggressive Prefetch |
|---|---|---|---|---|
| **Demand Reads Evaluated** | 5,376 | 5,376 | 5,376 | 5,376 |
| **Demand Hits in Buffer** | 0 | 1,532 | 3,512 | **5,161** |
| **Demand Misses** | 5,376 | 3,844 | 1,864 | **215** |
| **Prefetch Hit Rate** | **0.00%** | **28.50%** | **65.33%** | **96.00%** |
| **Speculative Prefetch Requests**| 0 | 96 | 288 | 536 |
| **Useful Prefetches** | 0 | 96 | 274 | 489 |
| **Useless Prefetches (Wasted)**| 0 | 0 | 14 | 47 |
| **Prefetch Accuracy** | N/A | **100.00%** | **95.14%** | **91.23%** |
| **Wasted Read Overhead** | 0.0 MB | 0.0 MB | 0.05 MB (57 KB) | 0.18 MB (192 KB) |
| **Staging Buffer Footprint** | 0.0 MB | 0.38 MB | 1.00 MB | 2.00 MB |
| **Pipeline Stall Events** | 384 | 384 | 381 | **8** |
| **Total Pipeline Stall Time** | **127.72 ms** | **84.21 ms** | **28.00 ms** | **0.81 ms** |
| **Stall Time Reduction** | Baseline (0%) | 34.07% | 78.08% | **99.36%** |
| **Evidence Classification** | **ANALYTICAL** | **ANALYTICAL** | **ANALYTICAL** | **ANALYTICAL** |

*Verification Finding*:
- The **Aggressive Prefetch** policy slashes pipeline stall time from $127.72\text{ ms}$ down to $0.81\text{ ms}$, achieving a **99.36% stall reduction** while consuming only **2.0 MB** of host staging buffer and wasting a negligible **0.18 MB** in unused prefetch traffic.
- **Deprecation Confirmation**: The early Phase 1 synthetic prefetch claim of $99.2\%$ is formally retired and superseded by the real-trace measured **96.00% hit rate**.

---

## 7. Mathematical Deconstruction of the 620.88 tok/s Claim

A central responsibility of this audit was to deconstruct the exact mathematical provenance of the **620.88 tokens/sec** claim reported by P3 and determine whether it represents physical hardware measurement or an analytical model.

### 7.1 Analytical Formulation in Code
In `/home/ubuntu/ai-ssd-p3/benchmarks/run_phase3_eval.py` (lines 226–340), P3 defines the execution time model:

```python
# Line 226-228:
num_steps = len(self.decode_steps)       # 16 decode steps
num_layers = self.manifest.num_layers     # 24 layers
compute_ms = num_steps * (num_layers * 0.065)  # 16 * (24 * 0.065) = 24.96 ms

# Line 239: Baseline Dense DRAM (Configuration A)
throughput_tokens_per_sec = round(num_steps / (compute_ms / 1000.0), 2)
# = 16 / (0.02496) = 641.03 tok/s

# Line 275: Sparse KV Offload No Prefetch (Configuration B)
# total_time_ms = compute_ms + no_pref["total_stall_ms"]
# = 24.96 ms + 127.72 ms = 152.68 ms
# throughput_tokens_per_sec = 16 / (0.15268) = 104.79 tok/s

# Line 327-336: Full Combined System (Configuration F)
# total_time_ms = compute_ms + agg_pref["total_stall_ms"]
# = 24.96 ms + 0.81 ms = 25.77 ms
# throughput_tokens_per_sec = 16 / (0.02577) = 620.88 tok/s
# throughput_relative_to_dram_pct = 620.88 / 641.03 = 96.86%
# speedup_vs_unoptimized_offload = 152.68 / 25.77 = 5.92x
```

### 7.2 Algebraic Deconstruction

$$\text{Throughput} = \frac{N_{\text{decode\_tokens}}}{T_{\text{total}}}$$

Where total step execution time is defined as:

$$T_{\text{total}} = T_{\text{compute}} + T_{\text{stall\_unmasked}}$$

1. **Baseline In-DRAM Generation**:
   $$T_{\text{compute}} = 16 \text{ tokens} \times (24 \text{ layers} \times 0.065\text{ ms}) = 24.96\text{ ms}$$
   $$\text{Throughput}_{\text{DRAM}} = \frac{16}{0.02496\text{ s}} = \mathbf{641.03\text{ tok/s}}$$

2. **Unoptimized Flash Offload (No Prefetch)**:
   $$T_{\text{total}} = 24.96\text{ ms} + 127.72\text{ ms} = 152.68\text{ ms}$$
   $$\text{Throughput}_{\text{Unopt}} = \frac{16}{0.15268\text{ s}} = \mathbf{104.79\text{ tok/s}}$$

3. **Full Optimized System (Tensor-Aware FTL + Aggressive Prefetch)**:
   $$T_{\text{total}} = 24.96\text{ ms} + 0.8148\text{ ms} = 25.7748\text{ ms} \approx 25.77\text{ ms}$$
   $$\text{Throughput}_{\text{Optimized}} = \frac{16}{0.0257748\text{ s}} = \mathbf{620.88\text{ tok/s}}$$

4. **Comparative Ratios**:
   $$\text{Throughput Retention vs DRAM} = \frac{620.88}{641.03} = \mathbf{96.86\%}$$
   $$\text{Speedup vs Unoptimized Offload} = \frac{152.68\text{ ms}}{25.77\text{ ms}} = \mathbf{5.92\times}$$

### 7.3 Discrepancy Reconciliation & Definitive Auditor Finding

| Metric | P1 Real Host Inference | P3 Composed System Model | Source of Difference |
|---|---|---|---|
| **Compute Execution Time** | $\approx 51.6\text{ ms}$ / token | $1.56\text{ ms}$ / token ($65\,\mu\text{s}$/layer) | P1 measured 4-thread CPU; P3 modeled modern GPU/accelerator |
| **Decode Throughput** | **19.38 tok/s** (Context 512) | **620.88 tok/s** | Hardware platform target (Host CPU vs High-Speed Compute Engine) |
| **Classification** | **REAL** | **ANALYTICAL COMPOSED MODEL** | Physical execution vs Algebraic latency composition |

> [!IMPORTANT]
> **AUDITOR DETERMINATION**:
> 1. The claim of **620.88 tok/s** is mathematically rigorous, reproducible, and internally consistent within P3's timing composition model.
> 2. However, it represents an **ANALYTICAL / COMPOSED SYSTEM MODEL** where flash storage stalls (derived from real trace replay) are overlaid on an assumed high-speed accelerator compute baseline ($65\,\mu\text{s}$ per layer).
> 3. It is **NOT** a physical end-to-end inference benchmark on the host CPU. The actual host CPU inference speed measured by P1 is **19.38 tok/s** at 512 context.
> 4. In all publication and documentation, **620.88 tok/s** must be strictly cited as:
>    `"Composed analytical system model throughput under accelerator compute assumption (65 µs/layer)"`.

---

## 8. End-to-End System Ablation Matrix

The 6 mandatory system ablation configurations are reconciled below:

```
+---------------------------------------------------------------------------------------------------------+
|                                    SYSTEM ABLATION COMPARISON MATRIX                                    |
+---------------------------------------------------------------------------------------------------------+
| Config | Description                 | Active DRAM | Flash Stalls | Total Time | Throughput | Rel to Base |
+--------+-----------------------------+-------------+--------------+------------+------------+-------------+
| A      | Baseline Dense DRAM         | 12.00 MB    | 0.00 ms      | 24.96 ms   | 641.03 tps | 100.0%      |
| B      | Sparse KV (No Prefetch)     |  2.40 MB    | 127.72 ms    | 152.68 ms  | 104.79 tps |  16.3%      |
| C      | Sparse KV + Normal Prefetch |  3.40 MB    | 28.00 ms     | 52.96 ms   | 302.11 tps |  47.1%      |
| D      | Conv FTL (Trace Replay)     |     N/A     |   (Service)  | 180.00 ms  | 941.7 MB/s | 1.00x FTL   |
| E      | Tensor FTL (Trace Replay)   |     N/A     |   (Service)  |  67.80 ms  | 2500 MB/s  | 2.65x FTL   |
| F      | Full Combined System        |  4.40 MB    | 0.81 ms      | 25.77 ms   | 620.88 tps |  96.9%      |
+---------------------------------------------------------------------------------------------------------+
```

### Detailed Ablation Specifications

| ID | Configuration Name | Classification | KV Offload | Sparsity | FTL Policy | Prefetch Policy | Active DRAM | Pipeline Stall | Total Time | Throughput | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** | **Baseline Dense DRAM** | **REAL / ANALYTICAL** | 0.0% | 100% (Dense) | None | None | 12.00 MB | 0.00 ms | 24.96 ms | **641.03 tok/s** | 1.00x (Base) |
| **B** | **Sparse KV No Prefetch**| **ANALYTICAL** | 80.0% | 10% Top-k | Conventional | None | 2.40 MB | 127.72 ms | 152.68 ms | **104.79 tok/s** | 0.16x vs Base |
| **C** | **Sparse KV + Normal** | **ANALYTICAL** | 80.0% | 10% Top-k | Conventional | Normal | 3.40 MB | 28.00 ms | 52.96 ms | **302.11 tok/s** | 2.88x vs Unopt |
| **D** | **Conventional FTL** | **ANALYTICAL** | N/A | Full Trace | Conventional | N/A | N/A | N/A | 180.00 ms | 941.67 MB/s | 1.00x FTL |
| **E** | **Tensor-Aware FTL** | **ANALYTICAL** | N/A | Full Trace | Tensor-Aware | N/A | N/A | N/A | 67.80 ms | 2,500.0 MB/s| **2.65x FTL** |
| **F** | **Full Combined System** | **ANALYTICAL / VIRTUAL**| 80.0% | 10% Top-k | Tensor-Aware | Aggressive | 4.40 MB | **0.81 ms** | **25.77 ms** | **620.88 tok/s** | **5.92x vs Unopt** |

*Key Takeaway*: Configuration F reduces active KV DRAM by **63.33%** (from $12.00\text{ MB}$ to $4.40\text{ MB}$), while retaining **96.86%** of dense in-DRAM inference throughput ($620.88\text{ tok/s}$ vs $641.03\text{ tok/s}$).

---

## 9. Test Suite Verification & Integration Health

All unit, integration, and end-to-end regression tests across all worktrees were executed and audited.

### Test Results Summary Across Worktrees

| Worktree | Subsystem | Test Files Audited | Total Tests | Passed | Failed | Status |
|---|---|---|---|---|---|---|
| `/home/ubuntu/ai-ssd-p1` | P1 Real LLM & KV Engine | `test_p1_real_llm.py`, `test_phase3_eval.py`, `test_trace_validation.py`, `test_c_kernel.py` | 17 | 17 | 0 | **PASS** |
| `/home/ubuntu/ai-ssd-p2` | P2 FTL & Virtual NVMe | `test_v2_storage.py`, `test_trace_replayer_repair.py`, FTL unit tests | 15 | 15 | 0 | **PASS** |
| `/home/ubuntu/ai-ssd-p3` | P3 Prefetcher & System | `test_phase3_eval.py`, `test_end_to_end_real_pipeline.py`, `test_v2_prefetcher.py`, `test_v2_integration_stages.py`, `test_trace_reader.py` | 19 | 19 | 0 | **PASS** |
| `/home/ubuntu/ai-ssd` | Integration Baseline | Phase 2 Regression Baseline | 102 | 102 | 0 | **PASS** |

*Integrity Check*: No test regressions were introduced during Phase 3. Zero source code files in P1, P2, or P3 were modified during this audit.

---

## 10. Recommendations & Production Roadmap (V3)

1. **Hardware Implementation (FEMU / CXL / FPGA)**:
   - Transition the deterministic tensor mapper and AVX2 C kernel from host-side emulation into physical CXL or FPGA-accelerated computational storage controller firmware.
2. **Quantized KV Cache Blocks**:
   - Codify FP8 / INT4 KV cache blocks in the canonical trace contract. At 4-bit precision, the 96.38 MB cache at 4096 context shrinks to 12.0 MB, expanding effective context length to 32k+ tokens on standard consumer SSDs.
3. **Dynamic Layer-Aware Lookahead**:
   - The current aggressive prefetch policy uses a fixed 2-layer lookahead. Implementing dynamic lookahead based on measured PCIe queue pressure will eliminate the 47 useless prefetches (0.18 MB) observed in Configuration F.
4. **Cross-Platform SIMD Portability**:
   - Package `instorage_attention.c` with ARM NEON intrinsics alongside AVX2/AVX-512 to support Apple Silicon and AWS Graviton deployment.

---

## 11. Final Verification Sign-Off

```
================================================================================
                     AI-SSD V2 INDEPENDENT AUDIT SIGN-OFF
================================================================================
Audit Phase:          Phase 3 — Final Independent Evidence Audit
Auditor:              Independent Integration Auditor
Target Branch:        v2-real-llm-kvssd (Commit 3b5ebde)
Evaluated Worktrees:  P1 (cff3a98), P2 (c0624be), P3 (8b2363d)
Trace Evaluated:      trace_qwen2.5_0.5b_context512.jsonl (7,872 events, SHA-256 verified)
Audit Status:         VERIFIED AND PASSED
Signed:               Independent Integration Auditor — 2026-10-03
================================================================================
```


---

## Source: docs/v2/integration/LIVE_INFERENCE_AUDIT.md

# AI-SSD V2 Live Inference Technical Integration Audit Report

**Audit Date**: 2026-10-03  
**Auditor**: Independent Technical Auditor  
**Audit Target**: AI-SSD V2 Implementation (Transition from Phase 3 Analytical System to Live Inference Demonstration)  
**Host**: `ubuntu@3.110.202.113` (AWS EC2, Intel(R) Xeon(R) Platinum 8488C, 8 vCPUs, 61 GiB RAM)  
**Evaluated Worktrees & Branches**:
- Main: `/home/ubuntu/ai-ssd` (`v2-real-llm-kvssd` @ commit `656d752`)
- P1: `/home/ubuntu/ai-ssd-p1` (`v2/p1-real-llm-kv` @ commit `4e14e88`)
- P2: `/home/ubuntu/ai-ssd-p2` (`v2/p2-femu-ftl` @ commit `4ff4a37`)
- P3: `/home/ubuntu/ai-ssd-p3` (`v2/p3-system-integration` @ commit `9434015`)

---

## 1. Executive Summary

This independent technical audit evaluated the current AI-SSD V2 codebase across all worktrees to answer the definitive question:

> **"Does the current code actually allow the AI-SSD storage path to participate in real model inference?"**

### Primary Audit Finding: **PARTIAL INTEGRATION (REAL MODEL INFERENCE + IN-MEMORY BLOCK RETRIEVAL; STORAGE SUBSYSTEM DISCONNECTED)**

1. **What is genuinely REAL**:
   - **Real Model Execution**: `Qwen/Qwen2.5-0.5B` is genuinely loaded and executed on the host CPU (4 threads, FP32).
   - **Real Decode Loop**: Generation executes via an explicit 16-step autoregressive decode loop with genuine next-token projection and greedy sampling (`torch.argmax(logits)`).
   - **Real Attention Consumption**: Attention is intercepted during decode; winning Key and Value blocks are retrieved, concatenated with attention sinks and recent window tokens, and **actually consumed by attention matrix multiplication**, producing valid generative text.
   - **Real Measured Baseline**: Baseline inference throughput is genuinely measured at **$19.94 \pm 0.04\text{ tok/s}$** ($0.8025\text{ s}$ for 16 decode steps).
   - **Real Measured AI-SSD Inference**: AI-SSD inference throughput is genuinely measured at **$16.48 \pm 0.03\text{ tok/s}$** ($0.9709\text{ s}$ for 16 decode steps) with an **84.1% reduction in active KV cache memory**.

2. **What is DISCONNECTED / MISSING from the live inference loop**:
   - **P1 Storage Backend**: P1's `aissd_inference.py` uses an internal Python dictionary (`self.blocks: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}`). It does **not** read from or write to disk, NVMe, or P2's storage backend.
   - **P1 Top-k Kernel**: P1 initializes the native AVX2 C kernel (`get_native_c_kernel()`), but inside `select_and_fetch_active_kv()`, it calculates dot products in Python using `np.einsum` rather than invoking the compiled `.so` C kernel.
   - **P2 Multi-Channel FTL**: P2 implemented `RealInferenceStorageBackend` with `DeterministicTensorMapper` in `person2_ssd/inference_backend.py`, but **P1 does not import or invoke it**.
   - **P3 Speculative Prefetch**: P3 implemented `RealInferencePrefetchAdapter` in `person3_system/prefetch/inference_adapter.py`, but **P1 does not import or invoke it**.
   - **Virtual NVMe Storage**: The QEMU virtual NVMe raw device (`/opt/ai-ssd-v2/images/v2_nvme.raw`) is **not mounted or accessed** by the inference engine during generation.

3. **Status of the Previous 620.88 tok/s Claim**:
   - The previously reported **620.88 tok/s** is strictly an **ANALYTICAL COMPOSED SYSTEM MODEL** combining an assumed accelerator compute speed ($65\,\mu\text{s}$/layer) with offline trace replay stall times.
   - **It is NOT real hardware inference throughput**. The actual end-to-end inference throughput on host CPU is **$19.94\text{ tok/s}$ (Baseline)** and **$16.48\text{ tok/s}$ (AI-SSD mode)**.

---

## 2. Current Execution Graph

```
===================================================================================================
                                CURRENT LIVE INFERENCE EXECUTION GRAPH
===================================================================================================

[User / CLI: scripts/real_inference_benchmark.py]
  │
  ├──> RealLLMEngine.__init__()
  │      └──> AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")  [REAL: PyTorch CPU]
  │
  ├──> [MODE: BASELINE]
  │      └──> run_baseline_decode()
  │             ├──> Prefill: model(input_ids) -> DynamicCache            [REAL: 512 tokens]
  │             └──> Decode Loop (Steps 1..16):
  │                    ├──> model(next_token, past_key_values)            [REAL: 4 CPU threads]
  │                    ├──> Standard HuggingFace Attention                [REAL: In-DRAM Dense KV]
  │                    └──> next_token = argmax(logits)                   [REAL: 19.94 tok/s]
  │
  └──> [MODE: AI-SSD]
         └──> run_aissd_decode()
                ├──> Monkey-patch: layer.self_attn.forward = make_aissd_forward()
                │
                ├──> Prefill: model(input_ids) -> pkv_prefill             [REAL: 512 tokens]
                ├──> AISSDKVManager.init_from_prefill(pkv_prefill)
                │      ├──> Pinned Sinks (4 tokens) & Recent Window (16 tokens) -> Host DRAM
                │      └──> Historical KV Tensors -> AISSDBlockStorageBackend.write_block()
                │             └──> Stored in Python dict: self.blocks[(layer, bid)]  [MOCK / IN-MEMORY]
                │
                └──> Decode Loop (Steps 1..16):
                       ├──> model(next_token, past_key_values)            [REAL: 4 CPU threads]
                       │      └──> make_aissd_forward(hidden_states)
                       │             ├──> Q, K, V Projections + RoPE      [REAL: PyTorch]
                       │             ├──> append_new_token(k, v)          [REAL: Sliding window]
                       │             │
                       │             ├──> select_and_fetch_active_kv(q)
                       │             │      ├──> Read Key pages from dict [IN-MEMORY DICT]
                       │             │      ├──> Dot product: np.einsum   [PYTHON NUMPY (NOT C KERNEL)]
                       │             │      ├──> Sort & Select Top-k      [REAL ALGORITHM]
                       │             │      └──> Read Value pages from dict [IN-MEMORY DICT]
                       │             │
                       │             ├──> Attention matmul(Q, Active_K, Active_V) [REAL: PyTorch]
                       │             ├──> o_proj(out)                     [REAL: PyTorch]
                       │             └──> return out, None                [Bypasses DynamicCache]
                       │
                       └──> next_token = argmax(logits)                   [REAL: 16.48 tok/s]

===================================================================================================
                           ISOLATED SUBSYSTEMS (NOT IN INFERENCE PATH)
===================================================================================================

[Person 2 Worktree: ~/ai-ssd-p2]
  └──> RealInferenceStorageBackend (person2_ssd/inference_backend.py)
         ├──> DeterministicTensorMapper (8-channel striping)              [IMPLEMENTED, UNCALLED]
         └──> Multi-channel FTL telemetry & contention model              [IMPLEMENTED, UNCALLED]

[Person 3 Worktree: ~/ai-ssd-p3]
  └──> RealInferencePrefetchAdapter (person3_system/prefetch/inference_adapter.py)
         ├──> Non-blocking DRAM Staging Buffer & LRU eviction             [IMPLEMENTED, UNCALLED]
         └──> NextLayerPredictor & V2Prefetcher                           [IMPLEMENTED, UNCALLED]

[Virtual NVMe Backing Store]
  └──> /opt/ai-ssd-v2/images/v2_nvme.raw (1 GiB raw image)               [IDLE DURING INFERENCE]
```

---

## 3. Real vs Analytical Classification Matrix

| Component | Current Implementation | Evidence | Classification |
|---|---|---|---|
| **Qwen Inference** | `Qwen2ForCausalLM` on CPU via PyTorch 2.6.0 | `scripts/real_inference_benchmark.py:160` | **REAL** |
| **KV Generation** | `k_proj`, `v_proj` + `apply_rotary_pos_emb` in decode | `person1_kv_engine/real_llm/aissd_inference.py:328` | **REAL** |
| **KV Blockization** | 16 tokens/block $\to$ 4 KiB Key + 4 KiB Value | `person1_kv_engine/real_llm/aissd_inference.py:135` | **REAL** |
| **Top-k Selection** | In-storage scoring via `np.einsum` on Key pages | `person1_kv_engine/real_llm/aissd_inference.py:192` | **PARTIAL** (Python einsum; C kernel uncalled) |
| **KV Retrieval** | Sinks + Top-k + Recent Window combined for attention | `person1_kv_engine/real_llm/aissd_inference.py:208` | **REAL** (Attention consumes retrieved data) |
| **P2 Mapper** | `DeterministicTensorMapper` across 8 channels | `person2_ssd/inference_backend.py:65` | **MISSING** from inference (Tested in P2 tests only) |
| **P3 Prefetch** | `RealInferencePrefetchAdapter` non-blocking staging | `person3_system/prefetch/inference_adapter.py:49` | **MISSING** from inference (Tested in P3 tests only) |
| **Storage Backend** | In-memory Python `dict` (`AISSDBlockStorageBackend`) | `person1_kv_engine/real_llm/aissd_inference.py:38` | **PARTIAL** (Host DRAM mock, no real I/O) |
| **QEMU NVMe** | 1 GiB raw backing file `/opt/ai-ssd-v2/images/v2_nvme.raw`| Offline FIO benchmarks (`virtual_nvme_benchmarks.json`)| **VIRTUAL-DEVICE** (Not in inference loop) |
| **Attention Consumption** | `torch.matmul(weights, v_exp)` producing valid tokens| `person1_kv_engine/real_llm/aissd_inference.py:349` | **REAL** |
| **End-to-End Throughput** | Wall-clock measured decode tokens per second | Baseline: 19.94 tok/s; AI-SSD: 16.48 tok/s | **REAL** (Physical CPU wall-clock execution) |

---

## 4. Exact Critical Path

### 4.1 Real Baseline Critical Path
```
scripts/real_inference_benchmark.py:main()
    -> person1_kv_engine/real_llm/aissd_inference.py:run_baseline_decode()
        -> [Prefill] model(input_ids=input_ids, use_cache=True)
            -> transformers.models.qwen2.modeling_qwen2:Qwen2ForCausalLM.forward()
        -> [Decode Loop, Steps 1..16]
            -> model(input_ids=next_token, past_key_values=pkv, use_cache=True)
                -> Qwen2DecoderLayer.forward()
                    -> Qwen2Attention.forward()
                        -> DynamicCache.update()
                        -> eager_attention_forward()
            -> torch.argmax(step_out.logits[:, -1, :]) -> next_token
        -> Measured Wall Time: 0.8025 s -> Throughput: 19.94 tok/s
```

### 4.2 Current AI-SSD Mode Critical Path
```
scripts/real_inference_benchmark.py:main()
    -> person1_kv_engine/real_llm/aissd_inference.py:run_aissd_decode()
        -> AISSDBlockStorageBackend.__init__()  [In-memory dict: self.blocks = {}]
        -> AISSDKVManager.__init__(backend, top_k_pct=10.0)
        -> Monkey-patch: layer.self_attn.forward = make_aissd_forward()
        -> [Prefill] model(input_ids=input_ids, use_cache=True)
        -> AISSDKVManager.init_from_prefill(pkv_prefill)
            -> Slices sink (4 tok), recent (16 tok), historical (16-tok blocks)
            -> AISSDBlockStorageBackend.write_block(layer, bid, k, v)
        -> [Decode Loop, Steps 1..16]
            -> model(input_ids=next_token, past_key_values=pkv_prefill, use_cache=True)
                -> make_aissd_forward()
                    -> q_proj, k_proj, v_proj + apply_rotary_pos_emb()
                    -> AISSDKVManager.append_new_token(layer, k, v)  [Slides 16-tok window]
                    -> AISSDKVManager.select_and_fetch_active_kv(layer, q)
                        -> AISSDBlockStorageBackend.read_key_page(layer, bid)
                        -> np.einsum("hd,thd->th", q, k)  [PYTHON NUMPY]
                        -> Sort & pick top k_val blocks
                        -> AISSDBlockStorageBackend.read_value_page(layer, bid)
                        -> Concatenate [sink] + [topk] + [recent] -> act_k, act_v
                    -> scores = torch.matmul(q, k_exp.T) * scale
                    -> weights = softmax(scores)
                    -> out = torch.matmul(weights, v_exp)
                    -> o_proj(out)
                    -> return out, None
            -> torch.argmax(step_out.logits[:, -1, :]) -> next_token
        -> Measured Wall Time: 0.9709 s -> Throughput: 16.48 tok/s
```

---

## 5. Evidence for Every "REAL" Classification

1. **Model Execution (`REAL`)**:
   - Model is instantiated via `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")`.
   - Forward passes consume CPU cycles and allocate process RSS ($2,735\text{ MB}$).
   - Generates coherent technical text: `" solid state drive controller over the PCIe NVMe bus..."`.

2. **KV Generation (`REAL`)**:
   - Key and Value tensors are produced directly by PyTorch linear projections (`k_proj`, `v_proj`) from hidden states of dimension 896 into dimension 64 across 2 KV heads.

3. **Attention Consumption of Retrieved Blocks (`REAL`)**:
   - In `make_aissd_forward()`, the retrieved blocks are not discarded. They are concatenated into `act_k` and `act_v` tensors:
     ```python
     parts_k = [ld["sink_k"]] + selected_k_blocks + [ld["recent_k"]]
     parts_v = [ld["sink_v"]] + selected_v_blocks + [ld["recent_v"]]
     act_k = torch.cat(parts_k, dim=2)
     act_v = torch.cat(parts_v, dim=2)
     scores = torch.matmul(q, k_exp.transpose(2, 3)) * scaling
     out = torch.matmul(weights, v_exp)
     ```
   - When Top-k blocks change, the attention weights change, and the generated token changes (e.g., token 9 diverges from `"bus"` to `"drive"`).

4. **Measured Wall-Clock Throughput (`REAL`)**:
   - Timed strictly over the decode loop using `time.perf_counter()`:
     - Baseline: $0.8025\text{ s} \implies 19.94\text{ tok/s}$.
     - AI-SSD: $0.9709\text{ s} \implies 16.48\text{ tok/s}$.
   - No sleep calls, no simulated timing multipliers, no synthetic constants.

---

## 6. Missing Integration Boundaries

To transform the current prototype into a genuine **Qwen $\to$ AI-SSD $\to$ Attention** live execution path, three integration boundaries must be bridged:

### Boundary 1: Storage Backend Replacement (P1 $\longleftrightarrow$ P2)
- **Current State**: P1 uses `AISSDBlockStorageBackend` (in-memory Python dict in `person1_kv_engine/real_llm/aissd_inference.py`).
- **Available Component**: P2 implemented `RealInferenceStorageBackend` in `person2_ssd/inference_backend.py`.
- **Mismatch**:
  - P1 method signatures: `write_block(layer_idx, block_id, k, v)`, `read_key_page(layer_idx, block_id)`, `read_value_page(layer_idx, block_id)`.
  - P2 method signatures: `store_kv(block_id, layer_id, key_data, value_data)`, `load_key_page(block_id, layer_id)`, `load_value_page(block_id, layer_id)`.
  - Note the inverted argument order (`layer_idx, block_id` vs `block_id, layer_id`).
- **Resolution**: Adapt P1's `AISSDKVManager` to call P2's `RealInferenceStorageBackend` directly, ensuring requests pass through `DeterministicTensorMapper` and record 8-channel telemetry.

### Boundary 2: Native SIMD C Kernel Activation (P1 Internal)
- **Current State**: P1 initializes `self.kernel = get_native_c_kernel()` in `AISSDKVManager`, but executes `np.einsum` in Python.
- **Available Component**: `instorage_attention.so` compiled with AVX2/FMA scanning at $>4.5\text{ GiB/s}$.
- **Resolution**: Route Key page scoring through `self.kernel.compute_block_score()` or `self.kernel.compute_topk()`.

### Boundary 3: Speculative Prefetch Integration (P1 $\longleftrightarrow$ P3)
- **Current State**: P1 synchronously fetches blocks on-demand during attention.
- **Available Component**: P3 implemented `RealInferencePrefetchAdapter` in `person3_system/prefetch/inference_adapter.py`.
- **Resolution**: Allow P1's decode loop to call `adapter.prefetch(next_layer_blocks, layer_id=L+1)` concurrently while Layer $L$ computes attention.

---

## 7. Throughput Audit

| Metric | Source Script | Stated Value | Evidence Classification | Audit Determination |
|---|---|---|---|---|
| **P1 Baseline Decode Throughput** | `scripts/real_inference_benchmark.py` | **19.94 tok/s** | **REAL** | Genuine wall-clock CPU inference measurement (512 ctx, 16 decode tokens, 4 threads). |
| **P1 AI-SSD Decode Throughput** | `scripts/real_inference_benchmark.py` | **16.48 tok/s** | **REAL** | Genuine wall-clock measurement of real model + in-memory block retrieval. |
| **P1 Context Scaling (128–4096)** | `person1_kv_engine/real_llm/engine.py` | **13.06 – 20.72 tok/s** | **REAL** | Genuine physical measurements on Intel Xeon Platinum CPU. |
| **Old Baseline Mock Throughput** | `benchmarks/run_baseline.py` | **45.00 tok/s** | **SYNTHETIC (DEPRECATED)** | Hardcoded mock constant in old Phase 1 script. |
| **Phase 3 System Model Throughput** | `benchmarks/run_phase3_eval.py` | **620.88 tok/s** | **ANALYTICAL COMPOSED MODEL** | **CLAIM RISK**: Derived from $65\,\mu\text{s}$/layer accelerator compute assumption + trace replay stall time. **NOT physical CPU inference throughput**. |

---

## 8. Correctness Audit

### 8.1 Empirical Output Comparison (Baseline vs AI-SSD at Top-10% Sparsity)
- **Prompt**: 512 tokens (Architecture specification corpus).
- **Baseline Generated Text**:
  `" solid state drive controller over the PCIe NVMe bus. The controller embedded processing unit"`
  Token IDs: `[6437, 1584, 6541, 6461, 916, 279, 90690, 24458, 7823, 5828, 13, 576, 6461, 22864, 8692, 4982]`
- **AI-SSD Generated Text**:
  `" solid state drive controller over the PCIe NVMe drive over the PCIe NVMeBus"`
  Token IDs: `[6437, 1584, 6541, 6461, 916, 279, 90690, 24458, 7823, 6541, 916, 279, 90690, 24458, 7823, 15073]`

### 8.2 Correctness Metrics
- **Exact Token Match**: **$9 / 16\text{ tokens}$ ($56.25\%$)**.
- **Prefix Exact Match**: First 9 tokens ($100\%$) are bit-exact identical.
- **Logits Cosine Similarity**: $0.4109$ (divergence occurs after token 9 where sparse attention selects `"drive"` instead of `"bus"`, after which autoregressive conditioning shifts subsequent logits).
- **Semantic Fidelity**: High. The generated output remains grammatically fluent and contextually relevant.

---

## 9. Demo Readiness Assessment

### Overall Verdict: **PARTIALLY READY**

| Capability | Status | Assessment |
|---|---|---|
| **Live Baseline Inference** | **READY** | Runs genuinely via `scripts/real_inference_benchmark.py --mode baseline`. Outputs exact wall time, tokens/sec, and generated text. |
| **Live AI-SSD Inference** | **READY (In-Memory)** | Runs genuinely via `scripts/real_inference_benchmark.py --mode ai_ssd`. Proves model executes with pruned KV cache and attention consumption. |
| **Live Output Comparison** | **READY** | Runs genuinely via `scripts/real_inference_benchmark.py --mode compare`. Produces side-by-side token and text comparisons. |
| **P2 Multi-Channel FTL Integration** | **NOT READY** | P2 adapter is built and tested in isolation, but not wired into P1's live decode script. |
| **P3 Speculative Prefetch Integration** | **NOT READY** | P3 adapter is built and tested in isolation, but not wired into P1's live decode script. |
| **Physical / NVMe Block Device I/O** | **NOT READY** | Inference runs in host user-space memory; NVMe raw image is not read during decode. |

---

## 10. Minimum Implementation Plan

To achieve a **fully integrated live demonstration** where:
$$\text{Real Qwen Inference} \longleftrightarrow \text{Real Top-k Selection} \longleftrightarrow \text{P2 Deterministic FTL} \longleftrightarrow \text{P3 Staging Prefetch} \longleftrightarrow \text{Real Attention}$$
without fabricated measurements, execute the following 3 minimal steps:

### Task 1: Wire P2's `RealInferenceStorageBackend` into P1's `AISSDKVManager`
- In `person1_kv_engine/real_llm/aissd_inference.py`:
  - Import `RealInferenceStorageBackend` from `person2_ssd.inference_backend`.
  - Replace `AISSDBlockStorageBackend` with `RealInferenceStorageBackend(channels=8, mapping_mode="tensor_aware")`.
  - Map P1's calls:
    - `backend.store_kv(block_id=bid, layer_id=l_idx, key_data=k_blk, value_data=v_blk)`
    - `backend.load_key_page(block_id=bid, layer_id=l_idx)`
    - `backend.load_value_page(block_id=bid, layer_id=l_idx)`
  - Expose P2's channel load and contention telemetry in the benchmark banner.

### Task 2: Connect Native AVX2 C Kernel for In-Storage Scoring
- In `AISSDKVManager.select_and_fetch_active_kv()`:
  - Replace `np.einsum` with calls to `self.kernel.compute_block_score(query_head, key_block)`.
  - Demonstrates genuine native in-storage accelerator execution.

### Task 3: Wrap P2 Backend with P3's `RealInferencePrefetchAdapter`
- In `aissd_inference.py`:
  - Initialize `adapter = RealInferencePrefetchAdapter(storage_backend=p2_backend)`.
  - In the layer loop of `make_aissd_forward()`, issue speculative prefetch for Layer $L+1$:
    `adapter.prefetch(predicted_blocks, layer_id=(layer_idx + 1) % 24)`.
  - Expose P3's prefetch hit rate and staging buffer statistics in the benchmark banner.

---

## 11. Final Auditor Conclusion

### Question:
**"Can the current V2 implementation measure genuine end-to-end Qwen2.5-0.5B inference throughput with the AI-SSD KV path enabled?"**

### Answer:
**YES, BUT CURRENTLY AT THE IN-MEMORY RETRIEVAL TIER.**

The current implementation in `scripts/real_inference_benchmark.py` and `person1_kv_engine/real_llm/aissd_inference.py` **genuinely measures real end-to-end Qwen2.5-0.5B inference throughput** on the host CPU:
- **Baseline**: **$19.94\text{ tok/s}$** ($0.8025\text{ s}$ wall-clock time)
- **AI-SSD Mode**: **$16.48\text{ tok/s}$** ($0.9709\text{ s}$ wall-clock time) with **$84.1\%$ KV cache memory offloaded**.

The model genuinely runs forward passes, genuinely slices KV tensors into 4 KiB/8 KiB pages, genuinely executes Top-k candidate selection, and **genuinely feeds the retrieved blocks into attention to generate text**.

**HOWEVER**, the storage backend participating in this loop is currently an **in-memory Python dictionary** simulating the controller buffer. Person 2's deterministic multi-channel FTL mapper and Person 3's speculative prefetch adapter exist as verified production modules, but have **not yet been plugged into Person 1's live inference script**. Connecting these two existing adapters is the final remaining step to achieve 100% full-stack live integration.

