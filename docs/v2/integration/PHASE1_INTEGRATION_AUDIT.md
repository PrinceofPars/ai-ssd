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
