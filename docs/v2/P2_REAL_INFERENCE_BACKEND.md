# Person 2 (P2) — Real Inference Storage Backend Adapter Specification

## Metadata
- **Owner**: Person 2 (P2 — Storage / FEMU / NVMe / FTL / NAND)
- **Target Consumer**: Person 1 (P1 — Live Qwen2.5-0.5B Inference Engine)
- **Worktree**: `/home/ubuntu/ai-ssd-p2`
- **Branch**: `v2/p2-femu-ftl`
- **Classification**: `ANALYTICAL`
  - Real payload retention (stores and returns actual KV tensors)
  - Analytical FTL multi-channel mapping and telemetry
  - Zero synthetic sleep latency injection

---

## 1. Overview & Architectural Role

The **Real Inference Storage Backend Adapter** (`person2_ssd.inference_backend.RealInferenceStorageBackend`) replaces mock in-memory dictionaries in P1's live inference loop with Person 2's deterministic tensor-aware FTL subsystem.

### Key Capabilities
1. **Real Tensor Persistence**: Stores and returns actual Key and Value tensors (`np.ndarray`), ensuring bit-for-bit numerical fidelity for downstream multi-head attention without corruption.
2. **Canonical & Model Geometry Support**:
   - Canonical single-head flash page: $16\text{ tokens} \times 1\text{ head} \times 64\text{ dim} \times \text{FP32} = 4,096\text{ bytes}$ per page ($8,192\text{ bytes}$ combined block).
   - Qwen2.5-0.5B grouped block: $(16\text{ tokens}, 2\text{ KV heads}, 64\text{ dim})$ float32 ($8,192\text{ bytes}$ per tensor).
3. **Deterministic Multi-Channel Striping**: Every block write and read maps through `DeterministicTensorMapper`, striping traffic across all 8 SSD channels.
4. **Zero Artificial Latency**: **Never calls `time.sleep()`**. Operates at line memory speed to allow accurate wall-clock profiling of P1's inference loop.
5. **Drop-in Compatibility**: Fully implements the exact method signatures, argument orders, and property counters expected by P1's `AISSDBlockStorageBackend`.

---

## 2. Exact Backend API

### Class: `RealInferenceStorageBackend`

```python
from person2_ssd.inference_backend import RealInferenceStorageBackend

backend = RealInferenceStorageBackend(
    channels=8,              # SSD concurrent channel count
    dies_per_channel=4,      # Dies per channel
    planes_per_die=2,        # Planes per die
    pages_per_block=256,     # Pages per flash block
    blocks_per_plane=1024,   # Blocks per plane
    num_layers=24,           # Qwen2.5-0.5B layer count
    num_heads=2,             # Qwen2.5-0.5B KV heads (GQA)
    tokens_per_block=16,     # Tokens per KVBlock
    head_dim=64,             # Head dimension
    dtype="float32",         # FP32 data representation
    mapping_mode="tensor_aware", # Tensor-aware multi-channel striping
)
```

### Methods

| Method Signature | Description | Return Type |
|---|---|---|
| `write_block(layer_idx: int, block_id: int, k_block: np.ndarray, v_block: np.ndarray, head_id: int = 0, token_start: int = 0) -> None` | Writes an 8 KiB KV block (Key + Value) into storage. Resolves placement across channels via `DeterministicTensorMapper`. | `None` |
| `read_key_page(layer_idx: int, block_id: int, head_id: int = 0, token_start: int = 0) -> np.ndarray` | Reads Key page (scanned during in-storage `TOPK_FILTER`). Returns actual stored numpy array. | `np.ndarray` |
| `read_value_page(layer_idx: int, block_id: int, head_id: int = 0, token_start: int = 0) -> np.ndarray` | Reads Value page over PCIe bus (`TOPK_FETCH`). Returns actual stored numpy array. | `np.ndarray` |
| `read_block(layer_idx: int, block_id: int, head_id: int = 0, token_start: int = 0) -> Tuple[np.ndarray, np.ndarray]` | Reads full KV block. Returns `(key_tensor, value_tensor)`. | `Tuple[np.ndarray, np.ndarray]` |
| `evict_block(layer_idx: int, block_id: int) -> bool` | Invalidates block when context terminates. | `bool` |
| `reset_stats() -> None` | Resets all telemetry counters to zero. | `None` |
| `get_telemetry() -> Dict[str, Any]` | Returns comprehensive hardware, channel distribution, and contention metrics. | `Dict[str, Any]` |

### Compatibility Properties (Directly read by P1)

| Property | Type | Description |
|---|---|---|
| `backend.bytes_read` | `int` | Total bytes transferred on read paths |
| `backend.bytes_written` | `int` | Total bytes transferred on write paths |
| `backend.blocks_read` | `int` | Total block read operations |
| `backend.blocks_written` | `int` | Total block write operations |
| `backend.requests` | `int` | Total I/O request count |

---

## 3. Example P1 Call Sequence

In P1's `person1_kv_engine/real_llm/aissd_inference.py`, simply swap the in-memory backend for P2's adapter:

```python
# 1. Instantiate P2's Real Inference Storage Backend
from person2_ssd.inference_backend import RealInferenceStorageBackend
backend = RealInferenceStorageBackend(num_layers=24, tokens_per_block=16, head_dim=64)

# 2. Prefill Phase: Offload historical KV blocks to storage
# Slices: k_blk and v_blk of shape (16, 2, 64)
for b_start in range(0, total_hist_tok, 16):
    b_end = min(b_start + 16, total_hist_tok)
    tok_count = b_end - b_start
    k_blk = np.zeros((16, 2, 64), dtype=np.float32)
    v_blk = np.zeros((16, 2, 64), dtype=np.float32)
    k_blk[:tok_count] = k_t[b_start:b_end]
    v_blk[:tok_count] = v_t[b_start:b_end]

    # Write block through P2 FTL mapper:
    backend.write_block(layer_idx=l_idx, block_id=bid, k_block=k_blk, v_block=v_blk)
    bid += 1

# 3. Decode Step: In-Storage Top-k Filter (Scans Key page)
for bid, actual_tokens in cand_bids:
    k_blk = backend.read_key_page(layer_idx=l_idx, block_id=bid)  # Returns actual [16, 2, 64] float32
    # Compute in-storage attention score with query...

# 4. Decode Step: Sparse Gather Winning Top-k Value Pages over PCIe
for _, bid, actual_tokens in top_bids:
    v_blk = backend.read_value_page(layer_idx=l_idx, block_id=bid)  # Returns actual [16, 2, 64] float32
    k_blk = backend.read_key_page(layer_idx=l_idx, block_id=bid)
    k_t = torch.from_numpy(k_blk[:actual_tokens]).permute(1, 0, 2).unsqueeze(0)
    v_t = torch.from_numpy(v_blk[:actual_tokens]).permute(1, 0, 2).unsqueeze(0)

# 5. Inspect Hardware Telemetry
stats = backend.get_telemetry()
print("Channel Distribution:", stats["channel_distribution"]["per_channel_total_requests"])
print("Contention Ratio:", stats["channel_distribution"]["contention_ratio"])
```

---

## 4. Geometry Verification

The adapter enforces and preserves canonical dimensions:

$$\text{Page Size} = N_{\text{tokens}} \times N_{\text{heads}} \times D_{\text{head}} \times \text{sizeof}(\text{FP32})$$

- **Key Page**: $16 \times 1 \times 64 \times 4\text{ B} = 4,096\text{ bytes}$ ($4\text{ KiB}$)
- **Value Page**: $16 \times 1 \times 64 \times 4\text{ B} = 4,096\text{ bytes}$ ($4\text{ KiB}$)
- **Combined Block**: $4,096\text{ B} + 4,096\text{ B} = 8,192\text{ bytes}$ ($8\text{ KiB}$)
- **Qwen Grouped Block**: $16 \times 2 \times 64 \times 4\text{ B} = 8,192\text{ bytes}$ per tensor ($16,384\text{ bytes}$ total for 2 heads).

---

## 5. Multi-Channel Hardware Mapping Verification

Physical channel assignment uses the proven tensor-aware co-design formula:
$$\text{Channel} = (L + h + b_{\text{idx}} + \lfloor b_{\text{idx}} / C \rfloor) \pmod C$$

Where:
- $L$: Transformer layer index ($0..23$)
- $h$: Attention head index ($0..1$ for GQA)
- $b_{\text{idx}}$: Block index within the head
- $C = 8$: Number of flash channels

### Load Balance Guarantee
Under multi-head inference across 48 blocks:
- All 8 channels receive requests ($0$ starved channels).
- Channel contention ratio remains within $1.01\times$ to $1.35\times$ (near-perfect theoretical parity).
- Completely prevents the 2-head GQA bottleneck that occurs under naive head-only striping.

---

## 6. Telemetry Schema Available to P1

Calling `backend.get_telemetry()` returns:
```json
{
  "backend_classification": "ANALYTICAL",
  "architecture": {
    "channels": 8,
    "dies_per_channel": 4,
    "planes_per_die": 2,
    "key_page_bytes": 4096,
    "value_page_bytes": 4096,
    "logical_block_bytes": 8192,
    "mapping_mode": "tensor_aware"
  },
  "requests": {
    "total": 128,
    "reads": 64,
    "writes": 64,
    "read_key_pages": 32,
    "read_value_pages": 32,
    "read_combined_blocks": 0
  },
  "bytes": {
    "total": 1048576,
    "reads": 524288,
    "writes": 524288
  },
  "channel_distribution": {
    "channel_read_counts": {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8, "7": 8},
    "channel_write_counts": {"0": 8, "1": 8, "2": 8, "3": 8, "4": 8, "5": 8, "6": 8, "7": 8},
    "per_channel_total_requests": {"0": 16, "1": 16, "2": 16, "3": 16, "4": 16, "5": 16, "6": 16, "7": 16},
    "contention_ratio": 1.0,
    "load_imbalance_percent": 0.0
  },
  "simulated_metrics": {
    "analytical_service_time_ms": 3.84,
    "sleep_latency_injected": false
  }
}
```

---

## 7. Known Limitations

1. **Analytical Classification**: The multi-channel queuing and latency are analytical simulations. The actual tensor payload is retained in host DRAM buffers, not on raw NAND flash cells.
2. **Controller DRAM Caching**: The current backend persists all offloaded blocks in its indexed store; hardware flash endurance degradation and garbage collection are modeled analytically, not physically worn.
3. **No Direct Kernel Block Device**: Live inference uses this direct Python-callable backend rather than communicating via `/dev/nvme0n1` ioctl. Physical virtual NVMe benchmarking remains available via `scripts/run_virtual_nvme_bench.py`.
