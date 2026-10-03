# Person 2 (P2) — Real Inference Storage Backend Adapter

## Metadata
- **Agent**: Person 2 (P2)
- **Role**: Storage / FEMU / NVMe / FTL / NAND
- **Worktree**: `/home/ubuntu/ai-ssd-p2`
- **Branch**: `v2/p2-femu-ftl`
- **Target Consumer**: Person 1 (P1 Real LLM Inference Engine)
- **Classification**: `ANALYTICAL`
  - Real payload retention (stores and returns actual KV tensors/bytes)
  - Analytical FTL multi-channel mapping and telemetry
  - Zero synthetic sleep latency injection

---

## 1. Overview & Purpose

The **Real Inference Storage Backend Adapter** (`person2_ssd.inference_backend.RealInferenceStorageBackend`) bridges Person 2's deterministic multi-channel FTL simulator with Person 1's actual Qwen2.5-0.5B inference execution loop.

Prior to Phase 5B, Person 2 evaluated storage traces offline or synthetically. In Phase 5B, Person 1 can invoke Person 2's storage backend directly during live prompt prefill and auto-regressive generation.

### Core Architecture Requirements
1. **Real Data Ingest & Retrieval**: Persists and returns actual Key and Value tensors (`np.ndarray`), ensuring numerical integrity for down-stream attention calculation.
2. **Page Geometry Contract**:
   - $\text{Key Page} = 4,096\text{ bytes}$ ($4\text{ KiB}$)
   - $\text{Value Page} = 4,096\text{ bytes}$ ($4\text{ KiB}$)
   - $\text{Combined Block} = 8,192\text{ bytes}$ ($8\text{ KiB}$)
   - Sized for Qwen2.5-0.5B: $16\text{ tokens} \times 1\text{ head} \times 64\text{ dim} \times 4\text{ bytes (FP32)} = 4,096\text{ bytes}$.
3. **Deterministic Mapping**: Every read and write is mapped through `DeterministicTensorMapper` across 8 channels, 4 dies, and 2 planes.
4. **Zero Simulated Sleep Latency**: To prevent artificial stalling of P1's inference pipeline, the backend **never calls `time.sleep()`**. Instead, analytical service times are computed mathematically and logged in telemetry.
5. **Full Hardware Telemetry**: Tracks per-channel request and byte loads, max/min channel utilization, and contention ratios.

---

## 2. Python API Specification (For P1)

### Import
```python
from person2_ssd.inference_backend import RealInferenceStorageBackend
```

### Initialization
```python
backend = RealInferenceStorageBackend(
    channels=8,              # Number of concurrent SSD channels
    dies_per_channel=4,      # Dies per channel
    planes_per_die=2,        # Planes per die
    pages_per_block=256,     # Pages per flash block
    blocks_per_plane=1024,   # Blocks per plane
    num_layers=24,           # Qwen2.5-0.5B layers
    num_heads=2,             # Qwen2.5-0.5B KV heads (GQA)
    tokens_per_block=16,     # Tokens per KVBlock
    head_dim=64,             # Head dimension
    dtype="float32",         # Data type
    mapping_mode="tensor_aware", # Multi-channel striping policy
)
```

### Core Operations

#### 1. Store KV Block (`store_kv`)
Stores real Key and Value payloads into the storage subsystem.
```python
success = backend.store_kv(
    block_id=block_id,      # int: Unique block index within layer
    layer_id=layer_id,      # int: Transformer layer index (0..23)
    key_data=k_tensor,      # np.ndarray or bytes: exactly 4,096 bytes
    value_data=v_tensor,    # np.ndarray or bytes: exactly 4,096 bytes
    head_id=head_id,        # int: Attention head index (0..1)
    token_start=token_start # int: Token offset in sequence
)
```

#### 2. Load Full KV Block (`load_kv`)
Retrieves both Key and Value tensors ($8,192\text{ bytes}$ combined) for attention decoding.
```python
k_tensor, v_tensor = backend.load_kv(
    block_id=block_id,
    layer_id=layer_id,
    head_id=head_id,
    token_start=token_start
)
# Returns: Tuple[np.ndarray, np.ndarray], each 4,096 bytes
```

#### 3. Load Key Page Only (`load_key_page`)
Optimized for `TOPK_FILTER`: reads only the $4,096\text{ B}$ Key page across PCIe.
```python
k_tensor = backend.load_key_page(
    block_id=block_id,
    layer_id=layer_id,
    head_id=head_id,
    token_start=token_start
)
# Returns: np.ndarray, 4,096 bytes
```

#### 4. Load Value Page Only (`load_value_page`)
Optimized for `TOPK_FETCH`: reads only the $4,096\text{ B}$ Value page for filtered tokens.
```python
v_tensor = backend.load_value_page(
    block_id=block_id,
    layer_id=layer_id,
    head_id=head_id,
    token_start=token_start
)
# Returns: np.ndarray, 4,096 bytes
```

#### 5. Evict KV Block (`evict_kv`)
Invalidates a block from storage when a context window or sequence terminates.
```python
backend.evict_kv(block_id=block_id, layer_id=layer_id)
```

#### 6. Telemetry & Hardware Counters (`get_telemetry`)
Returns complete metrics detailing channel balancing and simulated service time.
```python
stats = backend.get_telemetry()
print(f"Backend Classification : {stats['backend_classification']}")
print(f"Total Requests         : {stats['requests']['total']}")
print(f"Read Bytes             : {stats['bytes']['reads']}")
print(f"Contention Ratio       : {stats['channel_distribution']['contention_ratio']}")
print(f"Simulated Latency (ms) : {stats['simulated_metrics']['analytical_service_time_ms']}")
```

---

## 3. Physical Channel Placement Verification

The adapter maps requests through `DeterministicTensorMapper` using the proven tensor-aware striping formula:
$$\text{Channel} = (L + h + b_{\text{idx}} + \lfloor b_{\text{idx}} / C \rfloor) \pmod C$$

This guarantees that:
- Requests never collapse onto Channel 0 (as in conventional linear FTL).
- Even with Grouped-Query Attention ($H_{kv} = 2$), load is uniformly balanced across all 8 channels (contention ratio $\approx 1.01\times$ to $1.15\times$).

---

## 4. Verification & Testing

Person 2 provides comprehensive verification in `person2_ssd/tests/test_inference_backend.py`:
- `test_write_and_read_known_kv_blocks`: Validates 100% numerical data fidelity on persisted tensors.
- `test_preserve_exact_geometry_bytes`: Enforces strict 4096 B / 4096 B / 8192 B boundaries.
- `test_subpage_reads_filter_and_fetch`: Verifies independent Key and Value transfers.
- `test_multi_channel_striping_and_telemetry`: Verifies all 8 channels receive balanced traffic.
- `test_zero_simulated_sleep_latency`: Verifies 200 I/O operations execute in $< 50\text{ ms}$ without sleep.
- `test_eviction_and_state_management`: Validates residency queries and cache invalidation.
- `test_telemetry_reset`: Validates telemetry reset between evaluation runs.
