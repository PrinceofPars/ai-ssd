# AI-SSD API Specification

This document defines the common programmatic contracts connecting the KV Engine (Person 1), SSD/FTL Simulator (Person 2), and System API (Person 3).

---

## 1. Primary Data Contracts

### 1.1 `KVBlock`
Defined in `common/schemas/kv_block.py`:
- `block_id: int`: Globally unique block identifier.
- `layer_id: int`: Transformer layer index ($0 \le l < L$).
- `token_start: int`: Starting token position in context.
- `token_count: int`: Number of tokens in the block (default: 16).
- `kv_head_start: int`: Starting KV head index.
- `kv_head_count: int`: Number of KV heads bundled in block (default: 1).
- `head_dim: int`: Dimension per head (default: 128).
- `dtype: str`: Precision format (`"FP16"` = 2 bytes, `"FP8"` = 1 byte).
- `key_size_bytes: int`: Size of cached keys in bytes (default: 2048).
- `value_size_bytes: int`: Size of cached values in bytes (default: 2048).
- `storage_tier: str`: Current location (`"GPU"`, `"DRAM"`, `"SSD"`).
- `hotness: float`: Access frequency / salience score ($0.0 \le h \le 1.0$).
- `physical_location: Optional[str]`: Physical flash address (`"ch<C>_die<D>_pl<P>_blk<B>_pg<G>"`).

---

## 2. API Operations

### Supported Operations:
1. `KV_WRITE`: Write or offload blocks from higher tier (GPU/DRAM) to SSD.
2. `KV_READ`: Retrieve specific blocks from SSD to host DRAM.
3. `KV_EVICT`: Evict selected cold blocks from GPU/DRAM.
4. `KV_PREFETCH`: Asynchronously request blocks ahead of execution.
5. `KV_TOPK`: Filter candidate blocks using attention scoring and return top-$k$.

---

## 3. Subsystem Function Signatures

### Person 1: KV Engine Interface
```python
def create_kv_cache(context_length: int, layers: int, heads: int) -> Dict[int, KVBlock]: ...
def split_into_blocks(tokens: int, layer_id: int, head_id: int) -> List[KVBlock]: ...
def classify_blocks(blocks: List[KVBlock], window_size: int) -> Tuple[List[KVBlock], List[KVBlock]]: ...
def evict_blocks(blocks: List[KVBlock], target_tier: str) -> List[int]: ...
def calculate_attention_scores(query: Any, candidate_blocks: List[KVBlock]) -> List[float]: ...
def select_topk(scores: List[float], candidate_blocks: List[KVBlock], k: int) -> List[KVBlock]: ...
```

### Person 2: SSD Storage Interface
```python
def store_block(block: KVBlock) -> str: ...
def load_block(block_id: int) -> Optional[KVBlock]: ...
def get_location(block_id: int) -> Optional[str]: ...
def estimate_read_latency(block_ids: List[int]) -> float: ...
def tensor_aware_allocate(blocks: List[KVBlock]) -> Dict[int, str]: ...
```

### Person 3: Unified System Interface
```python
def execute_request(request: KVRequest) -> KVResponse: ...
def prefetch_kv(layer_id: int, candidate_blocks: List[int]) -> None: ...
def topk_kv(request: KVRequest) -> KVResponse: ...
```
