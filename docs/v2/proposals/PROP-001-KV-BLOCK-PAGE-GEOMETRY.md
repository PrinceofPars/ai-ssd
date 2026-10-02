# Proposal 001: Explicit Separation of Physical Key/Value Flash Pages and 8 KiB Logical KV Block Geometry

**Author**: Person 1 (P1 - Real LLM + Real KV Cache Engine)  
**Date**: October 2026  
**Status**: PROPOSED  
**Target File**: `common/schemas/kv_block.py` and `common/constants.py`  
**Affected Agents**: P1 (KV Engine), P2 (FTL / FEMU), P3 (Prefetch / System Integration)

---

## 1. Problem Statement

In the initial V1 baseline implementation (`common/constants.py` and `common/schemas/kv_block.py`), the default block size is defined as:
```python
# V1 constants.py
DEFAULT_KEY_SIZE_BYTES = (DEFAULT_BLOCK_TOKENS * DEFAULT_KV_HEADS_PER_BLOCK * DEFAULT_HEAD_DIM * BYTES_PER_FP16) // 2
DEFAULT_VALUE_SIZE_BYTES = DEFAULT_KEY_SIZE_BYTES
DEFAULT_BLOCK_SIZE_BYTES = DEFAULT_KEY_SIZE_BYTES + DEFAULT_VALUE_SIZE_BYTES  # 4096 bytes (4 KB)
```
And in `kv_block.py::create_default`:
```python
bytes_per_elem = 2 if dtype.upper() == "FP16" else 1
total_size = token_count * kv_head_count * head_dim * bytes_per_elem
half_size = total_size // 2
return cls(..., key_size_bytes=half_size, value_size_bytes=half_size, ...)
```

This computation exhibits a critical physical inconsistency:
- 16 tokens $\times$ 1 head $\times$ 128 head dimension $\times$ 2 bytes (FP16) = **4096 bytes (4 KiB) for the Key tensor alone**.
- 16 tokens $\times$ 1 head $\times$ 128 head dimension $\times$ 2 bytes (FP16) = **4096 bytes (4 KiB) for the Value tensor alone**.
- Dividing `total_size // 2` assigns 2048 bytes (2 KiB) to Key and 2048 bytes (2 KiB) to Value, which physically corresponds to only **8 tokens of Key and 8 tokens of Value**, while claiming `token_count = 16`.
- Furthermore, describing a combined K+V block as a single 4 KiB NAND page obscures the physical reality of NAND flash storage: a flash page is 4096 bytes. Therefore, a complete 16-token (dim=128, FP16) Key segment occupies **one complete 4 KiB flash page**, and the Value segment occupies **a separate 4 KiB flash page**, yielding an **8 KiB logical block**.
- In-storage Top-$k$ attention scoring reads **ONLY the Key page** during candidate scoring, leaving the Value page untouched in flash until post-selection. Conflating K and V into a single 4 KiB unit prevents modeling this read asymmetry in FTL (P2) and prefetchers (P3).

---

## 2. Current Contract

In `common/schemas/kv_block.py`:
```python
@dataclass
class KVBlock:
    block_id: int
    layer_id: int
    token_start: int
    token_count: int
    kv_head_start: int
    kv_head_count: int
    head_dim: int
    dtype: str
    key_size_bytes: int
    value_size_bytes: int
    storage_tier: str
    hotness: float = 1.0
    physical_location: Optional[str] = None
```

---

## 3. Proposed Contract Update

Maintain full backwards-compatibility with the existing dataclass attributes, while correcting the size calculation helper methods and providing explicit page properties:

```python
@dataclass
class KVBlock:
    block_id: int
    layer_id: int
    token_start: int
    token_count: int
    kv_head_start: int
    kv_head_count: int
    head_dim: int
    dtype: str
    key_size_bytes: int
    value_size_bytes: int
    storage_tier: str
    hotness: float = 1.0
    physical_location: Optional[str] = None

    @property
    def total_size_bytes(self) -> int:
        return self.key_size_bytes + self.value_size_bytes

    @property
    def key_page_count(self) -> int:
        """Number of 4 KiB NAND flash pages required for Key data."""
        return (self.key_size_bytes + 4095) // 4096

    @property
    def value_page_count(self) -> int:
        """Number of 4 KiB NAND flash pages required for Value data."""
        return (self.value_size_bytes + 4095) // 4096

    @classmethod
    def calculate_sizes(
        cls,
        token_count: int,
        kv_head_count: int,
        head_dim: int,
        dtype: str = "FP16"
    ) -> Tuple[int, int]:
        """Calculates accurate (key_size_bytes, value_size_bytes)."""
        bytes_per_elem = 4 if dtype.upper() in ("FP32", "FLOAT32") else (2 if dtype.upper() in ("FP16", "FLOAT16", "BF16") else 1)
        k_size = token_count * kv_head_count * head_dim * bytes_per_elem
        v_size = k_size
        return k_size, v_size

    @classmethod
    def create_default(
        cls,
        block_id: int,
        layer_id: int,
        token_start: int,
        token_count: int = 16,
        kv_head_start: int = 0,
        kv_head_count: int = 1,
        head_dim: int = 128,
        dtype: str = "FP16",
        storage_tier: str = "GPU",
        hotness: float = 1.0,
    ) -> KVBlock:
        k_bytes, v_bytes = cls.calculate_sizes(token_count, kv_head_count, head_dim, dtype)
        return cls(
            block_id=block_id,
            layer_id=layer_id,
            token_start=token_start,
            token_count=token_count,
            kv_head_start=kv_head_start,
            kv_head_count=kv_head_count,
            head_dim=head_dim,
            dtype=dtype,
            key_size_bytes=k_bytes,
            value_size_bytes=v_bytes,
            storage_tier=storage_tier,
            hotness=hotness,
        )
```

And update `common/constants.py`:
```python
DEFAULT_KEY_SIZE_BYTES = DEFAULT_BLOCK_TOKENS * DEFAULT_KV_HEADS_PER_BLOCK * DEFAULT_HEAD_DIM * BYTES_PER_FP16  # 4096 bytes (4 KiB)
DEFAULT_VALUE_SIZE_BYTES = DEFAULT_KEY_SIZE_BYTES  # 4096 bytes (4 KiB)
DEFAULT_BLOCK_SIZE_BYTES = DEFAULT_KEY_SIZE_BYTES + DEFAULT_VALUE_SIZE_BYTES  # 8192 bytes (8 KiB)
```

---

## 4. Reason for Change

1. **Physical Accuracy**: 16 tokens $\times$ 1 head $\times$ 128 dim $\times$ 2 bytes = 4096 bytes = exactly one physical 4 KiB flash page for Key. Value is an identical 4096 bytes = one 4 KiB flash page. Total logical block = 8 KiB.
2. **I/O Asymmetry in Top-$k$**: Enables P2 (FTL) to accurately model 4 KiB read traffic when fetching only Key pages for candidate evaluation, rather than an artificial 2 KiB slice.
3. **Multi-Architecture Support**: Supports varying `head_dim` (e.g. 64 for Qwen/TinyLlama, 128 for LLaMA-2/3) and dtypes (FP32, FP16, FP8) with exact byte formulas.

---

## 5. Compatibility Impact

- **Zero Breaking Schema Changes**: All existing fields (`block_id`, `layer_id`, `key_size_bytes`, `value_size_bytes`, etc.) remain identical.
- **P2 / FEMU Impact**: FTL will now map 4 KiB Key pages and 4 KiB Value pages to separate flash LBAs (or contiguous LBA pairs), correctly capturing the 4 KiB physical page granularity of real NAND flash.
- **P3 Impact**: Storage backend and prefetcher will allocate 8 KiB buffers per logical block (4 KiB Key + 4 KiB Value).
