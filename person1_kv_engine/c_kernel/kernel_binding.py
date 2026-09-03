"""Python ctypes interface to the compiled native C in-storage kernel."""

import ctypes
import os
from pathlib import Path
from typing import List, Tuple, Any, Optional
import numpy as np


class NativeCKernel:
    """Wrapper loading and binding to instorage_attention.dll."""

    def __init__(self, dll_path: Optional[Path] = None):
        if dll_path is None:
            kernel_dir = Path(__file__).parent.resolve()
            dll_path = kernel_dir / "instorage_attention.dll"

        self.dll_path = dll_path
        self._loaded = False
        self._lib = None

        if self.dll_path.exists():
            try:
                self._lib = ctypes.CDLL(str(self.dll_path))
                self._setup_function_signatures()
                self._loaded = True
            except Exception as e:
                print(f"[WARNING] Failed to load DLL {self.dll_path}: {e}")
                self._loaded = False

    def is_available(self) -> bool:
        return self._loaded

    def _setup_function_signatures(self) -> None:
        # compute_block_score
        self._lib.compute_block_score.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        self._lib.compute_block_score.restype = ctypes.c_float

        # instorage_topk_filter
        self._lib.instorage_topk_filter.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.instorage_topk_filter.restype = ctypes.c_int

    def compute_topk(
        self,
        query: np.ndarray,
        layer_blocks: List[Tuple[int, dict]],
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Invokes the native C in-storage filtering kernel.
        
        Args:
            query: Query array [heads, head_dim] (float32)
            layer_blocks: List of (block_id, entry)
            top_k: Desired number of top blocks
            
        Returns:
            Tuple of (topk_block_ids, topk_values, topk_scores)
        """
        if not self._loaded:
            raise RuntimeError("Native C kernel DLL is not loaded.")

        num_blocks = len(layer_blocks)
        effective_k = min(top_k, num_blocks)
        heads, head_dim = query.shape
        tokens_per_block = layer_blocks[0][1]["k"].shape[0]
        scale = float(1.0 / np.sqrt(head_dim))

        # Ensure query is contiguous float32
        query_c = np.ascontiguousarray(query, dtype=np.float32)

        # Pack Key blocks into a single contiguous buffer
        k_list = [entry["k"] for _, entry in layer_blocks]
        k_contiguous = np.ascontiguousarray(np.stack(k_list, axis=0), dtype=np.float32)

        # Allocate output buffers
        out_indices = np.zeros(effective_k, dtype=np.int32)
        out_scores = np.zeros(effective_k, dtype=np.float32)

        q_ptr = query_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        k_ptr = k_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        idx_ptr = out_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        sc_ptr = out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        status = self._lib.instorage_topk_filter(
            q_ptr,
            k_ptr,
            num_blocks,
            tokens_per_block,
            heads,
            head_dim,
            effective_k,
            scale,
            idx_ptr,
            sc_ptr,
        )

        if status != 0:
            raise RuntimeError(f"C kernel execution failed with code {status}")

        # Map internal indices back to block_ids
        topk_block_ids = np.array([layer_blocks[idx][0] for idx in out_indices], dtype=np.int32)
        topk_values = np.stack([layer_blocks[idx][1]["v"] for idx in out_indices], axis=0)

        return topk_block_ids, topk_values, out_scores


# Singleton instance
_KERNEL_INSTANCE: Optional[NativeCKernel] = None


def get_native_c_kernel() -> NativeCKernel:
    global _KERNEL_INSTANCE
    if _KERNEL_INSTANCE is None:
        _KERNEL_INSTANCE = NativeCKernel()
    return _KERNEL_INSTANCE
