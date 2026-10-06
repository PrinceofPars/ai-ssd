"""Real Qwen2.5-0.5B Inference Engine with AI-SSD KV-Cache Integration.

Implements two genuinely executable inference modes:
1. BASELINE: Normal Qwen inference using standard in-memory DynamicCache.
2. AI-SSD: Qwen inference where KV-cache access during decoding actually executes
   the in-storage Top-k block selection and retrieval path through the AI-SSD storage backend.

Measures actual wall-clock execution time strictly over the decode loop.
Zero analytical timing injection; zero sleep; 100% genuine model forward execution.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import os
import sys
import time
import math
import gc
import logging
import psutil
import threading
import torch
import numpy as np
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from person1_kv_engine.c_kernel.kernel_binding import get_native_c_kernel

logger = logging.getLogger(__name__)

# Resilient integration with Person 2's RealInferenceStorageBackend
_P2_AVAILABLE = False
RealInferenceStorageBackend = None

_candidate_p2_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
    "/home/ubuntu/ai-ssd",
    "/home/ubuntu/ai-ssd-p2",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../ai-ssd-p2")),
]
for _p2_path in _candidate_p2_paths:
    if os.path.isdir(_p2_path):
        if _p2_path not in sys.path:
            sys.path.append(_p2_path)
        try:
            import person2_ssd
            _p2_ssd_dir = os.path.join(_p2_path, "person2_ssd")
            if _p2_ssd_dir not in person2_ssd.__path__:
                person2_ssd.__path__.insert(0, _p2_ssd_dir)
            from person2_ssd.inference_backend import RealInferenceStorageBackend as _P2Backend
            RealInferenceStorageBackend = _P2Backend
            _P2_AVAILABLE = True
            break
        except Exception:
            pass

# Resilient integration with Person 3's RealInferencePrefetchAdapter
_P3_AVAILABLE = False
RealInferencePrefetchAdapter = None

_candidate_p3_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
    "/home/ubuntu/ai-ssd",
    "/home/ubuntu/ai-ssd-p3",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../ai-ssd-p3")),
]
for _p3_path in _candidate_p3_paths:
    if os.path.isdir(_p3_path):
        if _p3_path not in sys.path:
            sys.path.append(_p3_path)
        try:
            import person3_system
            _p3_sys_dir = os.path.join(_p3_path, "person3_system")
            if _p3_sys_dir not in person3_system.__path__:
                person3_system.__path__.insert(0, _p3_sys_dir)
            from person3_system.prefetch.inference_adapter import RealInferencePrefetchAdapter as _P3Adapter
            RealInferencePrefetchAdapter = _P3Adapter
            _P3_AVAILABLE = True
            break
        except Exception:
            pass


class ProcessMemorySampler:
    """Threaded high-resolution process RSS sampler.
    
    Samples actual OS process Resident Set Size (RSS) continuously during
    decode execution interval to accurately determine minimum, average,
    and peak memory without estimation.
    """
    def __init__(self, sample_interval_s: float = 0.002):
        self.sample_interval_s = sample_interval_s
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process()

    def start(self):
        self.samples = [self._proc.memory_info().rss / (1024.0 * 1024.0)]
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self.samples.append(self._proc.memory_info().rss / (1024.0 * 1024.0))
        return {
            "min_rss_mb": float(np.min(self.samples)),
            "avg_rss_mb": float(np.mean(self.samples)),
            "peak_rss_mb": float(np.max(self.samples)),
            "std_rss_mb": float(np.std(self.samples)),
            "sample_count": len(self.samples),
        }

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self._proc.memory_info().rss / (1024.0 * 1024.0))
            except Exception:
                pass
            time.sleep(self.sample_interval_s)


def get_current_rss_mb() -> float:
    """Returns the current process Resident Set Size (RSS) in megabytes."""
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


class AISSDBlockStorageBackend:
    """Executable block storage backend for AI-SSD V2.
    
    Represents physical flash page storage organized into 4 KiB Key pages
    and 4 KiB Value pages (8 KiB logical blocks). Backed by in-memory byte buffers
    on the host representing the computational storage controller NAND plane.
    """

    def __init__(self, num_layers: int = 24, tokens_per_block: int = 16, head_dim: int = 64):
        self.num_layers = num_layers
        self.tokens_per_block = tokens_per_block
        self.head_dim = head_dim
        # Key: (layer_idx, block_id) -> {"k": np.ndarray [tokens, heads, dim], "v": ...}
        self.blocks: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        self.bytes_read: int = 0
        self.bytes_written: int = 0
        self.blocks_read: int = 0
        self.blocks_written: int = 0
        self.requests: int = 0
        self.storage_batches: int = 0
        self.batched_requests: int = 0

    def write_block(self, layer_idx: int, block_id: int, k_block: np.ndarray, v_block: np.ndarray) -> None:
        """Writes an 8 KiB logical KV block (4 KiB Key + 4 KiB Value)."""
        self.blocks[(layer_idx, block_id)] = {
            "k": np.ascontiguousarray(k_block, dtype=np.float32),
            "v": np.ascontiguousarray(v_block, dtype=np.float32),
        }
        self.blocks_written += 1
        self.bytes_written += 8192

    def read_key_page(self, layer_idx: int, block_id: int) -> np.ndarray:
        """Controller internal read: scans 4 KiB Key page for in-storage filtering."""
        self.requests += 1
        self.bytes_read += 4096
        return self.blocks[(layer_idx, block_id)]["k"]

    def read_value_page(self, layer_idx: int, block_id: int) -> np.ndarray:
        """Host retrieval read: fetches 4 KiB Value page over PCIe interface."""
        self.requests += 1
        self.blocks_read += 1
        self.bytes_read += 4096
        return self.blocks[(layer_idx, block_id)]["v"]

    def read_key_page_batch(self, layer_idx: int, block_ids: List[int]) -> Dict[int, np.ndarray]:
        res = {}
        for bid in block_ids:
            res[bid] = self.read_key_page(layer_idx, bid)
        self.storage_batches += 1
        self.batched_requests += len(block_ids)
        return res

    def read_value_page_batch(self, layer_idx: int, block_ids: List[int]) -> Dict[int, np.ndarray]:
        res = {}
        for bid in block_ids:
            res[bid] = self.read_value_page(layer_idx, bid)
        self.storage_batches += 1
        self.batched_requests += len(block_ids)
        return res

    def read_block_batch(self, layer_idx: int, block_ids: List[int]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        res = {}
        for bid in block_ids:
            res[bid] = (self.read_key_page(layer_idx, bid), self.read_value_page(layer_idx, bid))
        self.storage_batches += 1
        self.batched_requests += len(block_ids)
        return res

    def reset_stats(self) -> None:
        self.bytes_read = 0
        self.bytes_written = 0
        self.blocks_read = 0
        self.blocks_written = 0
        self.requests = 0
        self.storage_batches = 0
        self.batched_requests = 0


def create_default_storage_backend(
    channels: int = 8,
    num_layers: int = 24,
    num_heads: int = 2,
    tokens_per_block: int = 16,
    head_dim: int = 64,
    dtype: str = "float32",
    mapping_mode: str = "tensor_aware",
    enable_prefetch: bool = True,
    buffer_capacity_blocks: int = 512,
    storage_mode: str = "file",
    enable_batching: bool = True,
    enable_async_pipeline: bool = False,
) -> Any:
    """Creates the production AI-SSD storage backend pipeline.
    
    If enable_prefetch or enable_async_pipeline is True and Person 3 is available:
        Person 3's RealInferencePrefetchAdapter wraps Person 2's RealInferenceStorageBackend.
    Otherwise:
        Person 2's RealInferenceStorageBackend (multi-channel FTL + tensor-aware mapping) is returned.
    Falls back to AISSDBlockStorageBackend if P2 is unavailable.
    """
    backend = None
    if _P2_AVAILABLE and RealInferenceStorageBackend is not None:
        backend = RealInferenceStorageBackend(
            channels=channels,
            num_layers=num_layers,
            num_heads=num_heads,
            tokens_per_block=tokens_per_block,
            head_dim=head_dim,
            dtype=dtype,
            mapping_mode=mapping_mode,
            storage_mode=storage_mode,
            enable_batching=enable_batching,
        )
    else:
        backend = AISSDBlockStorageBackend(
            num_layers=num_layers,
            tokens_per_block=tokens_per_block,
            head_dim=head_dim,
        )

    if (enable_prefetch or enable_async_pipeline) and _P3_AVAILABLE and RealInferencePrefetchAdapter is not None:
        bytes_per_elem = 4 if str(dtype).lower() in ("fp32", "float32") else 2
        block_bytes = tokens_per_block * num_heads * head_dim * bytes_per_elem * 2
        adapter = RealInferencePrefetchAdapter(
            storage_backend=backend,
            buffer_capacity_blocks=buffer_capacity_blocks,
            bytes_per_block=block_bytes,
            tokens_per_block=tokens_per_block,
            kv_heads_per_block=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            num_layers=num_layers,
            enable_async_pipeline=enable_async_pipeline,
        )
        return adapter

    return backend


class AISSDKVManager:
    """Manages the computational storage KV hierarchy for real LLM inference.
    
    Partitions KV blocks into:
    - Host DRAM Attention Sinks (first 4 tokens)
    - Host DRAM Recent Window (last 16 tokens)
    - Offloaded Historical Blocks (stored in AI-SSD storage backend)
    """

    def __init__(
        self,
        backend: Any,
        num_layers: int = 24,
        sink_tokens: int = 4,
        recent_tokens: int = 16,
        top_k_pct: float = 10.0,
        enable_computational_storage: bool = False,
        enable_prefetch: bool = True,
        enable_async_pipeline: bool = False,
    ):
        self.backend = backend
        self.num_layers = num_layers
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.top_k_pct = top_k_pct
        self.enable_computational_storage = enable_computational_storage
        self.enable_prefetch = enable_prefetch
        self.enable_async_pipeline = enable_async_pipeline
        self.tokens_per_block = 16
        self.head_dim = 64
        self.is_active = False
        self.layer_data: Dict[int, Dict[str, Any]] = {}
        self.kernel = get_native_c_kernel()
        self.candidate_k_bytes_to_host: int = 0
        self.winning_k_bytes_to_host: int = 0
        self.winning_v_bytes_to_host: int = 0
        self.topk_metadata_bytes_to_host: int = 0
        self.timings = {
            "qkv_proj_s": 0.0,
            "rope_s": 0.0,
            "candidate_k_reads_s": 0.0,
            "topk_scoring_s": 0.0,
            "candidate_selection_s": 0.0,
            "prefetch_s": 0.0,
            "winning_v_reads_s": 0.0,
            "tensor_recon_s": 0.0,
            "active_concat_s": 0.0,
            "attn_matmul_s": 0.0,
            "out_proj_s": 0.0,
            "mlp_and_norm_s": 0.0,
            "bookkeeping_s": 0.0,
        }

    def reset_timings(self) -> None:
        """Resets all sub-operation timings to zero."""
        for k in self.timings:
            self.timings[k] = 0.0
        self.candidate_k_bytes_to_host = 0
        self.winning_k_bytes_to_host = 0
        self.winning_v_bytes_to_host = 0
        self.topk_metadata_bytes_to_host = 0

    def init_from_prefill(self, past_key_values: Any) -> None:
        """Blockizes the prefill KV cache and offloads historical blocks to storage backend."""
        if hasattr(past_key_values, "layers"):
            self.num_layers = len(past_key_values.layers)
            first_k = past_key_values.layers[0].keys
        elif hasattr(past_key_values, "key_cache"):
            self.num_layers = len(past_key_values.key_cache)
            first_k = past_key_values.key_cache[0]
        else:
            self.num_layers = len(past_key_values)
            first_k = past_key_values[0][0]
        self.num_kv_heads = first_k.shape[1]
        self.head_dim = first_k.shape[3]
        self.dtype = first_k.dtype
        self.bytes_per_elem = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        self.page_bytes = self.tokens_per_block * self.num_kv_heads * self.head_dim * self.bytes_per_elem
        np_dtype = np.float16 if self.bytes_per_elem == 2 else np.float32
        for l_idx in range(self.num_layers):
            if hasattr(past_key_values, "layers"):
                k_tensor = past_key_values.layers[l_idx].keys
                v_tensor = past_key_values.layers[l_idx].values
            elif hasattr(past_key_values, "key_cache"):
                k_tensor = past_key_values.key_cache[l_idx]
                v_tensor = past_key_values.value_cache[l_idx]
            else:
                k_tensor = past_key_values[l_idx][0]
                v_tensor = past_key_values[l_idx][1]
            seq_len = k_tensor.shape[2]

            # 1. Attention Sinks: retain in host DRAM
            sink_k = k_tensor[:, :, :self.sink_tokens, :].clone()
            sink_v = v_tensor[:, :, :self.sink_tokens, :].clone()

            # 2. Recent Window: retain in host DRAM
            recent_k = k_tensor[:, :, -self.recent_tokens:, :].clone()
            recent_v = v_tensor[:, :, -self.recent_tokens:, :].clone()

            # 3. Offload historical blocks (between sink and recent window)
            hist_start = self.sink_tokens
            hist_end = seq_len - self.recent_tokens
            k_np = k_tensor[0].cpu().numpy()  # [num_kv_heads, seq_len, head_dim]
            v_np = v_tensor[0].cpu().numpy()

            cand_bids = []
            if hist_end > hist_start:
                k_t = np.transpose(k_np[:, hist_start:hist_end, :], (1, 0, 2))  # [hist_tokens, kv_heads, head_dim]
                v_t = np.transpose(v_np[:, hist_start:hist_end, :], (1, 0, 2))
                total_hist_tok = k_t.shape[0]

                bid = 0
                for b_start in range(0, total_hist_tok, self.tokens_per_block):
                    b_end = min(b_start + self.tokens_per_block, total_hist_tok)
                    tok_count = b_end - b_start
                    k_blk = np.zeros((self.tokens_per_block, self.num_kv_heads, self.head_dim), dtype=np_dtype)
                    v_blk = np.zeros((self.tokens_per_block, self.num_kv_heads, self.head_dim), dtype=np_dtype)
                    k_blk[:tok_count] = k_t[b_start:b_end]
                    v_blk[:tok_count] = v_t[b_start:b_end]
                    self.backend.write_block(l_idx, bid, k_blk, v_blk)
                    cand_bids.append((bid, tok_count))
                    bid += 1

            self.layer_data[l_idx] = {
                "sink_k": sink_k,
                "sink_v": sink_v,
                "recent_k": recent_k,
                "recent_v": recent_v,
                "candidate_blocks": cand_bids,
                "total_tokens": seq_len,
            }
        self.is_active = True

    def append_new_token(self, l_idx: int, k_tok: torch.Tensor, v_tok: torch.Tensor) -> None:
        """Slides the host recent window to incorporate the newly decoded token."""
        ld = self.layer_data[l_idx]
        ld["recent_k"] = torch.cat([ld["recent_k"][:, :, 1:, :], k_tok], dim=2)
        ld["recent_v"] = torch.cat([ld["recent_v"][:, :, 1:, :], v_tok], dim=2)
        ld["total_tokens"] += 1

    def select_and_fetch_active_kv(
        self,
        l_idx: int = 0,
        query_states: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes in-storage Top-k scoring on candidate Key pages and fetches winning Value pages."""
        if layer_idx is not None:
            l_idx = layer_idx
        if query_states is None:
            raise ValueError("query_states must be provided")
        ld = self.layer_data[l_idx]
        cand_bids = ld["candidate_blocks"]

        q_np = query_states[0, :, 0, :].to(torch.float32).cpu().numpy()
        scale = 1.0 / math.sqrt(self.head_dim)

        selected_k_blocks = []
        selected_v_blocks = []

        if cand_bids:
            k_val = max(1, int(math.ceil(len(cand_bids) * (self.top_k_pct / 100.0))))
            cand_ids = [bid for bid, _ in cand_bids]
            act_tokens_list = [actual_tokens for _, actual_tokens in cand_bids]

            if self.enable_computational_storage and hasattr(self.backend, "compute_topk_filter"):
                # Phase 7 Computational Storage:
                # Key scoring and Top-K filtering run directly inside the storage environment.
                # Candidate Key pages NEVER cross the storage bus to host!
                t_score_start = time.perf_counter()
                top_bids = self.backend.compute_topk_filter(
                    layer_idx=l_idx,
                    cand_bids=cand_bids,
                    query=q_np,
                    top_k=k_val,
                    scale=scale,
                    q_heads=q_np.shape[0],
                    kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                )
                self.timings["topk_scoring_s"] += time.perf_counter() - t_score_start

                # Record data movement:
                # Candidate Keys transferred to host = 0 bytes!
                self.topk_metadata_bytes_to_host += len(top_bids) * 12

                win_bids = [bid for _, bid, _ in top_bids]

                if self.enable_async_pipeline:
                    # Phase 8: Contiguous 8 KiB KV block retrieval
                    t_blk_start = time.perf_counter()
                    if hasattr(self.backend, "read_block_batch"):
                        loaded_blocks = self.backend.read_block_batch(l_idx, win_bids)
                        loaded_k_pages = {bid: loaded_blocks[bid][0] for bid in win_bids}
                        loaded_v_pages = {bid: loaded_blocks[bid][1] for bid in win_bids}
                    else:
                        loaded_k_pages = {bid: self.backend.read_key_page(l_idx, bid) for bid in win_bids}
                        loaded_v_pages = {bid: self.backend.read_value_page(l_idx, bid) for bid in win_bids}
                    blk_elapsed = time.perf_counter() - t_blk_start
                    self.timings["winning_v_reads_s"] += blk_elapsed / 2.0
                    self.timings["candidate_k_reads_s"] += blk_elapsed / 2.0
                    page_bytes = getattr(self, "page_bytes", 4096)
                    self.winning_v_bytes_to_host += len(win_bids) * page_bytes
                    self.winning_k_bytes_to_host += len(win_bids) * page_bytes

                    # Inter-layer pipelined async prefetch for Layer L+1:
                    # Dispatched after current layer blocks are read so background storage I/O
                    # overlaps with host attention matmul and MLP computation
                    t_pref_start = time.perf_counter()
                    if self.enable_prefetch and hasattr(self.backend, "predict_and_prefetch") and cand_bids:
                        self.backend.predict_and_prefetch(current_layer_id=l_idx, current_block_ids=win_bids, async_mode=True)
                    self.timings["prefetch_s"] += time.perf_counter() - t_pref_start
                else:
                    # Existing Phase 7 execution path
                    # Inter-layer speculative prefetch for Layer L+1:
                    t_pref_start = time.perf_counter()
                    if self.enable_prefetch and hasattr(self.backend, "predict_and_prefetch") and cand_bids:
                        winning_bids = [bid for _, bid, _ in top_bids]
                        self.backend.predict_and_prefetch(current_layer_id=l_idx, current_block_ids=winning_bids)
                    self.timings["prefetch_s"] += time.perf_counter() - t_pref_start

                    # Host retrieves winning blocks over PCIe (TOPK_FETCH):
                    # Fetch winning V pages
                    t_v_start = time.perf_counter()
                    if hasattr(self.backend, "read_value_page_batch"):
                        loaded_v_pages = self.backend.read_value_page_batch(l_idx, win_bids)
                    else:
                        loaded_v_pages = {bid: self.backend.read_value_page(l_idx, bid) for bid in win_bids}
                    self.timings["winning_v_reads_s"] += time.perf_counter() - t_v_start
                    page_bytes = getattr(self, "page_bytes", 4096)
                    self.winning_v_bytes_to_host += len(win_bids) * page_bytes

                    # Fetch winning K pages (only for winning blocks, NOT candidate blocks!)
                    t_k_start = time.perf_counter()
                    if hasattr(self.backend, "read_key_page_batch"):
                        loaded_k_pages = self.backend.read_key_page_batch(l_idx, win_bids)
                    else:
                        loaded_k_pages = {bid: self.backend.read_key_page(l_idx, bid) for bid in win_bids}
                    self.timings["candidate_k_reads_s"] += time.perf_counter() - t_k_start
                    self.winning_k_bytes_to_host += len(win_bids) * page_bytes

            else:
                # Host-side candidate streaming (Phase 5/6 baseline path)
                # Optimization C: Batch candidate Key page reads from storage
                t_k_start = time.perf_counter()
                if hasattr(self.backend, "read_key_page_batch"):
                    loaded_k_pages = self.backend.read_key_page_batch(l_idx, cand_ids)
                    k_blocks_list = [loaded_k_pages[bid] for bid in cand_ids]
                else:
                    loaded_k_pages = {}
                    k_blocks_list = []
                    for bid, actual_tokens in cand_bids:
                        k_blk = self.backend.read_key_page(l_idx, bid)
                        loaded_k_pages[bid] = k_blk
                        k_blocks_list.append(k_blk)
                self.timings["candidate_k_reads_s"] += time.perf_counter() - t_k_start
                page_bytes = getattr(self, "page_bytes", 4096)
                self.candidate_k_bytes_to_host += len(cand_ids) * page_bytes

                t_score_start = time.perf_counter()
                if self.kernel.is_available() and hasattr(self.kernel._lib, "instorage_topk_filter_gqa_avx2"):
                    top_indices, top_scores = self.kernel.compute_topk_gqa(
                        query=q_np,
                        k_blocks=k_blocks_list,
                        actual_tokens=act_tokens_list,
                        top_k=k_val,
                        q_heads=q_np.shape[0],
                        kv_heads=k_blocks_list[0].shape[1],
                        head_dim=self.head_dim,
                    )
                    self.timings["topk_scoring_s"] += time.perf_counter() - t_score_start

                    t_sel_start = time.perf_counter()
                    top_bids = [(float(top_scores[i]), cand_bids[idx][0], cand_bids[idx][1]) for i, idx in enumerate(top_indices)]
                    self.timings["candidate_selection_s"] += time.perf_counter() - t_sel_start
                else:
                    scores = []
                    for bid, actual_tokens in cand_bids:
                        k_blk = loaded_k_pages[bid]
                        dots = np.einsum("hd,thd->th", q_np, k_blk[:, [h // 7 for h in range(14)], :]) * scale
                        max_score = float(np.max(dots[:actual_tokens]))
                        scores.append((max_score, bid, actual_tokens))
                    self.timings["topk_scoring_s"] += time.perf_counter() - t_score_start

                    t_sel_start = time.perf_counter()
                    scores.sort(key=lambda x: x[0], reverse=True)
                    top_bids = scores[:k_val]
                    self.timings["candidate_selection_s"] += time.perf_counter() - t_sel_start

                # Inter-layer speculative prefetch for Layer L+1:
                t_pref_start = time.perf_counter()
                if self.enable_prefetch and hasattr(self.backend, "predict_and_prefetch") and cand_bids:
                    winning_bids = [bid for _, bid, _ in top_bids]
                    self.backend.predict_and_prefetch(current_layer_id=l_idx, current_block_ids=winning_bids)
                self.timings["prefetch_s"] += time.perf_counter() - t_pref_start

                # 2. Host retrieves winning blocks over PCIe (TOPK_FETCH)
                # Optimization C: Batch winning Value page reads from storage
                t_v_start = time.perf_counter()
                win_bids = [bid for _, bid, _ in top_bids]
                if hasattr(self.backend, "read_value_page_batch"):
                    loaded_v_pages = self.backend.read_value_page_batch(l_idx, win_bids)
                else:
                    loaded_v_pages = {bid: self.backend.read_value_page(l_idx, bid) for bid in win_bids}
                self.timings["winning_v_reads_s"] += time.perf_counter() - t_v_start
                page_bytes = getattr(self, "page_bytes", 4096)
                self.winning_v_bytes_to_host += len(win_bids) * page_bytes

            # Optimization B: Reuse Key pages already loaded during scoring, eliminating duplicate reads
            t_rec_start = time.perf_counter()
            for _, bid, actual_tokens in top_bids:
                v_blk = loaded_v_pages[bid]
                k_blk = loaded_k_pages[bid]
                target_dtype = ld["sink_k"].dtype
                target_device = ld["sink_k"].device
                k_t = torch.from_numpy(k_blk[:actual_tokens]).to(dtype=target_dtype, device=target_device).permute(1, 0, 2).unsqueeze(0)
                v_t = torch.from_numpy(v_blk[:actual_tokens]).to(dtype=target_dtype, device=target_device).permute(1, 0, 2).unsqueeze(0)
                selected_k_blocks.append(k_t)
                selected_v_blocks.append(v_t)
            self.timings["tensor_recon_s"] += time.perf_counter() - t_rec_start

        # 3. Concatenate active working set: Sinks + Top-k + Recent Window
        t_cat_start = time.perf_counter()
        parts_k = [ld["sink_k"]] + selected_k_blocks + [ld["recent_k"]]
        parts_v = [ld["sink_v"]] + selected_v_blocks + [ld["recent_v"]]

        active_k = torch.cat(parts_k, dim=2)
        active_v = torch.cat(parts_v, dim=2)
        self.timings["active_concat_s"] += time.perf_counter() - t_cat_start
        return active_k, active_v

    def get_memory_stats(self) -> Dict[str, Any]:
        """Calculates active DRAM memory vs full offloaded cache size."""
        if not self.layer_data:
            return {"active_dram_mb": 0.0, "total_kv_mb": 0.0, "offload_pct": 0.0}

        # Sample layer 0
        ld = self.layer_data[0]
        total_tokens = ld["total_tokens"]
        cand_bids = ld["candidate_blocks"]
        k_val = max(1, int(math.ceil(len(cand_bids) * (self.top_k_pct / 100.0)))) if cand_bids else 0

        # Active DRAM tokens = sinks (4) + recent (16) + selected Top-k (k_val * 16)
        active_tokens = self.sink_tokens + self.recent_tokens + (k_val * self.tokens_per_block)

        num_kv_heads = getattr(self, "num_kv_heads", 2)
        bytes_per_tok_all_layers = num_kv_heads * self.head_dim * 4 * 2 * self.num_layers
        total_kv_bytes = total_tokens * bytes_per_tok_all_layers
        active_dram_bytes = active_tokens * bytes_per_tok_all_layers

        total_kv_mb = total_kv_bytes / (1024.0 * 1024.0)
        active_dram_mb = active_dram_bytes / (1024.0 * 1024.0)
        offload_pct = max(0.0, 1.0 - (active_dram_bytes / max(1.0, float(total_kv_bytes)))) * 100.0

        return {
            "active_dram_mb": active_dram_mb,
            "total_kv_mb": total_kv_mb,
            "offload_pct": offload_pct,
            "active_tokens_per_layer": active_tokens,
            "total_tokens_per_layer": total_tokens,
        }


def run_baseline_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    decode_tokens: int = 16,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs genuine Qwen baseline inference using standard in-memory DynamicCache.
    
    Measures ONLY the generation execution interval (16 decode steps).
    """
    torch.manual_seed(seed)
    gc.collect()
    rss_before = get_current_rss_mb()

    # 1. Prefill step (excluded from decode timer)
    with torch.no_grad():
        prefill_out = model(input_ids=input_ids, use_cache=True)
    pkv = prefill_out.past_key_values
    next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)

    generated_tokens = [next_token.item()]
    step_logits = [prefill_out.logits[:, -1, :].clone()]

    # 2. Generation execution interval (measured strictly with high-res RSS sampler)
    sampler = ProcessMemorySampler(sample_interval_s=0.002)
    sampler.start()
    t_start = time.perf_counter()
    for step in range(1, decode_tokens):
        with torch.no_grad():
            step_out = model(input_ids=next_token, past_key_values=pkv, use_cache=True)
        pkv = step_out.past_key_values
        next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())
        step_logits.append(step_out.logits[:, -1, :].clone())
    t_end = time.perf_counter()
    proc_mem = sampler.stop()

    wall_time = t_end - t_start
    tps = len(generated_tokens) / max(1e-6, wall_time)

    # Dynamic KV-cache computation based on model architecture
    num_layers = getattr(model.config, "num_hidden_layers", 24)
    num_kv_heads = getattr(model.config, "num_key_value_heads", getattr(model.config, "num_attention_heads", 2))
    head_dim = getattr(model.config, "head_dim", 64)
    bytes_per_elem = 4 if getattr(model, "dtype", torch.float32) == torch.float32 else 2
    total_tokens = input_ids.shape[1] + decode_tokens
    total_kv_bytes = total_tokens * num_layers * num_kv_heads * head_dim * bytes_per_elem * 2
    total_kv_mb = total_kv_bytes / (1024.0 * 1024.0)

    generated_text = tokenizer.decode(generated_tokens)

    return {
        "mode": "BASELINE",
        "wall_time_s": wall_time,
        "generated_tokens": len(generated_tokens),
        "tokens_per_second": tps,
        "min_rss_mb": proc_mem["min_rss_mb"],
        "avg_rss_mb": proc_mem["avg_rss_mb"],
        "peak_rss_mb": proc_mem["peak_rss_mb"],
        "std_rss_mb": proc_mem["std_rss_mb"],
        "rss_sample_count": proc_mem["sample_count"],
        "rss_increment_mb": proc_mem["peak_rss_mb"] - rss_before,
        "kv_memory_mb": total_kv_mb,
        "kv_offloaded_pct": 0.0,
        "kv_blocks_read": 0,
        "storage_backend": "In-Memory DynamicCache (Host DRAM)",
        "storage_bytes_read": 0,
        "storage_requests": 0,
        "token_ids": generated_tokens,
        "generated_text": generated_text,
        "final_logits": step_logits[-1].cpu().numpy(),
    }


def run_aissd_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    decode_tokens: int = 16,
    top_k_pct: float = 10.0,
    seed: int = 42,
    storage_backend: Optional[Any] = None,
    enable_prefetch: bool = True,
    enable_computational_storage: bool = False,
    enable_async_pipeline: bool = False,
) -> Dict[str, Any]:
    """Runs genuine Qwen inference where KV access during decode executes the AI-SSD path.
    
    Measures ONLY the generation execution interval (16 decode steps).
    """
    torch.manual_seed(seed)
    gc.collect()
    rss_before = get_current_rss_mb()

    if storage_backend is None:
        backend = create_default_storage_backend(
            enable_prefetch=enable_prefetch,
            enable_async_pipeline=enable_async_pipeline,
        )
    else:
        backend = storage_backend

    kv_mgr = AISSDKVManager(
        backend,
        top_k_pct=top_k_pct,
        enable_computational_storage=enable_computational_storage,
        enable_prefetch=enable_prefetch,
        enable_async_pipeline=enable_async_pipeline,
    )

    model_timings = {
        "qkv_proj_s": 0.0,
        "rope_s": 0.0,
        "attn_matmul_s": 0.0,
        "out_proj_s": 0.0,
        "mlp_and_norm_s": 0.0,
        "bookkeeping_s": 0.0,
    }
    attn_forward_durations: List[float] = []

    # Wrap layer attention forward passes
    orig_forwards = {}
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        orig_forwards[i] = attn.forward

        def make_aissd_forward(layer_idx: int, original_fwd: Any):
            def forward(hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor] = None, past_key_values: Optional[Any] = None, **kwargs):
                if not kv_mgr.is_active or hidden_states.shape[1] > 1:
                    return original_fwd(hidden_states, position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)

                t_attn_fwd_start = time.perf_counter()
                attn_module = model.model.layers[layer_idx].self_attn
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_module.head_dim)

                t_qkv = time.perf_counter()
                q_raw = attn_module.q_proj(hidden_states).view(hidden_shape)
                k_raw = attn_module.k_proj(hidden_states).view(hidden_shape)
                if hasattr(attn_module, "q_norm"):
                    q_raw = attn_module.q_norm(q_raw)
                if hasattr(attn_module, "k_norm"):
                    k_raw = attn_module.k_norm(k_raw)

                q = q_raw.transpose(1, 2)
                k = k_raw.transpose(1, 2)
                v = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                model_timings["qkv_proj_s"] += time.perf_counter() - t_qkv

                t_rope = time.perf_counter()
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                model_timings["rope_s"] += time.perf_counter() - t_rope

                # Execute genuine AI-SSD path
                kv_mgr.append_new_token(layer_idx, k, v)
                act_k, act_v = kv_mgr.select_and_fetch_active_kv(layer_idx, q)

                # Attention computation on retrieved blocks (Optimized Native GQA SDPA)
                t_attn = time.perf_counter()
                scaling = attn_module.scaling
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, act_k, act_v, scale=scaling, enable_gqa=True
                )
                out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                model_timings["attn_matmul_s"] += time.perf_counter() - t_attn

                t_out = time.perf_counter()
                out = attn_module.o_proj(out)
                model_timings["out_proj_s"] += time.perf_counter() - t_out

                attn_forward_durations.append(time.perf_counter() - t_attn_fwd_start)
                return out, None

            return forward

        attn.forward = make_aissd_forward(i, orig_forwards[i])

    total_model_forward_s = 0.0
    try:
        # 1. Prefill step
        with torch.no_grad():
            prefill_out = model(input_ids=input_ids, use_cache=True)
        pkv_prefill = prefill_out.past_key_values
        kv_mgr.init_from_prefill(pkv_prefill)

        # Reset backend counters and KV manager timers to measure strictly decode traffic
        backend.reset_stats()
        kv_mgr.reset_timings()
        for k in model_timings:
            model_timings[k] = 0.0
        attn_forward_durations.clear()

        next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens = [next_token.item()]
        step_logits = [prefill_out.logits[:, -1, :].clone()]
        cur_seq_len = input_ids.shape[1]

        # Phase B: True Host-RAM Offload - Release prefill activations and original unpruned KV cache!
        del prefill_out
        del pkv_prefill
        gc.collect()

        # 2. Generation execution interval (measured strictly with high-res RSS sampler)
        sampler = ProcessMemorySampler(sample_interval_s=0.002)
        sampler.start()
        t_start = time.perf_counter()
        for step in range(1, decode_tokens):
            t_bk = time.perf_counter()
            pos_ids = torch.tensor([[cur_seq_len + step - 1]], device=input_ids.device)
            model_timings["bookkeeping_s"] += time.perf_counter() - t_bk

            t_step = time.perf_counter()
            with torch.no_grad():
                step_out = model(input_ids=next_token, position_ids=pos_ids, use_cache=False)
            total_model_forward_s += time.perf_counter() - t_step

            t_bk = time.perf_counter()
            next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            step_logits.append(step_out.logits[:, -1, :].clone())
            model_timings["bookkeeping_s"] += time.perf_counter() - t_bk
        t_end = time.perf_counter()
        proc_mem = sampler.stop()

    finally:
        # Always restore original forwards
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.forward = orig_forwards[i]

    wall_time = t_end - t_start
    tps = len(generated_tokens) / max(1e-6, wall_time)
    mem_stats = kv_mgr.get_memory_stats()

    # Calculate MLP and LayerNorm time as remaining model forward duration
    total_attn_fwd_s = sum(attn_forward_durations)
    model_timings["mlp_and_norm_s"] = max(0.0, total_model_forward_s - total_attn_fwd_s)

    if hasattr(backend, "buffer_capacity_blocks") or "Prefetch" in backend.__class__.__name__:
        storage_backend_name = "P1 -> P3 (DRAM Staging + Speculative Prefetch) -> P2 (Multi-Channel Flash FTL)"
    elif getattr(backend, "CLASSIFICATION", None) == "ANALYTICAL":
        storage_backend_name = "Person 2 Multi-Channel Flash FTL (Tensor-Aware)"
    else:
        storage_backend_name = "AI-SSD BlockStore (Controller Flash Buffer + PCIe Fetch)"

    generated_text = tokenizer.decode(generated_tokens)

    # Critical-path accounting reconciliation
    visible_storage_s = kv_mgr.timings["candidate_k_reads_s"] + kv_mgr.timings["winning_v_reads_s"]
    visible_compute_s = (
        model_timings["qkv_proj_s"] + model_timings["rope_s"] +
        kv_mgr.timings["topk_scoring_s"] + kv_mgr.timings["candidate_selection_s"] +
        kv_mgr.timings["prefetch_s"] + kv_mgr.timings["tensor_recon_s"] +
        kv_mgr.timings["active_concat_s"] + model_timings["attn_matmul_s"] +
        model_timings["out_proj_s"] + model_timings["mlp_and_norm_s"] +
        model_timings["bookkeeping_s"]
    )
    critical_path_total_s = visible_storage_s + visible_compute_s
    reconciliation_error_pct = (abs(wall_time - critical_path_total_s) / max(1e-6, wall_time)) * 100.0

    raw_storage_time_s = getattr(backend, "raw_storage_time_s", 0.0)
    overlap_hidden_s = max(0.0, raw_storage_time_s - visible_storage_s)

    # Compile unified timing breakdown across all measured components
    timing_breakdown = {
        "qkv_proj_s": round(model_timings["qkv_proj_s"], 4),
        "rope_s": round(model_timings["rope_s"], 4),
        "candidate_k_reads_s": round(kv_mgr.timings["candidate_k_reads_s"], 4),
        "topk_scoring_s": round(kv_mgr.timings["topk_scoring_s"], 4),
        "candidate_selection_s": round(kv_mgr.timings["candidate_selection_s"], 4),
        "prefetch_s": round(kv_mgr.timings["prefetch_s"], 4),
        "winning_v_reads_s": round(kv_mgr.timings["winning_v_reads_s"], 4),
        "tensor_recon_s": round(kv_mgr.timings["tensor_recon_s"], 4),
        "active_concat_s": round(kv_mgr.timings["active_concat_s"], 4),
        "attn_matmul_s": round(model_timings["attn_matmul_s"], 4),
        "out_proj_s": round(model_timings["out_proj_s"], 4),
        "mlp_and_norm_s": round(model_timings["mlp_and_norm_s"], 4),
        "bookkeeping_s": round(model_timings["bookkeeping_s"], 4),
        "visible_storage_s": round(visible_storage_s, 4),
        "visible_compute_s": round(visible_compute_s, 4),
        "critical_path_total_s": round(critical_path_total_s, 4),
        "reconciliation_error_pct": round(reconciliation_error_pct, 4),
        "raw_storage_time_s": round(raw_storage_time_s, 4),
        "overlap_hidden_s": round(overlap_hidden_s, 4),
        "total_measured_s": round(critical_path_total_s, 4),
        "wall_time_s": round(wall_time, 4),
    }

    result = {
        "mode": "AI-SSD",
        "wall_time_s": wall_time,
        "generated_tokens": len(generated_tokens),
        "tokens_per_second": tps,
        "min_rss_mb": proc_mem["min_rss_mb"],
        "avg_rss_mb": proc_mem["avg_rss_mb"],
        "peak_rss_mb": proc_mem["peak_rss_mb"],
        "std_rss_mb": proc_mem["std_rss_mb"],
        "rss_sample_count": proc_mem["sample_count"],
        "rss_increment_mb": proc_mem["peak_rss_mb"] - rss_before,
        "kv_memory_mb": mem_stats["active_dram_mb"],
        "kv_offloaded_pct": mem_stats["offload_pct"],
        "kv_blocks_read": getattr(backend, "blocks_read", 0),
        "storage_backend": storage_backend_name,
        "storage_bytes_read": getattr(backend, "bytes_read", 0),
        "storage_requests": getattr(backend, "requests", 0),
        "storage_batches": getattr(backend, "storage_batches", 0),
        "avg_batch_size": (getattr(backend, "requests", 0) / max(1, getattr(backend, "storage_batches", 1))) if getattr(backend, "storage_batches", 0) > 0 else 1.0,
        "token_ids": generated_tokens,
        "generated_text": generated_text,
        "final_logits": step_logits[-1].cpu().numpy(),
        "timing_breakdown": timing_breakdown,
        "enable_computational_storage": enable_computational_storage,
        "enable_async_pipeline": enable_async_pipeline,
        "visible_storage_s": visible_storage_s,
        "visible_compute_s": visible_compute_s,
        "critical_path_total_s": critical_path_total_s,
        "reconciliation_error_pct": reconciliation_error_pct,
        "raw_storage_time_s": raw_storage_time_s,
        "overlap_hidden_s": overlap_hidden_s,
        "candidate_k_bytes_to_host": kv_mgr.candidate_k_bytes_to_host,
        "winning_k_bytes_to_host": kv_mgr.winning_k_bytes_to_host,
        "winning_v_bytes_to_host": kv_mgr.winning_v_bytes_to_host,
        "topk_metadata_bytes_to_host": kv_mgr.topk_metadata_bytes_to_host,
        "total_data_movement_bytes": (
            kv_mgr.candidate_k_bytes_to_host
            + kv_mgr.winning_k_bytes_to_host
            + kv_mgr.winning_v_bytes_to_host
            + kv_mgr.topk_metadata_bytes_to_host
        ),
    }

    if hasattr(backend, "get_telemetry"):
        telem = backend.get_telemetry()
        result["telemetry"] = telem
        if "storage_backend" in telem:
            result["storage_telemetry"] = telem["storage_backend"]
            result["prefetch_telemetry"] = {k: v for k, v in telem.items() if k != "storage_backend"}
        if "nvme_telemetry" in telem:
            result["nvme_telemetry"] = telem["nvme_telemetry"]
        elif "storage_backend" in telem and "nvme_telemetry" in telem["storage_backend"]:
            result["nvme_telemetry"] = telem["storage_backend"]["nvme_telemetry"]

    return result
