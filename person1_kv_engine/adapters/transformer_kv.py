"""Transformer KV State Provider for standard and GQA/MQA/MHA Transformer models.

Implements model-agnostic blockization, offloading, top-k retrieval,
and tensor reconstruction for Transformer KV caches.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import math
import time
import numpy as np
import torch

from person1_kv_engine.adapters.descriptor import (
    StateDescriptor,
    StateType,
    ResidencyTier,
    ModelArchitectureConfig,
)
from person1_kv_engine.adapters.state_provider import StateProvider
from person1_kv_engine.c_kernel.kernel_binding import get_native_c_kernel


class TransformerKVStateProvider(StateProvider):
    """Generic StateProvider for causal Transformer architectures with separable KV cache."""

    def __init__(
        self,
        config: ModelArchitectureConfig,
        backend: Any,
        sink_tokens: int = 4,
        recent_tokens: int = 16,
        tokens_per_block: int = 16,
        top_k_pct: float = 10.0,
        enable_computational_storage: bool = False,
        enable_prefetch: bool = True,
        enable_async_pipeline: bool = False,
    ):
        super().__init__(
            config=config,
            backend=backend,
            tokens_per_block=tokens_per_block,
            top_k_pct=top_k_pct,
            enable_computational_storage=enable_computational_storage,
            enable_prefetch=enable_prefetch,
            enable_async_pipeline=enable_async_pipeline,
        )
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.num_layers = config.num_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.layer_data: Dict[int, Dict[str, Any]] = {}
        self.kernel = get_native_c_kernel()

        # Telemetry / data movement counters
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
        for k in self.timings:
            self.timings[k] = 0.0
        self.candidate_k_bytes_to_host = 0
        self.winning_k_bytes_to_host = 0
        self.winning_v_bytes_to_host = 0
        self.topk_metadata_bytes_to_host = 0

    def inspect_state(self, past_key_values: Any) -> List[StateDescriptor]:
        descriptors = []
        if hasattr(past_key_values, "layers"):
            layers_count = len(past_key_values.layers)
            sample_k = past_key_values.layers[0].keys
        elif hasattr(past_key_values, "key_cache"):
            layers_count = len(past_key_values.key_cache)
            sample_k = past_key_values.key_cache[0]
        else:
            layers_count = len(past_key_values)
            sample_k = past_key_values[0][0]

        dtype_str = "fp16" if sample_k.dtype in (torch.float16, torch.bfloat16) else "fp32"
        elem_bytes = 2 if dtype_str == "fp16" else 4
        page_bytes = self.tokens_per_block * self.num_kv_heads * self.head_dim * elem_bytes

        for l_idx in range(layers_count):
            descriptors.append(
                StateDescriptor(
                    state_type=StateType.ATTENTION_KV,
                    layer_id=l_idx,
                    residency=ResidencyTier.AI_SSD,
                    tensor_shape=(self.tokens_per_block, self.num_kv_heads, self.head_dim),
                    dtype=dtype_str,
                    byte_size=page_bytes * 2,  # K page + V page
                    layout="tokens_heads_dim",
                    block_size=self.tokens_per_block,
                )
            )
        return descriptors

    def init_from_prefill(self, past_key_values: Any) -> None:
        if isinstance(past_key_values, dict):
            # Dict mapping layer_idx -> (keys, values)
            first_layer_idx = next(iter(past_key_values.keys()))
            first_k = past_key_values[first_layer_idx][0]
            layers_dict = past_key_values
        elif hasattr(past_key_values, "layers"):
            self.num_layers = len(past_key_values.layers)
            layers_dict = {}
            first_k = None
            for idx, lyr in enumerate(past_key_values.layers):
                if hasattr(lyr, "keys") and lyr.keys is not None:
                    layers_dict[idx] = (lyr.keys, lyr.values)
                    if first_k is None:
                        first_k = lyr.keys
            if first_k is None:
                raise ValueError("No attention layers with keys found in past_key_values")
        elif hasattr(past_key_values, "key_cache"):
            self.num_layers = len(past_key_values.key_cache)
            first_k = past_key_values.key_cache[0]
            layers_dict = {i: (past_key_values.key_cache[i], past_key_values.value_cache[i]) for i in range(self.num_layers)}
        else:
            self.num_layers = len(past_key_values)
            first_k = past_key_values[0][0]
            layers_dict = {i: (past_key_values[i][0], past_key_values[i][1]) for i in range(self.num_layers)}

        self.num_kv_heads = first_k.shape[1]
        self.head_dim = first_k.shape[3]
        self.dtype = first_k.dtype
        self.bytes_per_elem = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        self.page_bytes = self.tokens_per_block * self.num_kv_heads * self.head_dim * self.bytes_per_elem
        np_dtype = np.float16 if self.bytes_per_elem == 2 else np.float32

        for l_idx, (k_tensor, v_tensor) in layers_dict.items():
            seq_len = k_tensor.shape[2]

            # 1. Attention Sinks: retain in host DRAM
            if seq_len <= (self.sink_tokens + self.recent_tokens):
                sink_k = k_tensor.detach().clone()
                sink_v = v_tensor.detach().clone()
                recent_k = torch.empty((k_tensor.shape[0], k_tensor.shape[1], 0, k_tensor.shape[3]), dtype=k_tensor.dtype, device=k_tensor.device)
                recent_v = torch.empty((v_tensor.shape[0], v_tensor.shape[1], 0, v_tensor.shape[3]), dtype=v_tensor.dtype, device=v_tensor.device)
                cand_bids = []
            else:
                sink_k = k_tensor[:, :, :self.sink_tokens, :].detach().clone()
                sink_v = v_tensor[:, :, :self.sink_tokens, :].detach().clone()
                recent_k = k_tensor[:, :, -self.recent_tokens:, :].detach().clone()
                recent_v = v_tensor[:, :, -self.recent_tokens:, :].detach().clone()

                # 3. Offload historical blocks
                hist_start = self.sink_tokens
                hist_end = seq_len - self.recent_tokens
                k_np = k_tensor[0].detach().cpu().numpy()  # [num_kv_heads, seq_len, head_dim]
                v_np = v_tensor[0].detach().cpu().numpy()
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

    def append_new_token(self, layer_idx: int, *state_tensors: Any) -> None:
        k_tok, v_tok = state_tensors[0], state_tensors[1]
        ld = self.layer_data[layer_idx]
        ld["recent_k"] = torch.cat([ld["recent_k"][:, :, 1:, :], k_tok], dim=2)
        ld["recent_v"] = torch.cat([ld["recent_v"][:, :, 1:, :], v_tok], dim=2)
        ld["total_tokens"] += 1

    def select_and_fetch_active_state(
        self,
        layer_idx: int,
        query: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if query is None:
            raise ValueError("query tensor must be provided for attention KV retrieval")

        ld = self.layer_data[layer_idx]
        cand_bids = ld["candidate_blocks"]
        q_np = query[0, :, 0, :].detach().to(torch.float32).cpu().numpy()
        scale = 1.0 / math.sqrt(self.head_dim)

        selected_k_blocks = []
        selected_v_blocks = []

        if cand_bids:
            k_val = max(1, int(math.ceil(len(cand_bids) * (self.top_k_pct / 100.0))))
            cand_ids = [bid for bid, _ in cand_bids]
            act_tokens_list = [actual_tokens for _, actual_tokens in cand_bids]

            if self.enable_computational_storage and hasattr(self.backend, "compute_topk_filter"):
                t_score_start = time.perf_counter()
                top_bids = self.backend.compute_topk_filter(
                    layer_idx=layer_idx,
                    cand_bids=cand_bids,
                    query=q_np,
                    top_k=k_val,
                    scale=scale,
                    q_heads=q_np.shape[0],
                    kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                )
                self.timings["topk_scoring_s"] += time.perf_counter() - t_score_start
                self.topk_metadata_bytes_to_host += len(top_bids) * 12

                win_bids = [bid for _, bid, _ in top_bids]

                if self.enable_async_pipeline:
                    t_blk_start = time.perf_counter()
                    if hasattr(self.backend, "read_block_batch"):
                        loaded_blocks = self.backend.read_block_batch(layer_idx, win_bids)
                        loaded_k_pages = {bid: loaded_blocks[bid][0] for bid in win_bids}
                        loaded_v_pages = {bid: loaded_blocks[bid][1] for bid in win_bids}
                    else:
                        loaded_k_pages = {bid: self.backend.read_key_page(layer_idx, bid) for bid in win_bids}
                        loaded_v_pages = {bid: self.backend.read_value_page(layer_idx, bid) for bid in win_bids}
                    blk_elapsed = time.perf_counter() - t_blk_start
                    self.timings["winning_v_reads_s"] += blk_elapsed / 2.0
                    self.timings["candidate_k_reads_s"] += blk_elapsed / 2.0
                    page_bytes = getattr(self, "page_bytes", 4096)
                    self.winning_v_bytes_to_host += len(win_bids) * page_bytes
                    self.winning_k_bytes_to_host += len(win_bids) * page_bytes

                    t_pref_start = time.perf_counter()
                    if self.enable_prefetch and hasattr(self.backend, "predict_and_prefetch") and cand_bids:
                        self.backend.predict_and_prefetch(current_layer_id=layer_idx, current_block_ids=win_bids, async_mode=True)
                    self.timings["prefetch_s"] += time.perf_counter() - t_pref_start
                else:
                    t_pref_start = time.perf_counter()
                    if self.enable_prefetch and hasattr(self.backend, "predict_and_prefetch") and cand_bids:
                        winning_bids = [bid for _, bid, _ in top_bids]
                        self.backend.predict_and_prefetch(current_layer_id=layer_idx, current_block_ids=winning_bids)
                    self.timings["prefetch_s"] += time.perf_counter() - t_pref_start

                    t_v_start = time.perf_counter()
                    if hasattr(self.backend, "read_value_page_batch"):
                        loaded_v_pages = self.backend.read_value_page_batch(layer_idx, win_bids)
                    else:
                        loaded_v_pages = {bid: self.backend.read_value_page(layer_idx, bid) for bid in win_bids}
                    self.timings["winning_v_reads_s"] += time.perf_counter() - t_v_start
                    page_bytes = getattr(self, "page_bytes", 4096)
                    self.winning_v_bytes_to_host += len(win_bids) * page_bytes

                    t_k_start = time.perf_counter()
                    if hasattr(self.backend, "read_key_page_batch"):
                        loaded_k_pages = self.backend.read_key_page_batch(layer_idx, win_bids)
                    else:
                        loaded_k_pages = {bid: self.backend.read_key_page(layer_idx, bid) for bid in win_bids}
                    self.timings["candidate_k_reads_s"] += time.perf_counter() - t_k_start
                    self.winning_k_bytes_to_host += len(win_bids) * page_bytes
            else:
                t_k_start = time.perf_counter()
                if hasattr(self.backend, "read_key_page_batch"):
                    loaded_k_pages = self.backend.read_key_page_batch(layer_idx, cand_ids)
                    k_blocks_list = [loaded_k_pages[bid] for bid in cand_ids]
                else:
                    loaded_k_pages = {}
                    k_blocks_list = []
                    for bid, actual_tokens in cand_bids:
                        k_blk = self.backend.read_key_page(layer_idx, bid)
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
                    gqa_ratio = self.config.gqa_ratio
                    for bid, actual_tokens in cand_bids:
                        k_blk = loaded_k_pages[bid]
                        dots = np.einsum("hd,thd->th", q_np, k_blk[:, [h // gqa_ratio for h in range(q_np.shape[0])], :]) * scale
                        max_score = float(np.max(dots[:actual_tokens]))
                        scores.append((max_score, bid, actual_tokens))
                    self.timings["topk_scoring_s"] += time.perf_counter() - t_score_start
                    t_sel_start = time.perf_counter()
                    scores.sort(key=lambda x: x[0], reverse=True)
                    top_bids = scores[:k_val]
                    self.timings["candidate_selection_s"] += time.perf_counter() - t_sel_start

                t_pref_start = time.perf_counter()
                if self.enable_prefetch and hasattr(self.backend, "predict_and_prefetch") and cand_bids:
                    winning_bids = [bid for _, bid, _ in top_bids]
                    self.backend.predict_and_prefetch(current_layer_id=layer_idx, current_block_ids=winning_bids)
                self.timings["prefetch_s"] += time.perf_counter() - t_pref_start

                t_v_start = time.perf_counter()
                win_bids = [bid for _, bid, _ in top_bids]
                if hasattr(self.backend, "read_value_page_batch"):
                    loaded_v_pages = self.backend.read_value_page_batch(layer_idx, win_bids)
                else:
                    loaded_v_pages = {bid: self.backend.read_value_page(layer_idx, bid) for bid in win_bids}
                self.timings["winning_v_reads_s"] += time.perf_counter() - t_v_start
                page_bytes = getattr(self, "page_bytes", 4096)
                self.winning_v_bytes_to_host += len(win_bids) * page_bytes

            t_rec_start = time.perf_counter()
            # Sort selected winning blocks by bid so temporal token ordering is preserved!
            sorted_top_bids = sorted(top_bids, key=lambda x: x[1])
            for _, bid, actual_tokens in sorted_top_bids:
                v_blk = loaded_v_pages[bid]
                k_blk = loaded_k_pages[bid]
                target_dtype = ld["sink_k"].dtype
                target_device = ld["sink_k"].device
                k_t = torch.from_numpy(k_blk[:actual_tokens]).to(dtype=target_dtype, device=target_device).permute(1, 0, 2).unsqueeze(0)
                v_t = torch.from_numpy(v_blk[:actual_tokens]).to(dtype=target_dtype, device=target_device).permute(1, 0, 2).unsqueeze(0)
                selected_k_blocks.append(k_t)
                selected_v_blocks.append(v_t)
            self.timings["tensor_recon_s"] += time.perf_counter() - t_rec_start

        t_cat_start = time.perf_counter()
        parts_k = [ld["sink_k"]] + selected_k_blocks + [ld["recent_k"]]
        parts_v = [ld["sink_v"]] + selected_v_blocks + [ld["recent_v"]]

        active_k = torch.cat(parts_k, dim=2)
        active_v = torch.cat(parts_v, dim=2)
        self.timings["active_concat_s"] += time.perf_counter() - t_cat_start
        return active_k, active_v

    def get_memory_stats(self) -> Dict[str, Any]:
        if not self.layer_data:
            return {"active_dram_mb": 0.0, "total_kv_mb": 0.0, "offload_pct": 0.0}

        first_layer_idx = next(iter(self.layer_data.keys()))
        ld = self.layer_data[first_layer_idx]
        total_tokens = ld["total_tokens"]
        cand_bids = ld["candidate_blocks"]
        k_val = max(1, int(math.ceil(len(cand_bids) * (self.top_k_pct / 100.0)))) if cand_bids else 0

        active_tokens = self.sink_tokens + self.recent_tokens + (k_val * self.tokens_per_block)
        bytes_per_elem = getattr(self, "bytes_per_elem", 4)
        num_attn_layers = len(self.layer_data)
        bytes_per_tok_all_layers = self.num_kv_heads * self.head_dim * bytes_per_elem * 2 * num_attn_layers
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
