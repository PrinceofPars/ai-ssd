#!/usr/bin/env python3
"""Detailed Micro-Profiling of AI-SSD V2 Live Inference Decode Loop.

Instruments every sub-component of the 16-step decode process:
1. Model forward outside attention (Embeddings, MLPs, Norms, Head)
2. Q/K/V Projection + RoPE
3. New token append to KV manager
4. Candidate Key page storage reads (P3/P2 backend)
5. Top-k scoring (np.einsum vs C kernel candidate)
6. Speculative prefetch prediction & trigger
7. Winning Key & Value page storage reads (P3/P2 backend)
8. NumPy -> Torch conversion & tensor permutations
9. Active working set concatenation (torch.cat)
10. Attention matmul, softmax, and output projection
11. Decode loop bookkeeping (sampling, logits clone)
"""

import sys
import os
from pathlib import Path
import time
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.aissd_inference import (
    AISSDKVManager,
    create_default_storage_backend,
    apply_rotary_pos_emb,
)
from scripts.real_inference_benchmark import build_prompt_for_length

class ProfileTimer:
    def __init__(self):
        self.timers: Dict[str, float] = {
            "qkv_proj_rope": 0.0,
            "append_token": 0.0,
            "topk_key_reads": 0.0,
            "topk_scoring_avx2": 0.0,
            "prefetch_predict": 0.0,
            "fetch_winning_pages": 0.0,
            "numpy_to_torch": 0.0,
            "concat_working_set": 0.0,
            "attn_matmul_oproj": 0.0,
            "other_layers_and_head": 0.0,
            "decode_bookkeeping": 0.0,
        }
        self.counts: Dict[str, int] = {k: 0 for k in self.timers}

    def record(self, key: str, duration: float):
        self.timers[key] += duration
        self.counts[key] += 1


def profile_aissd_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    decode_tokens: int = 16,
    top_k_pct: float = 10.0,
    enable_prefetch: bool = True,
    channels: int = 8,
    seed: int = 42,
) -> Tuple[Dict[str, float], float]:
    torch.manual_seed(seed)
    pt = ProfileTimer()

    backend = create_default_storage_backend(channels=channels, enable_prefetch=enable_prefetch)
    kv_mgr = AISSDKVManager(backend, top_k_pct=top_k_pct)

    orig_forwards = {}
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        orig_forwards[i] = attn.forward

        def make_profiled_forward(layer_idx: int, original_fwd: Any):
            def forward(hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor] = None, past_key_values: Optional[Any] = None, **kwargs):
                if not kv_mgr.is_active or hidden_states.shape[1] > 1:
                    return original_fwd(hidden_states, position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)

                attn_module = model.model.layers[layer_idx].self_attn
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_module.head_dim)

                # 1. QKV Proj + RoPE
                t0 = time.perf_counter()
                q = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                k = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                v = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                pt.record("qkv_proj_rope", time.perf_counter() - t0)

                # 2. Append new token
                t0 = time.perf_counter()
                kv_mgr.append_new_token(layer_idx, k, v)
                pt.record("append_token", time.perf_counter() - t0)

                # 3. Instrumented select_and_fetch_active_kv
                ld = kv_mgr.layer_data[layer_idx]
                cand_bids = ld["candidate_blocks"]
                q_np = q[0, :, 0, :].cpu().numpy()
                scale = 1.0 / math.sqrt(kv_mgr.head_dim)

                selected_k_blocks = []
                selected_v_blocks = []

                if cand_bids:
                    k_val = max(1, int(math.ceil(len(cand_bids) * (kv_mgr.top_k_pct / 100.0))))
                    scores = []
                    
                    # Read candidate Key pages (Optimization C: Batched)
                    cand_ids = [bid for bid, _ in cand_bids]
                    act_tokens_list = [actual_tokens for _, actual_tokens in cand_bids]
                    t_kr = time.perf_counter()
                    if hasattr(kv_mgr.backend, "read_key_page_batch"):
                        loaded_k_pages = kv_mgr.backend.read_key_page_batch(layer_idx, cand_ids)
                        k_blocks_list = [loaded_k_pages[bid] for bid in cand_ids]
                    else:
                        loaded_k_pages = {}
                        k_blocks_list = []
                        for bid, actual_tokens in cand_bids:
                            k_blk = kv_mgr.backend.read_key_page(layer_idx, bid)
                            loaded_k_pages[bid] = k_blk
                            k_blocks_list.append(k_blk)
                    pt.record("topk_key_reads", time.perf_counter() - t_kr)

                    # In-storage AVX2 Top-k scoring & selection
                    t_sc = time.perf_counter()
                    top_indices, top_scores = kv_mgr.kernel.compute_topk_gqa(
                        query=q_np,
                        k_blocks=k_blocks_list,
                        actual_tokens=act_tokens_list,
                        top_k=k_val,
                        q_heads=q_np.shape[0],
                        kv_heads=k_blocks_list[0].shape[1],
                        head_dim=kv_mgr.head_dim,
                    )
                    top_bids = [(float(top_scores[i]), cand_bids[idx][0], cand_bids[idx][1]) for i, idx in enumerate(top_indices)]
                    pt.record("topk_scoring_avx2", time.perf_counter() - t_sc)

                    # Speculative prefetch
                    t_pf = time.perf_counter()
                    if hasattr(kv_mgr.backend, "predict_and_prefetch") and cand_bids:
                        winning_bids = [bid for _, bid, _ in top_bids]
                        kv_mgr.backend.predict_and_prefetch(current_layer_id=layer_idx, current_block_ids=winning_bids)
                    pt.record("prefetch_predict", time.perf_counter() - t_pf)

                    # Fetch winning pages (Optimization C: Batched Value reads, Optimization B: reuse Key page)
                    t_fetch = time.perf_counter()
                    win_bids = [bid for _, bid, _ in top_bids]
                    if hasattr(kv_mgr.backend, "read_value_page_batch"):
                        loaded_v_pages = kv_mgr.backend.read_value_page_batch(layer_idx, win_bids)
                    else:
                        loaded_v_pages = {bid: kv_mgr.backend.read_value_page(layer_idx, bid) for bid in win_bids}
                    pt.record("fetch_winning_pages", time.perf_counter() - t_fetch)

                    for _, bid, actual_tokens in top_bids:
                        v_blk = loaded_v_pages[bid]
                        k_blk = loaded_k_pages[bid]
                        t_tc = time.perf_counter()
                        k_t = torch.from_numpy(k_blk[:actual_tokens]).permute(1, 0, 2).unsqueeze(0)
                        v_t = torch.from_numpy(v_blk[:actual_tokens]).permute(1, 0, 2).unsqueeze(0)
                        selected_k_blocks.append(k_t)
                        selected_v_blocks.append(v_t)
                        pt.record("numpy_to_torch", time.perf_counter() - t_tc)

                # Concatenate working set
                t_cat = time.perf_counter()
                parts_k = [ld["sink_k"]] + selected_k_blocks + [ld["recent_k"]]
                parts_v = [ld["sink_v"]] + selected_v_blocks + [ld["recent_v"]]
                act_k = torch.cat(parts_k, dim=2)
                act_v = torch.cat(parts_v, dim=2)
                pt.record("concat_working_set", time.perf_counter() - t_cat)

                # Attention matmul + o_proj
                t_mm = time.perf_counter()
                q_heads = attn_module.config.num_attention_heads
                kv_heads = attn_module.config.num_key_value_heads
                gqa = q_heads // kv_heads

                k_exp = act_k.repeat_interleave(gqa, dim=1) if gqa > 1 else act_k
                v_exp = act_v.repeat_interleave(gqa, dim=1) if gqa > 1 else act_v

                scaling = attn_module.scaling
                scores = torch.matmul(q, k_exp.transpose(2, 3)) * scaling
                weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

                out = torch.matmul(weights, v_exp)
                out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                out = attn_module.o_proj(out)
                pt.record("attn_matmul_oproj", time.perf_counter() - t_mm)
                return out, None

            return forward

        attn.forward = make_profiled_forward(i, orig_forwards[i])

    try:
        # Prefill
        with torch.no_grad():
            prefill_out = model(input_ids=input_ids, use_cache=True)
        pkv_prefill = prefill_out.past_key_values
        kv_mgr.init_from_prefill(pkv_prefill)

        next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens = [next_token.item()]

        t_start = time.perf_counter()
        for step in range(1, decode_tokens):
            t_step0 = time.perf_counter()
            with torch.no_grad():
                step_out = model(input_ids=next_token, past_key_values=pkv_prefill, use_cache=True)
            t_model_step = time.perf_counter() - t_step0

            t_bk = time.perf_counter()
            pkv_prefill = step_out.past_key_values
            next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            pt.record("decode_bookkeeping", time.perf_counter() - t_bk)
        t_end = time.perf_counter()
        total_decode_time = t_end - t_start

    finally:
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.forward = orig_forwards[i]

    # Calculate other layers by subtraction
    attn_sum = sum(pt.timers[k] for k in pt.timers if k not in ("other_layers_and_head", "decode_bookkeeping"))
    other_time = max(0.0, total_decode_time - attn_sum - pt.timers["decode_bookkeeping"])
    pt.timers["other_layers_and_head"] = other_time

    return pt.timers, total_decode_time


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Micro-profiler for AI-SSD decode loop")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = RealLLMEngine(model_name=args.model, device="cpu", dtype="FP32", num_threads=4)
    prompt = build_prompt_for_length(engine, target_tokens=args.context)
    inputs = engine.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print("=" * 70)
    print("        AI-SSD V2 DETAILED COMPONENT EXECUTION PROFILING")
    print("=" * 70)
    print(f"Model: {args.model} | Context: {args.context} | Decode: {args.decode} tokens | Threads: 4")
    print(f"Running {args.reps} profiling repetitions and averaging...")

    all_timers = []
    all_totals = []
    for rep in range(args.reps):
        timers, total = profile_aissd_decode(engine.model, engine.tokenizer, input_ids, decode_tokens=args.decode, seed=args.seed + rep)
        all_timers.append(timers)
        all_totals.append(total)

    avg_total = float(np.mean(all_totals))
    avg_timers = {k: float(np.mean([t[k] for t in all_timers])) for k in all_timers[0]}

    # Sort descending by latency
    sorted_components = sorted(avg_timers.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print(f"{'Component':<32} | {'Time (ms)':>10} | {'% of Total':>10} | {'Cumulative %':>12}")
    print("-" * 70)
    cum = 0.0
    for name, dur_s in sorted_components:
        ms = dur_s * 1000.0
        pct = (dur_s / avg_total) * 100.0
        cum += pct
        print(f"{name:<32} | {ms:>10.2f} | {pct:>9.1f}% | {cum:>11.1f}%")
    print("-" * 70)
    print(f"{'TOTAL DECODE TIME':<32} | {avg_total * 1000.0:>10.2f} | {100.0:>9.1f}% |")
    print(f"{'THROUGHPUT':<32} | {16.0 / avg_total:>10.2f} tok/s")
    print("=" * 70)


if __name__ == "__main__":
    main()
