#!/usr/bin/env python3
"""READ-ONLY Memory Accounting Audit for Qwen3-4B Live Inference Path.

Measures actual OS process memory via /proc/self/status and /proc/self/smaps_rollup
at every lifecycle checkpoint of Baseline and AI-SSD execution at Context=4096.

Zero analytical estimates. Zero modifications to production codebase.
"""

import sys
import os
import gc
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.aissd_inference import (
    create_default_storage_backend,
    run_baseline_decode,
    run_aissd_decode,
    AISSDKVManager,
)
from scripts.real_inference_benchmark import build_prompt_for_length


def get_proc_memory() -> Dict[str, float]:
    """Reads real memory telemetry from Linux /proc/self/status and smaps_rollup."""
    res = {
        "vm_rss_mb": 0.0,
        "rss_anon_mb": 0.0,
        "rss_file_mb": 0.0,
        "vm_peak_mb": 0.0,
        "pss_mb": 0.0,
    }
    # 1. /proc/self/status
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    res["vm_rss_mb"] = float(line.split()[1]) / 1024.0
                elif line.startswith("RssAnon:"):
                    res["rss_anon_mb"] = float(line.split()[1]) / 1024.0
                elif line.startswith("RssFile:"):
                    res["rss_file_mb"] = float(line.split()[1]) / 1024.0
                elif line.startswith("VmPeak:"):
                    res["vm_peak_mb"] = float(line.split()[1]) / 1024.0
    except Exception as e:
        pass

    # 2. /proc/self/smaps_rollup
    try:
        with open("/proc/self/smaps_rollup", "r") as f:
            for line in f:
                if line.startswith("Pss:"):
                    res["pss_mb"] = float(line.split()[1]) / 1024.0
    except Exception:
        pass

    return res


def inspect_storage_structures(backend: Any) -> Dict[str, Any]:
    """Audits the resident payload byte size in P2/P3 storage backend."""
    p3_payload_bytes = 0
    p3_meta_bytes = 0
    p3_staging_bytes = 0
    p2_storage_bytes = 0
    p2_access_log_len = 0

    cur = backend
    # If wrapped in P3 adapter:
    if hasattr(cur, "_block_payloads"):
        for k, v in cur._block_payloads.items():
            if "k" in v and isinstance(v["k"], np.ndarray):
                p3_payload_bytes += v["k"].nbytes
            if "v" in v and isinstance(v["v"], np.ndarray):
                p3_payload_bytes += v["v"].nbytes
            if "bytes" in v and isinstance(v["bytes"], (bytes, bytearray)):
                p3_payload_bytes += len(v["bytes"])
        p3_staging_bytes = cur.current_memory_bytes if hasattr(cur, "current_memory_bytes") else 0
        if hasattr(cur, "storage_backend"):
            cur = cur.storage_backend

    # If P2 backend:
    if hasattr(cur, "_storage"):
        for k, v in cur._storage.items():
            if "k" in v and isinstance(v["k"], np.ndarray):
                p2_storage_bytes += v["k"].nbytes
            if "v" in v and isinstance(v["v"], np.ndarray):
                p2_storage_bytes += v["v"].nbytes
            if "k_bytes" in v and isinstance(v["k_bytes"], (bytes, bytearray)):
                p2_storage_bytes += len(v["k_bytes"])
            if "v_bytes" in v and isinstance(v["v_bytes"], (bytes, bytearray)):
                p2_storage_bytes += len(v["v_bytes"])
        p2_access_log_len = len(cur._access_log) if hasattr(cur, "_access_log") else 0

    return {
        "p3_payload_mb": p3_payload_bytes / (1024.0 * 1024.0),
        "p3_staging_mb": p3_staging_bytes / (1024.0 * 1024.0),
        "p2_storage_mb": p2_storage_bytes / (1024.0 * 1024.0),
        "p2_access_log_entries": p2_access_log_len,
        "total_backend_resident_mb": (p3_payload_bytes + p2_storage_bytes) / (1024.0 * 1024.0),
    }


def audit_baseline(context_len: int = 4096, decode_tokens: int = 16):
    print("=" * 64)
    print("        AUDIT 1: BASELINE INFERENCE MEMORY LIFECYCLE")
    print("=" * 64)
    
    cp_start = get_proc_memory()
    print(f"[CHECKPOINT: process_start]       RSS: {cp_start['vm_rss_mb']:.1f} MB (Anon: {cp_start['rss_anon_mb']:.1f} MB)")

    engine = RealLLMEngine(model_name="Qwen/Qwen3-4B-Instruct-2507", device="cpu", dtype="float32", num_threads=4)
    cp_model = get_proc_memory()
    print(f"[CHECKPOINT: after_model_load]   RSS: {cp_model['vm_rss_mb']:.1f} MB (Anon: {cp_model['rss_anon_mb']:.1f} MB)")

    prompt = build_prompt_for_length(engine, target_tokens=context_len)
    inputs = engine.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Prefill step
    t0 = time.perf_counter()
    with torch.no_grad():
        prefill_out = engine.model(input_ids=input_ids, use_cache=True)
    t_prefill = time.perf_counter() - t0
    cp_prefill = get_proc_memory()
    print(f"[CHECKPOINT: after_prefill]      RSS: {cp_prefill['vm_rss_mb']:.1f} MB (Anon: {cp_prefill['rss_anon_mb']:.1f} MB, prefill time: {t_prefill:.1f}s)")

    pkv = prefill_out.past_key_values
    pkv_bytes = sum(l.keys.nelement() * 4 + l.values.nelement() * 4 for l in pkv.layers)
    print(f"  -> PyTorch Active KV size:     {pkv_bytes / (1024*1024):.2f} MB")

    cp_before_decode = get_proc_memory()
    print(f"[CHECKPOINT: before_decode]      RSS: {cp_before_decode['vm_rss_mb']:.1f} MB")

    # Autoregressive decode
    next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
    for step in range(1, decode_tokens):
        with torch.no_grad():
            step_out = engine.model(input_ids=next_token, past_key_values=pkv, use_cache=True)
        next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
        pkv = step_out.past_key_values

    cp_after_decode = get_proc_memory()
    print(f"[CHECKPOINT: after_decode]       RSS: {cp_after_decode['vm_rss_mb']:.1f} MB (Peak: {cp_after_decode['vm_peak_mb']:.1f} MB)")
    
    return {
        "process_start": cp_start,
        "after_model_load": cp_model,
        "after_prefill": cp_prefill,
        "before_decode": cp_before_decode,
        "after_decode": cp_after_decode,
        "pytorch_kv_mb": pkv_bytes / (1024*1024),
    }


def audit_aissd(context_len: int = 4096, decode_tokens: int = 16):
    print("\n" + "=" * 64)
    print("        AUDIT 2: FULL AI-SSD INFERENCE MEMORY LIFECYCLE")
    print("=" * 64)
    
    cp_start = get_proc_memory()
    print(f"[CHECKPOINT: process_start]       RSS: {cp_start['vm_rss_mb']:.1f} MB (Anon: {cp_start['rss_anon_mb']:.1f} MB)")

    engine = RealLLMEngine(model_name="Qwen/Qwen3-4B-Instruct-2507", device="cpu", dtype="float32", num_threads=4)
    cp_model = get_proc_memory()
    print(f"[CHECKPOINT: after_model_load]   RSS: {cp_model['vm_rss_mb']:.1f} MB (Anon: {cp_model['rss_anon_mb']:.1f} MB)")

    # Initialize P2 + P3 storage backend
    num_layers = getattr(engine.model.config, "num_hidden_layers", 36)
    num_kv_heads = getattr(engine.model.config, "num_key_value_heads", 8)
    head_dim = getattr(engine.model.config, "head_dim", 128)
    backend = create_default_storage_backend(
        channels=8,
        enable_prefetch=True,
        num_layers=num_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        dtype="float32",
    )
    cp_backend_init = get_proc_memory()
    print(f"[CHECKPOINT: after_backend_init] RSS: {cp_backend_init['vm_rss_mb']:.1f} MB")

    prompt = build_prompt_for_length(engine, target_tokens=context_len)
    inputs = engine.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Prefill step
    t0 = time.perf_counter()
    with torch.no_grad():
        prefill_out = engine.model(input_ids=input_ids, use_cache=True)
    t_prefill = time.perf_counter() - t0
    cp_prefill = get_proc_memory()
    print(f"[CHECKPOINT: after_prefill]      RSS: {cp_prefill['vm_rss_mb']:.1f} MB (prefill time: {t_prefill:.1f}s)")

    pkv_prefill = prefill_out.past_key_values
    orig_pkv_bytes = sum(l.keys.nelement() * 4 + l.values.nelement() * 4 for l in pkv_prefill.layers)
    print(f"  -> Original PyTorch Prefill KV size: {orig_pkv_bytes / (1024*1024):.2f} MB")

    # Initialize AI-SSD KV Manager and offload historical blocks
    kv_mgr = AISSDKVManager(backend, top_k_pct=10.0)
    kv_mgr.init_from_prefill(pkv_prefill)
    cp_after_offload = get_proc_memory()
    print(f"[CHECKPOINT: after_kv_offload]   RSS: {cp_after_offload['vm_rss_mb']:.1f} MB (Anon: {cp_after_offload['rss_anon_mb']:.1f} MB)")

    # Audit resident storage structures
    storage_stats = inspect_storage_structures(backend)
    print(f"  -> Storage Inspection:")
    print(f"     P3 Adapter resident bytes:        {storage_stats['p3_payload_mb']:.2f} MB")
    print(f"     P2 FTL Backend resident bytes:    {storage_stats['p2_storage_mb']:.2f} MB")
    print(f"     Total RAM in Storage Backends:    {storage_stats['total_backend_resident_mb']:.2f} MB")
    print(f"     P2 Access Log entries:            {storage_stats['p2_access_log_entries']}")

    mem_stats = kv_mgr.get_memory_stats()
    print(f"  -> Reported Active KV DRAM:          {mem_stats['active_dram_mb']:.2f} MB")
    print(f"  -> Reported Total KV Cache:          {mem_stats['total_kv_mb']:.2f} MB")
    print(f"  -> Reported Offload Pct:             {mem_stats['offload_pct']:.1f}%")

    # Phase B: Free prefill activations & original unpruned KV cache
    next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
    cur_seq_len = input_ids.shape[1]
    del prefill_out
    del pkv_prefill
    gc.collect()

    cp_before_decode = get_proc_memory()
    print(f"[CHECKPOINT: after_pkv_release]  RSS: {cp_before_decode['vm_rss_mb']:.1f} MB (Anon: {cp_before_decode['rss_anon_mb']:.1f} MB)")

    # Run AI-SSD decode step
    # Wrap layer forwards
    orig_forwards = {}
    from person1_kv_engine.real_llm.aissd_inference import apply_rotary_pos_emb
    for i, layer in enumerate(engine.model.model.layers):
        attn = layer.self_attn
        orig_forwards[i] = attn.forward

        def make_aissd_forward(layer_idx: int, original_fwd: Any):
            def forward(hidden_states: torch.Tensor, position_embeddings: Any, attention_mask: Any = None, past_key_values: Any = None, **kwargs):
                if not kv_mgr.is_active or hidden_states.shape[1] > 1:
                    return original_fwd(hidden_states, position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)

                attn_module = engine.model.model.layers[layer_idx].self_attn
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_module.head_dim)

                q = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                k = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                v = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                kv_mgr.append_new_token(layer_idx, k, v)
                act_k, act_v = kv_mgr.select_and_fetch_active_kv(layer_idx, q)

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
                return out, None

            return forward

        attn.forward = make_aissd_forward(i, orig_forwards[i])

    decode_rss_samples = []
    for step in range(1, decode_tokens):
        pos_ids = torch.tensor([[cur_seq_len + step - 1]], device=input_ids.device)
        with torch.no_grad():
            step_out = engine.model(input_ids=next_token, position_ids=pos_ids, use_cache=False)
        next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
        decode_rss_samples.append(get_proc_memory()["vm_rss_mb"])

    cp_after_decode = get_proc_memory()
    print(f"[CHECKPOINT: after_decode]       RSS: {cp_after_decode['vm_rss_mb']:.1f} MB (Peak: {cp_after_decode['vm_peak_mb']:.1f} MB)")
    print(f"  -> Decode RSS min/avg/max:          {min(decode_rss_samples):.1f} / {np.mean(decode_rss_samples):.1f} / {max(decode_rss_samples):.1f} MB")

    # Restore forwards
    for i, layer in enumerate(engine.model.model.layers):
        layer.self_attn.forward = orig_forwards[i]

    return {
        "process_start": cp_start,
        "after_model_load": cp_model,
        "after_backend_init": cp_backend_init,
        "after_prefill": cp_prefill,
        "after_kv_offload": cp_after_offload,
        "before_decode": cp_before_decode,
        "after_decode": cp_after_decode,
        "storage_stats": storage_stats,
        "orig_pkv_mb": orig_pkv_bytes / (1024*1024),
        "reported_active_kv_mb": mem_stats["active_dram_mb"],
        "reported_offload_pct": mem_stats["offload_pct"],
    }


def main():
    print("=" * 64)
    print("   AI-SSD V2: LIVE MEMORY ACCOUNTING AUDIT (QWEN3-4B @ 4096)")
    print("=" * 64)
    
    # Run Baseline in isolated subprocess or clean execution
    # To prevent interference between runs, we run Baseline then AI-SSD in separate processes
    pass

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--baseline":
        audit_baseline(context_len=4096, decode_tokens=16)
    elif len(sys.argv) > 1 and sys.argv[1] == "--aissd":
        audit_aissd(context_len=4096, decode_tokens=16)
    else:
        print("Usage: python memory_accounting_audit.py --baseline | --aissd")
