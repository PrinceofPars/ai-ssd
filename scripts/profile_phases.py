import sys
import os
import gc
import time
import json
import psutil
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.aissd_inference import (
    ProcessMemorySampler,
    get_current_rss_mb,
    create_default_storage_backend,
    run_baseline_decode,
    run_aissd_decode,
)
from person1_kv_engine.adapters.registry import ModelRegistry
from scripts.real_inference_benchmark import build_prompt_for_length


def profile_run(mode: str = "baseline", model_name: str = "Qwen/Qwen3-8B", context: int = 16384, decode: int = 16, dtype_str: str = "float16", threads: int = 4):
    print("=" * 60)
    print(f" PROFILING RUN: {mode.upper()} | {model_name} | ctx={context} | dtype={dtype_str} | threads={threads}")
    print("=" * 60)

    results = {}
    torch.set_num_threads(threads)
    gc.collect()

    mem_init = get_current_rss_mb()
    print(f"[Initial Process RSS]: {mem_init:.2f} MB")

    # PHASE A: Model Loading
    sampler = ProcessMemorySampler(sample_interval_s=0.002)
    sampler.start()
    t0 = time.perf_counter()
    rss_a_start = get_current_rss_mb()

    engine = RealLLMEngine(
        model_name=model_name,
        device="cpu",
        dtype=dtype_str,
        num_threads=threads,
    )
    t_a = time.perf_counter() - t0
    rss_a_end = get_current_rss_mb()
    mem_a = sampler.stop()
    results["Phase A (Model Loading)"] = {
        "wall_time_s": t_a,
        "rss_start_mb": rss_a_start,
        "rss_end_mb": rss_a_end,
        "peak_rss_mb": mem_a["peak_rss_mb"],
    }
    print(f"Phase A (Model Loading): time={t_a:.2f}s, start={rss_a_start:.1f}MB, end={rss_a_end:.1f}MB, peak={mem_a['peak_rss_mb']:.1f}MB")

    total_param_bytes = sum(p.numel() * p.element_size() for p in engine.model.parameters())
    print(f"  -> Model Parameter Bytes: {total_param_bytes / (1024*1024):.2f} MB")

    # PHASE C: Prompt Preparation
    sampler = ProcessMemorySampler(sample_interval_s=0.002)
    sampler.start()
    t0 = time.perf_counter()
    rss_c_start = get_current_rss_mb()

    prompt = build_prompt_for_length(engine, target_tokens=context)
    t_c = time.perf_counter() - t0
    rss_c_end = get_current_rss_mb()
    mem_c = sampler.stop()
    results["Phase C (Prompt Prep)"] = {
        "wall_time_s": t_c,
        "rss_start_mb": rss_c_start,
        "rss_end_mb": rss_c_end,
        "peak_rss_mb": mem_c["peak_rss_mb"],
    }
    print(f"Phase C (Prompt Prep): time={t_c:.4f}s, start={rss_c_start:.1f}MB, end={rss_c_end:.1f}MB, peak={mem_c['peak_rss_mb']:.1f}MB")

    # PHASE B: Tokenization
    sampler = ProcessMemorySampler(sample_interval_s=0.002)
    sampler.start()
    t0 = time.perf_counter()
    rss_b_start = get_current_rss_mb()

    inputs = engine.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    actual_tokens = input_ids.shape[1]
    t_b = time.perf_counter() - t0
    rss_b_end = get_current_rss_mb()
    mem_b = sampler.stop()
    results["Phase B (Tokenization)"] = {
        "wall_time_s": t_b,
        "rss_start_mb": rss_b_start,
        "rss_end_mb": rss_b_end,
        "peak_rss_mb": mem_b["peak_rss_mb"],
        "actual_tokens": actual_tokens,
    }
    print(f"Phase B (Tokenization): tokens={actual_tokens}, time={t_b:.4f}s, start={rss_b_start:.1f}MB, end={rss_b_end:.1f}MB, peak={mem_b['peak_rss_mb']:.1f}MB")

    if mode == "baseline":
        # BASELINE DETAILED PROFILING
        # PHASE D: Prefill Forward Pass
        sampler = ProcessMemorySampler(sample_interval_s=0.002)
        sampler.start()
        t0 = time.perf_counter()
        rss_d_start = get_current_rss_mb()

        with torch.no_grad():
            prefill_out = engine.model(input_ids=input_ids, use_cache=True)

        t_d = time.perf_counter() - t0
        rss_d_end = get_current_rss_mb()
        mem_d = sampler.stop()
        results["Phase D (Prefill Forward Pass)"] = {
            "wall_time_s": t_d,
            "rss_start_mb": rss_d_start,
            "rss_end_mb": rss_d_end,
            "peak_rss_mb": mem_d["peak_rss_mb"],
        }
        print(f"Phase D (Prefill Forward): time={t_d:.2f}s, start={rss_d_start:.1f}MB, end={rss_d_end:.1f}MB, peak={mem_d['peak_rss_mb']:.1f}MB")

        pkv = prefill_out.past_key_values
        logits_bytes = prefill_out.logits.numel() * prefill_out.logits.element_size()
        print(f"  -> Prefill Logits shape: {prefill_out.logits.shape}, bytes: {logits_bytes / (1024*1024):.2f} MB")

        num_layers = getattr(engine.model.config, "num_hidden_layers", 36)
        num_kv_heads = getattr(engine.model.config, "num_key_value_heads", 8)
        head_dim = getattr(engine.model.config, "head_dim", 128)
        bytes_per_elem = 2 if dtype_str == "float16" else 4
        kv_bytes = actual_tokens * num_layers * num_kv_heads * head_dim * bytes_per_elem * 2
        print(f"  -> Theoretical KV Bytes for {actual_tokens} tokens: {kv_bytes / (1024*1024):.2f} MB")

        # PHASE H: Transition prefill -> decode
        sampler = ProcessMemorySampler(sample_interval_s=0.002)
        sampler.start()
        t0 = time.perf_counter()
        rss_h_start = get_current_rss_mb()

        prefill_last_logits = prefill_out.logits[:, -1, :]
        next_token = torch.argmax(prefill_last_logits, dim=-1, keepdim=True)
        t_h = time.perf_counter() - t0
        rss_h_end = get_current_rss_mb()
        mem_h = sampler.stop()
        results["Phase H (Transition)"] = {
            "wall_time_s": t_h,
            "rss_start_mb": rss_h_start,
            "rss_end_mb": rss_h_end,
            "peak_rss_mb": mem_h["peak_rss_mb"],
        }
        print(f"Phase H (Transition): time={t_h:.4f}s, start={rss_h_start:.1f}MB, end={rss_h_end:.1f}MB, peak={mem_h['peak_rss_mb']:.1f}MB")

        # PHASE I: Decode (16 tokens)
        sampler = ProcessMemorySampler(sample_interval_s=0.002)
        sampler.start()
        t0 = time.perf_counter()
        rss_i_start = get_current_rss_mb()

        gen_tokens = [next_token.item()]
        for step in range(1, decode):
            with torch.no_grad():
                step_out = engine.model(input_ids=next_token, past_key_values=pkv, use_cache=True)
            pkv = step_out.past_key_values
            next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
            gen_tokens.append(next_token.item())

        t_i = time.perf_counter() - t0
        rss_i_end = get_current_rss_mb()
        mem_i = sampler.stop()
        results["Phase I (Decode)"] = {
            "wall_time_s": t_i,
            "rss_start_mb": rss_i_start,
            "rss_end_mb": rss_i_end,
            "peak_rss_mb": mem_i["peak_rss_mb"],
        }
        print(f"Phase I (Decode): time={t_i:.2f}s, start={rss_i_start:.1f}MB, end={rss_i_end:.1f}MB, peak={mem_i['peak_rss_mb']:.1f}MB")

        # PHASE J: Cleanup / Release
        sampler = ProcessMemorySampler(sample_interval_s=0.002)
        sampler.start()
        t0 = time.perf_counter()
        rss_j_start = get_current_rss_mb()

        del prefill_out
        del pkv
        gc.collect()

        t_j = time.perf_counter() - t0
        rss_j_end = get_current_rss_mb()
        mem_j = sampler.stop()
        results["Phase J (Cleanup)"] = {
            "wall_time_s": t_j,
            "rss_start_mb": rss_j_start,
            "rss_end_mb": rss_j_end,
            "peak_rss_mb": mem_j["peak_rss_mb"],
        }
        print(f"Phase J (Cleanup): time={t_j:.4f}s, start={rss_j_start:.1f}MB, end={rss_j_end:.1f}MB, peak={mem_j['peak_rss_mb']:.1f}MB")

    else:
        # AI-SSD DETAILED PROFILING
        m_cfg, comp_level, comp_reason = ModelRegistry.detect_model_config(getattr(engine.model.config, "_name_or_path", "qwen"))
        model_adapter = ModelRegistry.get_adapter(m_cfg)

        backend = create_default_storage_backend(
            channels=8,
            num_layers=m_cfg.num_layers if m_cfg else 36,
            num_heads=m_cfg.num_key_value_heads if m_cfg else 8,
            head_dim=m_cfg.head_dim if m_cfg else 128,
            dtype=dtype_str,
            storage_mode="file",
            enable_batching=True,
            enable_async_pipeline=False,
        )

        kv_mgr = model_adapter.create_state_provider(
            backend=backend,
            top_k_pct=10.0,
            enable_computational_storage=True,
            enable_prefetch=True,
            enable_async_pipeline=False,
        )

        model_timings = {"qkv_proj_s": 0.0, "rope_s": 0.0, "attn_matmul_s": 0.0, "out_proj_s": 0.0, "mlp_and_norm_s": 0.0, "bookkeeping_s": 0.0}
        attn_forward_durations = []
        orig_forwards, restore_func = model_adapter.wrap_model_for_aissd(
            model=engine.model,
            state_provider=kv_mgr,
            model_timings=model_timings,
            attn_forward_durations=attn_forward_durations,
        )

        # PHASE D: Prefill Forward Pass
        sampler = ProcessMemorySampler(sample_interval_s=0.002)
        sampler.start()
        t0 = time.perf_counter()
        rss_d_start = get_current_rss_mb()

        with torch.no_grad():
            prefill_out = engine.model(input_ids=input_ids, use_cache=True)

        t_d = time.perf_counter() - t0
        rss_d_end = get_current_rss_mb()
        mem_d = sampler.stop()
        results["Phase D (Prefill Forward Pass)"] = {
            "wall_time_s": t_d,
            "rss_start_mb": rss_d_start,
            "rss_end_mb": rss_d_end,
            "peak_rss_mb": mem_d["peak_rss_mb"],
        }
        print(f"Phase D (Prefill Forward): time={t_d:.2f}s, start={rss_d_start:.1f}MB, end={rss_d_end:.1f}MB, peak={mem_d['peak_rss_mb']:.1f}MB")

        pkv_prefill = prefill_out.past_key_values
        logits_bytes = prefill_out.logits.numel() * prefill_out.logits.element_size()
        print(f"  -> Prefill Logits shape: {prefill_out.logits.shape}, bytes: {logits_bytes / (1024*1024):.2f} MB")

        # PHASE E/F/G: KV Extraction, Blockization and Storage/Offload Initialization
        sampler = ProcessMemorySampler(sample_interval_s=0.002)
        sampler.start()
        t0 = time.perf_counter()
        rss_efg_start = get_current_rss_mb()

        kv_mgr.init_from_prefill(pkv_prefill)

        t_efg = time.perf_counter() - t0
        rss_efg_end = get_current_rss_mb()
        mem_efg = sampler.stop()
        results["Phase E/F/G (KV Blockization and Offload)"] = {
            "wall_time_s": t_efg,
            "rss_start_mb": rss_efg_start,
            "rss_end_mb": rss_efg_end,
            "peak_rss_mb": mem_efg["peak_rss_mb"],
        }
        print(f"Phase E/F/G (KV Blockization and Offload): time={t_efg:.2f}s, start={rss_efg_start:.1f}MB, end={rss_efg_end:.1f}MB, peak={mem_efg['peak_rss_mb']:.1f}MB")

        # PHASE H: Transition & Release of dense prefill tensors
        sampler = ProcessMemorySampler(sample_interval_s=0.002)
        sampler.start()
        t0 = time.perf_counter()
        rss_h_start = get_current_rss_mb()

        prefill_last_logits = prefill_out.logits[:, -1, :]
        next_token = torch.argmax(prefill_last_logits, dim=-1, keepdim=True)
        gen_tokens = [next_token.item()]
        cur_seq_len = input_ids.shape[1]

        del prefill_out
        del pkv_prefill
        gc.collect()

        t_h = time.perf_counter() - t0
        rss_h_end = get_current_rss_mb()
        mem_h = sampler.stop()
        results["Phase H (Transition and Release Prefill Tensors)"] = {
            "wall_time_s": t_h,
            "rss_start_mb": rss_h_start,
            "rss_end_mb": rss_h_end,
            "peak_rss_mb": mem_h["peak_rss_mb"],
        }
        print(f"Phase H (Transition and Release Prefill): time={t_h:.4f}s, start={rss_h_start:.1f}MB, end={rss_h_end:.1f}MB, peak={mem_h['peak_rss_mb']:.1f}MB")

        active_kv_mb = kv_mgr.get_resident_kv_bytes() / (1024*1024)
        print(f"  -> AI-SSD Active KV in Host RAM: {active_kv_mb:.2f} MB")

        # PHASE I: Decode (16 tokens)
        sampler = ProcessMemorySampler(sample_interval_s=0.002)
        sampler.start()
        t0 = time.perf_counter()
        rss_i_start = get_current_rss_mb()

        for step in range(1, decode):
            pos_ids = torch.tensor([[cur_seq_len + step - 1]], device=input_ids.device)
            with torch.no_grad():
                step_pkv = getattr(kv_mgr, "native_cache", None)
                if step_pkv is not None:
                    step_out = engine.model(input_ids=next_token, position_ids=pos_ids, past_key_values=step_pkv, use_cache=True)
                else:
                    step_out = engine.model(input_ids=next_token, position_ids=pos_ids, use_cache=False)
            next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
            gen_tokens.append(next_token.item())

        t_i = time.perf_counter() - t0
        rss_i_end = get_current_rss_mb()
        mem_i = sampler.stop()
        results["Phase I (Decode)"] = {
            "wall_time_s": t_i,
            "rss_start_mb": rss_i_start,
            "rss_end_mb": rss_i_end,
            "peak_rss_mb": mem_i["peak_rss_mb"],
        }
        print(f"Phase I (Decode): time={t_i:.2f}s, start={rss_i_start:.1f}MB, end={rss_i_end:.1f}MB, peak={mem_i['peak_rss_mb']:.1f}MB")

        restore_func()
        backend.close()

    overall_peak = max(v["peak_rss_mb"] for v in results.values())
    print("\n" + "=" * 60)
    print(f" OVERALL PEAK RSS FOR {mode.upper()}: {overall_peak:.2f} MB")
    print("=" * 60 + "\n")
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    profile_run(mode=mode)
