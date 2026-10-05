#!/usr/bin/env python3
"""
Smoke test for AI-SSD Llama 3 8B FP16 on Virtual NVMe.
"""

import sys
import os
import json
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.aissd_inference import (
    create_default_storage_backend,
    run_aissd_decode,
)

MODEL_NAME = "NousResearch/Meta-Llama-3-8B"
CONTEXT = 4096
DECODE = 4
THREADS = 4
SEED = 42

EXPECTED_FIRST_4 = [8668, 3160, 11, 35208]

def main():
    print("=" * 80)
    print("AI-SSD V2 — Optimization 3: Llama 3 8B FP16 Smoke Test")
    print("=" * 80)

    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)

    print(f"[1/4] Loading {MODEL_NAME} (FP16, {THREADS} threads)...")
    t0 = time.time()
    engine = RealLLMEngine(
        model_name=MODEL_NAME,
        device="cpu",
        dtype="FP16",
        num_threads=THREADS,
    )
    print(f"Model loaded in {time.time() - t0:.2f}s")

    print(f"[2/4] Constructing prompt for exactly {CONTEXT} tokens...")
    base_text = (
        "The development of computational storage architectures represents a fundamental paradigm shift "
        "in modern data-intensive computing systems. Traditional von Neumann computer architectures suffer "
        "from the classic memory wall problem, where the latency and energy required to transfer data "
        "across the memory bus significantly constrain end-to-end application throughput. "
        "In Large Language Model (LLM) autoregressive inference, the key-value (KV) cache grows linearly "
        "with sequence length, consuming tens of gigabytes of host DRAM and saturating PCIe bandwidth during retrieval. "
    )
    base_ids = engine.tokenizer.encode(base_text, add_special_tokens=True)
    repeats = (CONTEXT // len(base_ids)) + 2
    full_text = " ".join([base_text] * repeats)
    full_ids = engine.tokenizer.encode(full_text, add_special_tokens=True)
    input_ids = torch.tensor([full_ids[:CONTEXT]], dtype=torch.long)
    print(f"Prompt verified: {input_ids.shape[1]} tokens")

    cfg = engine.model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads

    print(f"[3/4] Initializing QEMU Virtual NVMe Storage Backend (L={num_layers}, KV={num_kv_heads}, D={head_dim}, FP16)...")
    backend = create_default_storage_backend(
        channels=8,
        enable_prefetch=False,
        num_layers=num_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        dtype="float16",
        mapping_mode="tensor_aware",
        storage_mode="nvme_qemu",
        enable_batching=True,
        enable_async_pipeline=True,
    )

    try:
        print(f"[4/4] Running AI-SSD decode ({DECODE} tokens, Top-K=10%, Async=ON, Storage=Virtual NVMe)...")
        res = run_aissd_decode(
            model=engine.model,
            tokenizer=engine.tokenizer,
            input_ids=input_ids,
            decode_tokens=DECODE,
            top_k_pct=10.0,
            seed=SEED,
            storage_backend=backend,
            enable_prefetch=False,
            enable_computational_storage=True,
            enable_async_pipeline=True,
        )

        tokens = res["token_ids"]
        print("\n" + "=" * 80)
        print("SMOKE TEST RESULTS:")
        print(f"  Wall time:             {res['wall_time_s']:.3f} s")
        print(f"  Throughput:            {res['tokens_per_second']:.3f} tok/s")
        print(f"  Candidate K to host:   {res.get('candidate_k_bytes_to_host', 0)} bytes")
        print(f"  Winning K to host:     {res.get('winning_k_bytes_to_host', 0)} bytes")
        print(f"  Winning V to host:     {res.get('winning_v_bytes_to_host', 0)} bytes")
        print(f"  Total bus movement:    {res.get('total_data_movement_bytes', 0)} bytes")
        print(f"  Generated tokens:      {tokens}")
        print(f"  Expected tokens:       {EXPECTED_FIRST_4}")
        match = (tokens == EXPECTED_FIRST_4)
        print(f"  Exact Bit-for-Bit Match: {match}")
        print("=" * 80)
        if match:
            print("[SUCCESS] Smoke test passed with 100% token match!")
        else:
            print("[WARNING] Token mismatch!")
    finally:
        if hasattr(backend, "close"):
            backend.close()

if __name__ == "__main__":
    main()
