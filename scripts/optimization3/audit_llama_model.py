#!/usr/bin/env python3
"""
AI-SSD V2 — Optimization 3: Llama 3 8B FP16 Architectural & Dtype Audit.
"""

import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/ubuntu/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B/snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061"


def main():
    print("=" * 80)
    print("AI-SSD V2 — Llama 3 8B FP16 Architectural & Dtype Audit")
    print("=" * 80)

    # 1. Tokenizer
    print("\n[1/5] Loading Tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    t_tok = time.time() - t0
    print(f"Tokenizer loaded in {t_tok:.2f}s:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Bos token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"  Eos token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    # 2. Model in FP16
    print("\n[2/5] Loading Model in FP16 on CPU (low_cpu_mem_usage=True)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    t_model = time.time() - t0
    print(f"Model loaded in {t_model:.2f}s")

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", hidden_size // num_heads)
    gqa_ratio = num_heads // num_kv_heads

    print("\n[3/5] Architectural Parameters:")
    print(f"  Model Architecture:      {cfg.architectures[0]}")
    print(f"  Hidden Layers (L):       {num_layers}")
    print(f"  Hidden Size (D):         {hidden_size}")
    print(f"  Intermediate Size (F):   {intermediate_size}")
    print(f"  Query Heads (Q):         {num_heads}")
    print(f"  Key/Value Heads (KV):    {num_kv_heads}")
    print(f"  Head Dimension:          {head_dim}")
    print(f"  GQA Ratio:               {gqa_ratio}")
    print(f"  RoPE Base Theta:         {getattr(cfg, 'rope_theta', 10000.0)}")

    # 3. Dtype and Parameter Audit
    print("\n[4/5] Dtype Distribution & Parameter Accounting:")
    total_params = 0
    fp16_params = 0
    fp32_params = 0
    other_params = 0
    total_bytes = 0

    dtype_counts = {}
    for name, param in model.named_parameters():
        num_el = param.numel()
        param_bytes = param.nbytes
        total_params += num_el
        total_bytes += param_bytes
        dt = str(param.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + num_el
        if param.dtype == torch.float16:
            fp16_params += num_el
        elif param.dtype == torch.float32:
            fp32_params += num_el
            print(f"  [WARNING] FP32 parameter detected: {name} ({num_el} elements)")
        else:
            other_params += num_el

    print(f"  Total Parameters:        {total_params:,} ({total_params / 1e9:.3f}B)")
    print(f"  Total Parameter Bytes:   {total_bytes:,} ({total_bytes / (1024**3):.2f} GB)")
    print(f"  FP16 Parameters:         {fp16_params:,} ({fp16_params / total_params * 100:.2f}%)")
    print(f"  FP32 Parameters:         {fp32_params:,} ({fp32_params / total_params * 100:.2f}%)")
    print(f"  Dtype breakdown:         {dtype_counts}")

    # Check buffers (e.g. rotary embeddings)
    for name, buf in model.named_buffers():
        print(f"  Buffer {name}: shape={buf.shape}, dtype={buf.dtype}")

    # 4. KV Cache Geometry Analysis
    bytes_per_elem = 2  # FP16
    tokens_per_block = 16
    k_page_bytes = tokens_per_block * num_kv_heads * head_dim * bytes_per_elem
    v_page_bytes = tokens_per_block * num_kv_heads * head_dim * bytes_per_elem
    block_bytes = k_page_bytes + v_page_bytes
    kv_bytes_per_token_layer = num_kv_heads * head_dim * bytes_per_elem * 2
    kv_bytes_per_token_all = kv_bytes_per_token_layer * num_layers

    print("\n[5/5] KV Cache Geometry Calculation (FP16):")
    print(f"  Tokens per Block:        {tokens_per_block}")
    print(f"  Key Page Bytes:          {k_page_bytes:,} B ({k_page_bytes / 1024:.1f} KiB)")
    print(f"  Value Page Bytes:        {v_page_bytes:,} B ({v_page_bytes / 1024:.1f} KiB)")
    print(f"  Total Block Bytes:       {block_bytes:,} B ({block_bytes / 1024:.1f} KiB)")
    print(f"  KV Bytes/token/layer:    {kv_bytes_per_token_layer:,} B")
    print(f"  KV Bytes/token/32 layers:{kv_bytes_per_token_all:,} B ({kv_bytes_per_token_all / 1024:.2f} KiB)")
    print(f"  For 4096 context length: {4096 * kv_bytes_per_token_all / (1024**2):.2f} MB total KV cache")
    print(f"  Number of blocks (4096): {4096 // tokens_per_block} blocks per layer")

    # 5. Sanity forward pass
    print("\n[Sanity Check] Running 1 forward token pass on CPU...")
    prompt = "The key difference between AI-SSD and traditional GPU inference is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    torch.set_num_threads(4)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    fwd_time = time.perf_counter() - t0
    next_id = out.logits[:, -1, :].argmax(dim=-1).item()
    next_tok = tokenizer.decode([next_id])
    print(f"  Forward pass: {fwd_time*1000:.1f} ms | Output next token: {next_id} ({repr(next_tok)})")
    print(f"  Past KV layers: {len(out.past_key_values)}, shape: {out.past_key_values[0][0].shape}, dtype: {out.past_key_values[0][0].dtype}")


if __name__ == "__main__":
    main()
