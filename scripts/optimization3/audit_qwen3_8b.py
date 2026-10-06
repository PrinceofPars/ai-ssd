#!/usr/bin/env python3
"""
AI-SSD V2 — Optimization 3: Qwen3-8B FP16 Architectural & Dtype Audit.
"""

import sys
import time
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_ID = "Qwen/Qwen3-8B"

def main():
    print("=" * 80)
    print("AI-SSD V2 — Qwen3-8B FP16 Architectural & Precision Audit")
    print("=" * 80)

    # 1. Config & Tokenizer
    print("\n[1/5] Loading Tokenizer & Config...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)
    print(f"Tokenizer & Config loaded in {time.time() - t0:.2f}s:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Bos token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"  Eos token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    # 2. Model Loading in FP16 on CPU
    print("\n[2/5] Loading Model in FP16 on CPU (low_cpu_mem_usage=True)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model_load_time = time.time() - t0
    print(f"Model loaded in {model_load_time:.2f}s")

    # 3. Architectural Parameters
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", hidden_size // num_heads)
    gqa_ratio = num_heads // num_kv_heads

    print("\n[3/5] Architectural Parameters:")
    print(f"  Model Architecture:      {config.architectures[0]}")
    print(f"  Hidden Layers (L):       {num_layers}")
    print(f"  Hidden Size (D):         {hidden_size}")
    print(f"  Intermediate Size (F):   {intermediate_size}")
    print(f"  Query Heads (Q):         {num_heads}")
    print(f"  Key/Value Heads (KV):    {num_kv_heads}")
    print(f"  Head Dimension:          {head_dim}")
    print(f"  GQA Ratio:               {gqa_ratio}")

    # Inspect self-attention layer structure
    first_attn = model.model.layers[0].self_attn
    print(f"  Has q_norm:              {hasattr(first_attn, 'q_norm')}")
    print(f"  Has k_norm:              {hasattr(first_attn, 'k_norm')}")
    print(f"  Attn scaling:            {getattr(first_attn, 'scaling', None)}")

    # 4. Parameter Count & Dtype Audit
    print("\n[4/5] Parameter Count & Dtype Audit:")
    total_params = 0
    fp16_params = 0
    fp32_params = 0
    other_params = 0
    total_param_bytes = 0

    param_details = []
    for name, param in model.named_parameters():
        cnt = param.numel()
        total_params += cnt
        total_param_bytes += param.nbytes
        if param.dtype == torch.float16:
            fp16_params += cnt
        elif param.dtype == torch.float32:
            fp32_params += cnt
        else:
            other_params += cnt

    print(f"  Total Parameters:        {total_params:,}")
    print(f"  FP16 Parameters:         {fp16_params:,} ({fp16_params/total_params*100:.2f}%)")
    print(f"  FP32 Parameters:         {fp32_params:,} ({fp32_params/total_params*100:.2f}%)")
    print(f"  Other Parameters:        {other_params:,}")
    print(f"  Total Weight Bytes:      {total_param_bytes:,} bytes ({total_param_bytes/(1024**3):.2f} GB)")

    # 5. KV Cache Geometry Analysis
    tokens_per_block = 16
    bytes_per_elem_fp16 = 2
    bytes_per_tok_layer = num_kv_heads * head_dim * bytes_per_elem_fp16 * 2  # K + V
    bytes_per_tok_all_layers = bytes_per_tok_layer * num_layers

    key_page_bytes = tokens_per_block * num_kv_heads * head_dim * bytes_per_elem_fp16
    val_page_bytes = tokens_per_block * num_kv_heads * head_dim * bytes_per_elem_fp16
    block_bytes = key_page_bytes + val_page_bytes

    print("\n[5/5] KV Cache Geometry Analysis (FP16):")
    print(f"  Tokens per block:        {tokens_per_block}")
    print(f"  KV heads:                {num_kv_heads}")
    print(f"  Head dimension:          {head_dim}")
    print(f"  KV bytes / token / layer:{bytes_per_tok_layer} bytes")
    print(f"  KV bytes / token / all L:{bytes_per_tok_all_layers} bytes ({bytes_per_tok_all_layers/1024:.2f} KiB)")
    print(f"  Key page bytes:          {key_page_bytes} bytes ({key_page_bytes/1024:.1f} KiB)")
    print(f"  Value page bytes:        {val_page_bytes} bytes ({val_page_bytes/1024:.1f} KiB)")
    print(f"  Logical block size:      {block_bytes} bytes ({block_bytes/1024:.1f} KiB)")

    for ctx in [4096, 8192, 16384, 32768]:
        dense_kv_bytes = (ctx + 16) * bytes_per_tok_all_layers
        dense_kv_mb = dense_kv_bytes / (1024 * 1024)
        num_blocks = (ctx - 20) // tokens_per_block
        active_blocks = max(1, int(num_blocks * 0.10))
        active_tokens = 4 + 16 + (active_blocks * tokens_per_block)
        active_kv_mb = (active_tokens * bytes_per_tok_all_layers) / (1024 * 1024)
        savings = (1.0 - (active_kv_mb / dense_kv_mb)) * 100.0
        print(f"  Context {ctx:>5}: Dense KV = {dense_kv_mb:>7.1f} MB | Active KV = {active_kv_mb:>6.1f} MB | Offload = {savings:>5.1f}%")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
