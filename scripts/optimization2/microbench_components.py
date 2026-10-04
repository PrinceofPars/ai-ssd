#!/usr/bin/env python3
"""
AI-SSD V2 — Micro-benchmark for Host CPU Components.
Measures isolated execution times for:
1. MLP (gate_proj + up_proj + silu + mul + down_proj)
2. RMSNorm (input_layernorm, post_attention_layernorm, final norm)
3. LM Head (linear projection 2560 -> vocab_size)
4. Active KV tensor concatenation vs pre-allocated buffer
"""

import sys
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.real_llm.engine import RealLLMEngine


def main():
    print("=" * 70)
    print("AI-SSD V2 — Host CPU Micro-Benchmark")
    print("=" * 70)

    torch.set_num_threads(4)
    print(f"PyTorch threads: {torch.get_num_threads()}")

    print("Loading model...")
    engine = RealLLMEngine(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        device="cpu",
        dtype="float32",
        num_threads=4,
    )
    model = engine.model
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    vocab_size = model.config.vocab_size

    print(f"Model parameters: layers={num_layers}, hidden_size={hidden_size}, vocab_size={vocab_size}")

    x = torch.randn(1, 1, hidden_size, dtype=torch.float32)

    # 1. Benchmark Layer 0 MLP
    mlp = model.model.layers[0].mlp
    for _ in range(10):
        _ = mlp(x)

    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = mlp(x)
    mlp_s = (time.perf_counter() - t0) / iters
    mlp_total_15steps = mlp_s * num_layers * 15

    # 2. Benchmark Layer 0 RMSNorm
    norm = model.model.layers[0].input_layernorm
    for _ in range(10):
        _ = norm(x)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = norm(x)
    norm_s = (time.perf_counter() - t0) / iters
    # 2 norms per layer * 36 layers = 72 norms per step, plus 1 final norm
    norm_total_15steps = norm_s * (num_layers * 2 + 1) * 15

    # 3. Benchmark LM Head
    lm_head = model.lm_head
    for _ in range(5):
        _ = lm_head(x)

    t0 = time.perf_counter()
    for _ in range(50):
        _ = lm_head(x)
    lm_head_s = (time.perf_counter() - t0) / 50
    lm_head_total_15steps = lm_head_s * 15

    # 4. Benchmark Active KV Concat vs Buffer Copy
    # Current active KV: sink (4 tokens) + 25 topk blocks (400 tokens) + recent (32 tokens) = 436 tokens
    # Key shape: [1, 2, 436, 64], float32
    sink_k = torch.randn(1, 2, 4, 64)
    blocks_k = [torch.randn(1, 2, 16, 64) for _ in range(25)]
    recent_k = torch.randn(1, 2, 32, 64)

    # Current path: torch.cat
    for _ in range(10):
        _ = torch.cat([sink_k] + blocks_k + [recent_k], dim=2)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.cat([sink_k] + blocks_k + [recent_k], dim=2)
    cat_s = (time.perf_counter() - t0) / iters
    cat_total_15steps = cat_s * 2 * num_layers * 15  # for K and V

    # Alternative: pre-allocated tensor with copy_
    total_tokens = 4 + 25 * 16 + 32
    buf = torch.empty(1, 2, total_tokens, 64)
    for _ in range(10):
        buf[:, :, :4, :].copy_(sink_k)
        offset = 4
        for blk in blocks_k:
            buf[:, :, offset:offset+16, :].copy_(blk)
            offset += 16
        buf[:, :, offset:offset+32, :].copy_(recent_k)

    t0 = time.perf_counter()
    for _ in range(iters):
        buf[:, :, :4, :].copy_(sink_k)
        offset = 4
        for blk in blocks_k:
            buf[:, :, offset:offset+16, :].copy_(blk)
            offset += 16
        buf[:, :, offset:offset+32, :].copy_(recent_k)
    copy_s = (time.perf_counter() - t0) / iters
    copy_total_15steps = copy_s * 2 * num_layers * 15

    print("\n--- RESULTS ---")
    print(f"MLP (single layer):          {mlp_s * 1000:6.3f} ms | Extrapolated 15 steps: {mlp_total_15steps:6.3f} s")
    print(f"RMSNorm (single layer):      {norm_s * 1000:6.3f} ms | Extrapolated 15 steps: {norm_total_15steps:6.3f} s")
    print(f"LM Head (vocab projection):  {lm_head_s * 1000:6.3f} ms | Extrapolated 15 steps: {lm_head_total_15steps:6.3f} s")
    print(f"MLP + Norm + LM Head Total:                         Extrapolated 15 steps: {mlp_total_15steps + norm_total_15steps + lm_head_total_15steps:6.3f} s")
    print(f"Active KV torch.cat:         {cat_s * 1000:6.3f} ms | Extrapolated 15 steps: {cat_total_15steps:6.3f} s")
    print(f"Active KV buffer copy_:      {copy_s * 1000:6.3f} ms | Extrapolated 15 steps: {copy_total_15steps:6.3f} s")


if __name__ == "__main__":
    main()
