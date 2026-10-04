#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path("/home/ubuntu/ai-ssd")
sys.path.insert(0, str(ROOT))

import torch
from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.aissd_inference import AISSDKVManager, create_default_storage_backend
from scripts.real_inference_benchmark import build_prompt_for_length

engine = RealLLMEngine(model_name="Qwen/Qwen3-4B-Instruct-2507", device="cpu", dtype="float32", num_threads=4)
prompt = build_prompt_for_length(engine, target_tokens=4096)
inputs = engine.tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

backend = create_default_storage_backend(channels=8, storage_mode="file", enable_batching=True, enable_async_pipeline=True)
kv_mgr = AISSDKVManager(backend, top_k_pct=0.10)

with torch.no_grad():
    out = engine.model(input_ids=input_ids, use_cache=True)
kv_mgr.init_from_prefill(out.past_key_values)

# Inspect cands_info for layer 0
storage = getattr(backend, "storage_backend", backend)
cands = getattr(storage, "_storage", {})
layer0_bids = [bid for (l, bid) in cands.keys() if l == 0]
layer0_bids.sort()
print(f"Layer 0 has {len(layer0_bids)} blocks")
for b in layer0_bids[:10]:
    entry = cands[(0, b)]
    print(f"  Block {b}: k_offset={entry['k_offset']}, k_size={entry['k_size']}, v_offset={entry['v_offset']}, v_size={entry['v_size']}")

layer1_bids = [bid for (l, bid) in cands.keys() if l == 1]
layer1_bids.sort()
print(f"Layer 1 has {len(layer1_bids)} blocks")
for b in layer1_bids[:10]:
    entry = cands[(1, b)]
    print(f"  Block {b}: k_offset={entry['k_offset']}, k_size={entry['k_size']}, v_offset={entry['v_offset']}, v_size={entry['v_size']}")
