import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/ubuntu/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B/snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)

inputs = tokenizer("Hello", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, use_cache=True)
pkv = out.past_key_values

next_tok = torch.tensor([[1]], dtype=torch.long)
pos_ids = torch.tensor([[1]], dtype=torch.long)

layer = model.model.layers[0]
hidden = torch.randn(1, 1, 4096, dtype=torch.float16)
cos = torch.randn(1, 1, 128, dtype=torch.float16)
sin = torch.randn(1, 1, 128, dtype=torch.float16)

# Test layer self_attn forward signature and return
import inspect
print("self_attn forward signature:", inspect.signature(layer.self_attn.forward))

# Check what decoder layer forward passes to self_attn
print("decoder layer forward signature:", inspect.signature(layer.forward))
