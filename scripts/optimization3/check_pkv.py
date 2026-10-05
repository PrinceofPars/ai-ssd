import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/ubuntu/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B/snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)

inputs = tokenizer("Hello world", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, use_cache=True)

pkv = out.past_key_values
print("PKV type:", type(pkv))
if hasattr(pkv, "key_cache"):
    print("key_cache[0] shape:", pkv.key_cache[0].shape, "dtype:", pkv.key_cache[0].dtype)
    print("value_cache[0] shape:", pkv.value_cache[0].shape, "dtype:", pkv.value_cache[0].dtype)
elif hasattr(pkv, "layers"):
    print("layers[0].keys shape:", pkv.layers[0].keys.shape, "dtype:", pkv.layers[0].keys.dtype)
