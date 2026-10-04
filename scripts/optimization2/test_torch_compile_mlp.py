import time
import torch
from person1_kv_engine.real_llm.engine import RealLLMEngine

engine = RealLLMEngine("Qwen/Qwen3-4B-Instruct-2507", "cpu", "float32", 4)
mlp = engine.model.model.layers[0].mlp
x = torch.randn(1, 1, 2560)

# Uncompiled baseline
for _ in range(5):
    y = mlp(x)

t0 = time.perf_counter()
for _ in range(100):
    y = mlp(x)
t_orig = (time.perf_counter() - t0) / 100

print(f"Uncompiled MLP: {t_orig*1000:.3f} ms")

try:
    c_mlp = torch.compile(mlp, mode="reduce-overhead")
    # Warmup compile
    for _ in range(3):
        y = c_mlp(x)
    t0 = time.perf_counter()
    for _ in range(100):
        y = c_mlp(x)
    t_comp = (time.perf_counter() - t0) / 100
    print(f"Compiled MLP (reduce-overhead): {t_comp*1000:.3f} ms (speedup: {t_orig/t_comp:.2f}x)")
except Exception as e:
    print("torch.compile failed:", e)
