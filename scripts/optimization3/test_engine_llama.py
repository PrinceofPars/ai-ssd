import torch
from person1_kv_engine.real_llm.engine import RealLLMEngine

engine = RealLLMEngine(
    model_name="NousResearch/Meta-Llama-3-8B",
    device="cpu",
    dtype="FP16",
    num_threads=4,
)

a = engine.model.model.layers[0].self_attn
print("self_attn attributes:")
print("  head_dim:", getattr(a, "head_dim", None))
print("  scaling:", getattr(a, "scaling", None))
print("  num_heads:", getattr(a, "num_heads", None))
print("  num_key_value_heads:", getattr(a, "num_key_value_heads", None))
print("  has q_norm:", hasattr(a, "q_norm"))
print("  has k_norm:", hasattr(a, "k_norm"))
