import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_rope
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen_rope
import inspect

print("llama_rope exists:", llama_rope is not None)
print("qwen_rope exists:", qwen_rope is not None)
print("llama signature:", inspect.signature(llama_rope))
print("qwen signature:", inspect.signature(qwen_rope))
