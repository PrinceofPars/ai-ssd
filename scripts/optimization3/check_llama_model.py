import sys
import huggingface_hub

models_to_check = [
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3-8B",
    "openlm-research/open_llama_7b",
]

for m in models_to_check:
    try:
        info = huggingface_hub.model_info(m)
        print(f"[OK] {m}: accessible (gated={getattr(info, 'gated', 'unknown')})")
    except Exception as e:
        print(f"[ERROR] {m}: {type(e).__name__} - {e}")
