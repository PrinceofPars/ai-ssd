import json
import huggingface_hub

repo = "NousResearch/Meta-Llama-3-8B"
try:
    cfg_file = huggingface_hub.hf_hub_download(repo_id=repo, filename="config.json")
    print(f"[SUCCESS] Downloaded config from {repo}")
    with open(cfg_file) as f:
        cfg = json.load(f)
    print("Architectural parameters:")
    for k in ["model_type", "architectures", "num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "torch_dtype"]:
        print(f"  {k}: {cfg.get(k)}")
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
