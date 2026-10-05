from huggingface_hub import HfApi

api = HfApi()

# Search for exact Llama-3-8B models
models = api.list_models(filter="llama-3", search="Meta-Llama-3-8B", limit=20)
print("Search results for Meta-Llama-3-8B:")
for m in models:
    gated = getattr(m, 'gated', None)
    print(f"  {m.id} | gated: {gated} | downloads: {m.downloads}")
