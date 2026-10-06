import time
from huggingface_hub import snapshot_download

print("Starting download of Qwen/Qwen3-8B...")
t0 = time.time()
path = snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    allow_patterns=["*.safetensors", "*.json", "*.txt"],
)
elapsed = time.time() - t0
print(f"Downloaded Qwen/Qwen3-8B in {elapsed:.2f}s to: {path}")
