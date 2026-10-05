#!/usr/bin/env python3
"""
Downloads NousResearch/Meta-Llama-3-8B into the HuggingFace cache.
"""
import time
from huggingface_hub import snapshot_download

print("[START] Downloading NousResearch/Meta-Llama-3-8B snapshot...")
t0 = time.time()
path = snapshot_download(
    repo_id="NousResearch/Meta-Llama-3-8B",
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    max_workers=8,
)
elapsed = time.time() - t0
print(f"[DONE] Downloaded to {path} in {elapsed:.1f}s")
