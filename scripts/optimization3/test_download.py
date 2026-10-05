import huggingface_hub

try:
    path = huggingface_hub.hf_hub_download(
        repo_id="meta-llama/Meta-Llama-3-8B",
        filename="config.json"
    )
    print("Downloaded config:", path)
except Exception as e:
    print("Download failed:", type(e).__name__, e)
