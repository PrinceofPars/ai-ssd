from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("NousResearch/Meta-Llama-3-8B")
print("Files in NousResearch/Meta-Llama-3-8B:")
for f in files:
    print(" ", f)
