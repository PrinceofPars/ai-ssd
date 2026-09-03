"""
Interactive Command-Line Demo for AI-SSD.
Simulates a complete long-context token generation step with real-time offloading and retrieval.
"""

import time
from common.utils import load_yaml
from person1_kv_engine.baseline.baseline_kv import BaselineKVCache
from person1_kv_engine.cache_manager.kv_cache import PagedKVCache
from person1_kv_engine.mock_ssd import MockSSD
from person2_ssd.storage_model.io_model import StorageSimulator
from person3_system.integration.orchestrator import SystemOrchestrator


def run_demo():
    print("=========================================================")
    print("        [+] AI-SSD CO-DESIGN SYSTEM DEMONSTRATION         ")
    print("=========================================================\n")

    context_length = 32768
    print(f"[*] Initializing LLM workload with Context Length: {context_length:,} tokens (FP16)")
    baseline = BaselineKVCache(layers=32, heads=32, head_dim=128, dtype="FP16")
    base_mb = baseline.get_memory_mb(context_length)
    print(f"[*] Dense Baseline KV Memory: {base_mb:,.1f} MB (~{base_mb/1024:.2f} GB)")

    print("\n[*] Initializing Tensor-Aware Storage Simulator (8 Channels, 4 Dies/Ch)...")
    ssd = StorageSimulator(mode="tensor_aware", channels=8)

    print("[*] Initializing Paged KV Cache Manager (16 tokens/block)...")
    kv_cache = PagedKVCache(layers=32, heads=32, head_dim=128, block_tokens=16, ssd_backend=ssd)
    total_blocks = kv_cache.initialize_context(context_length)
    print(f"[+] Total KV Blocks created: {total_blocks:,} blocks (4 KB each)")

    print("\n[*] Executing 80% KV Cache Offload to SSD...")
    hot_cnt, cold_cnt = kv_cache.run_offload_pass(context_length)
    print(f"    - Hot Blocks (in GPU VRAM): {hot_cnt:,}")
    print(f"    - Cold Blocks (Offloaded to SSD): {cold_cnt:,}")
    gpu_mb = (hot_cnt * 4096) / (1024 * 1024)
    print(f"    - New GPU VRAM Footprint: {gpu_mb:,.1f} MB (Reduction: {(1.0 - gpu_mb/base_mb)*100:.1f}%)")

    print("\n[*] Simulating Decode Step with Top-k Sparse Retrieval (Top 10%)...")
    k = max(1, int(cold_cnt * 0.10))
    sample_ids = [b.block_id for b in kv_cache.block_manager.all_blocks() if b.storage_tier == "SSD"][:k]
    read_lat = ssd.estimate_read_latency(sample_ids)
    print(f"    - Retrieved Top-k Cold Blocks: {len(sample_ids):,}")
    print(f"    - Estimated Parallel Read Latency: {read_lat:.1f} microseconds")
    print(f"    - I/O Traffic Saved: {(1.0 - (len(sample_ids)/cold_cnt))*100:.1f}%")

    print("\n=========================================================")
    print("[SUCCESS] System operational! Ready for parallel track development.")
    print("=========================================================\n")


if __name__ == "__main__":
    run_demo()
