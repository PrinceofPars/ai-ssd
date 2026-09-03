"""SanDisk Cerebrum 2026 - Hackathon Live Demonstration Script.

Runs an interactive autoregressive inference demonstration showcasing:
1. Full Ground-Truth Baseline Attention.
2. Hot/Cold KV Block Tiering (Host RAM <-> SSD).
3. Native C In-Storage Attention Pruning (Streaming Top-k over PCIe).
4. Live Telemetry: PCIe Bus Traffic, Host RAM savings, and Attention Fidelity.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.baseline.transformer_attention import AttentionConfig, ScaledDotProductAttention
from person1_kv_engine.baseline.kv_cache_baseline import BaselineKVCache
from person1_kv_engine.storage_backend.mock_ssd import MockSSDController
from person1_kv_engine.tiering.hot_cold_classifier import TieringPolicy
from person1_kv_engine.tiering.tiered_kv_manager import TieredKVManager
from person1_kv_engine.computational_storage.instorage_pruner import InStorageAttentionPruner


def run_live_hackathon_demo():
    print("=" * 80)
    print("       SANDISK CEREBRUM 2026 | FIRMWARE & COMPUTATIONAL STORAGE DEMO")
    print("       AI-Aware SSD Firmware for Long-Context LLM KV Cache Offloading")
    print("=" * 80)

    # 1. Setup Configuration
    # Modeling a 4-layer transformer with 8 heads, head_dim=64
    config = AttentionConfig(
        num_layers=4,
        num_heads=8,
        head_dim=64,
        dtype=np.float32,
    )
    elem_bytes = np.dtype(config.dtype).itemsize

    print(f"\n[CONFIG] Layers: {config.num_layers} | Heads: {config.num_heads} | Head Dim: {config.head_dim} | Dim Model: {config.d_model}")
    print(f"[CONFIG] Precision: FP32 ({elem_bytes} bytes/element) | Paged KV Block Size: 16 tokens")

    # 2. Initialize Subsystems
    attn_engine = ScaledDotProductAttention(config)
    baseline_kv = BaselineKVCache(config)

    ssd = MockSSDController(controller_dram_size_mb=1024)
    policy = TieringPolicy(
        sink_tokens=16,             # 16 tokens pinned in RAM
        sliding_window_tokens=64,   # 64 tokens hot in RAM
        max_hot_blocks_per_layer=5,
        tokens_per_block=16,
    )
    manager = TieredKVManager(config=config, storage=ssd, policy=policy)
    pruner = InStorageAttentionPruner(
        config=config,
        storage=ssd,
        manager=manager,
        default_top_k=3,
    )

    # 3. Simulate Long Context Prefill (512 tokens = 32 blocks)
    seq_len = 512
    print(f"\n---> [STEP 1] Prefilling Long Context Prompt: {seq_len} tokens...")

    # Define semantic topic centroids
    num_topics = 4
    scale_factor = np.sqrt(float(config.head_dim))
    topics = np.random.randn(num_topics, config.num_heads, config.head_dim).astype(np.float32)
    topics = (topics / np.linalg.norm(topics, axis=-1, keepdims=True)) * scale_factor

    for l in range(config.num_layers):
        k_seq = np.random.randn(seq_len, config.num_heads, config.head_dim).astype(np.float32) * 0.4
        v_seq = np.random.randn(seq_len, config.num_heads, config.head_dim).astype(np.float32)

        # Distribute topic concepts
        for t_idx in range(num_topics):
            start = 32 + t_idx * 64
            if start + 16 <= seq_len:
                k_seq[start : start + 16] += topics[t_idx] * 0.9

        baseline_kv.prefill(l, k_seq, v_seq)
        manager.prefill_sequence(l, k_seq, v_seq)

    base_mem_mb = baseline_kv.get_memory_usage_mb()
    host_ram_mb = manager.get_host_ram_usage_mb()
    ssd_mem_mb = manager.get_offloaded_storage_mb()
    ram_savings_pct = (1.0 - host_ram_mb / base_mem_mb) * 100.0

    print(f"     Prompt Ingested: 512 tokens partitioned into 32 PagedAttention blocks per layer.")
    print(f"     Baseline Host RAM Footprint: {base_mem_mb:6.2f} MB")
    print(f"     AI-SSD Host RAM Footprint:   {host_ram_mb:6.2f} MB (Active Hot blocks)")
    print(f"     Offloaded to SSD Flash:     {ssd_mem_mb:6.2f} MB (Cold historical blocks)")
    print(f"     >> HOST RAM FOOTPRINT SAVINGS: {ram_savings_pct:.1f}% <<")

    # 4. Interactive Autoregressive Token Generation
    print("\n---> [STEP 2] Simulating Autoregressive Decode Generation (10 tokens)...")
    print("-" * 80)
    print(f"{'Step':<6} | {'Hot Tok':<8} | {'Cold Blks':<10} | {'Top-k Retr':<10} | {'Pruning %':<10} | {'PCIe Bus Saved':<15} | {'Cosine Sim':<10}")
    print("-" * 80)

    for gen_step in range(1, 11):
        target_topic = topics[gen_step % num_topics]
        query = target_topic + np.random.randn(config.num_heads, config.head_dim).astype(np.float32) * 0.4

        ssd.reset_telemetry()

        step_similarities = []
        step_pruning_ratios = []

        for l in range(config.num_layers):
            # Ground truth
            base_k, base_v = baseline_kv.get_kv(l)
            out_base, _ = attn_engine.compute_decode_step(query, base_k, base_v)

            # In-Storage top-k attention
            out_pruned, stats = pruner.compute_decode_attention(query=query, layer_id=l, top_k=3)

            # Cosine similarity
            dot = np.sum(out_base * out_pruned, axis=-1)
            norm_b = np.linalg.norm(out_base, axis=-1)
            norm_p = np.linalg.norm(out_pruned, axis=-1)
            cos = float(np.mean(dot / (norm_b * norm_p + 1e-12)))

            step_similarities.append(cos)
            step_pruning_ratios.append(stats["pruning_ratio"])

            # Append generated token to cache
            k_new = query + np.random.randn(config.num_heads, config.head_dim).astype(np.float32) * 0.1
            v_new = np.random.randn(config.num_heads, config.head_dim).astype(np.float32)
            baseline_kv.append_token(l, k_new, v_new)
            manager.append_token(l, k_new, v_new)

        mean_cos = float(np.mean(step_similarities))
        mean_prune = float(np.mean(step_pruning_ratios)) * 100.0

        # Calculate PCIe bus traffic saved
        # Standard load: all cold KV transferred
        total_cold_tokens = stats["total_cold_tokens"]
        cold_bytes_full = config.num_layers * (2 * total_cold_tokens * config.num_heads * config.head_dim * elem_bytes)
        actual_pcie = ssd.get_telemetry()["total_pcie_bytes"]
        bus_saved_pct = ((cold_bytes_full - actual_pcie) / max(1, cold_bytes_full)) * 100.0

        print(f"{gen_step:<6} | {stats['hot_tokens']:<8} | {stats['cold_blocks_total']:<10} | {stats['cold_blocks_retrieved']:<10} | {mean_prune:6.1f}%    | {bus_saved_pct:6.1f}%          | {mean_cos:8.6f}")

    print("-" * 80)
    print("\n---> [STEP 3] Final Hackathon Metrics Summary:")
    print(f"     [+] Host RAM Reduction:          {ram_savings_pct:.1f}%")
    print(f"     [+] Cold Block Pruning Ratio:     {mean_prune:.1f}%")
    print(f"     [+] Host PCIe Bus Traffic Saved:  {bus_saved_pct:.1f}%")
    print(f"     [+] Output Mathematical Fidelity: {mean_cos:.6f} Cosine Similarity (Target >= 0.990)")
    print("\n[SUCCESS] AI-Aware SSD Computational KV Cache Engine Demonstration Complete.")
    print("=" * 80)


if __name__ == "__main__":
    run_live_hackathon_demo()
