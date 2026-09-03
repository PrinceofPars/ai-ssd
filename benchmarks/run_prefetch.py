"""
Benchmark: Speculative Prefetch Evaluation (Person 3 Subsystem)
Evaluates inter-layer speculative prefetching (Layer L -> L+1), cache hit rates,
pipeline bubble elimination, and DRAM staging overhead across 32 transformer layers.
Outputs benchmark metrics to results/raw/prefetch_results.csv.
"""

import sys
import csv
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person3_system.prefetch.prefetcher import SpeculativePrefetcher
from person3_system.adapters.physical_adapter import PhysicalTensorAwareStorageAdapter
from person2_ssd.storage_model.io_model import StorageSimulator


def run_prefetch_benchmark() -> List[Dict[str, Any]]:
    print("==================================================")
    print("       AI-SSD Speculative Prefetch Benchmark      ")
    print("==================================================")

    total_layers = 32
    tokens_per_block = 16
    context_length = 32768
    num_blocks_per_layer = context_length // tokens_per_block  # 2048 blocks

    # Initialize physical storage and speculative prefetcher
    ssd = StorageSimulator(mode="tensor_aware", channels=8)
    prefetcher = SpeculativePrefetcher(
        buffer_capacity_blocks=512,  # 2 MB host DRAM budget
        bytes_per_block=4096,
        gpu_compute_time_per_layer_us=65.0,
    )

    rows = []
    # Seed prompt topic blocks that persist across adjacent layers
    topic_blocks = [10, 11, 12, 13, 40, 41, 42, 80, 81, 82, 150, 151, 152, 200, 201, 202]

    from common.schemas.kv_block import KVBlock
    # Pre-store blocks in StorageSimulator so physical FTL and channel timing are modeled
    all_bids = set(topic_blocks + [300 + i for i in range(total_layers)])
    for bid in all_bids:
        blk = KVBlock.create_default(block_id=bid, layer_id=0, token_start=bid * 16)
        ssd.store_block(blk)

    for layer_id in range(total_layers):
        # 1. Blocks needed by current layer: 85% persistent topic blocks + 15% random explore
        needed_blocks = list(topic_blocks[:14])
        if layer_id % 4 == 0:
            needed_blocks.append(300 + layer_id)  # Occasional fresh block
        else:
            needed_blocks.append(topic_blocks[14])

        # Estimate un-prefetched flash latency through Person 2 physical model
        flash_lat_us = ssd.estimate_read_latency(needed_blocks)

        # 2. Check if prefetcher already has these blocks staged
        is_hit = prefetcher.is_staged(
            block_ids=needed_blocks,
            layer_id=layer_id,
            estimated_flash_latency_us=flash_lat_us,
        )

        effective_latency_us = 0.0 if is_hit else flash_lat_us
        bubble_saved_us = flash_lat_us if is_hit else 0.0

        rows.append({
            "layer_id": layer_id,
            "blocks_requested": len(needed_blocks),
            "cache_hit": is_hit,
            "flash_latency_us": round(flash_lat_us, 1),
            "effective_latency_us": round(effective_latency_us, 1),
            "bubble_saved_us": round(bubble_saved_us, 1),
            "current_hit_rate": round(prefetcher.hit_rate * 100.0, 1),
            "dram_staged_kb": round(prefetcher.used_capacity_bytes / 1024.0, 1),
        })

        # 3. Asynchronously prefetch next layer (L+1) while current layer computes
        prefetcher.prefetch_next_layer(
            current_layer_id=layer_id,
            active_block_ids=needed_blocks,
        )

    # Save benchmark results
    out_dir = PROJECT_ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "prefetch_results.csv"

    fieldnames = [
        "layer_id",
        "blocks_requested",
        "cache_hit",
        "flash_latency_us",
        "effective_latency_us",
        "bubble_saved_us",
        "current_hit_rate",
        "dram_staged_kb",
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    hit_pct = prefetcher.hit_rate * 100.0
    total_bubble_saved = sum(r["bubble_saved_us"] for r in rows)

    print(f"Total Layers Evaluated:    {total_layers}")
    print(f"Overall Prefetch Hit Rate: {hit_pct:.1f}% (Target: >= 80.0%)")
    print(f"Total Flash Bubble Saved:  {total_bubble_saved:.1f} us")
    print(f"Pipeline Stalls Incurred:  {prefetcher.pipeline_stalls}")
    print(f"Saved prefetch results to: {out_file}")
    print("==================================================\n")

    assert hit_pct >= 80.0, f"Prefetch hit rate {hit_pct:.1f}% < target 80.0%"
    return rows


if __name__ == "__main__":
    run_prefetch_benchmark()
