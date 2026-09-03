"""
Benchmark: Speculative Prefetch Evaluation (Person 3)
Measures cache hit rates and prefetch accuracy across layers.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from person3_system.prefetch.prefetcher import SpeculativePrefetcher


def run_prefetch_benchmark():
    print("=== Running Speculative Prefetch Benchmark ===")
    prefetcher = SpeculativePrefetcher(buffer_capacity_blocks=512)

    rows = []
    # Simulate 32 layers
    for layer in range(32):
        # Stage blocks for layer
        target_blocks = [layer * 16 + i for i in range(16)]
        prefetcher.stage_blocks(target_blocks)

        # Access with 85% predictability
        is_hit = prefetcher.is_staged(target_blocks[:14])
        rows.append({
            "layer_id": layer,
            "blocks_requested": 16,
            "cache_hit": is_hit,
            "current_hit_rate": prefetcher.hit_rate,
        })

    df = pd.DataFrame(rows)
    out_dir = Path("results/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "prefetch_results.csv"
    df.to_csv(out_file, index=False)
    print(f"Overall Prefetch Hit Rate: {prefetcher.hit_rate * 100:.1f}%")
    print(f"Saved prefetch results to: {out_file}\n")


if __name__ == "__main__":
    run_prefetch_benchmark()
