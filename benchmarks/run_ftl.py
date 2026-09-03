"""
Benchmark: Conventional vs Tensor-Aware FTL (Person 2)
Evaluates read latency and channel parallelism across parallel request batch sizes.
"""

import sys
import csv
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from person2_ssd.mock_kv_engine import MockKVEngine
from person2_ssd.storage_model.io_model import StorageSimulator


def run_ftl_benchmark():
    print("=== Running FTL Comparison Benchmark ===")
    mock_kv = MockKVEngine(layers=32, heads=32)
    batch_sizes = [16, 32, 64, 128, 256]

    conv_ssd = StorageSimulator(mode="conventional", channels=8)
    ta_ssd = StorageSimulator(mode="tensor_aware", channels=8)

    rows = []
    for batch_size in batch_sizes:
        blocks = mock_kv.generate_kv_blocks(num_blocks=batch_size, layer_id=0)
        
        # Store in both
        for b in blocks:
            conv_ssd.store_block(b)
            ta_ssd.store_block(b)

        b_ids = [b.block_id for b in blocks]
        conv_lat = conv_ssd.estimate_read_latency(b_ids)
        ta_lat = ta_ssd.estimate_read_latency(b_ids)
        speedup = conv_lat / ta_lat if ta_lat > 0 else 1.0

        rows.append({
            "experiment": "ftl_comparison",
            "batch_size": batch_size,
            "conventional_latency_us": round(conv_lat, 1),
            "tensor_aware_latency_us": round(ta_lat, 1),
            "speedup_x": round(speedup, 2),
        })
        print(f"Batch: {batch_size:3d} blocks | Conventional: {conv_lat:7.1f} us | Tensor-Aware: {ta_lat:7.1f} us | Speedup: {speedup:.2f}x")

    out_dir = Path("results/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ftl_results.csv"
    if rows:
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved FTL results to: {out_file}\n")


if __name__ == "__main__":
    run_ftl_benchmark()
