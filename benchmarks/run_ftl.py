"""
Benchmark: Conventional vs Tensor-Aware FTL (Person 2 Subsystem)
Evaluates read latency and channel parallelism across parallel request batch sizes.
Outputs benchmark metrics to results/raw/ftl_results.csv.
"""

import sys
import csv
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person2_ssd.mock_kv_engine import MockKVEngine
from person2_ssd.storage_model.io_model import StorageSimulator


def run_ftl_benchmark() -> list:
    print("==================================================")
    print("       AI-SSD FTL Comparison Benchmark           ")
    print("==================================================")
    mock_kv = MockKVEngine(layers=32, heads=32)
    batch_sizes = [16, 32, 64, 128, 256]

    rows = []
    for batch_size in batch_sizes:
        # Fresh isolated simulation per batch size to ensure clean repeatable metrics
        mock_kv.reset()
        conv_ssd = StorageSimulator(mode="conventional", channels=8)
        ta_ssd = StorageSimulator(mode="tensor_aware", channels=8)

        blocks = mock_kv.generate_kv_blocks(num_blocks=batch_size, layer_id=0, layout="token_major")

        for b in blocks:
            conv_ssd.store_block(b)
            ta_ssd.store_block(b)

        b_ids = [b.block_id for b in blocks]
        conv_lat = conv_ssd.estimate_read_latency(b_ids)
        ta_lat = ta_ssd.estimate_read_latency(b_ids)
        speedup = conv_lat / ta_lat if ta_lat > 0 else 1.0

        # Assert speedup >= 2.5x for target batch sizes (64, 128, 256)
        if batch_size in (64, 128, 256):
            assert speedup >= 2.5, (
                f"Benchmark target unmet: speedup {speedup:.2f}x < 2.5x at batch size {batch_size}"
            )

        rows.append({
            "experiment": "ftl_comparison",
            "batch_size": batch_size,
            "conventional_latency_us": round(conv_lat, 1),
            "tensor_aware_latency_us": round(ta_lat, 1),
            "speedup_x": round(speedup, 2),
        })
        print(
            f"Batch: {batch_size:3d} blocks | "
            f"Conventional: {conv_lat:7.1f} us | "
            f"Tensor-Aware: {ta_lat:7.1f} us | "
            f"Speedup: {speedup:5.2f}x"
        )

    out_dir = PROJECT_ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ftl_results.csv"

    fieldnames = [
        "experiment",
        "batch_size",
        "conventional_latency_us",
        "tensor_aware_latency_us",
        "speedup_x",
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("==================================================")
    print(f"[SUCCESS] Saved FTL benchmark results to: {out_file}")
    print(f"[VERIFIED] Speedup >= 2.5x achieved for all target batch sizes (64, 128, 256)")
    print("==================================================\n")
    return rows


if __name__ == "__main__":
    run_ftl_benchmark()
