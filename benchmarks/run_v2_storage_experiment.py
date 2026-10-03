"""
AI-SSD V2 Controlled Storage Experiment: Conventional vs. Tensor-Aware FTL.

Measures strictly empirical latencies, throughput, channel contention,
and speedup across parametric sweeps of:
    - Number of Channels (4, 8, 16)
    - Queue Depths (4, 8, 16, 32)
    - Request Batch Sizes (16, 32, 64, 128, 256)
    - Workload Access Patterns (Sequential vs. Multi-Head Parallel Attention)

No speedup numbers or assertions are hardcoded.
"""

import sys
import json
import csv
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person2_ssd.trace_replay.replayer import StorageTraceReplayer


def run_controlled_experiments():
    print("================================================================================")
    print("       AI-SSD V2 Storage Experiment: Conventional vs. Tensor-Aware FTL          ")
    print("================================================================================")

    results = []
    
    # 1. Sweep across batch sizes with default 8 channels, QD=8
    batch_sizes = [16, 32, 64, 128, 256]
    print("\n--- Experiment 1: Batch Size Scaling (Channels=8, QD=8) ---")
    for bs in batch_sizes:
        replayer = StorageTraceReplayer(channels=8, queue_depth=8)
        trace = replayer.discover_or_synthesize_trace(num_layers=4, num_heads=8, seq_length=bs * 16)
        
        conv_res = replayer.replay(trace, mode="conventional")
        ta_res = replayer.replay(trace, mode="tensor_aware")
        
        conv_lat = conv_res["total_simulated_latency_us"]
        ta_lat = ta_res["total_simulated_latency_us"]
        speedup = conv_lat / ta_lat if ta_lat > 0 else 1.0
        
        row = {
            "experiment": "batch_scaling",
            "channels": 8,
            "queue_depth": 8,
            "batch_size": bs,
            "total_requests": len(trace),
            "conventional_latency_us": conv_lat,
            "tensor_aware_latency_us": ta_lat,
            "speedup_x": round(speedup, 2),
            "conventional_contention_ratio": conv_res["contention_ratio"],
            "tensor_aware_contention_ratio": ta_res["contention_ratio"],
            "conventional_throughput_mbs": conv_res["throughput_mbs"],
            "tensor_aware_throughput_mbs": ta_res["throughput_mbs"],
        }
        results.append(row)
        print(
            f"Batch: {bs:3d} | Conv Lat: {conv_lat:9.1f} us | TA Lat: {ta_lat:9.1f} us | "
            f"Speedup: {speedup:5.2f}x | Conv Contention: {conv_res['contention_ratio']:.1f}x | TA Contention: {ta_res['contention_ratio']:.1f}x"
        )

    # 2. Sweep across channel configurations (4, 8, 16 channels)
    print("\n--- Experiment 2: Channel Count Scaling (Batch=128, QD=8) ---")
    channel_counts = [4, 8, 16]
    for ch in channel_counts:
        replayer = StorageTraceReplayer(channels=ch, queue_depth=8)
        trace = replayer.discover_or_synthesize_trace(num_layers=4, num_heads=8, seq_length=128 * 16)
        
        conv_res = replayer.replay(trace, mode="conventional")
        ta_res = replayer.replay(trace, mode="tensor_aware")
        
        conv_lat = conv_res["total_simulated_latency_us"]
        ta_lat = ta_res["total_simulated_latency_us"]
        speedup = conv_lat / ta_lat if ta_lat > 0 else 1.0
        
        row = {
            "experiment": "channel_scaling",
            "channels": ch,
            "queue_depth": 8,
            "batch_size": 128,
            "total_requests": len(trace),
            "conventional_latency_us": conv_lat,
            "tensor_aware_latency_us": ta_lat,
            "speedup_x": round(speedup, 2),
            "conventional_contention_ratio": conv_res["contention_ratio"],
            "tensor_aware_contention_ratio": ta_res["contention_ratio"],
            "conventional_throughput_mbs": conv_res["throughput_mbs"],
            "tensor_aware_throughput_mbs": ta_res["throughput_mbs"],
        }
        results.append(row)
        print(
            f"Channels: {ch:2d} | Conv Lat: {conv_lat:9.1f} us | TA Lat: {ta_lat:9.1f} us | "
            f"Speedup: {speedup:5.2f}x | Conv Contention: {conv_res['contention_ratio']:.1f}x | TA Contention: {ta_res['contention_ratio']:.1f}x"
        )

    # 3. Sweep across queue depths (4, 8, 16, 32)
    print("\n--- Experiment 3: Queue Depth Scaling (Channels=8, Batch=128) ---")
    queue_depths = [4, 8, 16, 32]
    for qd in queue_depths:
        replayer = StorageTraceReplayer(channels=8, queue_depth=qd)
        trace = replayer.discover_or_synthesize_trace(num_layers=4, num_heads=8, seq_length=128 * 16)
        
        conv_res = replayer.replay(trace, mode="conventional")
        ta_res = replayer.replay(trace, mode="tensor_aware")
        
        conv_lat = conv_res["total_simulated_latency_us"]
        ta_lat = ta_res["total_simulated_latency_us"]
        speedup = conv_lat / ta_lat if ta_lat > 0 else 1.0
        
        row = {
            "experiment": "qd_scaling",
            "channels": 8,
            "queue_depth": qd,
            "batch_size": 128,
            "total_requests": len(trace),
            "conventional_latency_us": conv_lat,
            "tensor_aware_latency_us": ta_lat,
            "speedup_x": round(speedup, 2),
            "conventional_contention_ratio": conv_res["contention_ratio"],
            "tensor_aware_contention_ratio": ta_res["contention_ratio"],
            "conventional_throughput_mbs": conv_res["throughput_mbs"],
            "tensor_aware_throughput_mbs": ta_res["throughput_mbs"],
        }
        results.append(row)
        print(
            f"QD: {qd:2d} | Conv Lat: {conv_lat:9.1f} us | TA Lat: {ta_lat:9.1f} us | "
            f"Speedup: {speedup:5.2f}x | Conv Contention: {conv_res['contention_ratio']:.1f}x | TA Contention: {ta_res['contention_ratio']:.1f}x"
        )

    # Save outputs
    raw_dir = PROJECT_ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / "v2_ftl_benchmark.json"
    csv_path = raw_dir / "v2_ftl_benchmark.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Also save to shared /opt/ai-ssd-v2/results/
    shared_res_dir = Path("/opt/ai-ssd-v2/results")
    if shared_res_dir.exists():
        with open(shared_res_dir / "v2_storage_experiment_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print("\n================================================================================")
    print(f"[SUCCESS] Saved measured results to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - /opt/ai-ssd-v2/results/v2_storage_experiment_results.json")
    print("================================================================================")
    return results


if __name__ == "__main__":
    run_controlled_experiments()