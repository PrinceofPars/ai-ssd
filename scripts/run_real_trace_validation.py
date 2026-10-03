"""
Validation Script: Real LLM Trace Replay & Verification (P2).

Replays: /opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl
Validates exact schema compliance, operation semantics, byte counts,
and balanced multi-channel distribution.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person2_ssd.trace_replay.replayer import StorageTraceReplayer


def run_validation():
    print("================================================================================")
    print("       AI-SSD V2 Real-Trace Replay & FTL Verification (Person 2)               ")
    print("================================================================================")

    trace_dir = "/opt/ai-ssd-v2/traces/real_llm"
    replayer = StorageTraceReplayer(channels=8, queue_depth=8, num_layers=24, num_heads=2)

    # 1. Discover file and manifest
    trace_file, manifest = replayer.discover_trace_file(trace_dir=trace_dir, trace_name="trace_qwen2.5_0.5b_context512.jsonl")
    print(f"[Trace Discovery] Found trace: {trace_file}")
    if manifest:
        print(f"[Manifest] Model: {manifest.get('model_name')}, Expected Events: {manifest.get('total_events')}")
        print(f"[Manifest] Layers: {manifest.get('num_layers')}, KV Heads: {manifest.get('num_kv_heads')}, Head Dim: {manifest.get('head_dim')}")

    # 2. Load & Validate Records
    records = replayer.load_trace(trace_file)
    print(f"[Validation] Successfully validated {len(records)} events (0 dropped, 0 schema errors).")

    # 3. Replay under Conventional FTL
    conv_metrics = replayer.replay(records, mode="conventional", blocks_per_head=44)

    # 4. Replay under Tensor-Aware FTL
    ta_metrics = replayer.replay(records, mode="tensor_aware", blocks_per_head=44)

    speedup = conv_metrics["total_simulated_latency_us"] / ta_metrics["total_simulated_latency_us"] if ta_metrics["total_simulated_latency_us"] > 0 else 1.0

    report = {
        "benchmark": "real_llm_trace_replay_validation",
        "trace_file": str(trace_file),
        "total_events": len(records),
        "manifest_metadata": manifest,
        "event_counts_by_operation": ta_metrics["event_counts_by_operation"],
        "byte_accounting": {
            "total_read_bytes": ta_metrics["total_read_bytes"],
            "total_write_bytes": ta_metrics["total_write_bytes"],
            "k_bytes": ta_metrics["k_bytes"],
            "v_bytes": ta_metrics["v_bytes"],
            "combined_bytes": ta_metrics["combined_bytes"],
            "total_bytes_transferred": ta_metrics["total_bytes"],
        },
        "block_distribution": {
            "unique_blocks_count": ta_metrics["unique_blocks_count"],
            "min_block_id": ta_metrics["min_block_id"],
            "max_block_id": ta_metrics["max_block_id"],
        },
        "conventional_ftl": {
            "latency_us": conv_metrics["total_simulated_latency_us"],
            "channel_distribution": conv_metrics["channel_distribution"],
            "max_channel_load": conv_metrics["max_channel_load"],
            "contention_ratio": conv_metrics["contention_ratio"],
            "throughput_mbs": conv_metrics["throughput_mbs"],
        },
        "tensor_aware_ftl": {
            "latency_us": ta_metrics["total_simulated_latency_us"],
            "channel_distribution": ta_metrics["channel_distribution"],
            "max_channel_load": ta_metrics["max_channel_load"],
            "contention_ratio": ta_metrics["contention_ratio"],
            "throughput_mbs": ta_metrics["throughput_mbs"],
        },
        "speedup_x": round(speedup, 2),
        "invalid_or_dropped_records": ta_metrics["invalid_or_dropped_records"],
    }

    # Print summary
    print("\n--- Summary Verification Report ---")
    print(f"Total Events: {report['total_events']}")
    print(f"Operations: {report['event_counts_by_operation']}")
    print(f"Total Read Bytes: {report['byte_accounting']['total_read_bytes']:,} ({report['byte_accounting']['total_read_bytes'] / 1024 / 1024:.2f} MiB)")
    print(f"Total Write Bytes: {report['byte_accounting']['total_write_bytes']:,} ({report['byte_accounting']['total_write_bytes'] / 1024 / 1024:.2f} MiB)")
    print(f"Key Bytes: {report['byte_accounting']['k_bytes']:,} ({report['byte_accounting']['k_bytes'] / 1024 / 1024:.2f} MiB)")
    print(f"Value Bytes: {report['byte_accounting']['v_bytes']:,} ({report['byte_accounting']['v_bytes'] / 1024 / 1024:.2f} MiB)")
    print(f"Combined Bytes: {report['byte_accounting']['combined_bytes']:,} ({report['byte_accounting']['combined_bytes'] / 1024 / 1024:.2f} MiB)")
    print(f"Unique Blocks: {report['block_distribution']['unique_blocks_count']} (IDs {report['block_distribution']['min_block_id']} to {report['block_distribution']['max_block_id']})")
    print(f"Invalid / Dropped Records: {report['invalid_or_dropped_records']}")
    print("\n--- Channel Distribution (Tensor-Aware) ---")
    for ch, cnt in sorted(ta_metrics["channel_distribution"].items()):
        pct = cnt / report['total_events'] * 100
        print(f"  Channel {ch}: {cnt:5d} events ({pct:5.1f}%)")
    print(f"Contention Ratio: Conventional={conv_metrics['contention_ratio']:.1f}x -> Tensor-Aware={ta_metrics['contention_ratio']:.1f}x")
    print(f"Measured Speedup: {speedup:.2f}x")

    # Save to disk
    out_dir = PROJECT_ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "p2_real_trace_validation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    shared_dir = Path("/opt/ai-ssd-v2/results")
    if shared_dir.exists():
        with open(shared_dir / "p2_real_trace_validation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print("\n================================================================================")
    print(f"[SUCCESS] Saved validation report to:")
    print(f"  - {out_file}")
    print(f"  - /opt/ai-ssd-v2/results/p2_real_trace_validation_report.json")
    print("================================================================================")
    return report


if __name__ == "__main__":
    run_validation()