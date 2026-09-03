"""
Benchmark: Full End-to-End System Evaluation (Person 3 Subsystem)
Wires real Person 1 KV engine + real Person 2 physical flash SSD + Person 3 prefetcher.
Executes multi-layer inference across context lengths (4K -> 32K) and outputs empirical metrics.json.
"""

import sys
import csv
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.schemas.metrics import (
    SystemMetrics,
    MemoryMetrics,
    LatencyMetrics,
    StorageMetrics,
    PrefetchMetrics,
    FTLMetrics,
)
from common.utils import save_json
from person3_system.integration.orchestrator import SystemOrchestrator
from person3_system.integration.pipeline import InferencePipeline


def run_full_system() -> Dict[str, Any]:
    print("==========================================================")
    print("       AI-SSD Full End-to-End System Benchmark            ")
    print("==========================================================")

    orchestrator = SystemOrchestrator(
        mode="tensor_aware",
        num_layers=32,
        num_heads=32,
        head_dim=128,
        dtype="FP16",
        enable_prefetch=True,
    )
    pipeline = InferencePipeline(orchestrator=orchestrator)

    context_lengths = [4096, 8192, 16384, 32768]
    scaling_rows = []
    primary_metrics = None

    for ctx in context_lengths:
        print(f"\n[*] Evaluating Context Length: {ctx:,} tokens (FP16)...")
        res = pipeline.run_simulation(
            context_length=ctx,
            num_layers=32,
            num_heads=32,
            head_dim=128,
            offload_pct=80.0,
            topk_pct=10.0,
            dtype="FP16",
        )

        mem = res["memory"]
        stor = res["storage"]
        ftl = res["ftl"]
        pref = res["prefetch"]
        lat = res["latency"]

        print(f"    - RAM Footprint:    {mem['baseline_mb']:,.1f} MB -> {mem['proposed_mb']:,.1f} MB ({mem['reduction_percent']}% reduction)")
        print(f"    - PCIe Bus Saved:   {stor['traffic_reduction_percent']}% traffic reduction")
        print(f"    - FTL Read Speedup: {ftl['speedup_x']}x ({ftl['baseline_read_us']} us -> {ftl['tensor_aware_read_us']} us)")
        print(f"    - Prefetch Hit Rate: {pref['cache_hit_rate']*100:.1f}%")
        print(f"    - E2E Latency:      {lat['baseline_ms']} ms -> {lat['proposed_ms']} ms (+{lat['overhead_percent']}%)")

        scaling_rows.append({
            "context_length": ctx,
            "baseline_ram_mb": mem["baseline_mb"],
            "proposed_ram_mb": mem["proposed_mb"],
            "ram_reduction_pct": mem["reduction_percent"],
            "bytes_requested": stor["bytes_requested"],
            "bytes_transferred": stor["bytes_transferred"],
            "traffic_reduction_pct": stor["traffic_reduction_percent"],
            "ftl_speedup_x": ftl["speedup_x"],
            "prefetch_hit_rate": pref["cache_hit_rate"],
            "e2e_latency_ms": lat["proposed_ms"],
            "e2e_overhead_pct": lat["overhead_percent"],
        })

        if ctx == 32768:
            primary_metrics = res

    # Write unified metrics.json for 32K context
    out_dir = PROJECT_ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / "metrics.json"

    sys_metrics = SystemMetrics(
        memory=MemoryMetrics(
            baseline_mb=primary_metrics["memory"]["baseline_mb"],
            proposed_mb=primary_metrics["memory"]["proposed_mb"],
            reduction_percent=primary_metrics["memory"]["reduction_percent"],
        ),
        latency=LatencyMetrics(
            baseline_ms=primary_metrics["latency"]["baseline_ms"],
            proposed_ms=primary_metrics["latency"]["proposed_ms"],
            overhead_percent=primary_metrics["latency"]["overhead_percent"],
        ),
        storage=StorageMetrics(
            bytes_requested=primary_metrics["storage"]["bytes_requested"],
            bytes_transferred=primary_metrics["storage"]["bytes_transferred"],
            traffic_reduction_percent=primary_metrics["storage"]["traffic_reduction_percent"],
        ),
        prefetch=PrefetchMetrics(
            prediction_accuracy=primary_metrics["prefetch"]["prediction_accuracy"],
            cache_hit_rate=primary_metrics["prefetch"]["cache_hit_rate"],
        ),
        ftl=FTLMetrics(
            baseline_read_us=primary_metrics["ftl"]["baseline_read_us"],
            tensor_aware_read_us=primary_metrics["ftl"]["tensor_aware_read_us"],
            speedup_x=primary_metrics["ftl"]["speedup_x"],
        ),
    )
    save_json(sys_metrics.to_dict(), metrics_file)

    # Save scaling CSV
    scaling_file = out_dir / "full_system_scaling.csv"
    with open(scaling_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scaling_rows[0].keys())
        writer.writeheader()
        writer.writerows(scaling_rows)

    print("\n==========================================================")
    print(f"[SUCCESS] Empirical metrics saved to: {metrics_file}")
    print(f"[SUCCESS] Scaling results saved to:   {scaling_file}")
    print("==========================================================\n")

    # Assert locked targets
    assert primary_metrics["memory"]["reduction_percent"] >= 80.0
    assert primary_metrics["storage"]["traffic_reduction_percent"] >= 80.0
    assert primary_metrics["ftl"]["speedup_x"] >= 7.0
    assert primary_metrics["prefetch"]["cache_hit_rate"] >= 0.80
    assert primary_metrics["latency"]["overhead_percent"] <= 18.0

    return primary_metrics


if __name__ == "__main__":
    run_full_system()
