"""
AI-SSD V2 Full System Evaluation CLI.
Runs standardized baselines and ablations across context lengths.
Writes structured JSON, JSONL, and CSV outputs to /opt/ai-ssd-v2/results/ and results/raw/.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person3_system.experiments.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="AI-SSD V2 Evaluation Runner")
    parser.add_argument("--context-length", type=int, default=4096, help="Sequence context length in tokens")
    parser.add_argument("--steps", type=int, default=5, help="Number of decode steps to simulate")
    parser.add_argument("--trace", type=str, default=None, help="Path to real P1 trace file if available")
    parser.add_argument("--out-dir", type=str, default="/opt/ai-ssd-v2/results", help="Directory for exported results")
    args = parser.parse_args()

    print("================================================================================")
    print("                      AI-SSD V2 SYSTEM EVALUATION SUITE                         ")
    print("================================================================================")
    print(f"Context Length : {args.context_length:,} tokens")
    print(f"Decode Steps   : {args.steps}")
    print(f"Trace Path     : {args.trace or 'Auto-generated synthetic P1-compliant trace'}")
    print(f"Results Output : {args.out_dir} & results/raw/")
    print("================================================================================\n")

    runner = ExperimentRunner(primary_results_dir=args.out_dir)
    results = runner.run_standard_ablations(
        context_length=args.context_length,
        trace_path=args.trace,
        num_steps=args.steps,
    )

    print("\n" + "=" * 105)
    print(f"{'Experiment ID':<28} | {'RAM Red%':<9} | {'Reads':<8} | {'Lat (us)':<10} | {'Pref Hits%':<11} | {'Time (ms)':<10} | {'Tok/s':<7}")
    print("=" * 105)
    for r in results:
        cfg = r.config
        sys_m = r.system
        stor_m = r.storage
        pref_m = r.prefetch
        print(
            f"{cfg.experiment_id:<28} | "
            f"{sys_m.ram_reduction_pct:>7.1f}% | "
            f"{stor_m.storage_reads:>8,d} | "
            f"{stor_m.storage_latency_us:>10,.1f} | "
            f"{pref_m.prefetch_hit_rate_pct:>9.1f}% | "
            f"{sys_m.total_execution_time_ms:>10.2f} | "
            f"{sys_m.throughput_tokens_per_sec:>7.1f}"
        )
    print("=" * 105)
    print(f"\n[✓] Results successfully exported to {args.out_dir}/ and results/raw/ (JSON, JSONL, CSV).\n")


if __name__ == "__main__":
    main()
