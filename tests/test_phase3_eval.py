"""
Test suite for AI-SSD V2 Phase 3 Real Prefetch and System Evaluation.
"""

from pathlib import Path
import json
import pytest

from benchmarks.run_phase3_eval import Phase3SystemEvaluator, RESULTS_DIR


def test_phase3_evaluation_execution_and_exports():
    """Verify that Phase 3 evaluator executes cleanly on the real trace and generates valid outputs."""
    evaluator = Phase3SystemEvaluator()
    results = evaluator.run_all_and_export()

    # Check structure
    assert "provenance" in results
    assert "prefetch_ablations" in results
    assert "system_ablations" in results
    assert "virtual_nvme_storage" in results
    assert "summary_claims" in results

    # Verify Prefetch metrics
    pref = results["prefetch_ablations"]
    assert "no_prefetch" in pref
    assert "conservative_prefetch" in pref
    assert "normal_prefetch" in pref
    assert "aggressive_prefetch" in pref

    # Check demand read invariants
    for p_id, p_data in pref.items():
        assert p_data["demand_reads"] == 5376
        assert p_data["demand_hits"] + p_data["demand_misses"] == 5376
        assert p_data["classification"] == "ANALYTICAL — real workload trace"

    # Monotonic progression of hit rate
    hit_rates = [
        pref["no_prefetch"]["prefetch_hit_rate_pct"],
        pref["conservative_prefetch"]["prefetch_hit_rate_pct"],
        pref["normal_prefetch"]["prefetch_hit_rate_pct"],
        pref["aggressive_prefetch"]["prefetch_hit_rate_pct"],
    ]
    assert hit_rates[0] == 0.0
    assert 20.0 <= hit_rates[1] <= 35.0
    assert 55.0 <= hit_rates[2] <= 75.0
    assert 90.0 <= hit_rates[3] <= 99.0
    assert hit_rates[0] < hit_rates[1] < hit_rates[2] < hit_rates[3]

    # Verify System Ablation metrics
    abl = results["system_ablations"]
    assert "A_baseline_dense_no_prefetch" in abl
    assert "B_sparse_kv_no_prefetch" in abl
    assert "C_sparse_kv_prefetch" in abl
    assert "D_conventional_ftl" in abl
    assert "E_tensor_aware_ftl" in abl
    assert "F_full_combined_system" in abl

    # Check speedup invariants
    assert abl["E_tensor_aware_ftl"]["speedup_vs_conventional"] > 2.0
    assert abl["F_full_combined_system"]["throughput_relative_to_dram_pct"] > 90.0

    # Verify Virtual NVMe storage baseline
    storage = results["virtual_nvme_storage"]
    assert storage["classification"] == "VIRTUAL-DEVICE"
    assert storage["file_backend_verified"] is True
    assert "qemu_guest_benchmarks" in storage

    # Verify generated artifact files on disk
    assert (RESULTS_DIR / "phase3_prefetch_ablations.json").exists()
    assert (RESULTS_DIR / "phase3_system_ablations.json").exists()
    assert (RESULTS_DIR / "unified_results.json").exists()
    assert (RESULTS_DIR / "prefetch_summary.csv").exists()
    assert (RESULTS_DIR / "system_ablations_summary.csv").exists()
