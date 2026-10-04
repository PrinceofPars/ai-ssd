"""Unit and Regression Tests for Phase 9: Final End-to-End Validation & Benchmark Matrix.

Validates:
1. Exact Token-ID matching across runs and context lengths.
2. Zero-candidate-K host data movement invariant under computational storage.
3. True host-RAM offload invariant: P2 and P3 resident payload is 0.0 MB.
4. Tensor-Aware FTL channel distribution balance (< 5% imbalance across 8 channels).
5. Critical-path reconciliation error bound (< 2.0%).
6. Monotonic physical DRAM savings under context scaling (4K -> 32K).
"""

import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "benchmarks" / "live_inference" / "results" / "final_benchmark_results.json"

EXPECTED_TOKENS = [11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13]


@pytest.fixture(scope="module")
def benchmark_results():
    if not RESULTS_FILE.exists():
        pytest.skip(f"Benchmark results file not found at {RESULTS_FILE}")
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


def test_phase9_canonical_token_match(benchmark_results):
    """Verify that canonical reproduction exactly matches the 16 reference token IDs."""
    canon = benchmark_results.get("canonical_reproduction", {})
    assert canon.get("token_ids") == EXPECTED_TOKENS, (
        f"Canonical token mismatch: {canon.get('token_ids')} != {EXPECTED_TOKENS}"
    )


def test_phase9_benchmark_matrix_token_match(benchmark_results):
    """Verify that all 4K benchmark matrix configurations match reference token IDs."""
    matrix = benchmark_results.get("benchmark_matrix", {})
    for run_key, run_data in matrix.items():
        assert run_data.get("token_ids") == EXPECTED_TOKENS, (
            f"Token mismatch in {run_key}: {run_data.get('token_ids')} != {EXPECTED_TOKENS}"
        )


def test_phase9_computational_storage_bus_movement_invariant(benchmark_results):
    """Verify that candidate Key bytes to host is strictly 0 in computational storage mode."""
    matrix = benchmark_results.get("benchmark_matrix", {})
    
    # Computational storage runs must have 0 candidate K to host
    for comp_key in ["run_b_file_backed", "run_d_qemu_comp_sync", "run_e_qemu_final_async", "run_f_qemu_async_prefetch"]:
        run_data = matrix[comp_key]
        cand_k = run_data.get("candidate_k_bytes_to_host", -1)
        assert cand_k == 0, f"{comp_key} sent {cand_k} bytes of candidate K to host (must be 0)"
    
    # Host-side Top-K run MUST have transferred candidate keys to host
    host_topk = matrix["run_c_qemu_host_topk"]
    assert host_topk.get("candidate_k_bytes_to_host", 0) > 500_000_000, (
        "Host Top-K run did not register expected candidate K host traffic"
    )


def test_phase9_true_host_ram_offload_invariants(benchmark_results):
    """Verify that P2 and P3 resident payload is 0.0 MB across all AI-SSD configurations."""
    matrix = benchmark_results.get("benchmark_matrix", {})
    for run_key, run_data in matrix.items():
        if "baseline" in run_key:
            continue
        p2_res = run_data.get("p2_resident_payload_mb", -1)
        p3_res = run_data.get("p3_resident_payload_mb", -1)
        assert p2_res == 0.0, f"{run_key} has non-zero P2 payload: {p2_res} MB"
        assert p3_res == 0.0, f"{run_key} has non-zero P3 payload: {p3_res} MB"


def test_phase9_ftl_channel_load_balance(benchmark_results):
    """Verify that Tensor-Aware FTL distributes requests evenly (< 5% load imbalance across 8 channels)."""
    ftl = benchmark_results.get("ftl_comparison", {})
    ta = ftl.get("tensor_aware", {})
    assert ta.get("load_imbalance_percent", 999.0) < 5.0, (
        f"Tensor-aware load imbalance exceeds 5%: {ta.get('load_imbalance_percent')}%"
    )
    
    # Conventional mapping must exhibit extreme imbalance
    conv = ftl.get("conventional", {})
    assert conv.get("load_imbalance_percent", 0.0) > 100.0, (
        f"Conventional mapping load imbalance expected > 100%, got {conv.get('load_imbalance_percent')}%"
    )


def test_phase9_critical_path_reconciliation_error(benchmark_results):
    """Verify that critical-path reconciliation error is strictly below 2.0%."""
    matrix = benchmark_results.get("benchmark_matrix", {})
    for run_key, run_data in matrix.items():
        if "baseline" in run_key:
            continue
        recon_err = run_data.get("reconciliation_error_pct", 999.0)
        assert recon_err < 2.0, f"{run_key} reconciliation error {recon_err}% exceeds 2.0% threshold"


def test_phase9_context_scaling_monotonic_memory_savings(benchmark_results):
    """Verify that memory savings scale monotonically with context length from 4K to 32K."""
    scaling = benchmark_results.get("context_scaling", {})
    contexts = [4096, 8192, 16384, 32768]
    prev_saved_mb = 0.0
    
    for ctx in contexts:
        ctx_str = str(ctx)
        assert ctx_str in scaling, f"Context {ctx} missing from scaling results"
        base_rss = scaling[ctx_str]["baseline"]["peak_rss_mb"]
        aissd_rss = scaling[ctx_str]["aissd"]["peak_rss_mb"]
        saved_mb = base_rss - aissd_rss
        
        assert saved_mb > prev_saved_mb, (
            f"Context {ctx} saved {saved_mb:.1f} MB, which is not greater than previous {prev_saved_mb:.1f} MB"
        )
        prev_saved_mb = saved_mb
        
        # Verify exact token match for both baseline and aissd
        b_tokens = scaling[ctx_str]["baseline"]["token_ids"]
        a_tokens = scaling[ctx_str]["aissd"]["token_ids"]
        assert b_tokens == a_tokens, f"Context {ctx} token mismatch between baseline and AI-SSD"
