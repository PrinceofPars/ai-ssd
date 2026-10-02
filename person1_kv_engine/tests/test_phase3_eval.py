"""Unit tests for Phase 3 evaluation artifacts, results schema, and trace integrity."""

import os
import json
import hashlib
import pytest
from common.schemas.trace import CanonicalTraceRecord

RESULTS_DIR = "/opt/ai-ssd-v2/results/p1"
RESULTS_JSON = os.path.join(RESULTS_DIR, "phase3_real_llm_results.json")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "phase3_summary.csv")
KERNEL_JSON = os.path.join(RESULTS_DIR, "kernel_ablation_results.json")
TRACES_DIR = "/opt/ai-ssd-v2/traces/real_llm"


def test_phase3_results_json_structure():
    """Validates structure and required fields of phase3_real_llm_results.json."""
    assert os.path.exists(RESULTS_JSON), f"Results JSON not found at {RESULTS_JSON}"

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Top-level keys
    for k in ["metadata", "baselines", "topk_evaluations", "kernel_ablations", "generated_traces"]:
        assert k in data, f"Missing top-level key '{k}'"

    # Context lengths
    expected_contexts = ["128", "256", "512", "1024", "2048", "4096"]
    for ctx in expected_contexts:
        assert ctx in data["baselines"], f"Context {ctx} missing from baselines"
        assert ctx in data["topk_evaluations"], f"Context {ctx} missing from topk_evaluations"

        base = data["baselines"][ctx]
        assert base["prefill_latency_s"]["mean"] > 0
        assert base["decode_throughput_tps"]["mean"] > 0
        assert base["kv_cache_geometry"]["total_blocks_all_layers"] > 0

        # Sparsity budgets
        topk = data["topk_evaluations"][ctx]["sparsity_evaluations"]
        expected_sparsities = ["1.0%", "5.0%", "10.0%", "20.0%", "50.0%"]
        for sp in expected_sparsities:
            assert sp in topk, f"Sparsity {sp} missing for context {ctx}"
            eval_entry = topk[sp]
            assert 0.0 <= eval_entry["cosine_similarity"]["mean"] <= 1.0
            assert eval_entry["pcie_traffic_reduction_percent"] >= 0.0
            assert 0.0 <= eval_entry["attention_mass_recall"]["mean"] <= 1.0


def test_kernel_ablation_results():
    """Validates the native AVX2 C kernel vs NumPy ablation results."""
    assert os.path.exists(KERNEL_JSON), f"Kernel JSON not found at {KERNEL_JSON}"

    with open(KERNEL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "evaluations" in data
    assert len(data["evaluations"]) >= 5

    for ev in data["evaluations"]:
        assert ev["num_blocks"] > 0
        assert ev["c_kernel_latency_us"] > 0
        assert ev["numpy_latency_us"] > 0
        assert ev["c_scan_throughput_gib_per_sec"] > 1.0  # At least 1 GB/s scan
        assert ev["numerical_parity"]["topk_id_match_percent"] == 100.0
        assert ev["numerical_parity"]["max_score_error"] < 1e-4


def test_generated_traces_integrity():
    """Validates that all generated traces exist, match manifests, and parse canonically."""
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    traces = data["generated_traces"]
    assert len(traces) >= 4, f"Expected at least 4 generated traces, got {len(traces)}"

    for tr in traces:
        t_path = tr["trace_file"]
        m_path = tr["manifest_file"]
        expected_sha = tr["sha256"]

        assert os.path.exists(t_path), f"Trace file missing: {t_path}"
        assert os.path.exists(m_path), f"Manifest file missing: {m_path}"

        # SHA-256 check
        sha = hashlib.sha256()
        rec_count = 0
        with open(t_path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        assert sha.hexdigest() == expected_sha, f"SHA mismatch for {t_path}"

        # Parse every record with CanonicalTraceRecord
        with open(t_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = CanonicalTraceRecord.from_dict(json.loads(line))
                rec_count += 1

        assert rec_count == tr["total_events"], f"Event count mismatch in {t_path}"


def test_summary_csv():
    """Validates that phase3_summary.csv exists and is well-formed."""
    assert os.path.exists(SUMMARY_CSV), f"Summary CSV missing: {SUMMARY_CSV}"
    with open(SUMMARY_CSV, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 1 + (6 * 5)  # Header + 6 contexts * 5 sparsities = 31 lines
    header = lines[0]
    assert "Context_Tokens" in header
    assert "Cosine_Similarity" in header
    assert "Classification" in header
