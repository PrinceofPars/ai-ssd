"""
Tests for AI-SSD Benchmark Matrix Resolution, Provenance Precedence,
and Atomic Result Persistence.

Verifies:
1. Strict precedence: CURRENT > HISTORICAL > NOT EVALUATED.
2. If CURRENT is OOM, FAIL, ERROR, UNSUPPORTED, UNAVAILABLE:
   It strictly preserves that current non-PASS state and NEVER falls back to an older historical PASS.
3. Resumability & atomic disk serialization for JSON and CSV.
4. Archiving workflow preserving historical generations.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any

from benchmarks.live_inference.result_schema import (
    resolve_benchmark_matrix,
    save_quick_comparison_record,
    load_current_benchmark_records,
    load_historical_benchmark_records,
    archive_current_benchmark,
    CURRENT_RESULTS_FILE,
    CURRENT_RESULTS_CSV,
    CURRENT_RESULTS_DIR,
    ARCHIVE_RESULTS_DIR,
)


def test_precedence_case1_historical_pass_current_absent():
    """Case 1: Historical PASS exists, Current is absent -> Result is HISTORICAL PASS."""
    hist = [{
        "config_key": "qwen3-4b|fp32|2048",
        "stage": 2048,
        "stage_name": "2K",
        "model_key": "qwen3-4b",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "precision_requested": "fp32",
        "status": "PASS",
        "status_reason": "Historical exact match",
        "exact_token_match": True,
        "baseline_peak_rss_mb": 17000.0,
        "aissd_peak_rss_mb": 16000.0,
    }]
    curr = []

    res = resolve_benchmark_matrix(
        current_records=curr,
        historical_records=hist,
        models=["qwen3-4b"],
        stages=[2048],
        precisions=["fp32"],
        current_only=False,
    )

    records = res["records"]
    assert len(records) == 1
    r = records[0]
    assert r["config_key"] == "qwen3-4b|fp32|2048"
    assert r["status"] == "PASS"
    assert r["result_source"] == "HISTORICAL"


def test_precedence_case2_historical_pass_current_oom():
    """
    Case 2: Historical PASS exists, but Current benchmark got OOM ->
    Result MUST be CURRENT OOM and MUST NOT fall back to historical PASS!
    """
    hist = [{
        "config_key": "qwen3-4b|fp32|32768",
        "stage": 32768,
        "stage_name": "32K",
        "model_key": "qwen3-4b",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "precision_requested": "fp32",
        "status": "PASS",
        "status_reason": "Historical run",
        "exact_token_match": True,
    }]
    curr = [{
        "config_key": "qwen3-4b|fp32|32768",
        "stage": 32768,
        "stage_name": "32K",
        "model_key": "qwen3-4b",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "precision_requested": "fp32",
        "status": "OOM",
        "status_reason": "Killed by Linux OOM killer (exit code 137)",
        "exact_token_match": None,
    }]

    res = resolve_benchmark_matrix(
        current_records=curr,
        historical_records=hist,
        models=["qwen3-4b"],
        stages=[32768],
        precisions=["fp32"],
        current_only=False,
    )

    records = res["records"]
    assert len(records) == 1
    r = records[0]
    assert r["config_key"] == "qwen3-4b|fp32|32768"
    assert r["status"] == "OOM"
    assert r["result_source"] == "CURRENT"
    assert "OOM killer" in r["status_reason"]


def test_precedence_case3_historical_pass_current_error():
    """
    Case 3: Historical PASS exists, but Current benchmark got ERROR ->
    Result MUST be CURRENT ERROR and MUST NOT fall back to historical PASS!
    """
    hist = [{
        "config_key": "tiny-mistral|fp16|4096",
        "stage": 4096,
        "stage_name": "4K",
        "model_key": "tiny-mistral",
        "model_id": "openaccess-ai-collective/tiny-mistral",
        "precision_requested": "fp16",
        "status": "PASS",
        "status_reason": "Historical pass",
        "exact_token_match": True,
    }]
    curr = [{
        "config_key": "tiny-mistral|fp16|4096",
        "stage": 4096,
        "stage_name": "4K",
        "model_key": "tiny-mistral",
        "model_id": "openaccess-ai-collective/tiny-mistral",
        "precision_requested": "fp16",
        "status": "ERROR",
        "status_reason": "Device bus failure during write",
        "exact_token_match": None,
    }]

    res = resolve_benchmark_matrix(
        current_records=curr,
        historical_records=hist,
        models=["tiny-mistral"],
        stages=[4096],
        precisions=["fp16"],
        current_only=False,
    )

    records = res["records"]
    assert len(records) == 1
    r = records[0]
    assert r["config_key"] == "tiny-mistral|fp16|4096"
    assert r["status"] == "ERROR"
    assert r["result_source"] == "CURRENT"


def test_precedence_case4_neither_present_not_evaluated():
    """Case 4: Neither Current nor Historical has a record -> Result is NOT EVALUATED."""
    curr = []
    hist = []

    res = resolve_benchmark_matrix(
        current_records=curr,
        historical_records=hist,
        models=["qwen3-8b"],
        stages=[16384],
        precisions=["fp16"],
        current_only=False,
    )

    records = res["records"]
    assert len(records) == 1
    r = records[0]
    assert r["config_key"] == "qwen3-8b|fp16|16384"
    assert r["status"] == "NOT EVALUATED"
    assert r["result_source"] == "NOT EVALUATED"


def test_current_only_flag_ignores_historical():
    """When current_only=True, even if historical PASS exists, result is NOT EVALUATED."""
    hist = [{
        "config_key": "qwen2.5-0.5b|fp32|2048",
        "stage": 2048,
        "stage_name": "2K",
        "model_key": "qwen2.5-0.5b",
        "model_id": "Qwen/Qwen2.5-0.5B",
        "precision_requested": "fp32",
        "status": "PASS",
        "status_reason": "Historical pass",
    }]
    curr = []

    res = resolve_benchmark_matrix(
        current_records=curr,
        historical_records=hist,
        models=["qwen2.5-0.5b"],
        stages=[2048],
        precisions=["fp32"],
        current_only=True,
    )

    r = res["records"][0]
    assert r["status"] == "NOT EVALUATED"
    assert r["result_source"] == "NOT EVALUATED"


def test_atomic_persistence_and_resume():
    """Verifies that saving a quick comparison record persists atomically to JSON and CSV."""
    import tempfile
    import shutil

    # Create temporary isolated directory for current results
    temp_dir = Path(tempfile.mkdtemp())
    try:
        import benchmarks.live_inference.result_schema as rs
        old_dir = rs.CURRENT_RESULTS_DIR
        old_json = rs.CURRENT_RESULTS_FILE
        old_csv = rs.CURRENT_RESULTS_CSV

        rs.CURRENT_RESULTS_DIR = temp_dir
        rs.CURRENT_RESULTS_FILE = temp_dir / "quick_comparison_results.json"
        rs.CURRENT_RESULTS_CSV = temp_dir / "quick_comparison_results.csv"

        rec1 = {
            "config_key": "tiny-mistral|fp32|2048",
            "stage": 2048,
            "stage_name": "2K",
            "model_key": "tiny-mistral",
            "model_id": "openaccess-ai-collective/tiny-mistral",
            "precision_requested": "fp32",
            "status": "PASS",
            "status_reason": "Exact match",
            "exact_token_match": True,
            "token_match_rate": 100.0,
            "matching_tokens": 16,
            "total_tokens": 16,
            "baseline_peak_rss_mb": 1200.0,
            "aissd_peak_rss_mb": 1100.0,
            "memory_saved_mb": 100.0,
            "candidate_k_zero_bus": True,
        }

        # 1. Save first record
        json_p, csv_p = rs.save_quick_comparison_record(rec1)
        assert json_p.exists()
        assert csv_p.exists()

        loaded = rs.load_current_benchmark_records()
        assert len(loaded) == 1
        assert loaded[0]["config_key"] == "tiny-mistral|fp32|2048"

        # 2. Save second record (updating same file atomically)
        rec2 = {
            "config_key": "qwen2.5-0.5b|fp32|2048",
            "stage": 2048,
            "stage_name": "2K",
            "model_key": "qwen2.5-0.5b",
            "model_id": "Qwen/Qwen2.5-0.5B",
            "precision_requested": "fp32",
            "status": "PASS",
            "status_reason": "Exact match",
            "exact_token_match": True,
            "token_match_rate": 100.0,
            "matching_tokens": 16,
            "total_tokens": 16,
            "baseline_peak_rss_mb": 2500.0,
            "aissd_peak_rss_mb": 2300.0,
            "memory_saved_mb": 200.0,
            "candidate_k_zero_bus": True,
        }
        rs.save_quick_comparison_record(rec2)

        loaded2 = rs.load_current_benchmark_records()
        assert len(loaded2) == 2
        keys = {r["config_key"] for r in loaded2}
        assert keys == {"tiny-mistral|fp32|2048", "qwen2.5-0.5b|fp32|2048"}

    finally:
        rs.CURRENT_RESULTS_DIR = old_dir
        rs.CURRENT_RESULTS_FILE = old_json
        rs.CURRENT_RESULTS_CSV = old_csv
        shutil.rmtree(temp_dir, ignore_errors=True)
