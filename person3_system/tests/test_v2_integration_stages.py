import pytest
from pathlib import Path
from person3_system.experiments.config import ExperimentConfig
from person3_system.experiments.runner import ExperimentRunner
from person3_system.trace.synthetic_generator import SyntheticTraceGenerator


def test_stage_1_synthetic_trace_mock_storage():
    """
    Stage 1: Synthetic P1-like trace -> Mock Storage -> P3 Experiment Runner.
    """
    cfg = ExperimentConfig(
        experiment_id="stage1_mock_test",
        context_length=2048,
        offload_pct=80.0,
        topk_pct=10.0,
        prefetch_enabled=True,
        ftl_mode="tensor_aware",
        storage_backend_type="mock",
    )
    runner = ExperimentRunner()
    result = runner.run_experiment(cfg, num_steps=2)

    assert result.system.ram_reduction_pct == 80.0
    assert result.storage.storage_reads > 0
    assert result.storage.bytes_read > 0
    assert result.prefetch.prefetch_requests > 0
    assert result.prefetch.demand_hits > 0


def test_stage_2_trace_analytical_p2_ftl():
    """
    Stage 2: Trace (Real P1 if available, else synthetic) -> Analytical P2 FTL -> P3.
    """
    real_trace_dir = Path("/opt/ai-ssd-v2/traces/real_llm")
    trace_file = None
    if real_trace_dir.exists():
        candidates = list(real_trace_dir.glob("*.jsonl"))
        if candidates:
            trace_file = candidates[0]

    cfg = ExperimentConfig(
        experiment_id="stage2_analytical_test",
        context_length=2048,
        offload_pct=80.0,
        topk_pct=10.0,
        prefetch_enabled=True,
        ftl_mode="tensor_aware",
        storage_backend_type="analytical",
    )
    runner = ExperimentRunner()
    result = runner.run_experiment(cfg, trace_path=trace_file, num_steps=2)

    assert result.system.ram_reduction_pct == 80.0
    assert result.storage.storage_reads > 0
    assert result.storage.storage_latency_us > 0.0
    assert result.prefetch.prefetch_requests > 0
