import pytest
import json
import csv
import tempfile
from pathlib import Path
from person3_system.experiments.config import ExperimentConfig
from person3_system.experiments.metadata import EnvironmentProvenance
from person3_system.experiments.runner import ExperimentRunner


def test_provenance_capture():
    prov = EnvironmentProvenance.capture()
    assert prov.hostname != ""
    assert prov.cpu_count >= 1
    assert prov.python_version.startswith("3.")
    assert "numpy" in prov.dependency_versions


def test_standard_ablations_and_export():
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = ExperimentRunner(primary_results_dir=tmp_dir)
        results = runner.run_standard_ablations(context_length=1024, num_steps=2)

        assert len(results) == 5

        r_dense = results[0]
        r_offload_conv = results[1]
        r_conv_topk = results[2]
        r_ta_topk = results[3]
        r_ta_prefetch = results[4]

        # 1. Dense DRAM Baseline
        assert r_dense.config.experiment_id == "1_dense_dram_baseline"
        assert r_dense.system.ram_reduction_pct == 0.0
        assert r_dense.storage.storage_reads == 0

        # 2. KV Offload Dense Conv
        assert r_offload_conv.system.ram_reduction_pct == 80.0
        assert r_offload_conv.storage.storage_reads > 0

        # 3. Conv Top-k vs 4. Tensor-Aware Top-k: TA should have equal or lower latency due to channel striping
        assert r_conv_topk.storage.storage_reads > 0
        assert r_ta_topk.storage.storage_reads > 0
        assert r_ta_topk.storage.storage_latency_us <= r_conv_topk.storage.storage_latency_us

        # 5. Prefetch enabled
        assert r_ta_prefetch.prefetch.prefetch_requests > 0
        assert r_ta_prefetch.prefetch.useful_prefetches > 0

        # Verify export files
        out_path = Path(tmp_dir)
        assert (out_path / "experiment_results.json").exists()
        assert (out_path / "experiment_results.jsonl").exists()
        assert (out_path / "experiment_results.csv").exists()

        # Check JSON integrity
        with open(out_path / "experiment_results.json") as f:
            data = json.load(f)
            assert len(data) == 5

        # Check CSV columns
        with open(out_path / "experiment_results.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 5
            assert "cfg_experiment_id" in rows[0]
            assert "sys_ram_reduction_pct" in rows[0]
            assert "git_commit" in rows[0]
