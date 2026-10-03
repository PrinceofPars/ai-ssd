import pytest
import json
import tempfile
from pathlib import Path
from person3_system.trace.trace_schema import TraceHeader, TraceModelMetadata, TraceRecord, TraceOperation
from person3_system.trace.trace_reader import TraceReader, TraceValidationError
from person3_system.trace.synthetic_generator import SyntheticTraceGenerator


def test_synthetic_generation_and_reading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trace_path = Path(tmp_dir) / "test_trace.jsonl"
        gen = SyntheticTraceGenerator(
            model_name="Llama-3-8B",
            num_layers=4,
            num_heads=8,
            context_length=1024,
        )
        count = gen.generate_trace_file(trace_path, num_steps=2, top_k=8)
        assert count == 8  # 2 steps * 4 layers

        reader = TraceReader(trace_path)
        records = reader.load_all()

        assert len(records) == 8
        assert reader.header.schema_version == "v2.0"
        assert reader.header.model_metadata.num_layers == 4
        assert records[0].seq_id == 0
        assert records[-1].seq_id == 7
        assert len(records[0].block_ids) > 0


def test_reject_unsupported_version():
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        tmp_file = f.name
        f.write('{"schema_version": "v99.0", "model_metadata": {"model_name": "x", "num_layers": 1, "num_heads": 1, "head_dim": 1, "dtype": "FP16", "context_length": 100, "tokens_per_block": 16, "block_size_bytes": 4096}}\n')

    try:
        reader = TraceReader(tmp_file)
        with pytest.raises(TraceValidationError, match="Unsupported schema_version"):
            reader.load_all()
    finally:
        Path(tmp_file).unlink()


def test_reject_non_monotonic_seq_id():
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        tmp_file = f.name
        hdr = {"schema_version": "v2.0", "model_metadata": {"model_name": "x", "num_layers": 2, "num_heads": 2, "head_dim": 64, "dtype": "FP16", "context_length": 100, "tokens_per_block": 16, "block_size_bytes": 4096}}
        r1 = {"seq_id": 1, "step_id": 0, "layer_id": 0, "head_id": 0, "operation": "KV_READ", "block_ids": [1], "byte_offset": 0, "byte_length": 4096}
        r2 = {"seq_id": 1, "step_id": 0, "layer_id": 0, "head_id": 0, "operation": "KV_READ", "block_ids": [2], "byte_offset": 0, "byte_length": 4096}
        f.write(json.dumps(hdr) + "\n")
        f.write(json.dumps(r1) + "\n")
        f.write(json.dumps(r2) + "\n")

    try:
        reader = TraceReader(tmp_file)
        with pytest.raises(TraceValidationError, match="Non-monotonic seq_id"):
            reader.load_all()
    finally:
        Path(tmp_file).unlink()


def test_reject_out_of_bounds_layer():
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        tmp_file = f.name
        hdr = {"schema_version": "v2.0", "model_metadata": {"model_name": "x", "num_layers": 2, "num_heads": 2, "head_dim": 64, "dtype": "FP16", "context_length": 100, "tokens_per_block": 16, "block_size_bytes": 4096}}
        r1 = {"seq_id": 0, "step_id": 0, "layer_id": 5, "head_id": 0, "operation": "KV_READ", "block_ids": [1], "byte_offset": 0, "byte_length": 4096}
        f.write(json.dumps(hdr) + "\n")
        f.write(json.dumps(r1) + "\n")

    try:
        reader = TraceReader(tmp_file)
        with pytest.raises(TraceValidationError, match="layer_id 5 out of bounds"):
            reader.load_all()
    finally:
        Path(tmp_file).unlink()


def test_reject_negative_block_id():
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        tmp_file = f.name
        hdr = {"schema_version": "v2.0", "model_metadata": {"model_name": "x", "num_layers": 2, "num_heads": 2, "head_dim": 64, "dtype": "FP16", "context_length": 100, "tokens_per_block": 16, "block_size_bytes": 4096}}
        r1 = {"seq_id": 0, "step_id": 0, "layer_id": 0, "head_id": 0, "operation": "KV_READ", "block_ids": [-3], "byte_offset": 0, "byte_length": 4096}
        f.write(json.dumps(hdr) + "\n")
        f.write(json.dumps(r1) + "\n")

    try:
        reader = TraceReader(tmp_file)
        with pytest.raises(TraceValidationError, match="Negative block_id"):
            reader.load_all()
    finally:
        Path(tmp_file).unlink()
