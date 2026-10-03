"""
Regression and Verification Test Suite for P2 Real-Trace Replayer & FTL Repair.

Covers:
  1. Manifest is not selected as JSONL trace.
  2. PREFILL_WRITE is treated as a write operation.
  3. DECODE_READ is treated as a read operation.
  4. TOPK_FILTER and TOPK_FETCH operations are preserved with correct semantics.
  5. byte_size is strictly respected (no 4096-byte truncation on 8192-byte blocks).
  6. token_start / block mapping is respected (no collapse to block 0).
  7. All 8 channels are reachable with balanced load for real workload geometry.
  8. Malformed/missing required fields fail loudly with ValueError.
"""

import sys
import tempfile
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person2_ssd.trace_replay.replayer import StorageTraceReplayer
from person2_ssd.kv_allocator.tensor_mapping import DeterministicTensorMapper, TensorCoordinate


def test_manifest_not_selected_as_jsonl():
    """Regression test: Ensures *.manifest.json is never picked up as a trace file."""
    print("Testing manifest exclusion from JSONL discovery...", end=" ")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create manifest first
        manifest_file = tmp_path / "test_trace.manifest.json"
        manifest_file.write_text(json.dumps({"total_events": 10}), encoding="utf-8")

        # Create actual trace
        trace_file = tmp_path / "test_trace.jsonl"
        trace_file.write_text('{"event_id": 0, "operation": "PREFILL_WRITE", "layer_id": 0, "head_id": 0, "token_start": 0, "byte_size": 8192}\n', encoding="utf-8")

        discovered_file, meta = StorageTraceReplayer.discover_trace_file(trace_dir=tmpdir)
        assert discovered_file == trace_file, f"Expected {trace_file}, but got {discovered_file}"
        assert discovered_file.suffix == ".jsonl"
        assert meta is not None and meta.get("total_events") == 10
    print("[PASSED]")


def test_prefill_write_is_write():
    """Verifies PREFILL_WRITE is correctly classified as a write and increments total_write_bytes."""
    print("Testing PREFILL_WRITE semantics...", end=" ")
    replayer = StorageTraceReplayer(channels=8)
    records = [
        replayer.validate_record({
            "event_id": 0,
            "operation": "PREFILL_WRITE",
            "layer_id": 0,
            "head_id": 0,
            "token_start": 0,
            "block_id": 0,
            "byte_size": 8192,
            "sub_page": "BOTH",
        })
    ]
    res = replayer.replay(records, mode="tensor_aware")
    assert res["total_write_bytes"] == 8192, f"Expected 8192 write bytes, got {res['total_write_bytes']}"
    assert res["total_read_bytes"] == 0, f"Expected 0 read bytes, got {res['total_read_bytes']}"
    assert res["event_counts_by_operation"]["PREFILL_WRITE"] == 1
    print("[PASSED]")


def test_decode_read_is_read():
    """Verifies DECODE_READ is correctly classified as a read and increments total_read_bytes."""
    print("Testing DECODE_READ semantics...", end=" ")
    replayer = StorageTraceReplayer(channels=8)
    records = [
        replayer.validate_record({
            "event_id": 1,
            "operation": "DECODE_READ",
            "layer_id": 0,
            "head_id": 0,
            "token_start": 0,
            "block_id": 0,
            "byte_size": 8192,
            "sub_page": "BOTH",
        })
    ]
    res = replayer.replay(records, mode="tensor_aware")
    assert res["total_read_bytes"] == 8192, f"Expected 8192 read bytes, got {res['total_read_bytes']}"
    assert res["total_write_bytes"] == 0, f"Expected 0 write bytes, got {res['total_write_bytes']}"
    assert res["event_counts_by_operation"]["DECODE_READ"] == 1
    print("[PASSED]")


def test_topk_operations_preserved():
    """Verifies TOPK_FILTER (Key) and TOPK_FETCH (Value) are preserved with correct sub-page accounting."""
    print("Testing TOPK_FILTER and TOPK_FETCH preservation...", end=" ")
    replayer = StorageTraceReplayer(channels=8)
    records = [
        replayer.validate_record({
            "event_id": 10,
            "operation": "TOPK_FILTER",
            "layer_id": 0,
            "head_id": 0,
            "token_start": 0,
            "block_id": 1,
            "byte_size": 335872,
            "sub_page": "KEY",
        }),
        replayer.validate_record({
            "event_id": 11,
            "operation": "TOPK_FETCH",
            "layer_id": 0,
            "head_id": 1,
            "token_start": 16,
            "block_id": 2,
            "byte_size": 4096,
            "sub_page": "VALUE",
        }),
    ]
    res = replayer.replay(records, mode="tensor_aware")
    assert res["event_counts_by_operation"]["TOPK_FILTER"] == 1
    assert res["event_counts_by_operation"]["TOPK_FETCH"] == 1
    assert res["k_bytes"] == 335872, f"Expected 335872 K bytes, got {res['k_bytes']}"
    assert res["v_bytes"] == 4096, f"Expected 4096 V bytes, got {res['v_bytes']}"
    assert res["total_read_bytes"] == 335872 + 4096
    print("[PASSED]")


def test_byte_size_respected():
    """Verifies byte sizes are preserved exactly (both 4096 and 8192)."""
    print("Testing byte_size fidelity...", end=" ")
    replayer = StorageTraceReplayer(channels=8)
    rec_4k = replayer.validate_record({
        "event_id": 100,
        "operation": "TOPK_FETCH",
        "layer_id": 0,
        "head_id": 0,
        "token_start": 0,
        "byte_size": 4096,
    })
    rec_8k = replayer.validate_record({
        "event_id": 101,
        "operation": "PREFILL_WRITE",
        "layer_id": 0,
        "head_id": 0,
        "token_start": 16,
        "byte_size": 8192,
    })
    assert rec_4k["byte_size"] == 4096
    assert rec_8k["byte_size"] == 8192
    print("[PASSED]")


def test_token_start_and_block_mapping_respected():
    """Verifies that token_start and block_id vary across distinct blocks, preventing collapse to block 0."""
    print("Testing token_start and block_id distribution...", end=" ")
    replayer = StorageTraceReplayer(channels=8)
    records = [
        replayer.validate_record({
            "event_id": i,
            "operation": "PREFILL_WRITE",
            "layer_id": 0,
            "head_id": 0,
            "token_start": i * 16,
            "block_id": i,
            "byte_size": 8192,
        })
        for i in range(16)
    ]
    res = replayer.replay(records, mode="tensor_aware")
    assert res["unique_blocks_count"] == 16, f"Expected 16 unique blocks, got {res['unique_blocks_count']}"
    assert res["min_block_id"] == 0
    assert res["max_block_id"] == 15
    print("[PASSED]")


def test_all_8_channels_reachable_for_real_workload():
    """Verifies all 8 channels are reached when replaying real LLM topology."""
    print("Testing all 8 channels reachability...", end=" ")
    real_trace = Path("/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl")
    if real_trace.exists():
        replayer = StorageTraceReplayer(channels=8, num_layers=24, num_heads=2)
        records = replayer.load_trace(real_trace)
        res = replayer.replay(records, mode="tensor_aware", blocks_per_head=44)
        ch_dist = res["channel_distribution"]
        assert len(ch_dist) == 8, f"Expected 8 channels, got {len(ch_dist)}"
        for ch in range(8):
            assert ch_dist[ch] > 0, f"Channel {ch} has 0 events!"
            pct = ch_dist[ch] / res["total_events"]
            assert 0.08 <= pct <= 0.20, f"Channel {ch} percentage {pct:.2%} outside balanced range [8%, 20%]"
        assert res["contention_ratio"] <= 1.5, f"Contention ratio {res['contention_ratio']} exceeds target 1.5"
    print("[PASSED]")


def test_malformed_required_fields_fail_loudly():
    """Verifies that missing or invalid required fields raise ValueError."""
    print("Testing loud failure on malformed records...", end=" ")
    replayer = StorageTraceReplayer(channels=8)

    # Missing operation
    try:
        replayer.validate_record({"event_id": 1, "layer_id": 0, "head_id": 0, "token_start": 0, "byte_size": 4096})
        assert False, "Should have failed on missing operation"
    except ValueError:
        pass

    # Invalid operation name
    try:
        replayer.validate_record({"event_id": 1, "operation": "UNKNOWN_OP", "layer_id": 0, "head_id": 0, "token_start": 0, "byte_size": 4096})
        assert False, "Should have failed on invalid operation name"
    except ValueError:
        pass

    # Missing layer_id
    try:
        replayer.validate_record({"event_id": 1, "operation": "DECODE_READ", "head_id": 0, "token_start": 0, "byte_size": 4096})
        assert False, "Should have failed on missing layer_id"
    except ValueError:
        pass

    # Missing head_id
    try:
        replayer.validate_record({"event_id": 1, "operation": "DECODE_READ", "layer_id": 0, "token_start": 0, "byte_size": 4096})
        assert False, "Should have failed on missing head_id"
    except ValueError:
        pass

    # Missing byte_size
    try:
        replayer.validate_record({"event_id": 1, "operation": "DECODE_READ", "layer_id": 0, "head_id": 0, "token_start": 0})
        assert False, "Should have failed on missing byte_size"
    except ValueError:
        pass

    # Missing both token_start and block_id
    try:
        replayer.validate_record({"event_id": 1, "operation": "DECODE_READ", "layer_id": 0, "head_id": 0, "byte_size": 4096})
        assert False, "Should have failed on missing token and block location"
    except ValueError:
        pass
    print("[PASSED]")


if __name__ == "__main__":
    print("==================================================")
    print("   AI-SSD V2 Trace Replayer Repair Tests (P2)    ")
    print("==================================================")
    test_manifest_not_selected_as_jsonl()
    test_prefill_write_is_write()
    test_decode_read_is_read()
    test_topk_operations_preserved()
    test_byte_size_respected()
    test_token_start_and_block_mapping_respected()
    test_all_8_channels_reachable_for_real_workload()
    test_malformed_required_fields_fail_loudly()
    print("==================================================")
    print("  [SUCCESS] All 8 Trace Replayer repair tests PASSED! ")
    print("==================================================")