"""
End-to-End Real Pipeline Test for AI-SSD V2.
Validates the complete integration path:
P1 Real LLM Trace -> Canonical TraceReader -> StorageRequest -> P2 Deterministic Tensor Mapping / FTL -> P3 Analytical Backend.
"""

from pathlib import Path
import pytest

from common.schemas.trace import CanonicalTraceRecord, TraceOperation, TraceManifest
from common.schemas.kv_block import KEY_PAGE_BYTES, VALUE_PAGE_BYTES, LOGICAL_BLOCK_BYTES
from person3_system.trace.trace_reader import TraceReader, TraceValidationError
from person3_system.storage.analytical_backend import AnalyticalFTLBackend
from person3_system.storage.backend import StorageRequest, StorageResult

REAL_TRACE_PATH = Path("/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl")
REAL_MANIFEST_PATH = Path("/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.manifest.json")


def test_reject_manifest_file_as_trace():
    """Verify that no manifest file is accidentally interpreted as a JSONL trace."""
    if not REAL_MANIFEST_PATH.exists():
        pytest.skip(f"Manifest file not found: {REAL_MANIFEST_PATH}")

    with pytest.raises(TraceValidationError):
        reader = TraceReader(REAL_MANIFEST_PATH)
        reader.load_all()


def test_end_to_end_real_llm_pipeline():
    """
    End-to-end integration test executing all 7,872 events from P1's real trace
    through P3's CanonicalTraceReader and AnalyticalFTLBackend with P2 tensor mapping.
    """
    if not REAL_TRACE_PATH.exists():
        pytest.skip(f"Real trace file not found: {REAL_TRACE_PATH}")

    # 1. Load manifest explicitly and verify contract
    assert REAL_MANIFEST_PATH.exists(), f"Missing manifest at {REAL_MANIFEST_PATH}"
    manifest = TraceManifest.from_file(REAL_MANIFEST_PATH)
    assert manifest.total_events == 7872
    assert manifest.num_layers == 24
    assert manifest.num_kv_heads == 2
    assert manifest.key_page_bytes == KEY_PAGE_BYTES == 4096
    assert manifest.value_page_bytes == VALUE_PAGE_BYTES == 4096
    assert manifest.logical_block_bytes == LOGICAL_BLOCK_BYTES == 8192

    # 2. Pass real trace through canonical TraceReader
    reader = TraceReader(REAL_TRACE_PATH)
    assert reader.manifest is not None
    assert reader.manifest.total_events == 7872

    events = reader.load_all()
    assert len(events) == 7872, f"Expected 7872 events, got {len(events)}"

    # 3. Verify operation type counts and byte sizes
    op_counts = {}
    subpage_counts = {}
    total_trace_bytes = 0
    pure_k_bytes = 0
    pure_v_bytes = 0
    combined_expected_bytes = 0

    for ev in events:
        op_counts[ev.operation] = op_counts.get(ev.operation, 0) + 1
        subpage_counts[ev.sub_page] = subpage_counts.get(ev.sub_page, 0) + 1
        total_trace_bytes += ev.byte_size

        if ev.sub_page == "KEY":
            pure_k_bytes += ev.byte_size
            assert ev.byte_size == 335872  # 82 candidate blocks * 4096 B
            assert ev.operation == TraceOperation.TOPK_FILTER.value
        elif ev.sub_page == "VALUE":
            pure_v_bytes += ev.byte_size
            assert ev.byte_size == 4096
            assert ev.operation == TraceOperation.TOPK_FETCH.value
        elif ev.sub_page == "BOTH":
            combined_expected_bytes += ev.byte_size
            assert ev.byte_size == 8192  # 4096 K + 4096 V
            assert ev.operation in (TraceOperation.PREFILL_WRITE.value, TraceOperation.DECODE_READ.value)
        else:
            pytest.fail(f"Unexpected sub_page: {ev.sub_page}")

    # Check exact operation counts
    assert op_counts[TraceOperation.TOPK_FETCH.value] == 3072
    assert op_counts[TraceOperation.DECODE_READ.value] == 2304
    assert op_counts[TraceOperation.PREFILL_WRITE.value] == 2112
    assert op_counts[TraceOperation.TOPK_FILTER.value] == 384
    assert sum(op_counts.values()) == 7872

    # Check exact subpage counts
    assert subpage_counts["VALUE"] == 3072
    assert subpage_counts["BOTH"] == 4416
    assert subpage_counts["KEY"] == 384

    # Check exact byte totals
    assert pure_k_bytes == 384 * 335872            # 128,974,848 B (Key pages filtered in storage)
    assert pure_v_bytes == 3072 * 4096             # 12,582,912 B (Winning Value pages fetched)
    assert combined_expected_bytes == 4416 * 8192  # 36,175,872 B (Combined 8KB K+V writes & reads)
    assert total_trace_bytes == 177733632

    # Total Key and Value across both pure and combined accesses:
    expected_total_k = pure_k_bytes + (combined_expected_bytes // 2)   # 147,062,784 B
    expected_total_v = pure_v_bytes + (combined_expected_bytes // 2)   # 30,670,848 B
    assert expected_total_k + expected_total_v == total_trace_bytes

    # 4. Pass through P3 StorageBackend with P2 Deterministic Tensor Mapping
    backend = AnalyticalFTLBackend(mode="tensor_aware", channels=8)

    for ev in events:
        req = StorageRequest(
            block_id=ev.block_id,
            offset=0,
            length=ev.byte_size,
            is_write=ev.is_write,
            layer_id=ev.layer_id,
            head_id=ev.head_id,
            metadata={
                "event_id": ev.event_id,
                "token_start": ev.token_start,
                "sub_page": ev.sub_page,
                "operation": ev.operation,
                "byte_size": ev.byte_size,
                "candidate_blocks": ev.candidate_blocks,
                "selected_blocks": ev.selected_blocks,
            },
        )
        res = backend.submit(req)
        assert res.success is True
        assert res.length == ev.byte_size

    # 5. Verify Backend Statistics and Channel Distribution
    stats = backend.get_stats()
    assert stats["total_requests"] == 7872
    assert stats["total_bytes"] == 177733632
    assert stats["k_bytes"] == expected_total_k       # 147,062,784 B
    assert stats["v_bytes"] == expected_total_v       # 30,670,848 B
    assert stats["combined_bytes"] == combined_expected_bytes  # 36,175,872 B

    # Verify channel distribution across 8-channel topology
    channel_counts = stats["channel_access_counts"]
    assert len(channel_counts) == 8, f"Expected 8 channels active, got {len(channel_counts)}"
    for ch in range(8):
        assert ch in channel_counts, f"Channel {ch} was never accessed"
        assert channel_counts[ch] > 0, f"Channel {ch} has 0 accesses"

    # Verify that load is distributed across channels
    total_channel_accesses = sum(channel_counts.values())
    assert total_channel_accesses == 7872
    for ch, count in channel_counts.items():
        assert count > 500, f"Channel {ch} received too few accesses: {count}"

    # Close backend cleanly
    backend.close()
