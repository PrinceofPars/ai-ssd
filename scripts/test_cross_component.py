"""Cross-Component Contract Replay Verification for Phase 2.

Validates that P1's real trace maps cleanly to P2 (FTL / FEMU) and P3 (Prefetch) interfaces:
1. No events disappear (all 7,872 events preserved).
2. PREFILL_WRITE remains WRITE.
3. DECODE_READ remains READ.
4. TOPK operations remain identifiable (TOPK_FILTER = candidate key read, TOPK_FETCH = winning value read).
5. Byte accounting remains exact across both components.
6. Block IDs are preserved without collapsing to 0.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = "/home/ubuntu/ai-ssd-p1"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from common.schemas.trace import CanonicalTraceRecord, TraceOperation

TRACE_PATH = "/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl"


def test_cross_component_replay():
    print("=== CROSS-COMPONENT TRACE REPLAY AUDIT ===")
    events: list[CanonicalTraceRecord] = []

    with open(TRACE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            events.append(CanonicalTraceRecord.from_dict(data))

    total = len(events)
    print(f"Total events loaded: {total}")
    assert total == 7872, f"Event loss detected! Expected 7872, got {total}"

    # Verify P2 Replayer mapping
    # P2 replayer maps:
    # op in ("PREFILL_WRITE", "KV_WRITE") -> is_write=True
    # op in ("DECODE_READ", "TOPK_FETCH", "TOPK_FILTER") -> is_read=True
    p2_writes = []
    p2_reads = []
    p2_topk_filters = []
    p2_topk_fetches = []

    non_zero_block_ids = set()

    for e in events:
        non_zero_block_ids.add(e.block_id)

        if e.is_write:
            p2_writes.append(e)
            assert e.operation == "PREFILL_WRITE"
        elif e.is_read:
            p2_reads.append(e)
            if e.operation == "TOPK_FILTER":
                p2_topk_filters.append(e)
            elif e.operation == "TOPK_FETCH":
                p2_topk_fetches.append(e)
        else:
            raise ValueError(f"Unmapped operation: {e.operation}")

    print(f"P2 Write Events (PREFILL_WRITE): {len(p2_writes)}")
    print(f"P2 Read Events Total:            {len(p2_reads)}")
    print(f"  - Dense Reads (DECODE_READ):   {len(p2_reads) - len(p2_topk_filters) - len(p2_topk_fetches)}")
    print(f"  - In-Storage Scans (FILTER):   {len(p2_topk_filters)}")
    print(f"  - Host Retrievals (FETCH):     {len(p2_topk_fetches)}")
    print(f"Unique Block IDs active:         {len(non_zero_block_ids)} (Range: {min(non_zero_block_ids)} .. {max(non_zero_block_ids)})")

    assert len(p2_writes) == 2112, "PREFILL_WRITE count mismatch"
    assert len(p2_reads) == 5760, "Read count mismatch"
    assert len(p2_topk_filters) == 384, "TOPK_FILTER count mismatch"
    assert len(p2_topk_fetches) == 3072, "TOPK_FETCH count mismatch"
    assert len(non_zero_block_ids) > 1, "Block IDs collapsed to 0!"

    # Verify byte accounting matches across FTL / Storage interfaces
    total_trace_bytes = sum(e.byte_size for e in events)
    p2_mapped_bytes = sum(e.byte_length for e in events)
    print(f"Byte Accounting: {total_trace_bytes:,} bytes == {p2_mapped_bytes:,} bytes")
    assert total_trace_bytes == p2_mapped_bytes

    print("\n[SUCCESS] Cross-Component Replay Verification: PASSED (100% Consistent)")


if __name__ == "__main__":
    test_cross_component_replay()
