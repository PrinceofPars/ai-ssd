"""Deep Validation Script for AI-SSD V2 Real KV Access Trace (Phase 2).

Validates:
1. Exact event count (7872 events) and total tokens (703 tokens).
2. Semantics of PREFILL_WRITE, DECODE_READ, TOPK_FILTER, and TOPK_FETCH.
3. Event sequence ordering and monotonic event_id / timestamps.
4. Physical KV geometry: 2 KV heads, head_dim=64, 16 tokens/block, K=4096B, V=4096B, K+V=8192B.
5. Canonical trace contract compatibility (CanonicalTraceRecord / P3 schema).
6. Manifest consistency and SHA-256 integrity verification.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import hashlib
from typing import Dict, Any, List

TRACE_DIR = "/opt/ai-ssd-v2/traces/real_llm"
TRACE_FILE = os.path.join(TRACE_DIR, "trace_qwen2.5_0.5b_context512.jsonl")
MANIFEST_FILE = os.path.join(TRACE_DIR, "trace_qwen2.5_0.5b_context512.manifest.json")
SHA_FILE = os.path.join(TRACE_DIR, "trace_qwen2.5_0.5b_context512.sha256")


def validate_trace() -> Dict[str, Any]:
    print("=================================================================")
    print("PHASE 2: REAL KV TRACE CONTRACT & SEMANTICS VALIDATION")
    print("=================================================================")
    print(f"Inspecting Trace: {TRACE_FILE}")
    print(f"Inspecting Manifest: {MANIFEST_FILE}")

    if not os.path.exists(TRACE_FILE):
        raise FileNotFoundError(f"Trace file not found: {TRACE_FILE}")
    if not os.path.exists(MANIFEST_FILE):
        raise FileNotFoundError(f"Manifest file not found: {MANIFEST_FILE}")

    # 1. SHA-256 Checksum Verification
    print("\n[Step 1] Verifying SHA-256 Checksum...")
    sha256 = hashlib.sha256()
    with open(TRACE_FILE, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    computed_digest = sha256.hexdigest()

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_digest = manifest.get("sha256_checksum")

    print(f"Computed SHA-256: {computed_digest}")
    print(f"Manifest SHA-256: {manifest_digest}")
    assert computed_digest == manifest_digest, f"SHA-256 mismatch! {computed_digest} != {manifest_digest}"
    print("[PASS] SHA-256 checksum matches manifest exactly.")

    # 2. Manifest Validation
    print("\n[Step 2] Validating Manifest Attributes...")
    print(f"Model: {manifest.get('model_name')}")
    print(f"Layers: {manifest.get('num_layers')}, Q Heads: {manifest.get('num_query_heads')}, KV Heads: {manifest.get('num_kv_heads')}")
    print(f"Head Dim: {manifest.get('head_dim')}, GQA Ratio: {manifest.get('gqa_ratio')}")
    print(f"Prompt Tokens: {manifest.get('prompt_tokens')}, Generated: {manifest.get('generated_tokens')}, Total: {manifest.get('total_tokens')}")
    print(f"Declared Events: {manifest.get('total_events')}")

    assert manifest.get("num_layers") == 24, "Expected 24 layers"
    assert manifest.get("num_kv_heads") == 2, "Expected 2 KV heads"
    assert manifest.get("head_dim") == 64, "Expected head_dim=64"
    assert manifest.get("tokens_per_block") == 16, "Expected 16 tokens/block"
    assert manifest.get("key_page_bytes") == 4096, "Expected 4096 bytes Key page"
    assert manifest.get("value_page_bytes") == 4096, "Expected 4096 bytes Value page"
    assert manifest.get("logical_block_bytes") == 8192, "Expected 8192 bytes combined block"
    assert manifest.get("total_tokens") == 703, "Expected 703 total tokens"
    assert manifest.get("total_events") == 7872, "Expected 7872 total events"
    print("[PASS] Manifest metadata completely verified.")

    # 3. Stream Parse and Record-Level Validation
    print("\n[Step 3] Parsing and Validating All 7,872 Events...")
    op_counts = {}
    op_bytes = {}
    sub_page_counts = {}
    layer_counts = {}
    head_counts = {}
    tier_counts = {}

    prev_event_id = -1
    prev_timestamp_ns = 0

    total_events = 0
    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            total_events += 1
            rec = json.loads(line)

            event_id = rec.get("event_id")
            assert event_id == prev_event_id + 1, f"Non-contiguous event_id at line {line_no}: {event_id} != {prev_event_id + 1}"
            prev_event_id = event_id

            ts = rec.get("timestamp_ns", 0)
            assert ts >= prev_timestamp_ns, f"Non-monotonic timestamp at line {line_no}: {ts} < {prev_timestamp_ns}"
            prev_timestamp_ns = ts

            op = rec.get("operation")
            byte_size = rec.get("byte_size")
            sub_page = rec.get("sub_page")
            layer_id = rec.get("layer_id")
            head_id = rec.get("head_id")
            tier = rec.get("tier")
            step = rec.get("step")
            b_id = rec.get("block_id")

            # Track distributions
            op_counts[op] = op_counts.get(op, 0) + 1
            op_bytes[op] = op_bytes.get(op, 0) + byte_size
            sub_page_counts[sub_page] = sub_page_counts.get(sub_page, 0) + 1
            layer_counts[layer_id] = layer_counts.get(layer_id, 0) + 1
            head_counts[head_id] = head_counts.get(head_id, 0) + 1
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

            # Semantic constraints per operation
            if op == "PREFILL_WRITE":
                assert step == 0, f"PREFILL_WRITE must be at step 0, got {step}"
                assert sub_page == "BOTH", f"PREFILL_WRITE must allocate BOTH K+V, got {sub_page}"
                assert byte_size == 8192, f"PREFILL_WRITE block must be 8192 bytes, got {byte_size}"
                assert head_id in (0, 1), f"head_id must be in (0, 1) for 2 KV heads, got {head_id}"
                assert rec.get("selected_blocks") == [b_id], "PREFILL_WRITE selected_blocks mismatch"

            elif op == "DECODE_READ":
                assert step >= 1, f"DECODE_READ must be step >= 1, got {step}"
                assert tier == "DRAM", f"DECODE_READ is for DRAM-resident blocks, got {tier}"
                assert sub_page == "BOTH", f"DECODE_READ fetches BOTH K+V, got {sub_page}"
                assert byte_size == 8192, f"DECODE_READ block must be 8192 bytes, got {byte_size}"

            elif op == "TOPK_FILTER":
                assert step >= 1, f"TOPK_FILTER must be step >= 1, got {step}"
                assert tier == "SSD", f"TOPK_FILTER is on SSD candidate blocks, got {tier}"
                assert sub_page == "KEY", f"TOPK_FILTER must scan ONLY Key pages, got {sub_page}"
                # byte_size must be candidates * 4096
                cands = rec.get("candidate_blocks", [])
                assert len(cands) > 0, "TOPK_FILTER candidate_blocks cannot be empty"
                assert byte_size == len(cands) * 4096, f"TOPK_FILTER bytes mismatch: {byte_size} != {len(cands) * 4096}"
                assert len(rec.get("selected_blocks", [])) > 0, "TOPK_FILTER selected_blocks cannot be empty"

            elif op == "TOPK_FETCH":
                assert step >= 1, f"TOPK_FETCH must be step >= 1, got {step}"
                assert tier == "SSD", f"TOPK_FETCH is from SSD, got {tier}"
                assert sub_page == "VALUE", f"TOPK_FETCH transfers Value pages, got {sub_page}"
                assert byte_size == 4096, f"TOPK_FETCH Value page must be 4096 bytes, got {byte_size}"
                assert b_id in rec.get("selected_blocks", []), f"Fetched block {b_id} not in selected_blocks"

            else:
                raise ValueError(f"Unknown operation: {op}")

    assert total_events == 7872, f"Expected 7872 events, got {total_events}"
    print(f"[PASS] Successfully verified all {total_events} events without errors.")

    print("\n[Step 4] Operation and Byte Accounting:")
    print(f"{'Operation':<16} | {'Count':<10} | {'Total Bytes':<14} | {'Total MB':<10}")
    print("-" * 56)
    total_bytes = 0
    for op, cnt in op_counts.items():
        b = op_bytes[op]
        total_bytes += b
        print(f"{op:<16} | {cnt:<10} | {b:<14,} | {b/(1024*1024):>8.2f} MB")
    print("-" * 56)
    print(f"{'TOTAL':<16} | {total_events:<10} | {total_bytes:<14,} | {total_bytes/(1024*1024):>8.2f} MB")

    print("\n[Step 5] Sub-page and Physical Geometry Verification:")
    for sp, cnt in sub_page_counts.items():
        print(f"  - Sub-page {sp:<6}: {cnt:>5} events")
    assert sub_page_counts.get("KEY", 0) > 0, "Must have KEY-only events (in-storage filtering)"
    assert sub_page_counts.get("VALUE", 0) > 0, "Must have VALUE-only events (top-k fetching)"
    assert sub_page_counts.get("BOTH", 0) > 0, "Must have BOTH events (prefill write & DRAM hits)"

    print("\n[Step 6] KV Head Distribution (GQA Verification):")
    for hid, cnt in head_counts.items():
        print(f"  - KV Head {hid}: {cnt:>5} events ({cnt/total_events*100:.1f}%)")
    assert set(head_counts.keys()).issubset({0, 1}), f"Unexpected KV heads: {head_counts.keys()}"

    # 4. Canonical Contract Deserialization Test (P3 Schema Test)
    print("\n[Step 7] Testing Canonical Contract Deserialization (CanonicalTraceRecord)...")
    # Verify we can parse all records with CanonicalTraceRecord
    from common.schemas.trace import CanonicalTraceRecord
    parsed_canonical = 0
    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            record = CanonicalTraceRecord.from_dict(data)
            assert record.event_id == data["event_id"]
            assert record.seq_id == data["event_id"]
            assert record.step == data["step"]
            assert record.step_id == data["step"]
            assert record.operation == data["operation"]
            assert record.byte_size == data["byte_size"]
            assert record.byte_length == data["byte_size"]
            parsed_canonical += 1

    assert parsed_canonical == 7872, f"Failed canonical parse: {parsed_canonical} != 7872"
    print(f"[PASS] 100% of events ({parsed_canonical}/{total_events}) parse into CanonicalTraceRecord without data loss.")

    print("\n=================================================================")
    print("PHASE 2 VALIDATION RESULT: ALL CHECKS PASSED (100% COMPLIANT)")
    print("=================================================================")

    return {
        "total_events": total_events,
        "total_bytes": total_bytes,
        "op_counts": op_counts,
        "op_bytes": op_bytes,
        "sub_page_counts": sub_page_counts,
        "head_counts": head_counts,
        "sha256_match": True,
        "canonical_contract_pass": True,
    }


if __name__ == "__main__":
    validate_trace()
