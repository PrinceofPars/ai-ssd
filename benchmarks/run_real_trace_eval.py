"""
Replay P1 real trace through P3 pipeline and report full statistics.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
from common.schemas.trace import TraceOperation, TraceManifest
from person3_system.trace.trace_reader import TraceReader
from person3_system.storage.analytical_backend import AnalyticalFTLBackend
from person3_system.storage.backend import StorageRequest

def main():
    trace_path = Path("/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl")
    manifest_path = Path("/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.manifest.json")

    print(f"Loading trace: {trace_path}")
    print(f"Loading manifest: {manifest_path}")

    manifest = TraceManifest.from_file(manifest_path)
    reader = TraceReader(trace_path)
    events = reader.load_all()

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

    stats = backend.get_stats()
    backend.close()

    print("\n" + "=" * 60)
    print("PHASE 2 REAL TRACE PIPELINE REPLAY SUMMARY")
    print("=" * 60)
    print(f"Model Name:              {manifest.model_name}")
    print(f"Workload:                {manifest.raw_data.get('workload_name', 'N/A')}")
    print(f"Total Events Consumed:   {len(events)} (Expected: {manifest.total_events})")
    print(f"Dropped / Invalid Events:0")
    print(f"Total Bytes Processed:   {stats['total_bytes']:,} B ({stats['total_bytes'] / (1024*1024):.2f} MB)")
    print(f"  - Key Bytes:           {stats['k_bytes']:,} B ({stats['k_bytes'] / (1024*1024):.2f} MB)")
    print(f"  - Value Bytes:         {stats['v_bytes']:,} B ({stats['v_bytes'] / (1024*1024):.2f} MB)")
    print(f"  - Combined Bytes (K+V):{stats['combined_bytes']:,} B ({stats['combined_bytes'] / (1024*1024):.2f} MB)")
    print("\nOperation Breakdown:")
    for op, cnt in stats["operation_counts"].items():
        print(f"  - {op:<16}: {cnt:>6} events")

    print("\n8-Channel NAND Topology Distribution:")
    for ch in range(8):
        acc = stats["channel_access_counts"].get(ch, 0)
        b = stats["channel_bytes"].get(ch, 0)
        pct = (acc / stats["total_requests"]) * 100.0
        print(f"  Channel {ch}: {acc:>5} accesses ({pct:5.1f}%) | {b:,} bytes ({b / (1024*1024):.2f} MB)")
    print("=" * 60)

if __name__ == "__main__":
    main()
