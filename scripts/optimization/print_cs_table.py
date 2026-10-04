#!/usr/bin/env python3
import json

with open("/home/ubuntu/ai-ssd/benchmarks/live_inference/results/final_benchmark_results.json") as f:
    d = json.load(f)

cs = d.get("context_scaling", {})
for ctx in ["4096", "8192", "16384", "32768"]:
    val = cs.get(ctx, {})
    b = val.get("baseline", {})
    a = val.get("aissd", {})
    print(f"\n=================== CONTEXT {ctx} ===================")
    print(f"Dense Baseline: Wall={b.get('wall_time_s'):.2f}s | Tok/s={b.get('tokens_per_second'):.3f} | Peak RSS={b.get('peak_rss_mb'):.1f} MB | Active KV={b.get('active_kv_mb'):.1f} MB")
    print(f"Phase 9 AI-SSD: Wall={a.get('wall_time_s'):.2f}s | Tok/s={a.get('tokens_per_second'):.3f} | Peak RSS={a.get('peak_rss_mb'):.1f} MB | Active KV={a.get('active_kv_mb'):.1f} MB")
    print(f"  - Cand K to Host: {a.get('candidate_k_bytes_to_host')} B | Bus Mvmt: {a.get('total_data_movement_bytes')/(1024*1024):.2f} MB | P2 Res: {a.get('p2_resident_payload_mb')} MB | P3 Res: {a.get('p3_resident_payload_mb')} MB")
    print(f"  - Top-K Time: {a.get('timing_breakdown', {}).get('topk_scoring_s', 0.0):.2f}s | Attn Time: {a.get('timing_breakdown', {}).get('attn_matmul_s', 0.0):.2f}s")
