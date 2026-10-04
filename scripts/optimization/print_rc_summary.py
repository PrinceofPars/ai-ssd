#!/usr/bin/env python3
import json

with open("/home/ubuntu/ai-ssd/benchmarks/live_inference/results/rc_validation_complete.json") as f:
    d = json.load(f)

csv = d.get("context_scaling_validation", {})
print("\n" + "=" * 105)
print("AI-SSD V2 — RELEASE CANDIDATE (cc3972e) CONTEXT SCALING VALIDATION")
print("=" * 105)
print(f"{'Context':<8} | {'Wall (s)':<9} | {'Tok/s':<7} | {'Peak RSS (MB)':<14} | {'Active KV':<10} | {'Top-K (s)':<10} | {'Attn (s)':<9} | {'Cand K':<8} | {'P2 Res':<8} | {'Match':<6}")
print("-" * 105)
for ctx, r in csv.items():
    tb = r.get("timing_breakdown", {})
    topk_s = tb.get("topk_scoring_s", 0.0)
    attn_s = tb.get("attn_matmul_s", 0.0)
    print(f"{ctx:<8} | {r['wall_time_s']:<9.2f} | {r['tokens_per_second']:<7.3f} | {r['peak_rss_mb']:<14.1f} | {r['active_kv_mb']:<10.1f} | {topk_s:<10.2f} | {attn_s:<9.2f} | {r['candidate_k_bytes_to_host']:<8} | {r['p2_resident_payload_mb']:<8.1f} | {str(r['token_match']):<6}")
print("=" * 105)

th = d.get("thread_scaling_sanity", {})
print("\n" + "=" * 80)
print("AI-SSD V2 — RELEASE CANDIDATE (cc3972e) THREAD SCALING SANITY")
print("=" * 80)
print(f"{'Threads':<8} | {'Wall (s)':<9} | {'Tok/s':<7} | {'Peak RSS (MB)':<14} | {'Match':<6}")
print("-" * 80)
for t, r in th.items():
    print(f"{t:<8} | {r['wall_time_s']:<9.2f} | {r['tokens_per_second']:<7.3f} | {r['peak_rss_mb']:<14.1f} | {str(r['token_match']):<6}")
print("=" * 80)
