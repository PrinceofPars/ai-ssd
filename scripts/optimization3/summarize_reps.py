#!/usr/bin/env python3
import glob
import json

def main():
    files = sorted(glob.glob('/tmp/qwen3_8b_fp16_rep*.json'))
    print(f"Found {len(files)} repetition files:")
    for p in files:
        with open(p) as f:
            d = json.load(f)
        wall = d.get("wall_time_s", 0.0)
        tps = d.get("tokens_per_second", 0.0)
        rss = d.get("peak_rss_mb", 0.0)
        cand_k = d.get("candidate_k_bytes_to_host", 0)
        p2_res = d.get("p2_resident_payload_mb", 0.0)
        print(f"  {p}: wall={wall:.3f}s | {tps:.3f} tok/s | Peak RSS: {rss:.1f} MB | Cand K: {cand_k} B | P2 Res: {p2_res} MB")

if __name__ == "__main__":
    main()
