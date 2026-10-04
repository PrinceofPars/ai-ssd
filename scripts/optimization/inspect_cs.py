#!/usr/bin/env python3
import json

with open("/home/ubuntu/ai-ssd/benchmarks/live_inference/results/final_benchmark_results.json") as f:
    d = json.load(f)

cs = d.get("context_scaling", {})
for ctx, val in cs.items():
    print(f"Context {ctx} keys:", list(val.keys()))
    if "aissd" in val:
        print(f"  aissd keys:", list(val["aissd"].keys()))
        print(f"  aissd wall: {val['aissd'].get('wall_time_s')}, tps: {val['aissd'].get('tokens_per_second')}")
    if "baseline" in val:
        print(f"  baseline wall: {val['baseline'].get('wall_time_s')}, tps: {val['baseline'].get('tokens_per_second')}")
