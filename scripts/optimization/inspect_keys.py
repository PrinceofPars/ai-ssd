#!/usr/bin/env python3
import json
from pathlib import Path

p = Path("/home/ubuntu/ai-ssd/benchmarks/live_inference/results/final_benchmark_results.json")
if p.exists():
    with open(p) as f:
        d = json.load(f)
    print("Keys in final_benchmark_results.json:")
    for k in d.keys():
        print(f"  {k}")
        if isinstance(d[k], dict):
            for subk in list(d[k].keys())[:5]:
                print(f"    - {subk}")
