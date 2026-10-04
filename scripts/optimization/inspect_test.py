#!/usr/bin/env python3
import json

with open("/tmp/test_opt3.json") as f:
    d = json.load(f)

print("Wall time:", d.get("wall_time_s"))
print("Tokens per sec:", d.get("tokens_per_second"))
print("Timing breakdown:")
for k, v in d.get("timing_breakdown", {}).items():
    print(f"  {k}: {v:.4f}s")
print("NVMe telemetry:")
for k, v in d.get("nvme_telemetry", {}).items():
    print(f"  {k}: {v}")
