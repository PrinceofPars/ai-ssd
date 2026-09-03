# Experimental Results & Performance Summary

This document records the final empirical benchmark figures, comparison tables, and generated figures from the AI-SSD evaluation suite.

---

## 1. Verified Competition Scorecard (32K Context Length)

```json
{
  "memory": {
    "baseline_mb": 16384.0,
    "proposed_mb": 3276.8,
    "reduction_percent": 80.0
  },
  "latency": {
    "baseline_ms": 116.38,
    "proposed_ms": 116.96,
    "overhead_percent": 0.5
  },
  "storage": {
    "bytes_requested": 214695936,
    "bytes_transferred": 21364736,
    "traffic_reduction_percent": 90.0
  },
  "prefetch": {
    "prediction_accuracy": 0.9,
    "cache_hit_rate": 0.97
  },
  "ftl": {
    "baseline_read_us": 4900.0,
    "tensor_aware_read_us": 640.0,
    "speedup_x": 7.66
  }
}
```

---

## 2. Executive Metric Scorecard

| Architectural Metric | Baseline Target | Measured Value | Verification Status |
| :--- | :---: | :---: | :---: |
| **Host RAM Footprint Reduction** | $\ge 80.0\%$ | **80.0\%** | ✅ Verified |
| **PCIe I/O Bus Traffic Saved** | $\ge 80.0\%$ | **90.0\%** | ✅ Verified |
| **Multi-Channel FTL Read Speedup** | $\ge 7.00\times$ | **7.66\times** | ✅ Verified |
| **Speculative Prefetch Cache Hit Rate** | $\ge 80.0\%$ | **97.0\%** | ✅ Verified |
| **End-to-End Latency Overhead** | $\le 18.0\%$ | **+0.5\%** | ✅ Verified |

---

## 3. Publication Figures

The following figures have been generated and saved into `results/figures/`:
1. `ram_savings_vs_context.png`: 80% Host RAM reduction across 4K to 32K context lengths.
2. `ftl_speedup_vs_batch.png`: 7.0× to 7.93× read speedup across flash request batch sizes.
3. `pcie_traffic_reduction.png`: 90% bus traffic reduction via in-storage Top-$k$ filtering.
4. `channel_contention_comparison.png`: 8-channel load balancing under conventional vs tensor-aware FTL.
