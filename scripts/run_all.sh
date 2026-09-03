#!/usr/bin/env bash
set -e

echo "=== Running AI-SSD Benchmark Suite ==="

echo "[1/6] Running Baseline..."
python3 benchmarks/run_baseline.py

echo "[2/6] Running Offload Benchmark..."
python3 benchmarks/run_offload.py

echo "[3/6] Running Top-k Benchmark..."
python3 benchmarks/run_topk.py

echo "[4/6] Running FTL Benchmark..."
python3 benchmarks/run_ftl.py

echo "[5/6] Running Prefetch Benchmark..."
python3 benchmarks/run_prefetch.py

echo "[6/6] Running Full End-to-End System Benchmark..."
python3 benchmarks/run_full_system.py

echo "=== All Benchmarks Completed! Results saved in results/raw/ ==="
