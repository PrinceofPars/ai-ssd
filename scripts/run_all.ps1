# Run all benchmarks for Windows PowerShell
Write-Host "=== Running AI-SSD Benchmark Suite ===" -ForegroundColor Cyan

Write-Host "[1/6] Running Baseline..." -ForegroundColor Yellow
python benchmarks/run_baseline.py

Write-Host "[2/6] Running Offload Benchmark..." -ForegroundColor Yellow
python benchmarks/run_offload.py

Write-Host "[3/6] Running Top-k Benchmark..." -ForegroundColor Yellow
python benchmarks/run_topk.py

Write-Host "[4/6] Running FTL Benchmark..." -ForegroundColor Yellow
python benchmarks/run_ftl.py

Write-Host "[5/6] Running Prefetch Benchmark..." -ForegroundColor Yellow
python benchmarks/run_prefetch.py

Write-Host "[6/6] Running Full End-to-End System Benchmark..." -ForegroundColor Yellow
python benchmarks/run_full_system.py

Write-Host "=== All Benchmarks Completed! Results saved in results/raw/ ===" -ForegroundColor Green
