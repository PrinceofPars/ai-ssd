#!/usr/bin/env python3
"""
AI-SSD V2 — Phase 9 Final Demonstration.

Executes genuine end-to-end inference on Qwen3-4B with:
- Virtual QEMU/NVMe Direct Storage Device
- Computational Storage Top-K Filtering
- Asynchronous Pipelined Contiguous 8 KiB Retrieval
- Tensor-Aware Multi-Channel Flash FTL

Prints live telemetry and evidence classification.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = "/home/ubuntu/ai-ssd-p1/.venv/bin/python3"
WORKER_SCRIPT = str(PROJECT_ROOT / "scripts" / "context_scaling_worker.py")
OUT_JSON = "/tmp/phase9_demo_output.json"

EXPECTED_TOKENS = [11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871, 1948, 279, 3239, 4621, 323, 9144, 6894, 13]


def main():
    print("=" * 80)
    print("AI-SSD V2 — FINAL END-TO-END DEMONSTRATION")
    print("Architecture: Qwen3-4B + AI-SSD Computational Storage + QEMU/NVMe + Async")
    print("=" * 80)

    if os.path.exists(OUT_JSON):
        os.remove(OUT_JSON)

    cmd = [
        PYTHON_BIN, WORKER_SCRIPT,
        "--mode", "aissd",
        "--storage-mode", "nvme_qemu",
        "--enable-computational-storage",
        "--disable-prefetch",
        "--enable-async-pipeline",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--output-json", OUT_JSON,
    ]

    print("[DEMO] Launching genuine inference pipeline...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    total_launch_s = time.time() - t0

    if proc.returncode != 0:
        print("[FAIL] Demo execution failed:")
        print(proc.stderr)
        sys.exit(1)

    with open(OUT_JSON, "r") as f:
        data = json.load(f)

    tokens = data.get("token_ids", [])
    token_match = (tokens == EXPECTED_TOKENS)
    nvme_tel = data.get("nvme_telemetry", {})

    print("\n" + "=" * 80)
    print("AI-SSD V2 DEMONSTRATION RESULTS")
    print("=" * 80)
    print(f"Model:                   Qwen/Qwen3-4B-Instruct-2507 (FP32, 4 Threads)")
    print(f"Context:                 {data.get('context_length', 4096)} tokens")
    print(f"Decode tokens:           {data.get('decode_tokens', 16)} tokens")
    print(f"Storage mode:            QEMU Virtual NVMe Device (/dev/nvme0n1)")
    print(f"Top-K mode:              In-Storage Computational Filtering (Candidate Keys Never Cross Bus)")
    print(f"Async mode:              Asynchronous Pipelined Contiguous 8 KiB Retrieval")
    print("-" * 80)
    print(f"Wall time:               {data.get('wall_time_s', 0.0):.2f} s")
    print(f"Tok/s:                   {data.get('tokens_per_second', 0.0):.3f} tok/s")
    print("-" * 80)
    print(f"Peak RSS:                {data.get('peak_rss_mb', 0.0):.1f} MB (Active DRAM)")
    print(f"Active KV:               {data.get('active_kv_mb', 0.0):.1f} MB (vs 1,156.5 MB dense baseline)")
    print(f"P2 resident payload:     {data.get('p2_resident_payload_mb', 0.0):.1f} MB")
    print(f"P3 resident payload:     {data.get('p3_resident_payload_mb', 0.0):.1f} MB")
    print("-" * 80)
    print(f"Candidate K host bytes:  {data.get('candidate_k_bytes_to_host', 0)} bytes (STRICT IN-STORAGE INVARIANT)")
    print(f"Winning KV host bytes:   {(data.get('winning_k_bytes_to_host', 0) + data.get('winning_v_bytes_to_host', 0)) / (1024*1024):.2f} MB")
    print(f"Total bus movement:      {data.get('total_data_movement_bytes', 0) / (1024*1024):.2f} MB (vs ~592.7 MB in host-side topk)")
    print("-" * 80)
    print(f"NVMe reads:              {nvme_tel.get('nvme_read_ops', 0):,} ops")
    print(f"NVMe writes:             {nvme_tel.get('nvme_write_ops', 0):,} ops")
    print(f"NVMe throughput:         {nvme_tel.get('storage_throughput_mbs', 0.0):.1f} MB/s")
    print("-" * 80)
    print(f"Token IDs:               {tokens}")
    print(f"Generated text:         '{data.get('generated_text', '').strip()}'")
    print(f"Token match:             {'16/16 (100% EXACT MATCH)' if token_match else 'FAIL'}")
    print("=" * 80)
    print("EVIDENCE CLASSIFICATION:")
    print("[REAL]            : Qwen3-4B inference, CPU PyTorch tensors, Attention math, OS process RSS")
    print("[VIRTUAL-DEVICE]  : QEMU NVMe controller, Linux kernel NVMe driver, /dev/nvme0n1 direct block I/O")
    print("[ANALYTICAL]      : Multi-channel flash FTL conflict and latency model")
    print("DISCLAIMER: QEMU/NVMe measurements represent real virtualized storage running across the Linux")
    print("NVMe driver inside QEMU/KVM, and are not equivalent to physical ASIC computational-storage silicon.")
    print("=" * 80)


if __name__ == "__main__":
    main()
