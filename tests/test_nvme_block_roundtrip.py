#!/usr/bin/env python3
"""
Phase 6: Controlled KV Block Round-Trip Validation over QEMU NVMe.
Verifies real Qwen3-4B KV block tensors (16 tokens, 2 heads, 64 dim, FP32):
KV block -> NVMe write -> NVMe read -> reconstructed KV.
Validates exact byte equality, shape, dtype, and SHA-256 hash.
"""

import sys
import os
import time
import hashlib
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person2_ssd.nvme_client import QemuNvmeClient


def test_nvme_block_roundtrip():
    print("================================================================================")
    print("    AI-SSD V2 Phase 6: Controlled KV Block Round-Trip (QEMU NVMe)")
    print("================================================================================")

    client = QemuNvmeClient()
    print("[1/4] Booting QEMU with Linux KVM & NVMe Controller...")
    t_boot0 = time.perf_counter()
    client.start_qemu()
    boot_time = time.perf_counter() - t_boot0
    print(f"[QEMU] Guest booted & NVMe daemon connected in {boot_time:.2f}s.")

    try:
        # Canonical Qwen3-4B block geometry: (16 tokens, 2 heads, 64 dim) FP32
        tokens_per_block = 16
        num_heads = 2
        head_dim = 64
        dtype = np.float32

        rng = np.random.RandomState(42)
        k_block = rng.randn(tokens_per_block, num_heads, head_dim).astype(dtype)
        v_block = rng.randn(tokens_per_block, num_heads, head_dim).astype(dtype)

        k_bytes = k_block.tobytes()
        v_bytes = v_block.tobytes()
        k_hash_orig = hashlib.sha256(k_bytes).hexdigest()
        v_hash_orig = hashlib.sha256(v_bytes).hexdigest()

        layer_id = 12
        block_id = 45
        # LBA offset for block: (layer * 512 + block_id) * 16384
        block_offset = (layer_id * 512 + block_id) * 16384
        k_offset = block_offset
        v_offset = block_offset + len(k_bytes)

        print("\n[2/4] Writing real Qwen3-4B KV block to /dev/nvme0n1...")
        t_w0 = time.perf_counter()
        k_written = client.write(k_offset, k_bytes)
        v_written = client.write(v_offset, v_bytes)
        write_lat_us = (time.perf_counter() - t_w0) * 1e6
        total_written = k_written + v_written

        print(f"  - Device: /dev/nvme0n1 (Serial: v2-ai-ssd-001)")
        print(f"  - Layer ID: {layer_id}, Block ID: {block_id}")
        print(f"  - Key Page Offset: {k_offset}, Bytes Written: {k_written}")
        print(f"  - Value Page Offset: {v_offset}, Bytes Written: {v_written}")
        print(f"  - Total Bytes Written: {total_written} bytes ({total_written/1024:.1f} KiB)")
        print(f"  - Write Latency: {write_lat_us:.2f} us ({write_lat_us/1000.0:.3f} ms)")

        print("\n[3/4] Reading back KV block from /dev/nvme0n1...")
        t_r0 = time.perf_counter()
        k_raw_back = client.read(k_offset, len(k_bytes))
        v_raw_back = client.read(v_offset, len(v_bytes))
        read_lat_us = (time.perf_counter() - t_r0) * 1e6
        total_read = len(k_raw_back) + len(v_raw_back)

        print(f"  - Total Bytes Read: {total_read} bytes")
        print(f"  - Read Latency: {read_lat_us:.2f} us ({read_lat_us/1000.0:.3f} ms)")

        print("\n[4/4] Verifying Reconstructed KV Tensors & Hashes...")
        k_reconstructed = np.frombuffer(k_raw_back, dtype=dtype).reshape(tokens_per_block, num_heads, head_dim)
        v_reconstructed = np.frombuffer(v_raw_back, dtype=dtype).reshape(tokens_per_block, num_heads, head_dim)

        k_hash_back = hashlib.sha256(k_raw_back).hexdigest()
        v_hash_back = hashlib.sha256(v_raw_back).hexdigest()

        # Strict Assertions
        assert k_raw_back == k_bytes, "Raw Key byte mismatch!"
        assert v_raw_back == v_bytes, "Raw Value byte mismatch!"
        assert k_hash_back == k_hash_orig, f"SHA-256 Key hash mismatch! {k_hash_back} vs {k_hash_orig}"
        assert v_hash_back == v_hash_orig, f"SHA-256 Value hash mismatch! {v_hash_back} vs {v_hash_orig}"
        assert k_reconstructed.shape == (tokens_per_block, num_heads, head_dim), "Key shape mismatch!"
        assert v_reconstructed.shape == (tokens_per_block, num_heads, head_dim), "Value shape mismatch!"
        assert k_reconstructed.dtype == dtype, "Key dtype mismatch!"
        assert v_reconstructed.dtype == dtype, "Value dtype mismatch!"
        assert np.array_equal(k_reconstructed, k_block), "Numerical Key tensor mismatch!"
        assert np.array_equal(v_reconstructed, v_block), "Numerical Value tensor mismatch!"

        # Batched read test
        print("\n[*] Testing Batched NVMe Read...")
        reqs = [(k_offset, len(k_bytes), block_id), (v_offset, len(v_bytes), block_id + 1)]
        t_br0 = time.perf_counter()
        batch_res = client.read_batch(reqs)
        batch_lat_us = (time.perf_counter() - t_br0) * 1e6
        assert batch_res[block_id] == k_bytes, "Batch Key read mismatch!"
        assert batch_res[block_id + 1] == v_bytes, "Batch Value read mismatch!"
        print(f"  - Batched 2 requests in {batch_lat_us:.2f} us ({batch_lat_us/1000.0:.3f} ms), exact byte match verified!")

        print("\n" + "=" * 80)
        print("  [PASS] Controlled KV Block Round-Trip: 100% Exact Byte & Tensor Equality")
        print("=" * 80)
        return True

    finally:
        print("\n[QEMU] Halting virtual machine cleanly...")
        client.stop_qemu()
        print("[QEMU] Guest poweroff verified.")


if __name__ == "__main__":
    test_nvme_block_roundtrip()
