#!/usr/bin/env python3
"""Test script to verify QEMU Virtual NVMe FP16 in-storage Top-K filtering.
"""

import sys
import math
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person2_ssd.nvme_client import QemuNvmeClient

def main():
    print("=" * 60)
    print("Testing QEMU NVMe In-Storage Top-K with FP16")
    print("=" * 60)

    client = QemuNvmeClient()
    print("[1] Starting/connecting to QEMU NVMe guest daemon...")
    client.start_qemu()
    print(f"Connected! Is alive: {client.is_alive()}")

    q_heads = 32
    kv_heads = 8
    head_dim = 128
    tokens_per_block = 16
    scale = 1.0 / math.sqrt(head_dim)
    gqa_ratio = q_heads // kv_heads

    # Generate synthetic FP16 data
    np.random.seed(42)
    query = np.random.randn(q_heads, head_dim).astype(np.float32)

    # 10 blocks of Keys
    num_blocks = 10
    k_blocks = []
    candidates = []
    base_offset = 0

    print(f"[2] Writing {num_blocks} FP16 Key blocks to virtual NVMe...")
    for bid in range(num_blocks):
        # [tokens, kv_heads, head_dim] in float16
        k_blk = (np.random.randn(tokens_per_block, kv_heads, head_dim) * 0.5).astype(np.float16)
        k_blocks.append(k_blk)
        k_bytes = k_blk.tobytes()
        offset = base_offset + bid * 65536  # 64 KiB spaced
        client.write(offset, k_bytes)
        candidates.append((offset, len(k_bytes), bid, tokens_per_block))

    print(f"[3] Dispatched in-storage Top-K (top_k=3, is_fp16=True)...")
    results = client.compute_topk(
        query=query,
        candidates=candidates,
        top_k=3,
        scale=scale,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        is_fp16=True,
    )

    print(f"Top-K results from NVMe guest: {results}")

    # Compute expected results using numpy (scalar reference)
    expected_scores = []
    for bid in range(num_blocks):
        k_blk = k_blocks[bid].astype(np.float32)
        # GQA dot product: for each token and each Q head:
        max_score = -1e30
        for t in range(tokens_per_block):
            for qh in range(q_heads):
                kh = qh // gqa_ratio
                dot = np.dot(query[qh], k_blk[t, kh]) * scale
                if dot > max_score:
                    max_score = dot
        expected_scores.append((float(max_score), bid, tokens_per_block))

    expected_scores.sort(key=lambda x: x[0], reverse=True)
    expected_top3 = expected_scores[:3]
    print(f"Expected Top-K (NumPy reference): {expected_top3}")

    # Verify
    for i in range(3):
        m_score, m_bid, _ = results[i]
        e_score, e_bid, _ = expected_top3[i]
        assert m_bid == e_bid, f"Rank {i} Block ID mismatch: measured {m_bid} vs expected {e_bid}"
        diff = abs(m_score - e_score)
        print(f"Rank {i}: BID={m_bid} | Measured={m_score:.6f} | Expected={e_score:.6f} | Diff={diff:.2e}")
        assert diff < 1e-3, f"Score difference too large: {diff}"

    print("\n[SUCCESS] In-storage Top-K FP16 kernel verified with 100% accuracy!")
    client.close()

if __name__ == "__main__":
    main()
