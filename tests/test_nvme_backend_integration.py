#!/usr/bin/env python3
"""
Unit test validating RealInferenceStorageBackend in nvme_qemu mode.
Verifies real end-to-end tensor read/write/batch through the Linux NVMe guest driver.
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person2_ssd.inference_backend import RealInferenceStorageBackend


def test_nvme_backend_integration():
    print("Testing RealInferenceStorageBackend(storage_mode='nvme_qemu')...")
    backend = RealInferenceStorageBackend(
        channels=8,
        num_layers=4,
        num_heads=2,
        tokens_per_block=16,
        head_dim=64,
        dtype="float32",
        storage_mode="nvme_qemu",
    )
    assert backend.CLASSIFICATION == "VIRTUAL-DEVICE"

    try:
        rng = np.random.RandomState(123)
        k_blocks = {}
        v_blocks = {}

        # Write 5 blocks
        for bid in range(5):
            k = rng.randn(16, 2, 64).astype(np.float32)
            v = rng.randn(16, 2, 64).astype(np.float32)
            k_blocks[bid] = k
            v_blocks[bid] = v
            backend.write_block(layer_idx=2, block_id=bid, k_block=k, v_block=v)

        # Single key read
        k0 = backend.read_key_page(layer_idx=2, block_id=0)
        np.testing.assert_array_equal(k0, k_blocks[0])

        # Single value read
        v0 = backend.read_value_page(layer_idx=2, block_id=0)
        np.testing.assert_array_equal(v0, v_blocks[0])

        # Batched key read
        k_batch = backend.read_key_page_batch(layer_idx=2, block_ids=[1, 2, 3])
        for bid in [1, 2, 3]:
            np.testing.assert_array_equal(k_batch[bid], k_blocks[bid])

        # Batched value read
        v_batch = backend.read_value_page_batch(layer_idx=2, block_ids=[2, 3, 4])
        for bid in [2, 3, 4]:
            np.testing.assert_array_equal(v_batch[bid], v_blocks[bid])

        # Batched block read
        b_batch = backend.read_block_batch(layer_idx=2, block_ids=[0, 4])
        np.testing.assert_array_equal(b_batch[0][0], k_blocks[0])
        np.testing.assert_array_equal(b_batch[0][1], v_blocks[0])
        np.testing.assert_array_equal(b_batch[4][0], k_blocks[4])
        np.testing.assert_array_equal(b_batch[4][1], v_blocks[4])

        # Telemetry verification
        telem = backend.get_telemetry()
        assert telem["backend_classification"] == "VIRTUAL-DEVICE"
        assert "nvme_telemetry" in telem
        nvme_t = telem["nvme_telemetry"]
        assert nvme_t["nvme_read_ops"] > 0
        assert nvme_t["nvme_write_ops"] > 0
        print(f"Backend NVMe Telemetry: {nvme_t}")
        print("[PASS] RealInferenceStorageBackend(storage_mode='nvme_qemu') fully verified!")

    finally:
        backend.close()


if __name__ == "__main__":
    test_nvme_backend_integration()
