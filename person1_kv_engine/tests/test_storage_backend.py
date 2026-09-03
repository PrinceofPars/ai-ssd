"""Unit tests for Storage Backend & Physical Flash Emulation (using standard unittest)."""

import unittest
import numpy as np

from person1_kv_engine.storage_backend.flash_model import FlashModel, FlashTimingConfig, EnergyConfig
from person1_kv_engine.storage_backend.mock_ssd import MockSSDController
from person1_kv_engine.storage_backend.kv_storage_api import KVBlockMetadata


class TestStorageBackend(unittest.TestCase):

    def test_flash_model_timing(self):
        """Validates physical flash timing and PCIe transfer calculations."""
        timing = FlashTimingConfig(
            t_read_us=35.0,
            t_prog_us=650.0,
            pcie_bandwidth_gbps=15.75,
            internal_channels=8,
            page_size_bytes=16384,
        )
        model = FlashModel(timing_config=timing)

        # Transfer of 1 MB over PCIe
        bytes_1mb = 1024 * 1024
        pcie_time = model.calculate_pcie_transfer_time_us(bytes_1mb)
        self.assertGreater(pcie_time, 1.2)  # Must include DMA base overhead
        # 1MB / (15.75 * 1000 MB/s) = ~66.5 us + 1.2 us = ~67.7 us
        self.assertTrue(60.0 < pcie_time < 80.0, f"PCIe time {pcie_time} not in [60, 80]")

        # Read 128 KB (8 pages of 16 KB) across 8 parallel channels -> 1 round of t_read
        bytes_128kb = 8 * 16384
        read_time = model.calculate_flash_read_time_us(bytes_128kb)
        # 1 round of 35 us + channel serialization
        self.assertTrue(35.0 < read_time < 60.0, f"Read time {read_time} not in [35, 60]")

    def test_flash_model_energy(self):
        """Validates energy calculation across tiers."""
        model = FlashModel()
        energy = model.calculate_energy_joules(
            host_dram_bytes=1000000,
            pcie_bytes=1000000,
            controller_dram_bytes=1000000,
            flash_read_bytes=1000000,
            flash_write_bytes=100000,
            compute_macs=500000,
        )
        self.assertGreater(energy["total_joules"], 0)
        self.assertGreater(energy["flash_read_joules"], energy["pcie_joules"])
        self.assertGreater(energy["pcie_joules"], energy["host_dram_joules"])

    def test_mock_ssd_store_and_load(self):
        """Validates store_kv and load_kv operations on simulated SSD."""
        ssd = MockSSDController()

        tokens = 16
        heads = 8
        head_dim = 64
        k_data = np.random.randn(tokens, heads, head_dim).astype(np.float32)
        v_data = np.random.randn(tokens, heads, head_dim).astype(np.float32)

        success = ssd.store_kv(block_id=0, layer_id=0, key_data=k_data, value_data=v_data)
        self.assertTrue(success)

        telemetry = ssd.get_telemetry()
        self.assertEqual(telemetry["host_to_device_bytes"], k_data.nbytes + v_data.nbytes)
        self.assertEqual(telemetry["nand_write_ops"], 1)
        self.assertGreater(telemetry["simulated_time_us"], 0)

        k_loaded, v_loaded = ssd.load_kv(block_id=0, layer_id=0)
        np.testing.assert_allclose(k_data, k_loaded)
        np.testing.assert_allclose(v_data, v_loaded)

        telemetry2 = ssd.get_telemetry()
        self.assertEqual(telemetry2["device_to_host_bytes"], k_data.nbytes + v_data.nbytes)
        self.assertEqual(telemetry2["nand_read_ops"], 1)

    def test_mock_ssd_in_storage_topk(self):
        """Validates in-storage top-k dot-product filtering."""
        ssd = MockSSDController()

        tokens = 16
        heads = 4
        head_dim = 32
        num_blocks = 10

        # Store 10 blocks
        for bid in range(num_blocks):
            k = np.random.randn(tokens, heads, head_dim).astype(np.float32)
            v = np.random.randn(tokens, heads, head_dim).astype(np.float32)
            ssd.store_kv(block_id=bid, layer_id=0, key_data=k, value_data=v)

        ssd.reset_telemetry()

        # Query vector
        query = np.random.randn(heads, head_dim).astype(np.float32)
        top_k = 3

        topk_ids, topk_vals, topk_logits, topk_scores = ssd.in_storage_topk_attention(query=query, layer_id=0, top_k=top_k)

        self.assertEqual(len(topk_ids), top_k)
        self.assertEqual(topk_vals.shape, (top_k, tokens, heads, head_dim))
        self.assertEqual(topk_logits.shape, (heads, top_k * tokens))
        self.assertEqual(len(topk_scores), top_k)
        # Top-k scores should be descending
        self.assertTrue(np.all(np.diff(topk_scores) <= 1e-5))

        telemetry = ssd.get_telemetry()
        # In-storage compute transfers ONLY query to device, and ONLY top-k values + scores to host
        total_pcie = telemetry["total_pcie_bytes"]
        full_load_bytes = 10 * (tokens * heads * head_dim * 4 * 2)
        self.assertLess(total_pcie, 0.4 * full_load_bytes, f"PCIe bytes {total_pcie} should be < 40% of full load {full_load_bytes}")


if __name__ == "__main__":
    unittest.main()
