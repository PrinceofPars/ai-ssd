"""Unit tests for Phase 2: KV Block Abstraction and Tiered Memory Management."""

import unittest
import numpy as np

from person1_kv_engine.baseline.transformer_attention import AttentionConfig
from person1_kv_engine.storage_backend.mock_ssd import MockSSDController
from person1_kv_engine.tiering.kv_block import KVBlock, KVBlockPool
from person1_kv_engine.tiering.hot_cold_classifier import HotColdClassifier, TieringPolicy
from person1_kv_engine.tiering.tiered_kv_manager import TieredKVManager


class TestKVBlockTiering(unittest.TestCase):

    def setUp(self):
        self.config = AttentionConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            dtype=np.float32,
        )
        self.ssd = MockSSDController()
        self.policy = TieringPolicy(
            sink_tokens=16,             # 1st block of 16 tokens is sink
            sliding_window_tokens=32,   # Last 2 blocks (32 tokens) are hot
            max_hot_blocks_per_layer=4,
            tokens_per_block=16,
        )
        self.manager = TieredKVManager(
            config=self.config,
            storage=self.ssd,
            policy=self.policy,
        )

    def test_block_allocation_and_chunking(self):
        """Validates that a 128-token prefill prompt is sliced into eight 16-token blocks."""
        seq_len = 128
        k_seq = np.random.randn(seq_len, 4, 32).astype(np.float32)
        v_seq = np.random.randn(seq_len, 4, 32).astype(np.float32)

        self.manager.prefill_sequence(layer_id=0, k_seq=k_seq, v_seq=v_seq)
        blocks = self.manager.block_pool.get_layer_blocks(layer_id=0)

        self.assertEqual(len(blocks), 8)
        self.assertEqual(blocks[0].token_start, 0)
        self.assertEqual(blocks[0].token_count, 16)
        self.assertTrue(blocks[0].is_pinned)  # Sink block must be pinned

    def test_hot_cold_eviction_to_ssd(self):
        """Validates that historical blocks are evicted to SSD while sink and window remain in RAM."""
        seq_len = 160  # 10 blocks of 16 tokens
        k_seq = np.random.randn(seq_len, 4, 32).astype(np.float32)
        v_seq = np.random.randn(seq_len, 4, 32).astype(np.float32)

        self.manager.prefill_sequence(layer_id=0, k_seq=k_seq, v_seq=v_seq)
        blocks = self.manager.block_pool.get_layer_blocks(layer_id=0)

        # Block 0: Sink (tokens 0-15) -> HOT (pinned)
        self.assertTrue(blocks[0].is_in_ram)
        self.assertTrue(blocks[0].is_pinned)

        # Blocks 1-7: Historical (tokens 16-127) -> COLD_SSD
        for b in blocks[1:8]:
            self.assertFalse(b.is_in_ram, f"Block {b.block_id} should be evicted from RAM")
            self.assertEqual(b.tier, "COLD_SSD")
            self.assertIsNone(b.k_data)
            self.assertIsNone(b.v_data)

        # Blocks 8-9: Sliding window (tokens 128-159) -> HOT
        self.assertTrue(blocks[8].is_in_ram)
        self.assertTrue(blocks[9].is_in_ram)

        # Verify SSD received the cold blocks
        telemetry = self.ssd.get_telemetry()
        self.assertGreater(telemetry["stored_blocks_count"], 0)
        self.assertGreater(telemetry["nand_write_ops"], 0)

        # Verify RAM savings
        ram_bytes = self.manager.get_host_ram_usage_bytes()
        offloaded_bytes = self.manager.get_offloaded_storage_bytes()
        self.assertGreater(offloaded_bytes, ram_bytes)

    def test_cold_block_retrieval_integrity(self):
        """Validates that fetching a cold block from SSD reconstructs exact original data."""
        seq_len = 64  # 4 blocks
        k_seq = np.random.randn(seq_len, 4, 32).astype(np.float32)
        v_seq = np.random.randn(seq_len, 4, 32).astype(np.float32)

        self.manager.prefill_sequence(layer_id=0, k_seq=k_seq, v_seq=v_seq)
        blocks = self.manager.block_pool.get_layer_blocks(layer_id=0)

        # Block 1 should be cold
        b1 = blocks[1]
        self.assertEqual(b1.tier, "COLD_SSD")

        # Fetch back
        fetched = self.manager.fetch_cold_block(block_id=b1.block_id, layer_id=0)
        self.assertTrue(fetched.is_in_ram)

        # Validate against original slice
        np.testing.assert_allclose(fetched.k_data, k_seq[16:32])
        np.testing.assert_allclose(fetched.v_data, v_seq[16:32])


if __name__ == "__main__":
    unittest.main()
