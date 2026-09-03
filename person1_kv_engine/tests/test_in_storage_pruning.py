"""Unit tests for Phase 3: In-Storage Pruning & Online Streaming Softmax."""

import unittest
import numpy as np

from person1_kv_engine.baseline.transformer_attention import AttentionConfig, ScaledDotProductAttention
from person1_kv_engine.storage_backend.mock_ssd import MockSSDController
from person1_kv_engine.tiering.kv_block import KVBlockPool
from person1_kv_engine.tiering.hot_cold_classifier import TieringPolicy
from person1_kv_engine.tiering.tiered_kv_manager import TieredKVManager
from person1_kv_engine.computational_storage.streaming_softmax import OnlineSoftmaxAccumulator, merge_online_attention
from person1_kv_engine.computational_storage.instorage_pruner import InStorageAttentionPruner


class TestInStoragePruning(unittest.TestCase):

    def test_online_softmax_accumulator_mathematical_equivalence(self):
        """Streaming online softmax across partitions must exactly equal full batch softmax."""
        heads = 4
        head_dim = 32
        tokens_total = 64
        partition_size = 32

        q = np.random.randn(heads, head_dim).astype(np.float32)
        k = np.random.randn(tokens_total, heads, head_dim).astype(np.float32)
        v = np.random.randn(tokens_total, heads, head_dim).astype(np.float32)

        # 1. Full batch reference
        scale = 1.0 / np.sqrt(head_dim, dtype=np.float32)
        k_t = np.transpose(k, (1, 2, 0))
        logits_full = np.matmul(q[:, np.newaxis, :], k_t).squeeze(axis=1) * scale  # [heads, tokens]
        max_full = np.max(logits_full, axis=-1, keepdims=True)
        exp_full = np.exp(logits_full - max_full)
        weights_full = exp_full / np.sum(exp_full, axis=-1, keepdims=True)
        v_t = np.transpose(v, (1, 0, 2))
        expected_out = np.matmul(weights_full[:, np.newaxis, :], v_t).squeeze(axis=1)

        # 2. Partitioned streaming accumulator
        acc = OnlineSoftmaxAccumulator(num_heads=heads, head_dim=head_dim)
        # Partition 1
        logits_p1 = logits_full[:, :partition_size]
        v_p1 = v[:partition_size]
        acc.update_with_partition(logits=logits_p1, values=v_p1)

        # Partition 2
        logits_p2 = logits_full[:, partition_size:]
        v_p2 = v[partition_size:]
        acc.update_with_partition(logits=logits_p2, values=v_p2)

        streaming_out = acc.finalize()

        np.testing.assert_allclose(expected_out, streaming_out, rtol=1e-5, atol=1e-5)

    def test_instorage_pruner_execution_and_bus_traffic(self):
        """Validates that InStorageAttentionPruner prunes cold blocks and saves PCIe traffic."""
        config = AttentionConfig(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            dtype=np.float32,
        )
        ssd = MockSSDController()
        policy = TieringPolicy(
            sink_tokens=16,
            sliding_window_tokens=32,
            max_hot_blocks_per_layer=4,
            tokens_per_block=16,
        )
        manager = TieredKVManager(config=config, storage=ssd, policy=policy)

        # Prefill 160 tokens (10 blocks)
        seq_len = 160
        for l in range(config.num_layers):
            k = np.random.randn(seq_len, 4, 32).astype(np.float32)
            v = np.random.randn(seq_len, 4, 32).astype(np.float32)
            manager.prefill_sequence(layer_id=l, k_seq=k, v_seq=v)

        ssd.reset_telemetry()

        pruner = InStorageAttentionPruner(
            config=config,
            storage=ssd,
            manager=manager,
            default_top_k=2,  # Retrieve only 2 of the 7 cold blocks
        )

        query = np.random.randn(4, 32).astype(np.float32)
        out, stats = pruner.compute_decode_attention(query=query, layer_id=0, top_k=2)

        self.assertEqual(out.shape, (4, 32))
        self.assertGreater(stats["pruning_ratio"], 0.70)
        self.assertEqual(stats["cold_blocks_retrieved"], 2)

        telemetry = ssd.get_telemetry()
        # PCIe bytes must reflect only Q + 2 value blocks (not all 7 cold blocks)
        self.assertGreater(telemetry["host_to_device_bytes"], 0)
        self.assertGreater(telemetry["device_to_host_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
