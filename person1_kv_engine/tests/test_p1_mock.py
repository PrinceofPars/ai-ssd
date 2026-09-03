"""Unit tests proving Person 1 module works independently.

Fully compatible with both pytest and python -m unittest.
"""

import unittest
import numpy as np
from person1_kv_engine.mock_ssd import MockSSD
from person1_kv_engine.cache_manager.kv_cache import PagedKVCache
from person1_kv_engine.attention.scoring import AttentionScorer
from person1_kv_engine.attention.attention import AttentionEngine, precompute_rope_frequencies, apply_rotary_pos_emb
from person1_kv_engine.topk.selector import TopKSelector
from person1_kv_engine.topk.evaluator import TopKEvaluator
from person1_kv_engine.baseline.baseline_kv import BaselineKVCache
from common.schemas.kv_block import StorageTier, KVBlock


# Standalone function tests for pytest compatibility
def test_baseline_memory_calculation():
    baseline = BaselineKVCache(layers=32, heads=32, head_dim=128, dtype="FP16")
    # 4096 context length: 2 * 32 * 32 * 128 * 4096 * 2 = 2,147,483,648 bytes = 2048 MB
    assert baseline.get_memory_mb(4096) == 2048.0


def test_paged_kv_cache_with_mock_ssd():
    mock_ssd = MockSSD()
    kv_cache = PagedKVCache(
        layers=2,
        heads=2,
        block_tokens=16,
        head_dim=128,
        sink_tokens=32,
        recent_tokens=64,
        ssd_backend=mock_ssd,
    )

    # Initialize 128 tokens
    total_blocks = kv_cache.initialize_context(128)
    assert total_blocks == 2 * 2 * (128 // 16)  # 32 blocks

    # Run offload pass
    hot_cnt, cold_cnt = kv_cache.run_offload_pass(128)
    assert hot_cnt + cold_cnt == total_blocks
    assert cold_cnt > 0
    assert mock_ssd.write_count == cold_cnt

    # Verify cold blocks now in SSD tier
    for block in kv_cache.block_manager.all_blocks():
        if block.storage_tier == StorageTier.SSD.value:
            assert mock_ssd.has_block(block.block_id)
            loaded = mock_ssd.load_block(block.block_id)
            assert loaded is not None


def test_topk_selection():
    scorer = AttentionScorer(head_dim=128)
    selector = TopKSelector()

    blocks = [
        KVBlock.create_default(block_id=i, layer_id=0, token_start=i * 16, hotness=0.1 * i)
        for i in range(10)
    ]
    scores = scorer.score_blocks(blocks)
    top_3 = selector.select(scores, blocks, k=3)
    assert len(top_3) == 3
    # Block 9, 8, 7 should be top
    top_ids = [b.block_id for b in top_3]
    assert top_ids == [9, 8, 7]


def test_topk_evaluator_recall():
    recall = TopKEvaluator.calculate_recall([1, 2, 3, 4], [2, 3, 5, 6])
    assert recall == 0.5


def test_rope_orthogonality():
    cos, sin = precompute_rope_frequencies(head_dim=64, max_seq_len=64)
    vec = np.random.randn(4, 64).astype(np.float32)
    norm_before = np.linalg.norm(vec, axis=-1)
    vec_rot = apply_rotary_pos_emb(vec, cos, sin, pos_offset=10)
    norm_after = np.linalg.norm(vec_rot, axis=-1)
    np.testing.assert_allclose(norm_before, norm_after, rtol=1e-5)


# Standard unittest wrapper
class TestPerson1TestSuite(unittest.TestCase):
    def test_baseline(self):
        test_baseline_memory_calculation()

    def test_paged_cache(self):
        test_paged_kv_cache_with_mock_ssd()

    def test_topk(self):
        test_topk_selection()

    def test_evaluator(self):
        test_topk_evaluator_recall()

    def test_rope(self):
        test_rope_orthogonality()


if __name__ == "__main__":
    unittest.main()
