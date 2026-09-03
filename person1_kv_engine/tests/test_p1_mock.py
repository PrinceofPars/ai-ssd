# Standard assertion tests compatible with pytest and unittest
from person1_kv_engine.mock_ssd import MockSSD
from person1_kv_engine.cache_manager.kv_cache import PagedKVCache
from person1_kv_engine.attention.scoring import AttentionScorer
from person1_kv_engine.topk.selector import TopKSelector
from person1_kv_engine.baseline.baseline_kv import BaselineKVCache
from common.schemas.kv_block import StorageTier


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

    from common.schemas.kv_block import KVBlock
    blocks = [
        KVBlock.create_default(block_id=i, layer_id=0, token_start=i*16, hotness=0.1 * i)
        for i in range(10)
    ]
    scores = scorer.score_blocks(blocks)
    top_3 = selector.select(scores, blocks, k=3)
    assert len(top_3) == 3
    # Block 9, 8, 7 should be top
    top_ids = [b.block_id for b in top_3]
    assert top_ids == [9, 8, 7]
