# Standard assertion tests compatible with pytest and unittest
from person2_ssd.mock_kv_engine import MockKVEngine
from person2_ssd.nand.page import FlashPage, PageState
from person2_ssd.nand.block import FlashBlock
from person2_ssd.ftl.conventional import ConventionalFTL
from person2_ssd.ftl.tensor_aware import TensorAwareFTL
from person2_ssd.storage_model.io_model import StorageSimulator


def test_flash_page_and_block():
    block = FlashBlock(block_id=0, pages_count=16)
    assert block.free_page_count == 16
    assert block.valid_page_count == 0

    p_idx = block.allocate_page(data_block_id=101)
    assert p_idx == 0
    assert block.free_page_count == 15
    assert block.valid_page_count == 1

    # Invalidate and erase
    block.pages[0].invalidate()
    assert block.valid_page_count == 0
    assert block.invalid_page_count == 1

    block.erase()
    assert block.free_page_count == 16
    assert block.erase_count == 1


def test_tensor_aware_vs_conventional_speedup():
    mock_kv = MockKVEngine(layers=4, heads=8)
    # Generate 16 blocks within the same attention layer across different heads
    blocks = mock_kv.generate_kv_blocks(num_blocks=16, layer_id=0)

    # Conventional simulator
    conv_ssd = StorageSimulator(mode="conventional", channels=8)
    for b in blocks:
        conv_ssd.store_block(b)
    block_ids = [b.block_id for b in blocks]
    conv_latency = conv_ssd.estimate_read_latency(block_ids)

    # Tensor-Aware simulator
    ta_ssd = StorageSimulator(mode="tensor_aware", channels=8)
    for b in blocks:
        ta_ssd.store_block(b)
    ta_latency = ta_ssd.estimate_read_latency(block_ids)

    # Tensor-aware striping should distribute across channels and have lower latency than conventional
    assert ta_latency <= conv_latency
    assert ta_latency > 0
