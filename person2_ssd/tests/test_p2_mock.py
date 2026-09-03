"""
Standalone Verification Suite for Person 2 (AI-SSD & FTL Subsystem).
Zero external dependencies, pure Python standard library.
Directly discoverable by scripts/run_tests.py.
"""

import re
from common.schemas.kv_block import KVBlock, StorageTier
from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    SSD_PAGES_PER_BLOCK,
    T_R_US,
    T_PROG_US,
    T_BERS_US,
    BUS_TRANSFER_US_PER_PAGE,
    PCIE_OVERHEAD_US,
)
from person2_ssd.nand.page import FlashPage, PageState
from person2_ssd.nand.block import FlashBlock
from person2_ssd.nand.nand import FlashDie, FlashPlane
from person2_ssd.channels.channel import FlashChannel, ChannelTransferRequest
from person2_ssd.ftl.conventional import ConventionalFTL
from person2_ssd.ftl.tensor_aware import TensorAwareFTL
from person2_ssd.storage_model.latency import LatencyModel
from person2_ssd.storage_model.io_model import StorageSimulator, parse_physical_location
from person2_ssd.mock_kv_engine import MockKVEngine


def test_flash_page_and_block():
    """
    Suite 1: Flash Page & Block lifecycle, state transitions, double-programming rejection,
    int-equality for allocated FlashPage, erase cycles, and wear limits.
    """
    # 1. Page State Transitions: FREE -> VALID -> INVALID -> FREE
    page = FlashPage(page_id=0, size_bytes=4096)
    assert page.state == PageState.FREE
    assert page.data_block_id is None
    assert page.read_count == 0
    assert page.program_count == 0

    page.program(data_block_id=101)
    assert page.state == PageState.VALID
    assert page.data_block_id == 101
    assert page.program_count == 1

    # Double programming rejection
    caught_double_prog = False
    try:
        page.program(data_block_id=102)
    except ValueError:
        caught_double_prog = True
    assert caught_double_prog, "Programming a non-FREE page must raise ValueError"

    # Read and read count tracking
    read_data = page.read(current_time_us=50.0)
    assert read_data == 101
    assert page.read_count == 1
    assert page.last_accessed_us == 50.0

    # Invalidate page
    page.invalidate()
    assert page.state == PageState.INVALID
    assert page.data_block_id is None
    assert page.read() is None

    # Erase page back to FREE
    page.erase()
    assert page.state == PageState.FREE
    assert page.data_block_id is None
    assert page.read_count == 0

    # 2. Block Lifecycle, Page Allocation & Integer-Equality
    block = FlashBlock(block_id=0, pages_count=16, max_erase_cycles=3000)
    assert block.free_page_count == 16
    assert block.valid_page_count == 0
    assert block.invalid_page_count == 0
    assert block.is_empty is True
    assert block.is_full is False
    assert block.is_bad_block is False
    assert block.garbage_ratio == 0.0
    assert len(block) == 16

    # Allocation returning FlashPage with int-equality support
    p_idx = block.allocate_page(logical_block_id=101)
    assert isinstance(p_idx, FlashPage)
    assert p_idx == 0  # int-equality check
    assert int(p_idx) == 0
    assert p_idx.page_id == 0
    assert p_idx.state == PageState.VALID
    assert p_idx.data_block_id == 101
    assert block.free_page_count == 15
    assert block.valid_page_count == 1

    # Invalidate and check garbage ratio
    block.pages[0].invalidate()
    assert block.valid_page_count == 0
    assert block.invalid_page_count == 1
    assert block.garbage_ratio == 1 / 16

    # Fill remaining pages to test is_full and over-allocation
    for i in range(15):
        p = block.allocate_page(logical_block_id=200 + i)
        assert p == i + 1

    assert block.is_full is True
    assert block.free_page_count == 0
    assert block.allocate_page(logical_block_id=999) is None

    # Block erase and cycle counting
    block.erase()
    assert block.free_page_count == 16
    assert block.valid_page_count == 0
    assert block.invalid_page_count == 0
    assert block.erase_count == 1
    assert block.is_full is False
    assert block.is_bad_block is False

    # Wear limits (is_bad_block at 3000 cycles)
    block.erase_count = 2999
    assert block.is_bad_block is False
    block.erase()
    assert block.erase_count == 3000
    assert block.is_bad_block is True


def test_physical_hierarchy_and_channels():
    """
    Suite 2: 5-level physical hierarchy (Channel -> Die -> Plane -> Block -> Page),
    timing scheduling, ChannelTransferRequest dataclass, and FIFO queue processing.
    """
    channel = FlashChannel(channel_id=0, dies_per_channel=4, planes_per_die=2, blocks_per_plane=16)
    assert channel.channel_id == 0
    assert len(channel.dies) == 4
    assert len(channel) == 4

    # Level 2: FlashDie
    die = channel[0]
    assert isinstance(die, FlashDie)
    assert die.die_id == 0
    assert len(die.planes) == 2
    assert len(die) == 2

    # Level 3: FlashPlane
    plane = die[0]
    assert isinstance(plane, FlashPlane)
    assert plane.plane_id == 0
    assert len(plane.blocks) == 16
    assert len(plane) == 16

    # Level 4 & 5: FlashBlock & FlashPage
    blk = plane[0]
    assert isinstance(blk, FlashBlock)
    assert isinstance(blk[0], FlashPage)

    # Die timing schedules (t_R, t_PROG, t_BERS)
    finish_r = die.schedule_read(start_time_us=0.0)
    assert finish_r == T_R_US
    assert die.is_busy(10.0) is True
    assert die.is_busy(30.0) is False

    finish_p = die.schedule_program(start_time_us=30.0)
    assert finish_p == 30.0 + T_PROG_US

    finish_e = die.schedule_erase(start_time_us=finish_p)
    assert finish_e == finish_p + T_BERS_US

    die.reset()
    assert die.busy_until_us == 0.0

    # ChannelTransferRequest and transfer queue serialization
    channel.reset()
    req1 = ChannelTransferRequest(
        request_id="req1",
        die_id=0,
        plane_id=0,
        block_id=0,
        page_id=0,
        op_type="READ",
        arrival_time_us=0.0,
    )
    req2 = ChannelTransferRequest(
        request_id="req2",
        die_id=1,
        plane_id=0,
        block_id=0,
        page_id=0,
        op_type="READ",
        arrival_time_us=0.0,
    )
    channel.enqueue_request(req1)
    channel.enqueue_request(req2)
    assert len(channel.queue) == 2

    completion_time = channel.process_queue(base_time_us=0.0)
    assert len(channel.queue) == 0
    assert len(channel.completed_transfers) == 2

    # req1 sensing finishes at 25.0 us, bus transfer 25.0 -> 30.0 us
    assert req1.start_time_us == 25.0
    assert req1.completion_time_us == 30.0

    # req2 sensing finishes at 25.0 us on die 1, bus busy until 30.0 us, bus transfer 30.0 -> 35.0 us
    assert req2.start_time_us == 30.0
    assert req2.completion_time_us == 35.0
    assert completion_time == 35.0
    assert channel.bus_busy_until_us == 35.0

    # Channel reset cleans up queues and bus
    channel.reset()
    assert channel.bus_busy_until_us == 0.0
    assert len(channel.completed_transfers) == 0


def test_channel_contention_timing_proof():
    """
    Suite 3: Concurrently reading N blocks mapped to the same channel takes strictly
    longer than reading N blocks striped across N distinct channels for all N in [2..8].
    """
    latency_model = LatencyModel(channels=8)

    # 1. Analytical Latency Model Proof across all N in [2..8]
    for n in range(2, 9):
        # All N blocks mapped to channel 0
        locs_contended = [f"ch0_die{i % 4}_pl0_blk0_pg{i}" for i in range(n)]
        # N blocks striped across N distinct channels
        locs_striped = [f"ch{i}_die0_pl0_blk0_pg0" for i in range(n)]

        lat_contended = latency_model.calculate_batch_read_latency(locs_contended)
        lat_striped = latency_model.calculate_batch_read_latency(locs_striped)

        assert lat_contended > lat_striped, (
            f"Contention proof violated for N={n}: contended={lat_contended}us <= striped={lat_striped}us"
        )

        # Exact formula checks
        expected_striped = PCIE_OVERHEAD_US + 1 * (T_R_US + BUS_TRANSFER_US_PER_PAGE)
        expected_contended = PCIE_OVERHEAD_US + n * (T_R_US + BUS_TRANSFER_US_PER_PAGE)
        assert lat_striped == expected_striped
        assert lat_contended == expected_contended
        assert lat_contended - lat_striped == (n - 1) * (T_R_US + BUS_TRANSFER_US_PER_PAGE)

    # 2. Physical Channel Queue Serialization Proof across all N in [2..8]
    for n in range(2, 9):
        # Single channel serializing N read requests
        ch_single = FlashChannel(channel_id=0, dies_per_channel=4)
        for i in range(n):
            ch_single.enqueue_request(
                ChannelTransferRequest(
                    request_id=i, die_id=i % 4, op_type="READ", arrival_time_us=0.0
                )
            )
        time_single = ch_single.process_queue(base_time_us=0.0)

        # N independent channels processing 1 request concurrently
        striped_channels = [FlashChannel(channel_id=i, dies_per_channel=4) for i in range(n)]
        time_striped = 0.0
        for i in range(n):
            striped_channels[i].enqueue_request(
                ChannelTransferRequest(
                    request_id=i, die_id=0, op_type="READ", arrival_time_us=0.0
                )
            )
            t_finish = striped_channels[i].process_queue(base_time_us=0.0)
            time_striped = max(time_striped, t_finish)

        assert time_single > time_striped, (
            f"Physical queue contention violated for N={n}: single={time_single}us <= striped={time_striped}us"
        )


def test_latency_model_equations():
    """
    Suite 4: Analytical read, write, and erase latency equations, diagnostic breakdowns,
    and regex parsing of canonical and prefixed physical address strings.
    """
    model = LatencyModel()
    assert model.pcie_overhead_us == 10.0
    assert model.t_r_us == 25.0
    assert model.bus_transfer_us == 5.0
    assert model.t_prog_us == 200.0
    assert model.t_bers_us == 2000.0
    assert model.channels == 8

    # Empty inputs
    assert model.calculate_batch_read_latency([]) == 0.0
    assert model.calculate_batch_write_latency([]) == 0.0
    assert model.calculate_batch_erase_latency([]) == 0.0

    # Locations with 4 requests on ch0, 2 on ch1, 1 on ch2 -> max channel load = 4
    locs = [
        "ch0_die0_pl0_blk0_pg0",
        "ch0_die1_pl0_blk0_pg1",
        "ch0_die2_pl0_blk0_pg2",
        "ch0_die3_pl0_blk0_pg3",
        "ch1_die0_pl0_blk0_pg0",
        "ch1_die1_pl0_blk0_pg1",
        "ch2_die0_pl0_blk0_pg0",
    ]

    # Read: T = t_pcie + max_c (N_c * (t_R + t_bus)) = 10.0 + 4 * (25.0 + 5.0) = 130.0 us
    assert model.calculate_batch_read_latency(locs) == 130.0

    # Write: T = t_pcie + max_c (N_c * (t_bus + t_PROG)) = 10.0 + 4 * (5.0 + 200.0) = 830.0 us
    assert model.calculate_batch_write_latency(locs) == 830.0

    # Erase: T = t_pcie + max_c (N_c * t_BERS) = 10.0 + 4 * 2000.0 = 8010.0 us
    assert model.calculate_batch_erase_latency(locs) == 8010.0

    # Regex parsing of canonical and prefixed address strings
    assert LatencyModel.extract_channel("ch0_die0_pl0_blk0_pg0") == 0
    assert LatencyModel.extract_channel("ch7_die3_pl1_blk63_pg127") == 7
    assert LatencyModel.extract_channel("ssd_ch3_die0_pl0_blk1_pg2") == 3
    assert LatencyModel.extract_channel("pba_CH5_die1_pl0_blk0_pg0") == 5
    assert LatencyModel.extract_channel("NODE_CH6_BLOCK") == 6
    assert LatencyModel.extract_channel("invalid_address_format") == 0
    assert LatencyModel.extract_channel("") == 0
    assert LatencyModel.extract_channel(None) == 0

    # Latency breakdown diagnostic report
    breakdown = model.get_latency_breakdown(locs)
    assert breakdown["total_requests"] == 7
    assert breakdown["bottleneck_channel"] == 0
    assert breakdown["max_channel_load"] == 4
    assert breakdown["pcie_overhead_us"] == 10.0
    assert breakdown["read_channel_time_us"] == 120.0
    assert breakdown["total_read_latency_us"] == 130.0
    assert breakdown["write_channel_time_us"] == 820.0
    assert breakdown["total_write_latency_us"] == 830.0
    assert breakdown["total_erase_latency_us"] == 8010.0


def test_canonical_address_format_and_translation():
    """
    Suite 5: Canonical address format matching 'ch<C>_die<D>_pl<P>_blk<B>_pg<G>',
    bidirectional translation (translate and reverse_translate), and mapping table snapshots.
    """
    canonical_pattern = re.compile(
        r"^ch(?P<ch>\d+)_die(?P<die>\d+)_pl(?P<pl>\d+)_blk(?P<blk>\d+)_pg(?P<pg>\d+)$"
    )

    # Parser utility test
    parsed = parse_physical_location("ch3_die2_pl1_blk15_pg63")
    assert parsed == (3, 2, 1, 15, 63)
    assert parse_physical_location("invalid") is None
    assert parse_physical_location("") is None
    assert parse_physical_location(None) is None

    # 1. ConventionalFTL allocation and translation
    conv_ftl = ConventionalFTL(channels=8, dies_per_channel=4, planes_per_die=2, blocks_per_plane=64)
    b0 = KVBlock.create_default(block_id=10, layer_id=0, token_start=0, token_count=16, kv_head_start=0)
    b1 = KVBlock.create_default(block_id=11, layer_id=0, token_start=16, token_count=16, kv_head_start=1)

    loc0 = conv_ftl.allocate(b0)
    loc1 = conv_ftl.allocate(b1)

    assert canonical_pattern.match(loc0), f"Address {loc0} does not match canonical format"
    assert canonical_pattern.match(loc1), f"Address {loc1} does not match canonical format"
    assert b0.physical_location == loc0
    assert b1.physical_location == loc1

    # Bidirectional translation
    assert conv_ftl.translate(10) == loc0
    assert conv_ftl.get_location(10) == loc0
    assert conv_ftl.reverse_translate(loc0) == 10
    assert conv_ftl.translate(9999) is None
    assert conv_ftl.reverse_translate("ch0_die0_pl0_blk99_pg99") is None

    # Mapping table snapshots
    map_snapshot = conv_ftl.get_mapping_table()
    assert map_snapshot[10] == loc0
    assert map_snapshot[11] == loc1
    rev_snapshot = conv_ftl.get_reverse_mapping_table()
    assert rev_snapshot[loc0] == 10
    assert rev_snapshot[loc1] == 11

    # 2. TensorAwareFTL allocation and translation
    ta_ftl = TensorAwareFTL(channels=8, dies_per_channel=4, planes_per_die=2, blocks_per_plane=64)
    b2 = KVBlock.create_default(block_id=20, layer_id=1, token_start=0, token_count=16, kv_head_start=2)
    b3 = KVBlock.create_default(block_id=21, layer_id=1, token_start=16, token_count=16, kv_head_start=3)

    loc2 = ta_ftl.allocate(b2)
    loc3 = ta_ftl.allocate(b3)

    assert canonical_pattern.match(loc2), f"Address {loc2} does not match canonical format"
    assert canonical_pattern.match(loc3), f"Address {loc3} does not match canonical format"
    assert b2.physical_location == loc2
    assert b3.physical_location == loc3

    assert ta_ftl.translate(20) == loc2
    assert ta_ftl.get_location(20) == loc2
    assert ta_ftl.reverse_translate(loc2) == 20
    assert ta_ftl.translate(8888) is None
    assert ta_ftl.reverse_translate("ch7_die3_pl1_blk99_pg99") is None

    ta_map = ta_ftl.get_mapping_table()
    assert ta_map[20] == loc2
    ta_rev = ta_ftl.get_reverse_mapping_table()
    assert ta_rev[loc2] == 20


def test_tensor_aware_vs_conventional_speedup():
    """
    Suite 6: Explicitly assert that Tensor-Aware achieves >= 2.5x speedup over
    Conventional FTL for 64 and 256 parallel KV blocks across 8 channels.
    """
    mock_kv = MockKVEngine(layers=32, heads=32)

    # 1. Batch Size 64 blocks
    blocks_64 = mock_kv.generate_kv_blocks(num_blocks=64, layer_id=0, layout="token_major")
    conv_ssd_64 = StorageSimulator(mode="conventional", channels=8)
    ta_ssd_64 = StorageSimulator(mode="tensor_aware", channels=8)

    for b in blocks_64:
        conv_ssd_64.store_block(b)
        ta_ssd_64.store_block(b)

    b_ids_64 = [b.block_id for b in blocks_64]
    conv_lat_64 = conv_ssd_64.estimate_read_latency(b_ids_64)
    ta_lat_64 = ta_ssd_64.estimate_read_latency(b_ids_64)
    speedup_64 = conv_lat_64 / ta_lat_64 if ta_lat_64 > 0 else 1.0

    assert speedup_64 >= 2.5, (
        f"Speedup for 64 blocks ({speedup_64:.2f}x) failed requirement >= 2.5x "
        f"(Conv: {conv_lat_64}us, TA: {ta_lat_64}us)"
    )
    # Verify read_blocks interface consistency
    assert conv_ssd_64.read_blocks(blocks_64) == conv_lat_64
    assert ta_ssd_64.read_blocks(blocks_64) == ta_lat_64

    # 2. Batch Size 256 blocks
    mock_kv.reset()
    blocks_256 = mock_kv.generate_kv_blocks(num_blocks=256, layer_id=0, layout="token_major")
    conv_ssd_256 = StorageSimulator(mode="conventional", channels=8)
    ta_ssd_256 = StorageSimulator(mode="tensor_aware", channels=8)

    for b in blocks_256:
        conv_ssd_256.store_block(b)
        ta_ssd_256.store_block(b)

    b_ids_256 = [b.block_id for b in blocks_256]
    conv_lat_256 = conv_ssd_256.estimate_read_latency(b_ids_256)
    ta_lat_256 = ta_ssd_256.estimate_read_latency(b_ids_256)
    speedup_256 = conv_lat_256 / ta_lat_256 if ta_lat_256 > 0 else 1.0

    assert speedup_256 >= 2.5, (
        f"Speedup for 256 blocks ({speedup_256:.2f}x) failed requirement >= 2.5x "
        f"(Conv: {conv_lat_256}us, TA: {ta_lat_256}us)"
    )
    assert conv_ssd_256.read_blocks(blocks_256) == conv_lat_256
    assert ta_ssd_256.read_blocks(blocks_256) == ta_lat_256

    # Verify monotonic scaling and validity
    assert speedup_256 >= speedup_64
    assert conv_lat_64 > 0 and ta_lat_64 > 0
    assert conv_lat_256 > 0 and ta_lat_256 > 0


def test_mock_kv_engine_isolation():
    """
    Suite 7: MockKVEngine standalone isolation, zero external dependencies,
    Token-Major vs Head-Major layouts, 100% uniform 8-channel distribution,
    and attention trace generation.
    """
    engine = MockKVEngine(layers=4, heads=8)

    # 1. Token-Major layout: heads cycle consecutively for the same token block
    tm_blocks = engine.generate_kv_blocks(num_blocks=16, token_count=16, layout="token_major")
    assert len(tm_blocks) == 16
    for i, b in enumerate(tm_blocks):
        assert b.kv_head_start == i % 8
        assert b.token_start == (i // 8) * 16

    # 2. Head-Major layout: consecutive tokens for the same head before advancing
    engine.reset()
    hm_blocks = engine.generate_kv_blocks(num_blocks=16, token_count=16, layout="head_major")
    assert len(hm_blocks) == 16
    for i, b in enumerate(hm_blocks):
        assert b.kv_head_start == (i // 2) % 8
        assert b.token_start == (i % 2) * 16

    # 3. 100% Uniform 8-Channel Striping Distribution Check
    engine_32 = MockKVEngine(layers=32, heads=32)
    model = LatencyModel(channels=8)

    # 64 blocks: exactly 8 blocks per channel (64 / 8 = 8)
    blocks_64 = engine_32.generate_kv_blocks(num_blocks=64, layer_id=0, layout="token_major")
    ta_ftl = TensorAwareFTL(channels=8)
    locs_64 = [ta_ftl.allocate(b) for b in blocks_64]
    loads_64 = model._get_channel_loads(locs_64)
    assert len(loads_64) == 8, f"Expected all 8 channels loaded, got {len(loads_64)}"
    for ch in range(8):
        assert loads_64[ch] == 8, f"Channel {ch} has load {loads_64.get(ch, 0)}, expected 8"

    # 256 blocks: exactly 32 blocks per channel (256 / 8 = 32)
    ta_ftl.reset()
    blocks_256 = engine_32.generate_kv_blocks(num_blocks=256, layer_id=0, layout="token_major")
    locs_256 = [ta_ftl.allocate(b) for b in blocks_256]
    loads_256 = model._get_channel_loads(locs_256)
    assert len(loads_256) == 8, f"Expected all 8 channels loaded, got {len(loads_256)}"
    for ch in range(8):
        assert loads_256[ch] == 32, f"Channel {ch} has load {loads_256.get(ch, 0)}, expected 32"

    # 4. Attention Trace Generation
    trace_heads = engine.generate_attention_trace(
        layer_id=1, total_blocks=32, k=8, pattern="concurrent_heads"
    )
    assert trace_heads == list(range(32, 40))

    trace_strided = engine.generate_attention_trace(
        layer_id=0, total_blocks=32, k=8, pattern="strided"
    )
    assert trace_strided == [0, 4, 8, 12, 16, 20, 24, 28]

    # 5. Sparse Attention Request (Sink + Recent)
    sparse_ids = engine_32.generate_sparse_attention_request(blocks_64, k=16, sink_ratio=0.25)
    assert len(sparse_ids) == 16
    assert sparse_ids[:4] == [b.block_id for b in blocks_64[:4]]
    assert sparse_ids[4:] == [b.block_id for b in blocks_64[-12:]]
