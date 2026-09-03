"""
Challenger M1.2: Empirical Contention & Address Parsing Stress Suite.
Focus areas:
1. Contention inequality validation across batch sizes N in [2..64].
2. Address parsing with irregular prefixes and edge case strings.
3. StorageSimulator concurrent queue serialization and physical state tracking.
4. Boundary conditions, polymorphic inputs, and exhaustion limits.
"""

import sys
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.schemas.kv_block import KVBlock, StorageTier
from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    SSD_PAGES_PER_BLOCK,
    SSD_PAGE_SIZE_BYTES,
    T_R_US,
    T_PROG_US,
    T_BERS_US,
    BUS_TRANSFER_US_PER_PAGE,
    PCIE_OVERHEAD_US,
)
from person2_ssd.nand.page import FlashPage, PageState
from person2_ssd.nand.block import FlashBlock
from person2_ssd.nand.nand import FlashPlane, FlashDie
from person2_ssd.channels.channel import FlashChannel, ChannelTransferRequest
from person2_ssd.storage_model.latency import LatencyModel
from person2_ssd.storage_model.io_model import (
    StorageSimulator,
    parse_physical_location,
)
from person2_ssd.mock_kv_engine import MockKVEngine


def test_contention_inequality_n_2_to_64():
    """Verify T_single > T_striped strictly holds for all N in [2..64]."""
    lm = LatencyModel()
    for n in range(2, 65):
        single_locs = [f"ch0_die0_pl0_blk0_pg{i % 128}" for i in range(n)]
        t_single = lm.calculate_batch_read_latency(single_locs)

        striped_locs = [
            f"ch{i % 8}_die{(i // 8) % 4}_pl0_blk{i // 32}_pg{i % 128}"
            for i in range(n)
        ]
        t_striped = lm.calculate_batch_read_latency(striped_locs)

        expected_t_single = PCIE_OVERHEAD_US + n * (T_R_US + BUS_TRANSFER_US_PER_PAGE)
        expected_max_load = math.ceil(n / 8.0)
        expected_t_striped = PCIE_OVERHEAD_US + expected_max_load * (T_R_US + BUS_TRANSFER_US_PER_PAGE)

        assert t_single > t_striped, f"Contention inequality violated at N={n}: {t_single} <= {t_striped}"
        assert abs(t_single - expected_t_single) < 1e-6, f"Mismatch in t_single at N={n}"
        assert abs(t_striped - expected_t_striped) < 1e-6, f"Mismatch in t_striped at N={n}"


def test_speedup_acceptance_at_64():
    """Verify >= 2.5x speedup at batch size 64 across 8 channels."""
    lm = LatencyModel()
    single_locs = [f"ch0_die0_pl0_blk0_pg{i % 128}" for i in range(64)]
    striped_locs = [
        f"ch{i % 8}_die{(i // 8) % 4}_pl0_blk{i // 32}_pg{i % 128}"
        for i in range(64)
    ]
    t_single = lm.calculate_batch_read_latency(single_locs)
    t_striped = lm.calculate_batch_read_latency(striped_locs)
    speedup = t_single / t_striped
    assert speedup >= 2.5, f"Expected speedup >= 2.5x at N=64, got {speedup:.2f}x"
    assert t_single == 1930.0
    assert t_striped == 250.0


def test_address_parsing_irregular_prefixes():
    """Verify parsing robustness across prefixes, casing, and variations."""
    cases = [
        ("ch0_die0_pl0_blk0_pg0", 0, (0, 0, 0, 0, 0)),
        ("ch7_die3_pl1_blk511_pg127", 7, (7, 3, 1, 511, 127)),
        ("ssd_ch3_die0_pl0_blk0_pg0", 3, (3, 0, 0, 0, 0)),
        ("nvme_CH5_die1_pl0_blk0_pg0", 5, (5, 1, 0, 0, 0)),
        ("pba_ch1_die2_pl1_blk5_pg10", 1, (1, 2, 1, 5, 10)),
        ("device0_ch4_die0_pl0_blk0_pg0", 4, (4, 0, 0, 0, 0)),
        ("ssd_host_pcie_ch2_die3_pl0_blk10_pg20", 2, (2, 3, 0, 10, 20)),
        ("CH0_DIE0_PL0_BLK0_PG0", 0, (0, 0, 0, 0, 0)),
        ("Ch6_Die1_Pl1_Blk2_Pg3", 6, (6, 1, 1, 2, 3)),
        ("ssd_ch03_die0_pl0_blk0_pg0", 3, (3, 0, 0, 0, 0)),
        ("ch07_die0_pl0_blk0_pg0", 7, (7, 0, 0, 0, 0)),
        ("ch3_die0_plane1_blk2_pg4", 3, (3, 0, 1, 2, 4)),
        ("prefix_CH4_die0_plane0_blk0_pg0", 4, (4, 0, 0, 0, 0)),
    ]
    for loc_str, exp_ch, exp_parsed in cases:
        assert LatencyModel.extract_channel(loc_str) == exp_ch
        assert parse_physical_location(loc_str) == exp_parsed

    # Malformed inputs
    for malformed in ["", None, 12345, "invalid_str", "ch_die_pl_blk_pg"]:
        assert LatencyModel.extract_channel(malformed) == 0
        assert parse_physical_location(malformed) is None


def test_channel_queue_fifo_serialization():
    """Verify ChannelTransferRequest queue timing and serialization on same die."""
    ch = FlashChannel(channel_id=0, dies_per_channel=4)
    ch.enqueue_request(ChannelTransferRequest(request_id=1, die_id=0, transfer_time_us=5.0))
    ch.enqueue_request(ChannelTransferRequest(request_id=2, die_id=0, transfer_time_us=5.0))
    ch.enqueue_request(ChannelTransferRequest(request_id=3, die_id=0, transfer_time_us=5.0))
    assert len(ch.queue) == 3

    ch.process_queue(base_time_us=0.0)
    assert len(ch.queue) == 0
    assert len(ch.completed_transfers) == 3

    t1, t2, t3 = ch.completed_transfers
    assert (t1.start_time_us, t1.completion_time_us) == (25.0, 30.0)
    assert (t2.start_time_us, t2.completion_time_us) == (50.0, 55.0)
    assert (t3.start_time_us, t3.completion_time_us) == (75.0, 80.0)
    assert ch.bus_busy_until_us == 80.0


def test_channel_multi_die_parallel_sensing():
    """Verify multi-die parallel sensing with channel bus serialization."""
    ch = FlashChannel(channel_id=1, dies_per_channel=4)
    ch.enqueue_request(ChannelTransferRequest(request_id=10, die_id=0, transfer_time_us=5.0))
    ch.enqueue_request(ChannelTransferRequest(request_id=11, die_id=1, transfer_time_us=5.0))
    ch.process_queue(base_time_us=0.0)

    p1, p2 = ch.completed_transfers
    # Both dies sense 0..25 in parallel; bus transfers serialize 25..30 and 30..35
    assert (p1.start_time_us, p1.completion_time_us) == (25.0, 30.0)
    assert (p2.start_time_us, p2.completion_time_us) == (30.0, 35.0)
    assert ch.bus_busy_until_us == 35.0


def test_storage_simulator_e2e_batches():
    """Verify StorageSimulator read_blocks across batch sizes and FTL modes."""
    for n in [2, 4, 8, 16, 32, 64]:
        sim_conv = StorageSimulator(mode="conventional")
        sim_ta = StorageSimulator(mode="tensor_aware")

        engine = MockKVEngine()
        blocks = engine.generate_kv_blocks(num_blocks=n)

        for b in blocks:
            sim_conv.store_block(b)
            sim_ta.store_block(b)

        lat_conv = sim_conv.read_blocks(blocks)
        lat_ta = sim_ta.read_blocks(blocks)
        speedup = lat_conv / lat_ta

        assert lat_conv > lat_ta, f"Expected conv > ta at N={n}"
        if n >= 64:
            assert speedup >= 2.5, f"Expected speedup >= 2.5x at N={n}, got {speedup:.2f}x"


def test_storage_simulator_physical_tracking():
    """Verify FlashPage physical state and read_count updates on read_blocks."""
    sim = StorageSimulator(mode="tensor_aware")
    blk = KVBlock.create_default(101, 0, 0)
    loc = sim.store_block(blk)
    ch, die, pl, b_idx, pg = parse_physical_location(loc)
    page = sim.channels[ch].dies[die].planes[pl].blocks[b_idx].pages[pg]

    assert page.state == PageState.VALID
    assert page.data_block_id == 101
    assert page.read_count == 0

    for _ in range(3):
        sim.read_blocks([blk])

    assert page.read_count == 3


def test_boundary_edge_cases():
    """Verify empty reads, non-existent blocks, and polymorphic input types."""
    sim = StorageSimulator()
    assert sim.read_blocks([]) == 0.0
    assert sim.read_blocks([999999, 888888]) == 0.0

    b1 = KVBlock.create_default(201, 0, 0)
    b2 = KVBlock.create_default(202, 0, 1)
    sim.store_block(b1)
    sim.store_block(b2)

    lat_obj = sim.read_blocks([b1, b2])
    lat_int = sim.read_blocks([201, 202])
    lat_mix = sim.read_blocks([b1, 202])
    assert lat_obj == lat_int == lat_mix
    assert lat_obj > 0.0


def main():
    print("Running M1.2 Contention & Address Parsing Stress Suite...")
    test_contention_inequality_n_2_to_64()
    print("  [PASS] test_contention_inequality_n_2_to_64")
    test_speedup_acceptance_at_64()
    print("  [PASS] test_speedup_acceptance_at_64")
    test_address_parsing_irregular_prefixes()
    print("  [PASS] test_address_parsing_irregular_prefixes")
    test_channel_queue_fifo_serialization()
    print("  [PASS] test_channel_queue_fifo_serialization")
    test_channel_multi_die_parallel_sensing()
    print("  [PASS] test_channel_multi_die_parallel_sensing")
    test_storage_simulator_e2e_batches()
    print("  [PASS] test_storage_simulator_e2e_batches")
    test_storage_simulator_physical_tracking()
    print("  [PASS] test_storage_simulator_physical_tracking")
    test_boundary_edge_cases()
    print("  [PASS] test_boundary_edge_cases")
    print("All 8 M1.2 stress tests PASSED cleanly!")


if __name__ == "__main__":
    main()
