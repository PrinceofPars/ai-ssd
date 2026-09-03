"""
Challenger M1.1 Comprehensive Empirical Stress Test Suite.
Rigorous verification of NAND physical hierarchy, state machines, block allocation,
double programming rejection, endurance limits, dunder compatibility, and contention timing.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from person2_ssd.nand.page import FlashPage, PageState
from person2_ssd.nand.block import FlashBlock
from person2_ssd.nand.nand import FlashPlane, FlashDie
from person2_ssd.channels.channel import FlashChannel, ChannelTransferRequest
from person2_ssd.storage_model.latency import LatencyModel
from person2_ssd.storage_model.io_model import StorageSimulator, parse_physical_location
from common.schemas.kv_block import KVBlock
from common.constants import (
    SSD_PAGE_SIZE_BYTES,
    SSD_PAGES_PER_BLOCK,
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    T_R_US,
    T_PROG_US,
    T_BERS_US,
    BUS_TRANSFER_US_PER_PAGE,
    PCIE_OVERHEAD_US,
)


def run_suite_1_state_transitions():
    print("\n--- Suite 1: Physical State Transitions (FREE -> VALID -> INVALID -> FREE) ---")
    page = FlashPage(page_id=42, size_bytes=4096)
    
    # Initial state
    assert page.state == PageState.FREE, f"Expected FREE, got {page.state}"
    assert page.data_block_id is None
    assert page.program_count == 0
    assert page.read_count == 0
    assert page.last_accessed_us == 0.0
    print("  [PASS] Page initialized to PageState.FREE")

    # Read unprogrammed page returns None, tracks read_count
    val = page.read(current_time_us=5.0)
    assert val is None, f"Expected None on reading FREE page, got {val}"
    assert page.read_count == 1
    assert page.last_accessed_us == 5.0
    print("  [PASS] Reading FREE page returns None and increments read_count")

    # Transition FREE -> VALID via program
    page.program(data_block_id=1001)
    assert page.state == PageState.VALID
    assert page.data_block_id == 1001
    assert page.program_count == 1
    print("  [PASS] Program transitioned FREE -> VALID with data_block_id")

    # Read VALID page returns data_block_id
    val = page.read(current_time_us=12.5)
    assert val == 1001
    assert page.read_count == 2
    assert page.last_accessed_us == 12.5
    print("  [PASS] Reading VALID page returns stored data_block_id")

    # Programming with data_block_id=0 (falsy zero boundary test)
    zero_page = FlashPage(page_id=0)
    zero_page.program(data_block_id=0)
    assert zero_page.state == PageState.VALID
    assert zero_page.data_block_id == 0
    assert zero_page.read() == 0
    print("  [PASS] Program with data_block_id=0 correctly stores and returns 0")

    # Program with block_id keyword compatibility
    alias_page = FlashPage(page_id=1)
    alias_page.program(block_id=2002)
    assert alias_page.data_block_id == 2002
    print("  [PASS] Program with block_id alias works correctly")

    # Transition VALID -> INVALID via invalidate
    page.invalidate()
    assert page.state == PageState.INVALID
    assert page.data_block_id is None
    print("  [PASS] Invalidate transitioned VALID -> INVALID and cleared data_block_id")

    # Read INVALID page returns None, tracks read_count
    val = page.read(current_time_us=20.0)
    assert val is None
    assert page.read_count == 3
    print("  [PASS] Reading INVALID page returns None and increments read_count")

    # Transition INVALID -> FREE via erase
    page.erase()
    assert page.state == PageState.FREE
    assert page.data_block_id is None
    assert page.read_count == 0  # Erase resets read disturb counter
    assert page.program_count == 1  # Lifetime program count preserved
    print("  [PASS] Erase transitioned INVALID -> FREE, reset read_count, preserved program_count")


def run_suite_2_double_programming():
    print("\n--- Suite 2: Double Programming & Illegal State Rejection ---")
    
    # 2.1 Rejection on VALID page
    p1 = FlashPage(page_id=10)
    p1.program(100)
    try:
        p1.program(200)
        assert False, "Double programming on VALID page should have raised ValueError"
    except ValueError as e:
        assert "Cannot program non-free page" in str(e)
        print(f"  [PASS] Rejected double program on VALID page: {e}")

    # 2.2 Rejection on INVALID page without erase
    p2 = FlashPage(page_id=20)
    p2.program(100)
    p2.invalidate()
    try:
        p2.program(300)
        assert False, "Programming on INVALID page without erase should have raised ValueError"
    except ValueError as e:
        assert "Cannot program non-free page" in str(e)
        print(f"  [PASS] Rejected program on INVALID page without prior erase: {e}")

    # 2.3 Rejection via FlashBlock allocate_page
    block = FlashBlock(block_id=0, pages_count=4)
    allocated_p0 = block.allocate_page(data_block_id=500)
    assert allocated_p0.state == PageState.VALID
    try:
        allocated_p0.program(600)
        assert False, "Directly re-programming allocated block page should raise ValueError"
    except ValueError as e:
        assert "Cannot program non-free page" in str(e)
        print(f"  [PASS] Re-programming allocated block page rejected: {e}")


def run_suite_3_block_allocation_exhaustion():
    print("\n--- Suite 3: Block Sequential Allocation & Exhaustion ---")
    pages_count = 8
    block = FlashBlock(block_id=1, pages_count=pages_count)
    assert block.is_empty
    assert not block.is_full
    assert block.free_page_count == pages_count
    assert block.valid_page_count == 0
    assert block.invalid_page_count == 0

    allocated_pages = []
    for i in range(pages_count):
        p = block.allocate_page(data_block_id=i * 10)
        assert p is not None, f"Allocation at index {i} returned None"
        assert p.page_id == i, f"Expected page_id {i}, got {p.page_id}"
        assert p.state == PageState.VALID
        assert p.data_block_id == i * 10
        allocated_pages.append(p)
        assert block.free_page_index == i + 1
        assert block.valid_page_count == i + 1
        assert block.free_page_count == pages_count - (i + 1)

    print(f"  [PASS] Successfully allocated all {pages_count} pages sequentially")
    assert block.is_full
    assert not block.is_empty
    assert block.free_page_count == 0
    assert block.valid_page_count == pages_count

    # Allocation beyond exhaustion must return None cleanly
    p_overflow = block.allocate_page(data_block_id=999)
    assert p_overflow is None, f"Expected None on exhausted block, got {p_overflow}"
    p_overflow_2 = block.allocate_page(data_block_id=1000)
    assert p_overflow_2 is None, "Repeated allocation on exhausted block must return None"
    assert block.free_page_index == pages_count, "free_page_index should not exceed pages_count"
    print("  [PASS] Allocation on exhausted block returns None without state corruption")

    # Invalidate some pages and check counts
    block.pages[0].invalidate()
    block.pages[2].invalidate()
    block.pages[4].invalidate()
    assert block.valid_page_count == pages_count - 3
    assert block.invalid_page_count == 3
    assert block.free_page_count == 0
    assert block.garbage_ratio == 3.0 / pages_count
    print(f"  [PASS] Invalidation correctly updated counts: valid={block.valid_page_count}, invalid={block.invalid_page_count}, garbage={block.garbage_ratio:.3f}")

    # Erase block and verify full reset
    block.erase()
    assert block.is_empty
    assert not block.is_full
    assert block.free_page_index == 0
    assert block.erase_count == 1
    assert block.free_page_count == pages_count
    assert block.valid_page_count == 0
    assert block.invalid_page_count == 0
    assert block.garbage_ratio == 0.0
    print("  [PASS] Erase reset block state, free_page_index=0, erase_count=1, garbage_ratio=0.0")

    # Verify re-allocation works after erase
    p_new = block.allocate_page(data_block_id=777)
    assert p_new is not None
    assert p_new.page_id == 0
    assert p_new.state == PageState.VALID
    print("  [PASS] Re-allocation after erase succeeds seamlessly")


def run_suite_4_endurance_and_wear():
    print("\n--- Suite 4: Endurance Limits & Wear Limits (max_erase_cycles) ---")
    limit = 100
    block = FlashBlock(block_id=5, pages_count=4, max_erase_cycles=limit)
    assert not block.is_bad_block
    assert block.erase_count == 0

    # Erase up to limit - 1
    for c in range(1, limit):
        block.erase()
        assert block.erase_count == c
        assert not block.is_bad_block, f"Block prematurely marked bad at cycle {c}"

    print(f"  [PASS] Block remained healthy for {limit - 1} erase cycles")

    # Hit endurance limit exactly at limit
    block.erase()
    assert block.erase_count == limit
    assert block.is_bad_block, f"Block should be marked bad at cycle {limit}"
    print(f"  [PASS] Block correctly flagged is_bad_block=True at cycle {limit}")

    # Beyond endurance limit
    block.erase()
    assert block.erase_count == limit + 1
    assert block.is_bad_block
    print(f"  [PASS] Block remains is_bad_block=True beyond limit ({limit + 1} cycles)")

    # Default 3000 cycle verification
    default_block = FlashBlock(block_id=99, pages_count=2)
    assert default_block.max_erase_cycles == 3000
    default_block.erase_count = 2999
    assert not default_block.is_bad_block
    default_block.erase()
    assert default_block.erase_count == 3000
    assert default_block.is_bad_block
    print("  [PASS] Default 3000 max_erase_cycles invariant verified")


def run_suite_5_garbage_ratio():
    print("\n--- Suite 5: Garbage Ratio & GC Metric Accuracy ---")
    N = 16
    block = FlashBlock(block_id=10, pages_count=N)
    
    # 0% garbage initially
    assert block.garbage_ratio == 0.0

    # Allocate all pages
    for i in range(N):
        block.allocate_page(i)
    assert block.garbage_ratio == 0.0  # 100% valid is 0% garbage

    # Invalidate step-by-step
    expected_ratios = [
        (4, 4.0 / N),    # 25%
        (8, 8.0 / N),    # 50%
        (12, 12.0 / N),  # 75%
        (16, 1.0),       # 100%
    ]
    for count, expected_ratio in expected_ratios:
        for j in range(count):
            if block.pages[j].state == PageState.VALID:
                block.pages[j].invalidate()
        assert abs(block.garbage_ratio - expected_ratio) < 1e-6, (
            f"Expected garbage_ratio {expected_ratio}, got {block.garbage_ratio}"
        )
        print(f"  [PASS] Garbage ratio {block.invalid_page_count}/{N} = {block.garbage_ratio:.4f}")

    # Boundary: empty block with 0 pages
    empty_block = FlashBlock(block_id=11, pages_count=0)
    assert empty_block.garbage_ratio == 0.0
    print("  [PASS] 0-page block garbage_ratio safely evaluates to 0.0 without ZeroDivisionError")


def run_suite_6_dunder_and_int_compatibility():
    print("\n--- Suite 6: FlashPage Int-Compatibility & Dunder Methods ---")
    block = FlashBlock(block_id=7, pages_count=8)
    page = block.allocate_page(data_block_id=555)

    # 1. Type identity
    assert isinstance(page, FlashPage), "Must be an instance of FlashPage"
    print("  [PASS] isinstance(page, FlashPage) is True")

    # 2. Equality with integer (worker backward-compatibility claim)
    assert page == 0, f"page == 0 failed: page={page}"
    assert 0 == page, f"0 == page failed: page={page}"
    assert page != 1
    assert 1 != page
    print("  [PASS] Bidirectional integer equality: (page == 0) and (0 == page)")

    # 3. int() conversion
    assert int(page) == 0, f"int(page) returned {int(page)}"
    print("  [PASS] int(page) == 0")

    # 4. __index__ support for slicing, indexing, and range
    sample_list = ["apple", "banana", "cherry", "date"]
    assert sample_list[page] == "apple", f"Indexing list with FlashPage failed: {sample_list[page]}"
    assert list(range(page)) == []
    print("  [PASS] FlashPage supports __index__ for container indexing and range()")

    # 5. Hashability and set/dict membership
    page_dict = {page: "first_page"}
    assert page_dict[page] == "first_page"
    assert hash(page) == hash(0)
    print("  [PASS] FlashPage is hashable and works as dict key")

    # 6. Container membership with mixed ints and pages
    assert page in [0, 1, 2], "page in [0, 1, 2] should be True"
    assert 0 in [page], "0 in [page] should be True"
    print("  [PASS] Mixed membership: (page in [0, 1, 2]) and (0 in [page])")

    # 7. FlashBlock container dunders
    assert len(block) == 8
    assert block[0] == page
    assert [p.page_id for p in block] == list(range(8))
    print("  [PASS] FlashBlock supports __len__, __getitem__, and __iter__")


def run_suite_7_hierarchy_and_timing_physics():
    print("\n--- Suite 7: Physical Hierarchy & Contention Timing Invariants ---")
    
    # 7.1 Plane and Die aggregation
    plane = FlashPlane(plane_id=0, blocks_per_plane=4)
    assert len(plane) == 4
    assert plane.total_free_pages == 4 * SSD_PAGES_PER_BLOCK
    assert plane.total_valid_pages == 0
    assert plane.total_invalid_pages == 0
    
    # Allocate in plane blocks
    p0 = plane[0].allocate_page(1)
    p1 = plane[1].allocate_page(2)
    assert plane.total_valid_pages == 2
    assert plane.total_free_pages == 4 * SSD_PAGES_PER_BLOCK - 2
    p0.invalidate()
    assert plane.total_valid_pages == 1
    assert plane.total_invalid_pages == 1
    print("  [PASS] FlashPlane accurately aggregates valid, invalid, and free pages across blocks")

    # 7.2 FlashDie operation scheduling
    die = FlashDie(die_id=0, planes_per_die=2, blocks_per_plane=4)
    assert not die.is_busy(0.0)
    t1 = die.schedule_read(start_time_us=10.0)
    assert t1 == 10.0 + T_R_US  # 10.0 + 25.0 = 35.0
    assert die.is_busy(30.0)
    assert not die.is_busy(35.0)

    # Sequential operation without idle gap
    t2 = die.schedule_program(start_time_us=30.0)
    assert t2 == 35.0 + T_PROG_US  # 35.0 + 200.0 = 235.0

    t3 = die.schedule_erase(start_time_us=200.0)
    assert t3 == 235.0 + T_BERS_US  # 235.0 + 2000.0 = 2235.0
    die.reset()
    assert die.busy_until_us == 0.0
    print(f"  [PASS] FlashDie scheduling accurately enforces t_R ({T_R_US}us), t_PROG ({T_PROG_US}us), t_BERS ({T_BERS_US}us)")

    # 7.3 Channel contention queue
    channel = FlashChannel(channel_id=0, dies_per_channel=2, planes_per_die=1, blocks_per_plane=2)
    for req_i in range(4):
        channel.enqueue_request(ChannelTransferRequest(
            request_id=f"req_{req_i}",
            die_id=0,
            op_type="READ",
            arrival_time_us=0.0,
            transfer_time_us=BUS_TRANSFER_US_PER_PAGE,
        ))
    total_time = channel.process_queue(base_time_us=0.0)
    assert len(channel.completed_transfers) == 4
    assert len(channel.queue) == 0
    print(f"  [PASS] FlashChannel queue processed 4 serialized transfers, total_bus_busy={total_time}us")

    # 7.4 LatencyModel Contention Inequality
    lm = LatencyModel()
    for N in range(2, 9):
        single_locs = [f"ch0_die0_pl0_blk0_pg{i}" for i in range(N)]
        striped_locs = [f"ch{i}_die0_pl0_blk0_pg0" for i in range(N)]
        t_single = lm.calculate_batch_read_latency(single_locs)
        t_striped = lm.calculate_batch_read_latency(striped_locs)
        expected_single = PCIE_OVERHEAD_US + N * (T_R_US + BUS_TRANSFER_US_PER_PAGE)
        expected_striped = PCIE_OVERHEAD_US + 1 * (T_R_US + BUS_TRANSFER_US_PER_PAGE)
        assert abs(t_single - expected_single) < 1e-6
        assert abs(t_striped - expected_striped) < 1e-6
        assert t_single > t_striped, f"Contention inequality violated for N={N}: {t_single} <= {t_striped}"
        speedup = t_single / t_striped
        print(f"  [PASS] Contention verified for N={N}: single={t_single:.1f}us > striped={t_striped:.1f}us (speedup={speedup:.2f}x)")

    # 7.5 Address Regex Parsing Robustness
    test_cases = [
        ("ch3_die1_pl0_blk2_pg15", 3),
        ("ssd_ch5_die0_pl0_blk0_pg0", 5),
        ("PBA_CH7_DIE2_PL1_BLK10_PG60", 7),
        ("prefix_ch0_suffix", 0),
        ("invalid_no_channel", 0),
        ("", 0),
        (None, 0),
    ]
    for loc_str, expected_ch in test_cases:
        actual_ch = LatencyModel.extract_channel(loc_str)
        assert actual_ch == expected_ch, f"extract_channel('{loc_str}') expected {expected_ch}, got {actual_ch}"
    print("  [PASS] LatencyModel.extract_channel handles prefixes, casing, and malformed inputs")


def run_suite_8_high_volume_stress():
    print("\n--- Suite 8: High-Volume Stress & Invariant Fuzzing ---")
    num_cycles = 500
    block = FlashBlock(block_id=999, pages_count=16)
    
    t0 = time.perf_counter()
    for cycle in range(num_cycles):
        # Invariant 1: Fresh or erased block must have all free pages
        assert block.free_page_count == 16
        assert block.valid_page_count == 0
        assert block.invalid_page_count == 0
        assert block.garbage_ratio == 0.0

        # Fill all pages
        for p in range(16):
            allocated = block.allocate_page(data_block_id=cycle * 100 + p)
            assert allocated == p
            assert allocated.state == PageState.VALID

        # Invariant 2: Full block must have free_page_count == 0
        assert block.is_full
        assert block.free_page_count == 0
        assert block.valid_page_count == 16

        # Invalidate half of the pages
        for p in range(0, 16, 2):
            block.pages[p].invalidate()

        # Invariant 3: valid + invalid + free == total pages
        assert block.valid_page_count + block.invalid_page_count + block.free_page_count == 16
        assert block.invalid_page_count == 8
        assert block.garbage_ratio == 0.5

        # Erase
        block.erase()
        assert block.erase_count == cycle + 1

    dt = time.perf_counter() - t0
    print(f"  [PASS] Completed {num_cycles} full block allocation/invalidation/erase cycles in {dt:.3f}s")
    print(f"  [PASS] Average cycle latency: {dt / num_cycles * 1e6:.2f} us/cycle")
    print(f"  [PASS] Final erase count: {block.erase_count}")


def main():
    print("================================================================")
    print("  CHALLENGER M1.1: EMPIRICAL NAND INVARIANT & STRESS HARNESS   ")
    print("================================================================")
    
    start_time = time.perf_counter()
    try:
        run_suite_1_state_transitions()
        run_suite_2_double_programming()
        run_suite_3_block_allocation_exhaustion()
        run_suite_4_endurance_and_wear()
        run_suite_5_garbage_ratio()
        run_suite_6_dunder_and_int_compatibility()
        run_suite_7_hierarchy_and_timing_physics()
        run_suite_8_high_volume_stress()
        
        elapsed = time.perf_counter() - start_time
        print("\n================================================================")
        print(f"  ALL STRESS TESTS PASSED CLEANLY in {elapsed:.3f} seconds!     ")
        print("================================================================")
        return 0
    except Exception as e:
        print(f"\n[FATAL STRESS FAILURE]: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
