"""
NAND Flash Hierarchy Model: Die and Plane containers.
"""

from typing import List, Iterator
from person2_ssd.nand.block import FlashBlock
from common.constants import (
    SSD_PLANES_PER_DIE,
    T_R_US,
    T_PROG_US,
    T_BERS_US,
)


class FlashPlane:
    def __init__(self, plane_id: int, blocks_per_plane: int = 64):
        self.plane_id = plane_id
        self.blocks: List[FlashBlock] = [FlashBlock(i) for i in range(blocks_per_plane)]

    @property
    def total_valid_pages(self) -> int:
        return sum(b.valid_page_count for b in self.blocks)

    @property
    def total_invalid_pages(self) -> int:
        return sum(b.invalid_page_count for b in self.blocks)

    @property
    def total_free_pages(self) -> int:
        return sum(b.free_page_count for b in self.blocks)

    def get_block(self, block_id: int) -> FlashBlock:
        return self.blocks[block_id]

    def __getitem__(self, idx: int) -> FlashBlock:
        return self.blocks[idx]

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self) -> Iterator[FlashBlock]:
        return iter(self.blocks)

    def __repr__(self) -> str:
        return f"<FlashPlane id={self.plane_id} blocks={len(self.blocks)} valid_pages={self.total_valid_pages}>"


class FlashDie:
    def __init__(
        self,
        die_id: int,
        planes_per_die: int = SSD_PLANES_PER_DIE,
        blocks_per_plane: int = 64,
    ):
        self.die_id = die_id
        self.planes: List[FlashPlane] = [
            FlashPlane(p, blocks_per_plane) for p in range(planes_per_die)
        ]
        self.busy_until_us: float = 0.0

    def is_busy(self, current_time_us: float) -> bool:
        return current_time_us < self.busy_until_us

    def schedule_read(self, start_time_us: float = 0.0) -> float:
        """Schedules page read sensing (t_R). Returns completion time in us."""
        start = max(start_time_us, self.busy_until_us)
        self.busy_until_us = start + T_R_US
        return self.busy_until_us

    def schedule_program(self, start_time_us: float = 0.0) -> float:
        """Schedules page program (t_PROG). Returns completion time in us."""
        start = max(start_time_us, self.busy_until_us)
        self.busy_until_us = start + T_PROG_US
        return self.busy_until_us

    def schedule_erase(self, start_time_us: float = 0.0) -> float:
        """Schedules block erase (t_BERS). Returns completion time in us."""
        start = max(start_time_us, self.busy_until_us)
        self.busy_until_us = start + T_BERS_US
        return self.busy_until_us

    def reset(self) -> None:
        self.busy_until_us = 0.0

    def __getitem__(self, idx: int) -> FlashPlane:
        return self.planes[idx]

    def __len__(self) -> int:
        return len(self.planes)

    def __iter__(self) -> Iterator[FlashPlane]:
        return iter(self.planes)

    def __repr__(self) -> str:
        return f"<FlashDie id={self.die_id} planes={len(self.planes)} busy_until={self.busy_until_us:.1f}us>"

