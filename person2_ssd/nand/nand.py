"""
NAND Flash Hierarchy Model: Die and Plane containers.
"""

from typing import List
from person2_ssd.nand.block import FlashBlock


class FlashPlane:
    def __init__(self, plane_id: int, blocks_per_plane: int = 64):
        self.plane_id = plane_id
        self.blocks = [FlashBlock(i) for i in range(blocks_per_plane)]


class FlashDie:
    def __init__(self, die_id: int, planes_per_die: int = 2, blocks_per_plane: int = 64):
        self.die_id = die_id
        self.planes = [FlashPlane(p, blocks_per_plane) for p in range(planes_per_die)]
        self.busy_until_us: float = 0.0

    def is_busy(self, current_time_us: float) -> bool:
        return current_time_us < self.busy_until_us
