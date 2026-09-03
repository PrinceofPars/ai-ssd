"""
Flash Channel Model.
Models channel bus serialization and contention across multiple dies.
"""

from typing import List
from person2_ssd.nand.nand import FlashDie
from common.constants import BUS_TRANSFER_US_PER_PAGE


class FlashChannel:
    def __init__(self, channel_id: int, dies_per_channel: int = 4):
        self.channel_id = channel_id
        self.dies: List[FlashDie] = [FlashDie(d) for d in range(dies_per_channel)]
        self.bus_busy_until_us: float = 0.0

    def schedule_transfer(self, current_time_us: float, transfer_time_us: float = BUS_TRANSFER_US_PER_PAGE) -> float:
        """
        Schedules a data transfer over the channel bus. Returns completion time in microseconds.
        """
        start_time = max(current_time_us, self.bus_busy_until_us)
        completion_time = start_time + transfer_time_us
        self.bus_busy_until_us = completion_time
        return completion_time
