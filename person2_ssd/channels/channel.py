"""
Flash Channel Model.
Models channel bus serialization and contention across multiple dies.
"""

from dataclasses import dataclass
from typing import List, Union, Iterator
from person2_ssd.nand.nand import FlashDie
from common.constants import (
    SSD_DIES_PER_CHANNEL,
    SSD_PLANES_PER_DIE,
    BUS_TRANSFER_US_PER_PAGE,
    T_R_US,
    T_PROG_US,
    T_BERS_US,
)


@dataclass
class ChannelTransferRequest:
    request_id: Union[str, int]
    die_id: int
    plane_id: int = 0
    block_id: int = 0
    page_id: int = 0
    op_type: str = "READ"  # "READ", "PROGRAM", "ERASE"
    arrival_time_us: float = 0.0
    transfer_time_us: float = BUS_TRANSFER_US_PER_PAGE
    start_time_us: float = 0.0
    completion_time_us: float = 0.0


class FlashChannel:
    def __init__(
        self,
        channel_id: int,
        dies_per_channel: int = SSD_DIES_PER_CHANNEL,
        planes_per_die: int = SSD_PLANES_PER_DIE,
        blocks_per_plane: int = 64,
    ):
        self.channel_id = channel_id
        self.dies: List[FlashDie] = [
            FlashDie(d, planes_per_die, blocks_per_plane) for d in range(dies_per_channel)
        ]
        self.bus_busy_until_us: float = 0.0
        self.queue: List[ChannelTransferRequest] = []
        self.completed_transfers: List[ChannelTransferRequest] = []

    def schedule_transfer(
        self, current_time_us: float, transfer_time_us: float = BUS_TRANSFER_US_PER_PAGE
    ) -> float:
        """
        Schedules a data transfer over the channel bus. Returns completion time in microseconds.
        """
        start_time = max(current_time_us, self.bus_busy_until_us)
        completion_time = start_time + transfer_time_us
        self.bus_busy_until_us = completion_time
        return completion_time

    def enqueue_request(self, req: ChannelTransferRequest) -> None:
        """Enqueues a transfer request into the channel queue."""
        self.queue.append(req)

    def process_queue(self, base_time_us: float = 0.0) -> float:
        """
        Processes all queued transfers in FIFO order.
        Simulates die sensing and channel bus serialization.
        """
        for req in self.queue:
            die = self.dies[req.die_id] if 0 <= req.die_id < len(self.dies) else None
            if req.op_type == "READ":
                sensing_finish = (
                    die.schedule_read(req.arrival_time_us + base_time_us)
                    if die
                    else (req.arrival_time_us + base_time_us + T_R_US)
                )
                bus_start = max(sensing_finish, self.bus_busy_until_us)
                bus_finish = bus_start + req.transfer_time_us
            elif req.op_type == "PROGRAM":
                bus_start = max(req.arrival_time_us + base_time_us, self.bus_busy_until_us)
                bus_finish = bus_start + req.transfer_time_us
                if die:
                    die.schedule_program(bus_finish)
            elif req.op_type == "ERASE":
                if die:
                    die.schedule_erase(req.arrival_time_us + base_time_us)
                bus_start = max(req.arrival_time_us + base_time_us, self.bus_busy_until_us)
                bus_finish = bus_start
            else:
                bus_start = max(req.arrival_time_us + base_time_us, self.bus_busy_until_us)
                bus_finish = bus_start + req.transfer_time_us

            self.bus_busy_until_us = max(self.bus_busy_until_us, bus_finish)
            req.start_time_us = bus_start
            req.completion_time_us = bus_finish
            self.completed_transfers.append(req)

        self.queue.clear()
        return self.bus_busy_until_us

    def reset(self) -> None:
        """Resets channel bus and die state."""
        self.bus_busy_until_us = 0.0
        self.queue.clear()
        self.completed_transfers.clear()
        for d in self.dies:
            d.reset()

    def __getitem__(self, idx: int) -> FlashDie:
        return self.dies[idx]

    def __len__(self) -> int:
        return len(self.dies)

    def __iter__(self) -> Iterator[FlashDie]:
        return iter(self.dies)

    def __repr__(self) -> str:
        return (
            f"<FlashChannel id={self.channel_id} dies={len(self.dies)} "
            f"bus_busy_until={self.bus_busy_until_us:.1f}us queue_len={len(self.queue)}>"
        )

