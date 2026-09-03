"""
Analytical Latency Model:
Models Host/NVMe request -> PCIe bus transfer -> FTL lookup -> Channel contention
-> Die/Plane parallelism -> NAND operation (tR, tPROG, tBERS) -> SSD DRAM -> Host/GPU.
"""

from typing import List, Dict
from common.constants import (
    T_R_US,
    T_PROG_US,
    T_BERS_US,
    BUS_TRANSFER_US_PER_PAGE,
    PCIE_OVERHEAD_US,
    SSD_CHANNELS,
)


class LatencyModel:
    def __init__(
        self,
        t_r_us: float = T_R_US,
        t_prog_us: float = T_PROG_US,
        t_bers_us: float = T_BERS_US,
        bus_transfer_us: float = BUS_TRANSFER_US_PER_PAGE,
        pcie_overhead_us: float = PCIE_OVERHEAD_US,
        channels: int = SSD_CHANNELS,
    ):
        self.t_r_us = t_r_us
        self.t_prog_us = t_prog_us
        self.t_bers_us = t_bers_us
        self.bus_transfer_us = bus_transfer_us
        self.pcie_overhead_us = pcie_overhead_us
        self.channels = channels

    def calculate_batch_read_latency(self, physical_locations: List[str]) -> float:
        """
        Calculates the total parallel read latency for a batch of physical locations.
        Takes into account channel contention: if multiple requests hit the same channel,
        their bus transfers and NAND operations serialize.
        """
        if not physical_locations:
            return 0.0

        # Group by channel: "ch<X>_die<Y>..."
        channel_loads: Dict[int, int] = {}
        for loc in physical_locations:
            ch = 0
            if "ch" in loc:
                try:
                    ch_part = loc.split("_")[0]
                    ch = int(ch_part.replace("ch", ""))
                except Exception:
                    ch = 0
            channel_loads[ch] = channel_loads.get(ch, 0) + 1

        # Max load on any single channel dictates the bottleneck
        max_channel_load = max(channel_loads.values())

        # Parallel read time: PCIe overhead + (max_channel_load * (tR + bus_transfer))
        # When striping works, max_channel_load is minimized (total_requests / channels)
        channel_time = max_channel_load * (self.t_r_us + self.bus_transfer_us)
        return self.pcie_overhead_us + channel_time
