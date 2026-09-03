"""
Analytical Latency Model:
Models Host/NVMe request -> PCIe bus transfer -> FTL lookup -> Channel contention
-> Die/Plane parallelism -> NAND operation (tR, tPROG, tBERS) -> SSD DRAM -> Host/GPU.
"""

import re
from typing import List, Dict, Any, Optional
from common.constants import (
    T_R_US,
    T_PROG_US,
    T_BERS_US,
    BUS_TRANSFER_US_PER_PAGE,
    PCIE_OVERHEAD_US,
    SSD_CHANNELS,
)

CHANNEL_REGEX = re.compile(r"(?:^|_)ch(?P<ch>\d+)", re.IGNORECASE)


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

    @staticmethod
    def extract_channel(loc: str) -> int:
        """
        Extracts channel index from physical location string using regex.
        Robust against prefixes (e.g. 'ssd_ch0...', 'pba_ch1...') and casing ('CH2...').
        Defaults to 0 if no match is found.
        """
        if not loc or not isinstance(loc, str):
            return 0
        m = CHANNEL_REGEX.search(loc)
        return int(m.group("ch")) if m else 0

    def _get_channel_loads(self, physical_locations: List[str]) -> Dict[int, int]:
        """Groups physical locations by channel index and counts load per channel."""
        channel_loads: Dict[int, int] = {}
        for loc in physical_locations:
            ch = self.extract_channel(loc)
            channel_loads[ch] = channel_loads.get(ch, 0) + 1
        return channel_loads

    def calculate_batch_read_latency(self, physical_locations: List[str]) -> float:
        """
        Calculates the total parallel read latency for a batch of physical locations.
        Formula: T = t_pcie + max_c (N_c * (t_R + t_bus))
        Takes into account channel contention: if multiple requests hit the same channel,
        their bus transfers and NAND operations serialize.
        """
        if not physical_locations:
            return 0.0

        channel_loads = self._get_channel_loads(physical_locations)
        max_channel_load = max(channel_loads.values()) if channel_loads else 0

        # Parallel read time: PCIe overhead + (max_channel_load * (tR + bus_transfer))
        channel_time = max_channel_load * (self.t_r_us + self.bus_transfer_us)
        return self.pcie_overhead_us + channel_time

    def calculate_batch_write_latency(self, physical_locations: List[str]) -> float:
        """
        Calculates the total parallel write latency for a batch of physical locations.
        Formula: T = t_pcie + max_c (N_c * (t_bus + t_PROG))
        """
        if not physical_locations:
            return 0.0

        channel_loads = self._get_channel_loads(physical_locations)
        max_channel_load = max(channel_loads.values()) if channel_loads else 0

        channel_time = max_channel_load * (self.bus_transfer_us + self.t_prog_us)
        return self.pcie_overhead_us + channel_time

    def calculate_batch_erase_latency(self, physical_locations: List[str]) -> float:
        """
        Calculates the total parallel erase latency for a batch of physical block locations.
        Formula: T = t_pcie + max_c (N_c * t_BERS)
        """
        if not physical_locations:
            return 0.0

        channel_loads = self._get_channel_loads(physical_locations)
        max_channel_load = max(channel_loads.values()) if channel_loads else 0

        channel_time = max_channel_load * self.t_bers_us
        return self.pcie_overhead_us + channel_time

    def get_latency_breakdown(self, physical_locations: List[str]) -> Dict[str, Any]:
        """
        Provides detailed diagnostic breakdown of channel loads and bottlenecks.
        """
        channel_loads = self._get_channel_loads(physical_locations)
        max_load = max(channel_loads.values()) if channel_loads else 0
        bottleneck_channel = max(channel_loads, key=channel_loads.get) if channel_loads else None

        read_lat = self.calculate_batch_read_latency(physical_locations)
        write_lat = self.calculate_batch_write_latency(physical_locations)
        erase_lat = self.calculate_batch_erase_latency(physical_locations)

        return {
            "total_requests": len(physical_locations),
            "channel_loads": channel_loads,
            "bottleneck_channel": bottleneck_channel,
            "max_channel_load": max_load,
            "pcie_overhead_us": self.pcie_overhead_us,
            "read_channel_time_us": max_load * (self.t_r_us + self.bus_transfer_us),
            "total_read_latency_us": read_lat,
            "write_channel_time_us": max_load * (self.bus_transfer_us + self.t_prog_us),
            "total_write_latency_us": write_lat,
            "total_erase_latency_us": erase_lat,
        }

