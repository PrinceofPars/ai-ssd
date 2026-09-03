"""Physical NAND Flash and SSD Controller Hardware Model.

Models physical latencies, channel interleaving, bus transfer bandwidth,
and multi-tier energy consumption for modern enterprise PCIe NVMe SSDs.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class FlashTimingConfig:
    """NAND Flash and controller physical timing parameters."""
    t_read_us: float = 35.0             # 3D TLC NAND tR read latency in microseconds
    t_prog_us: float = 650.0           # 3D TLC NAND tPROG page program latency in microseconds
    t_erase_us: float = 3500.0         # 3D TLC NAND tBERS block erase latency in microseconds
    controller_dram_latency_ns: float = 50.0  # SSD controller LPDDR4/5 access latency
    pcie_bandwidth_gbps: float = 15.75  # PCIe Gen4 x4 bidirectional throughput in GB/sec
    internal_channels: int = 8         # Number of parallel NAND channels
    dies_per_channel: int = 4          # Number of dies per channel (32 dies total)
    page_size_bytes: int = 16384       # 16 KB physical flash page size
    pages_per_block: int = 128         # Physical pages per erase block (2 MB block)


@dataclass
class EnergyConfig:
    """Energy consumption parameters across system tiers in pJ / nJ."""
    host_dram_pj_per_bit: float = 4.5       # Host DDR5 read/write (pJ/bit) -> 0.036 nJ/byte
    pcie_pj_per_bit: float = 12.0           # PCIe Gen4 SerDes link energy (pJ/bit) -> 0.096 nJ/byte
    controller_dram_pj_per_bit: float = 3.5 # SSD Controller internal LPDDR (pJ/bit) -> 0.028 nJ/byte
    nand_read_nj_per_bit: float = 3.2       # Flash read sense energy (nJ/bit) -> 25.6 nJ/byte
    nand_write_nj_per_bit: float = 14.5     # Flash high-voltage charge pump program (nJ/bit) -> 116.0 nJ/byte
    compute_pj_per_mac: float = 0.5         # Controller embedded DSP/MAC operation energy (pJ/op)


class FlashModel:
    """Accurately calculates latency and energy for physical flash and controller operations."""

    def __init__(self, timing_config: FlashTimingConfig = None, energy_config: EnergyConfig = None):
        self.timing = timing_config or FlashTimingConfig()
        self.energy = energy_config or EnergyConfig()

    def calculate_pcie_transfer_time_us(self, num_bytes: int) -> float:
        """Calculates PCIe DMA transmission time in microseconds."""
        if num_bytes <= 0:
            return 0.0
        # bandwidth in GB/s = bytes / microsecond (1 GB/s = 10^9 B/s = 1000 B/us)
        bytes_per_us = self.timing.pcie_bandwidth_gbps * 1e3
        transfer_us = num_bytes / bytes_per_us
        # Base PCIe roundtrip overhead (descriptor fetch + interrupt / doorbell handshake)
        dma_overhead_us = 1.2
        return dma_overhead_us + transfer_us

    def calculate_flash_read_time_us(self, num_bytes: int) -> float:
        """Calculates physical NAND read time accounting for channel and die interleaving."""
        if num_bytes <= 0:
            return 0.0
        pages_needed = max(1, math.ceil(num_bytes / self.timing.page_size_bytes))
        # Parallelism factor across channels
        effective_channels = min(pages_needed, self.timing.internal_channels)
        rounds = math.ceil(pages_needed / effective_channels)
        # Each round costs t_read plus internal channel bus serialization
        channel_transfer_us = (self.timing.page_size_bytes / (1.2 * 1e3))  # 1.2 GB/s NV-DDR3 bus per channel
        total_read_us = rounds * (self.timing.t_read_us + channel_transfer_us)
        return total_read_us

    def calculate_flash_write_time_us(self, num_bytes: int) -> float:
        """Calculates physical NAND program time accounting for multi-channel striping."""
        if num_bytes <= 0:
            return 0.0
        pages_needed = max(1, math.ceil(num_bytes / self.timing.page_size_bytes))
        effective_channels = min(pages_needed, self.timing.internal_channels)
        rounds = math.ceil(pages_needed / effective_channels)
        return rounds * self.timing.t_prog_us

    def calculate_energy_joules(
        self,
        host_dram_bytes: int = 0,
        pcie_bytes: int = 0,
        controller_dram_bytes: int = 0,
        flash_read_bytes: int = 0,
        flash_write_bytes: int = 0,
        compute_macs: int = 0
    ) -> Dict[str, float]:
        """Calculates total energy consumption (in Joules) broken down by component.
        
        Conversion reference:
        1 pJ = 1e-12 Joules
        1 nJ = 1e-9 Joules
        """
        e_host_dram = (host_dram_bytes * 8) * (self.energy.host_dram_pj_per_bit * 1e-12)
        e_pcie = (pcie_bytes * 8) * (self.energy.pcie_pj_per_bit * 1e-12)
        e_ctrl_dram = (controller_dram_bytes * 8) * (self.energy.controller_dram_pj_per_bit * 1e-12)
        e_flash_read = (flash_read_bytes * 8) * (self.energy.nand_read_nj_per_bit * 1e-9)
        e_flash_write = (flash_write_bytes * 8) * (self.energy.nand_write_nj_per_bit * 1e-9)
        e_compute = compute_macs * (self.energy.compute_pj_per_mac * 1e-12)

        total_joules = (
            e_host_dram +
            e_pcie +
            e_ctrl_dram +
            e_flash_read +
            e_flash_write +
            e_compute
        )

        return {
            "total_joules": total_joules,
            "host_dram_joules": e_host_dram,
            "pcie_joules": e_pcie,
            "controller_dram_joules": e_ctrl_dram,
            "flash_read_joules": e_flash_read,
            "flash_write_joules": e_flash_write,
            "compute_joules": e_compute,
        }
