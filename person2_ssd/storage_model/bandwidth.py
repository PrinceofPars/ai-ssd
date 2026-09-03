"""
Bandwidth model for PCIe Gen4/Gen5 and NAND channel busses.
"""

class BandwidthModel:
    def __init__(self, pcie_gbps: float = 7.0, channel_mbs: float = 1200.0):
        self.pcie_gbps = pcie_gbps
        self.channel_mbs = channel_mbs
