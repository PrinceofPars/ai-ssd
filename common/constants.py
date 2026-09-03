"""
System-wide constants and physical parameters for AI-SSD.
"""

# Default KV Block Geometry
DEFAULT_BLOCK_TOKENS = 16
DEFAULT_HEAD_DIM = 128
DEFAULT_KV_HEADS_PER_BLOCK = 1
DEFAULT_DTYPE = "FP16"

# Calculated Sizes (bytes)
BYTES_PER_FP16 = 2
BYTES_PER_FP8 = 1

# Default 16 tokens * 1 head * 128 dim * 2 bytes = 4096 bytes total (2048 bytes Key + 2048 bytes Value)
DEFAULT_KEY_SIZE_BYTES = (DEFAULT_BLOCK_TOKENS * DEFAULT_KV_HEADS_PER_BLOCK * DEFAULT_HEAD_DIM * BYTES_PER_FP16) // 2
DEFAULT_VALUE_SIZE_BYTES = DEFAULT_KEY_SIZE_BYTES
DEFAULT_BLOCK_SIZE_BYTES = DEFAULT_KEY_SIZE_BYTES + DEFAULT_VALUE_SIZE_BYTES  # 4096 bytes (4 KB)

# Standard NAND Physical Constants
SSD_PAGE_SIZE_BYTES = 4096      # 4 KB physical flash page
SSD_PAGES_PER_BLOCK = 128       # 128 pages per flash erase block (512 KB)
SSD_CHANNELS = 8
SSD_DIES_PER_CHANNEL = 4
SSD_PLANES_PER_DIE = 2

# Latencies (microseconds)
T_R_US = 25.0                   # NAND Read
T_PROG_US = 200.0               # NAND Program (Write)
T_BERS_US = 2000.0              # NAND Block Erase
BUS_TRANSFER_US_PER_PAGE = 5.0  # Channel Bus Transfer
PCIE_OVERHEAD_US = 10.0         # PCIe / NVMe Driver Overhead

# Cache Hot/Cold Defaults
DEFAULT_ATTENTION_SINK_TOKENS = 64
DEFAULT_RECENT_WINDOW_TOKENS = 512
DEFAULT_TOPK_PERCENT = 10.0
