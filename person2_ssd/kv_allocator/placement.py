"""
Placement policy definitions.
"""

from enum import Enum


class PlacementPolicy(str, Enum):
    CONVENTIONAL = "conventional"
    TENSOR_AWARE = "tensor_aware"
