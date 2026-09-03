"""
NAND Flash Page Model.
"""

from enum import Enum
from typing import Optional


class PageState(str, Enum):
    FREE = "FREE"
    VALID = "VALID"
    INVALID = "INVALID"


class FlashPage:
    def __init__(self, page_id: int, size_bytes: int = 4096):
        self.page_id = page_id
        self.size_bytes = size_bytes
        self.state = PageState.FREE
        self.data_block_id: Optional[int] = None

    def program(self, data_block_id: int) -> None:
        if self.state != PageState.FREE:
            raise ValueError(f"Cannot program non-free page {self.page_id} (state: {self.state})")
        self.state = PageState.VALID
        self.data_block_id = data_block_id

    def invalidate(self) -> None:
        self.state = PageState.INVALID
        self.data_block_id = None

    def erase(self) -> None:
        self.state = PageState.FREE
        self.data_block_id = None
