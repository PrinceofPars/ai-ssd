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
        self.state: PageState = PageState.FREE
        self.data_block_id: Optional[int] = None
        self.program_count: int = 0
        self.read_count: int = 0
        self.last_accessed_us: float = 0.0

    def program(self, data_block_id: Optional[int] = None, block_id: Optional[int] = None) -> None:
        """
        Programs the flash page. Can accept either data_block_id or block_id.
        Raises ValueError if the page is not in FREE state.
        """
        bid = data_block_id if data_block_id is not None else block_id
        if self.state != PageState.FREE:
            raise ValueError(f"Cannot program non-free page {self.page_id} (state: {self.state})")
        self.state = PageState.VALID
        self.data_block_id = bid
        self.program_count += 1

    def read(self, current_time_us: float = 0.0) -> Optional[int]:
        """
        Reads data block ID from page and tracks read count (read disturb metric).
        Returns None if page is not VALID.
        """
        self.read_count += 1
        self.last_accessed_us = current_time_us
        return self.data_block_id if self.state == PageState.VALID else None

    def invalidate(self) -> None:
        """Marks page as INVALID (stale data)."""
        self.state = PageState.INVALID
        self.data_block_id = None

    def erase(self) -> None:
        """Resets page to FREE state and clears read count."""
        self.state = PageState.FREE
        self.data_block_id = None
        self.read_count = 0

    def __eq__(self, other: object) -> bool:
        """
        Supports comparison against integers (page_id) for backwards compatibility
        with tests expecting integer index, as well as FlashPage instances.
        """
        if isinstance(other, int):
            return self.page_id == other
        if isinstance(other, FlashPage):
            return (
                self.page_id == other.page_id
                and self.state == other.state
                and self.data_block_id == other.data_block_id
            )
        return False

    def __int__(self) -> int:
        return self.page_id

    def __index__(self) -> int:
        return self.page_id

    def __hash__(self) -> int:
        return hash(self.page_id)

    def __repr__(self) -> str:
        return f"<FlashPage id={self.page_id} state={self.state.value} data_block={self.data_block_id}>"

