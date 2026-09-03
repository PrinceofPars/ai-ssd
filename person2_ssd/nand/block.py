"""
NAND Flash Block Model.
Tracks erase cycles, valid page count, and garbage collection eligibility.
"""

from typing import List, Optional
from person2_ssd.nand.page import FlashPage, PageState
from common.constants import SSD_PAGES_PER_BLOCK, SSD_PAGE_SIZE_BYTES


class FlashBlock:
    def __init__(self, block_id: int, pages_count: int = SSD_PAGES_PER_BLOCK):
        self.block_id = block_id
        self.pages: List[FlashPage] = [FlashPage(i, SSD_PAGE_SIZE_BYTES) for i in range(pages_count)]
        self.erase_count: int = 0
        self.free_page_index: int = 0

    @property
    def valid_page_count(self) -> int:
        return sum(1 for p in self.pages if p.state == PageState.VALID)

    @property
    def invalid_page_count(self) -> int:
        return sum(1 for p in self.pages if p.state == PageState.INVALID)

    @property
    def free_page_count(self) -> int:
        return sum(1 for p in self.pages if p.state == PageState.FREE)

    def allocate_page(self, data_block_id: int) -> Optional[int]:
        """Sequential page write within erase block."""
        if self.free_page_index >= len(self.pages):
            return None
        page_idx = self.free_page_index
        self.pages[page_idx].program(data_block_id)
        self.free_page_index += 1
        return page_idx

    def erase(self) -> None:
        for p in self.pages:
            p.erase()
        self.free_page_index = 0
        self.erase_count += 1
