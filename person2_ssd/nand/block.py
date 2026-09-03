"""
NAND Flash Block Model.
Tracks erase cycles, valid page count, and garbage collection eligibility.
"""

from typing import List, Optional, Iterator
from person2_ssd.nand.page import FlashPage, PageState
from common.constants import SSD_PAGES_PER_BLOCK, SSD_PAGE_SIZE_BYTES


class FlashBlock:
    def __init__(
        self,
        block_id: int,
        pages_count: int = SSD_PAGES_PER_BLOCK,
        max_erase_cycles: int = 3000,
    ):
        self.block_id = block_id
        self.pages_count = pages_count
        self.max_erase_cycles = max_erase_cycles
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

    @property
    def is_full(self) -> bool:
        return self.free_page_index >= len(self.pages)

    @property
    def is_empty(self) -> bool:
        return self.free_page_count == len(self.pages)

    @property
    def is_bad_block(self) -> bool:
        return self.erase_count >= self.max_erase_cycles

    @property
    def garbage_ratio(self) -> float:
        """Proportion of invalid pages, used for GC victim selection."""
        return (self.invalid_page_count / len(self.pages)) if self.pages else 0.0

    def allocate_page(
        self,
        data_block_id: Optional[int] = None,
        logical_block_id: Optional[int] = None,
    ) -> Optional[FlashPage]:
        """
        Sequential page write within erase block.
        Returns the allocated FlashPage instance (satisfies integer comparison).
        """
        bid = data_block_id if data_block_id is not None else logical_block_id
        if self.free_page_index >= len(self.pages):
            return None
        page = self.pages[self.free_page_index]
        page.program(data_block_id=bid)
        self.free_page_index += 1
        return page

    def erase(self) -> None:
        """Erases all pages in the block, resets free_page_index, and increments erase_count."""
        for p in self.pages:
            p.erase()
        self.free_page_index = 0
        self.erase_count += 1

    def __getitem__(self, idx: int) -> FlashPage:
        return self.pages[idx]

    def __len__(self) -> int:
        return len(self.pages)

    def __iter__(self) -> Iterator[FlashPage]:
        return iter(self.pages)

    def __repr__(self) -> str:
        return (
            f"<FlashBlock id={self.block_id} pages={len(self.pages)} "
            f"free={self.free_page_count} valid={self.valid_page_count} "
            f"erase_count={self.erase_count}>"
        )

