"""NAND flash physical structure models."""
from person2_ssd.nand.page import FlashPage, PageState
from person2_ssd.nand.block import FlashBlock

__all__ = ["FlashPage", "PageState", "FlashBlock"]
