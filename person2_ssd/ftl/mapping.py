"""
Mapping Table: Logical Block Address (LBA / Block ID) to Physical Address (PBA).
"""

from typing import Dict, Optional


class MappingTable:
    def __init__(self):
        # Maps block_id -> "ch{c}_die{d}_pl{p}_blk{b}_pg{g}"
        self._map: Dict[int, str] = {}

    def set_mapping(self, block_id: int, physical_loc: str) -> None:
        self._map[block_id] = physical_loc

    def get_mapping(self, block_id: int) -> Optional[str]:
        return self._map.get(block_id)

    def remove_mapping(self, block_id: int) -> Optional[str]:
        return self._map.pop(block_id, None)

    def total_mapped(self) -> int:
        return len(self._map)
