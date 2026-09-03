"""
History buffer tracking layer-by-layer sequence patterns.
"""

from typing import List, Dict, Any


class AccessHistory:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._history: List[Dict[str, Any]] = []

    def record_access(self, layer_id: int, block_ids: List[int]) -> None:
        self._history.append({"layer_id": layer_id, "block_ids": block_ids})
        if len(self._history) > self.capacity:
            self._history.pop(0)

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        return self._history[-n:]
