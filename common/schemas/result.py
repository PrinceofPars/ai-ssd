from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any


class OperationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    CACHE_HIT = "CACHE_HIT"
    EVICTED = "EVICTED"
    ERROR = "ERROR"


@dataclass
class KVResponse:
    """
    Standard response format for all AI-SSD operations.
    """
    request_id: str
    status: OperationStatus

    selected_blocks: List[int] = field(default_factory=list)

    bytes_requested: int = 0
    bytes_transferred: int = 0

    latency_us: float = 0.0

    cache_hit: bool = False
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value if isinstance(self.status, OperationStatus) else self.status
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KVResponse:
        st = data.get("status")
        if isinstance(st, str):
            data["status"] = OperationStatus(st)
        return cls(**data)
