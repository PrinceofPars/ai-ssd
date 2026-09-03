from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any


class KVOperation(str, Enum):
    KV_WRITE = "KV_WRITE"
    KV_READ = "KV_READ"
    KV_EVICT = "KV_EVICT"
    KV_PREFETCH = "KV_PREFETCH"
    KV_TOPK = "KV_TOPK"


@dataclass
class KVRequest:
    """
    Standard request envelope passed across API and subsystem boundaries.
    """
    request_id: str
    operation: KVOperation
    layer_id: int
    head_id: int
    query_id: Optional[int] = None
    candidate_blocks: List[int] = field(default_factory=list)
    top_k: Optional[int] = None
    blocks_to_write: List[int] = field(default_factory=list)
    target_tier: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["operation"] = self.operation.value if isinstance(self.operation, KVOperation) else self.operation
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KVRequest:
        op = data.get("operation")
        if isinstance(op, str):
            data["operation"] = KVOperation(op)
        return cls(**data)
