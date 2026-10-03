"""
Trace Schema Definitions for Real and Synthetic LLM KV Traces (P1/P3 Contract).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum


class TraceOperation(str, Enum):
    KV_READ = "KV_READ"
    KV_WRITE = "KV_WRITE"
    KV_TOPK = "KV_TOPK"
    KV_PREFETCH = "KV_PREFETCH"
    KV_EVICT = "KV_EVICT"


@dataclass
class TraceModelMetadata:
    """Metadata describing the LLM model that generated the trace."""
    model_name: str
    num_layers: int
    num_heads: int
    head_dim: int
    dtype: str
    context_length: int
    tokens_per_block: int = 16
    block_size_bytes: int = 4096

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraceModelMetadata:
        return cls(**data)


@dataclass
class TraceRecord:
    """Single I/O or KV access event in the trace."""
    seq_id: int
    step_id: int
    layer_id: int
    head_id: int
    operation: TraceOperation
    block_ids: List[int]
    byte_offset: int = 0
    byte_length: int = 4096
    top_k: Optional[int] = None
    timestamp_us: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["operation"] = self.operation.value if isinstance(self.operation, TraceOperation) else str(self.operation)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraceRecord:
        op = data.get("operation")
        if isinstance(op, str):
            data["operation"] = TraceOperation(op)
        return cls(**data)


@dataclass
class TraceHeader:
    """Header of a trace file establishing version and metadata."""
    schema_version: str
    model_metadata: TraceModelMetadata
    total_records: int
    created_at: str
    source: str = "P1_REAL_LLM"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["model_metadata"] = self.model_metadata.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraceHeader:
        mm = TraceModelMetadata.from_dict(data["model_metadata"])
        return cls(
            schema_version=data["schema_version"],
            model_metadata=mm,
            total_records=data.get("total_records", 0),
            created_at=data.get("created_at", ""),
            source=data.get("source", "P1_REAL_LLM"),
        )
