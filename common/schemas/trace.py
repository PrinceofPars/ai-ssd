"""
Canonical Shared Trace Contract for AI-SSD V2.
Unifies P1 real LLM traces, P2 FTL replayer, and P3 storage integration.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


class TraceOperation(str, Enum):
    """
    Explicit operation semantics across the computational storage hierarchy.
    """
    PREFILL_WRITE = "PREFILL_WRITE"  # Host offloads initial KV block (K+V) to SSD
    DECODE_READ = "DECODE_READ"      # Host reads KV blocks directly (unpruned/dense)
    TOPK_FILTER = "TOPK_FILTER"      # In-storage scan of candidate Key pages
    TOPK_FETCH = "TOPK_FETCH"        # Host fetches winning Value pages (sparse Top-k)
    KV_PREFETCH = "KV_PREFETCH"      # Speculative host DRAM pre-staging
    KV_EVICT = "KV_EVICT"            # Host/Controller cache eviction


@dataclass
class CanonicalTraceRecord:
    """
    Canonical trace record format for AI-SSD V2.
    Supports both P1 production trace fields and P3 synthetic aliases.
    """
    event_id: int
    step: int
    layer_id: int
    head_id: int
    operation: str
    block_id: int

    token_start: int = 0
    token_end: Optional[int] = None
    token_count: int = 16
    sub_page: str = "BOTH"  # "KEY", "VALUE", "BOTH"
    byte_size: int = 8192   # 4096 for single page, 8192 for combined K+V
    tier: str = "SSD"       # "DRAM", "SSD"
    hotness: float = 1.0

    candidate_blocks: List[int] = field(default_factory=list)
    selected_blocks: List[int] = field(default_factory=list)
    query_id: Optional[int] = None
    timestamp_ns: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- Canonical Aliases for P3 / Legacy Compatibility ---
    @property
    def seq_id(self) -> int:
        return self.event_id

    @property
    def step_id(self) -> int:
        return self.step

    @property
    def kv_head_id(self) -> int:
        return self.head_id

    @property
    def byte_length(self) -> int:
        return self.byte_size

    @property
    def timestamp_us(self) -> float:
        return self.timestamp_ns / 1000.0

    @property
    def block_ids(self) -> List[int]:
        if self.selected_blocks:
            return self.selected_blocks
        if self.candidate_blocks:
            return self.candidate_blocks
        return [self.block_id]

    @property
    def is_write(self) -> bool:
        return self.operation == TraceOperation.PREFILL_WRITE.value or self.operation == "KV_WRITE"

    @property
    def is_read(self) -> bool:
        return self.operation in (
            TraceOperation.DECODE_READ.value,
            TraceOperation.TOPK_FETCH.value,
            TraceOperation.TOPK_FILTER.value,
            "KV_READ",
            "KV_TOPK",
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanonicalTraceRecord:
        """
        Parses dictionary accepting either P1 real trace fields or P3 synthetic fields.
        """
        # Event / Sequence ID
        event_id = data.get("event_id")
        if event_id is None:
            event_id = data.get("seq_id")
        if event_id is None:
            raise KeyError("Missing required field 'event_id' or 'seq_id'")

        # Step ID
        step = data.get("step")
        if step is None:
            step = data.get("step_id", 0)

        # Layer & Head
        layer_id = data.get("layer_id", 0)
        head_id = data.get("head_id")
        if head_id is None:
            head_id = data.get("kv_head_id", 0)

        # Operation
        raw_op = data.get("operation") or data.get("op", "DECODE_READ")
        # Normalize string representation
        if isinstance(raw_op, Enum):
            op_str = raw_op.value
        else:
            op_str = str(raw_op)

        # Block ID / Block IDs
        b_id = data.get("block_id")
        b_ids = data.get("block_ids", [])
        if b_id is None:
            b_id = b_ids[0] if b_ids else 0

        # Tokens
        t_start = data.get("token_start")
        if t_start is None:
            t_start = data.get("token_idx", b_id * 16)
        t_end = data.get("token_end")
        t_count = data.get("token_count", 16)

        # Sub-page
        sub_page = data.get("sub_page", "BOTH")

        # Byte sizing
        byte_size = data.get("byte_size")
        if byte_size is None:
            byte_size = data.get("byte_length")
        if byte_size is None:
            byte_size = data.get("bytes", 8192 if sub_page == "BOTH" else 4096)

        tier = data.get("tier", "SSD")
        hotness = float(data.get("hotness", 1.0))
        cand = list(data.get("candidate_blocks", []))
        sel = list(data.get("selected_blocks", []))
        qid = data.get("query_id")
        ts = int(data.get("timestamp_ns", data.get("timestamp_us", 0) * 1000))

        # Preserve any unmapped metadata
        meta = data.get("metadata", {})

        return cls(
            event_id=int(event_id),
            step=int(step),
            layer_id=int(layer_id),
            head_id=int(head_id),
            operation=op_str,
            block_id=int(b_id),
            token_start=int(t_start),
            token_end=int(t_end) if t_end is not None else None,
            token_count=int(t_count),
            sub_page=str(sub_page),
            byte_size=int(byte_size),
            tier=str(tier),
            hotness=hotness,
            candidate_blocks=cand,
            selected_blocks=sel,
            query_id=qid,
            timestamp_ns=ts,
            metadata=meta,
        )


@dataclass
class TraceManifest:
    """
    Metadata manifest associated with a P1 real trace file.
    """
    model_name: str
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    tokens_per_block: int = 16
    key_page_bytes: int = 4096
    value_page_bytes: int = 4096
    logical_block_bytes: int = 8192
    total_events: int = 0
    total_tokens: int = 0
    trace_format_version: str = "2.0"
    gqa_ratio: int = 1
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraceManifest:
        return cls(
            model_name=data.get("model_name", "unknown"),
            num_layers=data.get("num_layers", 32),
            num_query_heads=data.get("num_query_heads", 32),
            num_kv_heads=data.get("num_kv_heads", 32),
            head_dim=data.get("head_dim", 128),
            dtype=data.get("dtype", "FP16"),
            tokens_per_block=data.get("tokens_per_block", 16),
            key_page_bytes=data.get("key_page_bytes", 4096),
            value_page_bytes=data.get("value_page_bytes", 4096),
            logical_block_bytes=data.get("logical_block_bytes", 8192),
            total_events=data.get("total_events", 0),
            total_tokens=data.get("total_tokens", 0),
            trace_format_version=data.get("trace_format_version", "2.0"),
            gqa_ratio=data.get("gqa_ratio", 1),
            raw_data=data,
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> TraceManifest:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
