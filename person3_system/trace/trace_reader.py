"""
Trace Reader and Strict Validator for AI-SSD V2.
Consumes real P1 traces (/opt/ai-ssd-v2/traces/real_llm/) and synthetic traces.
Automatically associates .manifest.json metadata and produces CanonicalTraceRecords.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional, Union

from common.schemas.trace import (
    CanonicalTraceRecord,
    TraceManifest,
    TraceOperation,
)
from person3_system.trace.trace_schema import TraceHeader, TraceModelMetadata


class TraceValidationError(ValueError):
    """Raised when a trace violates schema, ordering, or metadata bounds."""
    pass


class TraceReader:
    """
    Validating trace parser for JSON and JSONL KV cache traces.
    Seamlessly consumes P1 production traces and P3 synthetic traces.
    """

    SUPPORTED_VERSIONS = {"v2.0", "v1.0", "2.0", "1.0"}

    def __init__(self, trace_path: Union[str, Path]):
        self.trace_path = Path(trace_path)
        self.manifest: Optional[TraceManifest] = None
        self.header: Optional[TraceHeader] = None
        self._last_event_id: int = -1

        # Reject manifest files directly supplied as trace files
        if self.trace_path.name.endswith(".manifest.json"):
            raise TraceValidationError(f"Cannot read manifest file '{self.trace_path.name}' as a JSONL trace stream.")

        # Resolve associated manifest if present
        self._discover_manifest()

    def _discover_manifest(self) -> None:
        """Looks for adjacent .manifest.json file."""
        stem = self.trace_path.stem
        # If filename is foo.jsonl, check foo.manifest.json or foo.jsonl.manifest.json
        cand1 = self.trace_path.with_name(f"{stem}.manifest.json")
        cand2 = self.trace_path.with_name(f"{self.trace_path.name}.manifest.json")
        
        for cand in [cand1, cand2]:
            if cand.exists():
                try:
                    self.manifest = TraceManifest.from_file(cand)
                    # Sync to header object for backwards compatibility
                    self.header = TraceHeader(
                        schema_version=self.manifest.trace_format_version,
                        model_metadata=TraceModelMetadata(
                            model_name=self.manifest.model_name,
                            num_layers=self.manifest.num_layers,
                            num_heads=self.manifest.num_kv_heads,
                            head_dim=self.manifest.head_dim,
                            dtype=self.manifest.dtype,
                            context_length=self.manifest.tokens_per_block * 256,
                            tokens_per_block=self.manifest.tokens_per_block,
                            block_size_bytes=self.manifest.logical_block_bytes,
                        ),
                        total_records=self.manifest.total_events,
                        created_at="",
                        source="P1_MANIFEST",
                    )
                    break
                except Exception:
                    pass

    def _validate_record(self, record: CanonicalTraceRecord) -> None:
        """Strict validation of individual records."""
        # 1. Monotonic ordering
        if record.event_id <= self._last_event_id:
            raise TraceValidationError(
                f"Non-monotonic seq_id / event_id: current {record.event_id} <= previous {self._last_event_id}"
            )
        self._last_event_id = record.event_id

        # 2. Layer boundary (if manifest or header available)
        max_layers = self.manifest.num_layers if self.manifest else (self.header.model_metadata.num_layers if self.header else 128)
        if record.layer_id < 0 or record.layer_id >= max_layers:
            raise TraceValidationError(
                f"layer_id {record.layer_id} out of bounds [0, {max_layers - 1}] at event {record.event_id}"
            )

        # 3. Head boundary
        max_heads = self.manifest.num_kv_heads if self.manifest else (self.header.model_metadata.num_heads if self.header else 64)
        if record.head_id < 0 or record.head_id >= max_heads:
            raise TraceValidationError(
                f"head_id {record.head_id} out of bounds [0, {max_heads - 1}] at event {record.event_id}"
            )

        # 4. Byte sizing
        if record.byte_size <= 0:
            raise TraceValidationError(
                f"Non-positive byte_size {record.byte_size} at event {record.event_id}"
            )

        # 5. Non-negative block ID
        if record.block_id < 0:
            raise TraceValidationError(
                f"Negative block_id {record.block_id} at event {record.event_id}"
            )

    def read_records(self) -> Generator[CanonicalTraceRecord, None, None]:
        """Streams and strictly validates records one-by-one."""
        if not self.trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {self.trace_path}")

        self._last_event_id = -1
        with open(self.trace_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line:
                raise TraceValidationError("Empty trace file")

            try:
                first_data = json.loads(first_line)
            except json.JSONDecodeError as e:
                raise TraceValidationError(f"Malformed JSON in line 1: {e}")

            # Check if line 1 is a synthetic Header or a data record
            if "schema_version" in first_data and "model_metadata" in first_data:
                # Synthetic in-file header
                if first_data["schema_version"] not in self.SUPPORTED_VERSIONS:
                    raise TraceValidationError(
                        f"Unsupported schema_version: {first_data['schema_version']}. Supported: {self.SUPPORTED_VERSIONS}"
                    )
                self.header = TraceHeader.from_dict(first_data)
                start_line = 2
            else:
                # Real P1 trace line 1 is the first record!
                record = CanonicalTraceRecord.from_dict(first_data)
                self._validate_record(record)
                yield record
                start_line = 2

            for line_no, line in enumerate(f, start=start_line):
                line = line.strip()
                if not line:
                    continue
                try:
                    record_dict = json.loads(line)
                except json.JSONDecodeError as e:
                    raise TraceValidationError(f"Line {line_no}: Malformed JSON in trace record: {e}")

                record = CanonicalTraceRecord.from_dict(record_dict)
                self._validate_record(record)
                yield record

    def load_all(self) -> List[CanonicalTraceRecord]:
        """Loads and strictly validates all records in the trace."""
        return list(self.read_records())
