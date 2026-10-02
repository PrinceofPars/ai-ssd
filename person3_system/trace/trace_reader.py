"""
Trace Reader and Strict Validator for AI-SSD V2.
Consumes real traces from P1 (/opt/ai-ssd-v2/traces/real_llm/) or synthetic traces.
"""

import json
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional, Union
from person3_system.trace.trace_schema import (
    TraceHeader,
    TraceRecord,
    TraceModelMetadata,
    TraceOperation,
)


class TraceValidationError(ValueError):
    """Raised when a trace violates schema, ordering, or metadata bounds."""
    pass


class TraceReader:
    """
    Validating trace parser for JSON and JSONL KV cache traces.
    Enforces strict schema validation, monotonic ordering, and boundary checking.
    """

    SUPPORTED_VERSIONS = {"v2.0", "v1.0"}

    def __init__(self, trace_path: Union[str, Path]):
        self.trace_path = Path(trace_path)
        self.header: Optional[TraceHeader] = None
        self._last_seq_id: int = -1

    def _validate_metadata(self, meta: TraceModelMetadata) -> None:
        if meta.num_layers <= 0:
            raise TraceValidationError(f"Invalid num_layers: {meta.num_layers} (must be > 0)")
        if meta.num_heads <= 0:
            raise TraceValidationError(f"Invalid num_heads: {meta.num_heads} (must be > 0)")
        if meta.head_dim <= 0:
            raise TraceValidationError(f"Invalid head_dim: {meta.head_dim} (must be > 0)")
        if meta.context_length <= 0:
            raise TraceValidationError(f"Invalid context_length: {meta.context_length} (must be > 0)")
        if meta.tokens_per_block <= 0:
            raise TraceValidationError(f"Invalid tokens_per_block: {meta.tokens_per_block} (must be > 0)")

    def validate_record(self, record: TraceRecord, header: TraceHeader) -> None:
        """Strict validation of an individual trace record."""
        # 1. Monotonic ordering
        if record.seq_id <= self._last_seq_id:
            raise TraceValidationError(
                f"Non-monotonic seq_id: current {record.seq_id} <= previous {self._last_seq_id}"
            )
        self._last_seq_id = record.seq_id

        # 2. Layer boundary
        if record.layer_id < 0 or record.layer_id >= header.model_metadata.num_layers:
            raise TraceValidationError(
                f"layer_id {record.layer_id} out of bounds [0, {header.model_metadata.num_layers - 1}]"
            )

        # 3. Head boundary
        if record.head_id < 0 or record.head_id >= header.model_metadata.num_heads:
            raise TraceValidationError(
                f"head_id {record.head_id} out of bounds [0, {header.model_metadata.num_heads - 1}]"
            )

        # 4. Operation check
        if not isinstance(record.operation, TraceOperation):
            try:
                record.operation = TraceOperation(record.operation)
            except ValueError:
                raise TraceValidationError(f"Unknown operation type: {record.operation}")

        # 5. Block IDs validation
        if not record.block_ids and record.operation in (TraceOperation.KV_READ, TraceOperation.KV_TOPK):
            raise TraceValidationError(f"Empty block_ids for operation {record.operation} at seq {record.seq_id}")

        max_blocks = (header.model_metadata.context_length // header.model_metadata.tokens_per_block) + 1
        for bid in record.block_ids:
            if bid < 0:
                raise TraceValidationError(f"Negative block_id {bid} at seq {record.seq_id}")
            if bid > max_blocks * 2:  # Safe margin for active sliding window
                raise TraceValidationError(
                    f"block_id {bid} exceeds expected capacity {max_blocks} at seq {record.seq_id}"
                )

        # 6. Byte range
        if record.byte_offset < 0:
            raise TraceValidationError(f"Negative byte_offset {record.byte_offset} at seq {record.seq_id}")
        if record.byte_length <= 0:
            raise TraceValidationError(f"Non-positive byte_length {record.byte_length} at seq {record.seq_id}")

    def read_records(self) -> Generator[TraceRecord, None, None]:
        """Streams and validates records one-by-one from the trace file."""
        if not self.trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {self.trace_path}")

        self._last_seq_id = -1
        with open(self.trace_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line:
                raise TraceValidationError("Empty trace file")

            try:
                header_data = json.loads(first_line)
            except json.JSONDecodeError as e:
                raise TraceValidationError(f"Malformed JSON in trace header: {e}")

            if "schema_version" not in header_data:
                raise TraceValidationError("Missing 'schema_version' in trace header")

            if header_data["schema_version"] not in self.SUPPORTED_VERSIONS:
                raise TraceValidationError(
                    f"Unsupported schema_version: {header_data['schema_version']}. Supported: {self.SUPPORTED_VERSIONS}"
                )

            self.header = TraceHeader.from_dict(header_data)
            self._validate_metadata(self.header.model_metadata)

            for line_no, line in enumerate(f, start=2):
                line = line.strip()
                if not line:
                    continue
                try:
                    record_dict = json.loads(line)
                except json.JSONDecodeError as e:
                    raise TraceValidationError(f"Line {line_no}: Malformed JSON in trace record: {e}")

                record = TraceRecord.from_dict(record_dict)
                self.validate_record(record, self.header)
                yield record

    def load_all(self) -> List[TraceRecord]:
        """Loads all records into memory after validation."""
        return list(self.read_records())
