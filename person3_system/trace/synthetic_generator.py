"""
Synthetic Trace Generator for Early Integration & Deterministic Testing.
Produces valid V2 JSONL traces simulating realistic multi-layer transformer decoding.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Union, List
from person3_system.trace.trace_schema import (
    TraceHeader,
    TraceRecord,
    TraceModelMetadata,
    TraceOperation,
)


class SyntheticTraceGenerator:
    """
    Generates synthetic KV cache trace files mimicking LLM autoregressive decoding.
    Incorporates attention sink tokens, sliding window tokens, and realistic candidate block sets.
    """

    def __init__(
        self,
        model_name: str = "Llama-3-8B-Simulated",
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype: str = "FP16",
        context_length: int = 4096,
        tokens_per_block: int = 16,
    ):
        self.metadata = TraceModelMetadata(
            model_name=model_name,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            context_length=context_length,
            tokens_per_block=tokens_per_block,
            block_size_bytes=tokens_per_block * head_dim * (2 if dtype == "FP16" else 4),
        )

    def generate_trace_file(
        self,
        output_path: Union[str, Path],
        num_steps: int = 5,
        top_k: int = 16,
    ) -> int:
        """
        Generates a validated JSONL trace file.
        Returns total number of records generated.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        total_blocks = self.metadata.context_length // self.metadata.tokens_per_block
        sink_blocks = list(range(min(4, total_blocks)))
        window_size = 16
        seq_id = 0

        with open(path, "w", encoding="utf-8") as f:
            # Write header
            header = TraceHeader(
                schema_version="v2.0",
                model_metadata=self.metadata,
                total_records=num_steps * self.metadata.num_layers,
                created_at=datetime.utcnow().isoformat(),
                source="P3_SYNTHETIC_GENERATOR",
            )
            f.write(json.dumps(header.to_dict()) + "\n")

            for step in range(num_steps):
                current_context_blocks = min(total_blocks, (step + 1) * 32 + 64)
                recent_start = max(0, current_context_blocks - window_size)
                recent_blocks = list(range(recent_start, current_context_blocks))

                for l in range(self.metadata.num_layers):
                    # Candidate blocks consist of sinks + recent + subset of middle blocks
                    stride_offset = (l * 3) % max(1, (recent_start - 4)) if recent_start > 4 else 0
                    mid_blocks = list(range(4 + stride_offset, min(recent_start, 4 + stride_offset + 12)))
                    candidates = sorted(list(set(sink_blocks + mid_blocks + recent_blocks)))
                    
                    rec = TraceRecord(
                        seq_id=seq_id,
                        step_id=step,
                        layer_id=l,
                        head_id=0,
                        operation=TraceOperation.KV_TOPK,
                        block_ids=candidates,
                        top_k=min(top_k, len(candidates)),
                        byte_offset=0,
                        byte_length=len(candidates) * self.metadata.block_size_bytes,
                        timestamp_us=float(seq_id * 50.0),
                    )
                    f.write(json.dumps(rec.to_dict()) + "\n")
                    seq_id += 1

        return seq_id
