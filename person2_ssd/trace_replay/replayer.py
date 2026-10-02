"""
Trace Replay Engine for AI-SSD V2 (Level 2 Storage Evaluation).

Consumes real LLM KV traces from /opt/ai-ssd-v2/traces/real_llm/ or generates
deterministic synthetic traces if P1 is still generating traces.
Replays requests through Conventional and Tensor-Aware storage backends,
measuring request count, throughput, channel/die utilization, and contention.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from common.constants import (
    SSD_CHANNELS,
    SSD_DIES_PER_CHANNEL,
    DEFAULT_BLOCK_SIZE_BYTES,
    T_R_US,
    BUS_TRANSFER_US_PER_PAGE,
    PCIE_OVERHEAD_US,
)
from person2_ssd.kv_allocator.tensor_mapping import DeterministicTensorMapper, TensorCoordinate


class StorageTraceReplayer:
    """
    Replays KV access traces against Conventional and Tensor-Aware FTL storage models.
    """
    def __init__(
        self,
        channels: int = SSD_CHANNELS,
        dies_per_channel: int = SSD_DIES_PER_CHANNEL,
        queue_depth: int = 8,
    ):
        self.channels = channels
        self.dies_per_channel = dies_per_channel
        self.queue_depth = queue_depth
        self.mapper = DeterministicTensorMapper(channels=channels, dies_per_channel=dies_per_channel)

    def discover_or_synthesize_trace(
        self,
        trace_dir: str = "/opt/ai-ssd-v2/traces/real_llm",
        num_layers: int = 32,
        num_heads: int = 32,
        seq_length: int = 2048,
    ) -> List[Dict[str, Any]]:
        """
        Loads real LLM traces if available; otherwise synthesizes an authentic trace.
        """
        trace_path = Path(trace_dir)
        real_files = list(trace_path.glob("*.json")) + list(trace_path.glob("*.jsonl")) if trace_path.exists() else []

        if real_files:
            print(f"[TraceReplayer] Discovered real LLM trace: {real_files[0]}")
            requests = []
            with open(real_files[0], "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        requests.append(json.loads(line))
            return requests

        # Synthetic trace representing multi-layer decoding attention access
        print("[TraceReplayer] No real trace in /opt/ai-ssd-v2/traces/real_llm/. Synthesizing realistic KV trace...")
        synthetic_requests = []
        req_id = 0

        # Prefill phase: sequential writes across layers and heads
        for l in range(min(num_layers, 4)):
            for h in range(min(num_heads, 8)):
                for t in range(0, min(seq_length, 256), 16):
                    synthetic_requests.append({
                        "request_id": f"req_w_{req_id}",
                        "op": "WRITE",
                        "layer_id": l,
                        "head_id": h,
                        "token_idx": t,
                        "token_count": 16,
                        "bytes": DEFAULT_BLOCK_SIZE_BYTES,
                    })
                    req_id += 1

        # Decoding phase: concurrent reads across heads (Top-k cold block retrievals)
        for step in range(16):
            for l in range(min(num_layers, 4)):
                for h in range(min(num_heads, 8)):
                    # Simulates attention retrieval on cold token blocks
                    token_offset = (step * 16 + h * 32) % 256
                    synthetic_requests.append({
                        "request_id": f"req_r_{req_id}",
                        "op": "READ",
                        "layer_id": l,
                        "head_id": h,
                        "token_idx": token_offset,
                        "token_count": 16,
                        "bytes": DEFAULT_BLOCK_SIZE_BYTES,
                    })
                    req_id += 1

        return synthetic_requests

    def replay(self, trace: List[Dict[str, Any]], mode: str = "tensor_aware") -> Dict[str, Any]:
        """
        Executes trace replay against the specified storage mode.
        """
        start_wall_time = time.perf_counter()
        channel_counters = {c: 0 for c in range(self.channels)}
        channel_access_counts = {c: 0 for c in range(self.channels)}
        die_access_counts = {d: 0 for d in range(self.dies_per_channel)}

        total_bytes_read = 0
        total_bytes_written = 0
        read_latencies_us = []
        batch_queue = []

        def process_batch(batch: List[Dict[str, Any]]) -> float:
            if not batch:
                return 0.0
            loads = {}
            for item in batch:
                ch = item["ch"]
                loads[ch] = loads.get(ch, 0) + 1
            max_load = max(loads.values()) if loads else 0
            # Latency = PCIe overhead + (max_channel_load * (tR + t_bus))
            batch_lat = PCIE_OVERHEAD_US + (max_load * (T_R_US + BUS_TRANSFER_US_PER_PAGE))
            return batch_lat

        for item in trace:
            op = item.get("op", "READ")
            layer = item.get("layer_id", 0)
            head = item.get("head_id", 0)
            token_idx = item.get("token_idx", 0)
            b_size = item.get("bytes", DEFAULT_BLOCK_SIZE_BYTES)

            coord = TensorCoordinate(layer_id=layer, head_id=head, token_idx=token_idx)
            phys = self.mapper.tensor_to_nand_physical(coord, mode=mode, channel_counters=channel_counters)

            ch = phys.channel
            die = phys.die
            channel_counters[ch] += 1
            channel_access_counts[ch] += 1
            die_access_counts[die] += 1

            if op == "WRITE":
                total_bytes_written += b_size
            else:
                total_bytes_read += b_size
                batch_queue.append({"ch": ch, "die": die})
                if len(batch_queue) >= self.queue_depth:
                    lat = process_batch(batch_queue)
                    read_latencies_us.append(lat)
                    batch_queue = []

        if batch_queue:
            lat = process_batch(batch_queue)
            read_latencies_us.append(lat)

        total_simulated_lat_us = sum(read_latencies_us)
        elapsed_wall_s = max(time.perf_counter() - start_wall_time, 1e-6)

        avg_channel_load = sum(channel_access_counts.values()) / max(1, self.channels)
        max_channel_load = max(channel_access_counts.values()) if channel_access_counts else 0
        contention_ratio = max_channel_load / max(1e-6, avg_channel_load)

        total_mb = (total_bytes_read + total_bytes_written) / (1024 * 1024)
        throughput_mbs = total_mb / (total_simulated_lat_us / 1e6) if total_simulated_lat_us > 0 else 0.0

        return {
            "mode": mode,
            "total_requests": len(trace),
            "bytes_read": total_bytes_read,
            "bytes_written": total_bytes_written,
            "queue_depth": self.queue_depth,
            "total_simulated_latency_us": round(total_simulated_lat_us, 2),
            "avg_latency_per_batch_us": round(sum(read_latencies_us) / max(1, len(read_latencies_us)), 2) if read_latencies_us else 0.0,
            "throughput_mbs": round(throughput_mbs, 2),
            "channel_access_counts": channel_access_counts,
            "die_access_counts": die_access_counts,
            "max_channel_load": max_channel_load,
            "avg_channel_load": round(avg_channel_load, 2),
            "contention_ratio": round(contention_ratio, 2),
            "replay_wall_time_s": round(elapsed_wall_s, 4),
        }