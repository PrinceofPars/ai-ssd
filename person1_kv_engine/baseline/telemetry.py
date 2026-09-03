"""Inference telemetry tracker for baseline execution."""

from typing import Dict, Any


class InferenceTelemetry:
    def __init__(self):
        self.step_latencies = []
        self.tokens_generated = 0
        self.total_memory_bytes = 0

    def record_step(self, latency_ms: float, memory_bytes: int):
        self.step_latencies.append(latency_ms)
        self.tokens_generated += 1
        self.total_memory_bytes = memory_bytes

    def get_summary(self) -> Dict[str, Any]:
        avg_lat = sum(self.step_latencies) / max(1, len(self.step_latencies))
        return {
            "tokens_generated": self.tokens_generated,
            "avg_latency_ms": avg_lat,
            "total_memory_mb": self.total_memory_bytes / (1024 * 1024),
        }
