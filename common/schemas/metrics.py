from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class MemoryMetrics:
    baseline_mb: float
    proposed_mb: float
    reduction_percent: float


@dataclass
class LatencyMetrics:
    baseline_ms: float
    proposed_ms: float
    overhead_percent: float


@dataclass
class StorageMetrics:
    bytes_requested: int
    bytes_transferred: int
    traffic_reduction_percent: float


@dataclass
class PrefetchMetrics:
    prediction_accuracy: float
    cache_hit_rate: float


@dataclass
class FTLMetrics:
    baseline_read_us: float
    tensor_aware_read_us: float
    speedup_x: float = 1.0


@dataclass
class SystemMetrics:
    """
    Standard aggregated system performance metrics.
    """
    memory: MemoryMetrics
    latency: LatencyMetrics
    storage: StorageMetrics
    prefetch: PrefetchMetrics
    ftl: FTLMetrics

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SystemMetrics:
        return cls(
            memory=MemoryMetrics(**data["memory"]),
            latency=LatencyMetrics(**data["latency"]),
            storage=StorageMetrics(**data["storage"]),
            prefetch=PrefetchMetrics(**data["prefetch"]),
            ftl=FTLMetrics(**data["ftl"]),
        )
