"""
Machine-readable Result schemas for AI-SSD V2 experiments.
Distinguishes compute, storage, prefetch, model quality, and system metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional
from person3_system.experiments.config import ExperimentConfig
from person3_system.experiments.metadata import EnvironmentProvenance


@dataclass
class ModelQualityMetrics:
    topk_recall: Optional[float] = None
    attention_similarity: Optional[float] = None


@dataclass
class ComputeMetrics:
    compute_time_ms: float = 0.0
    mac_count: int = 0


@dataclass
class StorageMetrics:
    storage_reads: int = 0
    storage_writes: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    storage_latency_us: float = 0.0
    avg_read_latency_us: float = 0.0


@dataclass
class PrefetchMetrics:
    prefetch_requests: int = 0
    useful_prefetches: int = 0
    useless_prefetches: int = 0
    late_prefetches: int = 0
    demand_hits: int = 0
    demand_misses: int = 0
    prefetch_hit_rate_pct: float = 0.0
    prefetch_accuracy_pct: float = 0.0
    extra_bytes_read: int = 0
    pipeline_stalls: int = 0
    pipeline_stall_us: float = 0.0
    peak_memory_mb: float = 0.0


@dataclass
class SystemMetrics:
    baseline_dram_mb: float = 0.0
    active_dram_mb: float = 0.0
    ram_reduction_pct: float = 0.0
    total_execution_time_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    provenance: EnvironmentProvenance
    system: SystemMetrics
    storage: StorageMetrics
    prefetch: PrefetchMetrics
    compute: ComputeMetrics
    quality: ModelQualityMetrics
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "provenance": self.provenance.to_dict(),
            "system": asdict(self.system),
            "storage": asdict(self.storage),
            "prefetch": asdict(self.prefetch),
            "compute": asdict(self.compute),
            "quality": asdict(self.quality),
            "timestamp": self.timestamp,
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattens nested metrics into a single row for CSV logging."""
        row = {}
        for k, v in self.config.to_dict().items():
            row[f"cfg_{k}"] = v
        for k, v in asdict(self.system).items():
            row[f"sys_{k}"] = v
        for k, v in asdict(self.storage).items():
            row[f"stor_{k}"] = v
        for k, v in asdict(self.prefetch).items():
            row[f"pref_{k}"] = v
        for k, v in asdict(self.compute).items():
            row[f"comp_{k}"] = v
        row["git_commit"] = self.provenance.git_commit
        row["hostname"] = self.provenance.hostname
        row["timestamp"] = self.timestamp
        return row
