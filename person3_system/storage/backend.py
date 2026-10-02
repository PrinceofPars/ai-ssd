"""
Storage Backend Abstraction Layer for AI-SSD V2.

Provides a unified interface decoupling higher-level orchestration,
prefetching, and benchmarking from the specific storage mechanism
(Mock, File/Direct I/O, Analytical FTL, or Physical NVMe/FEMU).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from concurrent.futures import Future, ThreadPoolExecutor
import time


@dataclass
class StorageRequest:
    """Standard storage I/O request descriptor."""
    block_id: int
    offset: int = 0
    length: int = 4096
    is_write: bool = False
    payload: Optional[bytes] = None
    layer_id: int = 0
    head_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageResult:
    """Standard storage I/O result descriptor."""
    block_id: int
    length: int
    latency_us: float
    success: bool = True
    error_msg: Optional[str] = None
    data: Optional[bytes] = None
    is_write: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class StorageBackend(ABC):
    """
    Abstract interface for all AI-SSD V2 storage backends.
    """

    def __init__(self, name: str = "StorageBackend"):
        self.name = name
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.total_reads: int = 0
        self.total_writes: int = 0
        self.bytes_read: int = 0
        self.bytes_written: int = 0
        self.total_latency_us: float = 0.0

    @abstractmethod
    def read(
        self,
        block_id: int,
        offset: int = 0,
        length: int = 4096,
        layer_id: int = 0,
        head_id: int = 0,
        **kwargs,
    ) -> StorageResult:
        """Synchronously read bytes from a block."""
        pass

    @abstractmethod
    def write(
        self,
        block_id: int,
        offset: int = 0,
        data: bytes = b"",
        layer_id: int = 0,
        head_id: int = 0,
        **kwargs,
    ) -> StorageResult:
        """Synchronously write bytes to a block."""
        pass

    def submit_batch(self, requests: List[StorageRequest]) -> List[StorageResult]:
        """
        Submit a batch of storage requests. Backends may optimize batching
        (e.g., parallel channel distribution or vector I/O).
        """
        results = []
        for req in requests:
            if req.is_write:
                res = self.write(
                    block_id=req.block_id,
                    offset=req.offset,
                    data=req.payload or b"",
                    layer_id=req.layer_id,
                    head_id=req.head_id,
                    **req.metadata,
                )
            else:
                res = self.read(
                    block_id=req.block_id,
                    offset=req.offset,
                    length=req.length,
                    layer_id=req.layer_id,
                    head_id=req.head_id,
                    **req.metadata,
                )
            results.append(res)
        return results

    def async_read(
        self,
        block_id: int,
        offset: int = 0,
        length: int = 4096,
        layer_id: int = 0,
        head_id: int = 0,
        callback: Optional[Callable[[StorageResult], None]] = None,
        **kwargs,
    ) -> Future[StorageResult]:
        """
        Asynchronously issue a read request returning a Future.
        """
        def _task() -> StorageResult:
            res = self.read(
                block_id=block_id,
                offset=offset,
                length=length,
                layer_id=layer_id,
                head_id=head_id,
                **kwargs,
            )
            if callback:
                callback(res)
            return res

        return self._executor.submit(_task)

    def async_write(
        self,
        block_id: int,
        offset: int = 0,
        data: bytes = b"",
        layer_id: int = 0,
        head_id: int = 0,
        callback: Optional[Callable[[StorageResult], None]] = None,
        **kwargs,
    ) -> Future[StorageResult]:
        """
        Asynchronously issue a write request returning a Future.
        """
        def _task() -> StorageResult:
            res = self.write(
                block_id=block_id,
                offset=offset,
                data=data,
                layer_id=layer_id,
                head_id=head_id,
                **kwargs,
            )
            if callback:
                callback(res)
            return res

        return self._executor.submit(_task)

    def get_telemetry(self) -> Dict[str, Any]:
        """Return backend telemetry counters."""
        avg_lat = (self.total_latency_us / (self.total_reads + self.total_writes)) if (self.total_reads + self.total_writes) > 0 else 0.0
        return {
            "backend_name": self.name,
            "total_reads": self.total_reads,
            "total_writes": self.total_writes,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "total_latency_us": self.total_latency_us,
            "avg_latency_us": avg_lat,
        }

    def reset_stats(self) -> None:
        """Reset performance counters."""
        self.total_reads = 0
        self.total_writes = 0
        self.bytes_read = 0
        self.bytes_written = 0
        self.total_latency_us = 0.0

    def close(self) -> None:
        """Clean up threadpool and resources."""
        self._executor.shutdown(wait=False)
