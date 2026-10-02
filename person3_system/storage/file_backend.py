"""
File-backed and Direct-I/O Storage Backend for Linux/NVMe testing.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from person3_system.storage.backend import StorageBackend, StorageResult


class FileStorageBackend(StorageBackend):
    """
    Storage backend that persists KV data to a backing file or block device.
    Supports standard buffered POSIX I/O and direct I/O (O_DIRECT).
    """

    def __init__(
        self,
        filepath: str,
        block_size: int = 4096,
        direct_io: bool = False,
        create_if_missing: bool = True,
        name: str = "FileStorageBackend",
    ):
        super().__init__(name=name)
        self.filepath = filepath
        self.block_size = block_size
        self.direct_io = direct_io

        flags = os.O_RDWR
        if create_if_missing and not os.path.exists(filepath):
            flags |= os.O_CREAT

        if direct_io:
            if hasattr(os, "O_DIRECT"):
                flags |= os.O_DIRECT
            else:
                pass  # Fall back to standard buffered if O_DIRECT unsupported

        self._fd = os.open(filepath, flags, 0o644)

    def _block_offset(self, block_id: int, offset: int) -> int:
        return (block_id * self.block_size) + offset

    def read(
        self,
        block_id: int,
        offset: int = 0,
        length: int = 4096,
        layer_id: int = 0,
        head_id: int = 0,
        **kwargs,
    ) -> StorageResult:
        file_pos = self._block_offset(block_id, offset)
        t0 = time.perf_counter_ns()
        try:
            data = os.pread(self._fd, length, file_pos)
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
            
            self.total_reads += 1
            self.bytes_read += len(data)
            self.total_latency_us += elapsed_us

            return StorageResult(
                block_id=block_id,
                length=len(data),
                latency_us=elapsed_us,
                success=True,
                data=data,
            )
        except Exception as e:
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
            return StorageResult(
                block_id=block_id,
                length=0,
                latency_us=elapsed_us,
                success=False,
                error_msg=str(e),
            )

    def write(
        self,
        block_id: int,
        offset: int = 0,
        data: bytes = b"",
        layer_id: int = 0,
        head_id: int = 0,
        **kwargs,
    ) -> StorageResult:
        file_pos = self._block_offset(block_id, offset)
        t0 = time.perf_counter_ns()
        try:
            bytes_written = os.pwrite(self._fd, data, file_pos)
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0

            self.total_writes += 1
            self.bytes_written += bytes_written
            self.total_latency_us += elapsed_us

            return StorageResult(
                block_id=block_id,
                length=bytes_written,
                latency_us=elapsed_us,
                success=True,
                is_write=True,
            )
        except Exception as e:
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
            return StorageResult(
                block_id=block_id,
                length=0,
                latency_us=elapsed_us,
                success=False,
                error_msg=str(e),
                is_write=True,
            )

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
        super().close()
