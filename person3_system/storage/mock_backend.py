"""
Deterministic Mock Storage Backend for Testing and Stage 1 Integration.
"""

from typing import Dict, Any, Optional
from person3_system.storage.backend import StorageBackend, StorageResult


class MockStorageBackend(StorageBackend):
    """
    Configurable in-memory mock storage backend with deterministic latency simulation.
    """

    def __init__(
        self,
        read_latency_us: float = 25.0,
        write_latency_us: float = 200.0,
        fail_block_id: Optional[int] = None,
        name: str = "MockStorageBackend",
    ):
        super().__init__(name=name)
        self.read_latency_us = read_latency_us
        self.write_latency_us = write_latency_us
        self.fail_block_id = fail_block_id
        self._store: Dict[int, bytearray] = {}

    def read(
        self,
        block_id: int,
        offset: int = 0,
        length: int = 4096,
        layer_id: int = 0,
        head_id: int = 0,
        **kwargs,
    ) -> StorageResult:
        if self.fail_block_id is not None and block_id == self.fail_block_id:
            return StorageResult(
                block_id=block_id,
                length=0,
                latency_us=self.read_latency_us,
                success=False,
                error_msg=f"Simulated read failure on block {block_id}",
            )

        self.total_reads += 1
        self.bytes_read += length
        self.total_latency_us += self.read_latency_us

        data = None
        if block_id in self._store:
            buf = self._store[block_id]
            data = bytes(buf[offset : offset + length])
        else:
            data = bytes(length)

        return StorageResult(
            block_id=block_id,
            length=length,
            latency_us=self.read_latency_us,
            success=True,
            data=data,
            is_write=False,
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
        if self.fail_block_id is not None and block_id == self.fail_block_id:
            return StorageResult(
                block_id=block_id,
                length=0,
                latency_us=self.write_latency_us,
                success=False,
                error_msg=f"Simulated write failure on block {block_id}",
                is_write=True,
            )

        self.total_writes += 1
        length = len(data)
        self.bytes_written += length
        self.total_latency_us += self.write_latency_us

        if block_id not in self._store:
            self._store[block_id] = bytearray(max(4096, offset + length))
        
        target = self._store[block_id]
        if len(target) < offset + length:
            target.extend(b"\x00" * (offset + length - len(target)))
        target[offset : offset + length] = data

        return StorageResult(
            block_id=block_id,
            length=length,
            latency_us=self.write_latency_us,
            success=True,
            is_write=True,
        )
