"""
Analytical FTL Storage Backend (P2 StorageSimulator Integration).
Supports both 'conventional' and 'tensor_aware' FTL placement policies.
"""

from typing import List, Dict, Any, Optional
from person3_system.storage.backend import StorageBackend, StorageRequest, StorageResult
from person2_ssd.storage_model.io_model import StorageSimulator
from common.schemas.kv_block import KVBlock, StorageTier


class AnalyticalFTLBackend(StorageBackend):
    """
    Storage backend powered by Person 2's multi-channel analytical flash simulator.
    Evaluates realistic flash bus transfer, die contention, and channel striping latencies.
    """

    def __init__(
        self,
        mode: str = "tensor_aware",
        channels: int = 8,
        dies_per_channel: int = 4,
        planes_per_die: int = 2,
        blocks_per_plane: int = 64,
        name: Optional[str] = None,
    ):
        backend_name = name or f"AnalyticalFTLBackend({mode})"
        super().__init__(name=backend_name)
        self.mode = mode
        self.simulator = StorageSimulator(
            mode=mode,
            channels=channels,
            dies_per_channel=dies_per_channel,
            planes_per_die=planes_per_die,
            blocks_per_plane=blocks_per_plane,
        )
        self._stored_blocks: set = set()

    def _ensure_block_placed(
        self,
        block_id: int,
        layer_id: int = 0,
        head_id: int = 0,
        token_count: int = 16,
    ) -> None:
        """Ensures the block is registered in the FTL mapping table."""
        if block_id not in self._stored_blocks:
            blk = KVBlock.create_default(
                block_id=block_id,
                layer_id=layer_id,
                token_start=block_id * token_count,
                token_count=token_count,
                kv_head_start=head_id,
                storage_tier=StorageTier.SSD.value,
            )
            self.simulator.store_block(blk)
            self._stored_blocks.add(block_id)

    def read(
        self,
        block_id: int,
        offset: int = 0,
        length: int = 4096,
        layer_id: int = 0,
        head_id: int = 0,
        **kwargs,
    ) -> StorageResult:
        self._ensure_block_placed(block_id, layer_id, head_id)
        latency_us = self.simulator.estimate_read_latency([block_id])

        self.total_reads += 1
        self.bytes_read += length
        self.total_latency_us += latency_us

        return StorageResult(
            block_id=block_id,
            length=length,
            latency_us=latency_us,
            success=True,
            metadata={"mode": self.mode, "location": self.simulator.get_location(block_id)},
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
        self._ensure_block_placed(block_id, layer_id, head_id)
        # Standard TLC program latency approx 200 us
        write_lat_us = 200.0

        self.total_writes += 1
        length = len(data) if data else 4096
        self.bytes_written += length
        self.total_latency_us += write_lat_us

        return StorageResult(
            block_id=block_id,
            length=length,
            latency_us=write_lat_us,
            success=True,
            is_write=True,
            metadata={"mode": self.mode, "location": self.simulator.get_location(block_id)},
        )

    def submit_batch(self, requests: List[StorageRequest]) -> List[StorageResult]:
        """
        Batch read optimization: evaluates multi-channel parallelism across all requested blocks simultaneously.
        """
        read_bids = []
        read_reqs = []
        write_reqs = []

        for req in requests:
            if req.is_write:
                write_reqs.append(req)
            else:
                self._ensure_block_placed(req.block_id, req.layer_id, req.head_id)
                read_bids.append(req.block_id)
                read_reqs.append(req)

        results = []
        if read_bids:
            # Multi-channel contention evaluated jointly across the batch
            batch_lat_us = self.simulator.estimate_read_latency(read_bids)
            per_block_lat = batch_lat_us / len(read_bids)

            for req in read_reqs:
                self.total_reads += 1
                self.bytes_read += req.length
                results.append(
                    StorageResult(
                        block_id=req.block_id,
                        length=req.length,
                        latency_us=per_block_lat,
                        success=True,
                        metadata={"mode": self.mode, "batch_latency_us": batch_lat_us},
                    )
                )
            self.total_latency_us += batch_lat_us

        for req in write_reqs:
            w_res = self.write(
                block_id=req.block_id,
                offset=req.offset,
                data=req.payload or b"",
                layer_id=req.layer_id,
                head_id=req.head_id,
                **req.metadata,
            )
            results.append(w_res)

        return results
