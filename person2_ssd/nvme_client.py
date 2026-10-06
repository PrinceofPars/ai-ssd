#!/usr/bin/env python3
"""
QEMU NVMe Guest Daemon Client for Person 2 Storage Backend.
Provides high-performance socket communication with the Linux kernel NVMe driver
running inside the QEMU KVM guest environment.
"""

import os
import sys
import time
import socket
import struct
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

RAW_IMG = "/opt/ai-ssd-v2/images/v2_nvme.raw"
INITRD = "/opt/ai-ssd-v2/images/initramfs.cpio.gz"
KERNEL = "/opt/ai-ssd-v2/images/vmlinuz"
DEFAULT_PORT = 9999
MAGIC = 0x4E564D45

OP_WRITE = 1
OP_READ = 2
OP_PING = 3
OP_FLUSH = 4
OP_SHUTDOWN = 5
OP_BATCH_READ = 6
OP_COMPUTE_TOPK = 7

HEADER_SIZE = 20  # struct.calcsize("<IBBHQI")


class QemuNvmeClient:
    """
    Client connecting host Python processes directly to the QEMU Virtual NVMe guest.
    Dispatches real block I/O requests through the Linux in-kernel NVMe driver.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self._qemu_proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self.total_read_bytes: int = 0
        self.total_write_bytes: int = 0
        self.read_ops: int = 0
        self.write_ops: int = 0
        self.total_read_time_s: float = 0.0
        self.total_write_time_s: float = 0.0
        self.total_pack_time_s: float = 0.0
        self.total_send_time_s: float = 0.0
        self.total_wait_time_s: float = 0.0
        self.total_recv_time_s: float = 0.0
        self.batch_count: int = 0
        self.batch_sizes: List[int] = []
        self.topk_compute_ops: int = 0
        self.topk_internal_scanned_bytes: int = 0
        self.topk_query_transferred_bytes: int = 0
        self.topk_metadata_transferred_bytes: int = 0

    def start_qemu(self, raw_img: str = RAW_IMG, timeout_s: float = 20.0) -> None:
        """Starts QEMU with KVM and the virtual NVMe device, waiting for the guest daemon."""
        if self.is_alive():
            return

        cmd = [
            "qemu-system-x86_64",
            "-enable-kvm",
            "-cpu", "host",
            "-m", "2048",
            "-smp", "2",
            "-no-reboot",
            "-kernel", KERNEL,
            "-initrd", INITRD,
            "-drive", f"file={raw_img},format=raw,if=none,id=nvme0",
            "-device", "nvme,drive=nvme0,serial=v2-ai-ssd-001,num_queues=8,logical_block_size=4096,physical_block_size=4096",
            "-netdev", f"user,id=net0,hostfwd=tcp:{self.host}:{self.port}-:9999",
            "-device", "virtio-net-pci,netdev=net0",
            "-nographic",
            "-append", "console=ttyS0 panic=-1 quiet loglevel=3 daemon=1",
        ]

        self._qemu_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        t0 = time.time()
        connected = False
        while time.time() - t0 < timeout_s:
            try:
                self.connect()
                if self.is_alive():
                    connected = True
                    break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.2)

        if not connected:
            self.stop_qemu()
            raise TimeoutError(f"Failed to connect to QEMU NVMe guest daemon on {self.host}:{self.port} within {timeout_s}s")

    def connect(self) -> None:
        """Establishes persistent TCP socket connection with TCP_NODELAY enabled."""
        with self._lock:
            if self.sock is not None:
                try:
                    self.sock.close()
                except OSError:
                    pass
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(10.0)
            s.connect((self.host, self.port))
            self.sock = s

    def is_alive(self) -> bool:
        """Sends PING opcode and verifies guest controller response."""
        with self._lock:
            if self.sock is None:
                try:
                    self.connect()
                except OSError:
                    return False
            try:
                req = struct.pack("<IBBHQI", MAGIC, OP_PING, 0, 0, 0, 0)
                self.sock.sendall(req)
                resp = self._recv_exact(HEADER_SIZE)
                magic, status, op, _, _, length = struct.unpack("<IBBHQI", resp)
                return magic == MAGIC and status == 0
            except OSError:
                return False

    def _recv_exact(self, n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionResetError("Connection closed unexpectedly by QEMU guest NVMe daemon")
            data.extend(chunk)
        return bytes(data)

    def write(self, offset: int, data: bytes) -> int:
        """
        Dispatches a write operation to /dev/nvme0n1 in the QEMU guest.
        Returns the number of bytes written.
        """
        with self._lock:
            if self.sock is None:
                self.connect()
            length = len(data)
            req = struct.pack("<IBBHQI", MAGIC, OP_WRITE, 0, 0, offset, length)
            t0 = time.perf_counter()
            self.sock.sendall(req + data)
            resp = self._recv_exact(HEADER_SIZE)
            elapsed = time.perf_counter() - t0

            magic, status, op, _, resp_offset, bytes_written = struct.unpack("<IBBHQI", resp)
            if magic != MAGIC or status != 0:
                raise IOError(f"NVMe write failed at offset {offset}, length {length}, status {status}")

            self.write_ops += 1
            self.total_write_bytes += bytes_written
            self.total_write_time_s += elapsed
            return bytes_written

    def read(self, offset: int, length: int) -> bytes:
        """
        Dispatches a read operation to /dev/nvme0n1 in the QEMU guest.
        Returns raw byte buffer.
        """
        with self._lock:
            if self.sock is None:
                self.connect()
            req = struct.pack("<IBBHQI", MAGIC, OP_READ, 0, 0, offset, length)
            t_send = time.perf_counter()
            self.sock.sendall(req)
            send_elapsed = time.perf_counter() - t_send

            t_wait = time.perf_counter()
            resp = self._recv_exact(HEADER_SIZE)
            wait_elapsed = time.perf_counter() - t_wait

            magic, status, op, _, resp_offset, data_len = struct.unpack("<IBBHQI", resp)
            if magic != MAGIC or status != 0:
                raise IOError(f"NVMe read failed at offset {offset}, length {length}, status {status}")

            t_recv = time.perf_counter()
            data = self._recv_exact(data_len)
            recv_elapsed = time.perf_counter() - t_recv

            total_elapsed = send_elapsed + wait_elapsed + recv_elapsed
            self.read_ops += 1
            self.total_read_bytes += data_len
            self.total_read_time_s += total_elapsed
            self.total_send_time_s += send_elapsed
            self.total_wait_time_s += wait_elapsed
            self.total_recv_time_s += recv_elapsed
            return data

    def read_batch(self, requests: List[Tuple[int, int, int]]) -> Dict[int, bytes]:
        """
        Dispatches a batch of read requests to /dev/nvme0n1 in a single transaction.
        Args:
            requests: List of (offset, length, block_id) tuples.
        Returns:
            Dict mapping block_id -> bytes data.
        """
        if not requests:
            return {}
        with self._lock:
            if self.sock is None:
                self.connect()

            t_pack = time.perf_counter()
            num_items = len(requests)
            req_hdr = struct.pack("<IBBHQI", MAGIC, OP_BATCH_READ, 0, 0, 0, num_items)
            items_payload = bytearray()
            total_req_bytes = 0
            for offset, length, bid in requests:
                items_payload.extend(struct.pack("<QII", offset, length, bid))
                total_req_bytes += length
            pack_elapsed = time.perf_counter() - t_pack

            t_send = time.perf_counter()
            self.sock.sendall(req_hdr + items_payload)
            send_elapsed = time.perf_counter() - t_send

            t_wait = time.perf_counter()
            resp = self._recv_exact(HEADER_SIZE)
            wait_elapsed = time.perf_counter() - t_wait

            magic, status, op, _, _, resp_items = struct.unpack("<IBBHQI", resp)
            if magic != MAGIC or status != 0:
                raise IOError(f"NVMe batch read failed, status {status}")

            t_recv = time.perf_counter()
            results: Dict[int, bytes] = {}
            for _ in range(resp_items):
                item_hdr = self._recv_exact(9)
                item_status, bid, item_len = struct.unpack("<BII", item_hdr)
                if item_status != 0:
                    raise IOError(f"NVMe batch read item failed for block {bid}")
                data = self._recv_exact(item_len)
                results[bid] = data
            recv_elapsed = time.perf_counter() - t_recv

            total_elapsed = pack_elapsed + send_elapsed + wait_elapsed + recv_elapsed
            self.read_ops += num_items
            self.total_read_bytes += total_req_bytes
            self.total_read_time_s += total_elapsed
            self.total_pack_time_s += pack_elapsed
            self.total_send_time_s += send_elapsed
            self.total_wait_time_s += wait_elapsed
            self.total_recv_time_s += recv_elapsed
            self.batch_count += 1
            self.batch_sizes.append(num_items)
            return results

    def compute_topk(
        self,
        query: Any,
        candidates: List[Tuple[int, int, int, int]],  # (offset, length, block_id, actual_tokens)
        top_k: int,
        scale: float,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        is_fp16: bool = False,
    ) -> List[Tuple[float, int, int]]:
        """
        Dispatches in-storage Top-K filtering to /dev/nvme0n1 inside the QEMU guest VM.
        Transfers ONLY the Query vector Q and candidate block descriptors over the storage bus.
        The guest daemon reads candidate Key pages directly from NVMe, computes GQA dot-products,
        and returns ONLY the top_k selected block IDs and scores.
        Candidate Key pages are NEVER transferred to the host!

        Returns:
            List of (score, block_id, actual_tokens) tuples sorted descending by score.
        """
        if not candidates:
            return []
        with self._lock:
            if self.sock is None:
                self.connect()

            import numpy as np

            t_pack = time.perf_counter()
            num_cands = len(candidates)
            # 1. Header: magic, op, flags (1 for fp16, 0 for fp32), reserved, offset, length (num_cands)
            flags = 1 if is_fp16 else 0
            req_hdr = struct.pack("<IBBHQI", MAGIC, OP_COMPUTE_TOPK, flags, 0, 0, num_cands)
            # 2. Top-K parameters header: num_candidates, top_k, q_heads, kv_heads, head_dim, scale
            topk_hdr = struct.pack("<IIIIIf", num_cands, top_k, q_heads, kv_heads, head_dim, float(scale))
            # 3. Query array bytes (float32 contiguous)
            q_contiguous = np.ascontiguousarray(query, dtype=np.float32)
            q_bytes = q_contiguous.tobytes()
            # 4. Candidate items: offset (Q), length (I), block_id (I), actual_tokens (I)
            cands_payload = bytearray()
            total_internal_k_bytes = 0
            for offset, length, bid, act_tok in candidates:
                cands_payload.extend(struct.pack("<QIII", offset, length, bid, act_tok))
                total_internal_k_bytes += length

            pack_elapsed = time.perf_counter() - t_pack

            t_send = time.perf_counter()
            self.sock.sendall(req_hdr + topk_hdr + q_bytes + cands_payload)
            send_elapsed = time.perf_counter() - t_send

            t_wait = time.perf_counter()
            resp = self._recv_exact(HEADER_SIZE)
            wait_elapsed = time.perf_counter() - t_wait

            magic, status, op, _, _, resp_items = struct.unpack("<IBBHQI", resp)
            if magic != MAGIC or status != 0:
                raise IOError(f"NVMe in-storage topk failed, status {status}")

            t_recv = time.perf_counter()
            topk_results: List[Tuple[float, int, int]] = []
            if resp_items > 0:
                # Each item: block_id (uint32), score (float32), actual_tokens (uint32) = 12 bytes
                resp_bytes = self._recv_exact(resp_items * 12)
                for i in range(resp_items):
                    bid, score, act_tok = struct.unpack_from("<IfI", resp_bytes, i * 12)
                    topk_results.append((float(score), int(bid), int(act_tok)))
            recv_elapsed = time.perf_counter() - t_recv

            total_elapsed = pack_elapsed + send_elapsed + wait_elapsed + recv_elapsed
            self.read_ops += num_cands
            self.total_read_bytes += total_internal_k_bytes
            self.total_read_time_s += total_elapsed
            self.total_pack_time_s += pack_elapsed
            self.total_send_time_s += send_elapsed
            self.total_wait_time_s += wait_elapsed
            self.total_recv_time_s += recv_elapsed
            self.batch_count += 1
            self.batch_sizes.append(num_cands)

            self.topk_compute_ops += 1
            self.topk_internal_scanned_bytes += total_internal_k_bytes
            self.topk_query_transferred_bytes += len(q_bytes) + len(cands_payload)
            self.topk_metadata_transferred_bytes += resp_items * 12
            return topk_results

    def flush(self) -> None:
        """Flushes volatile write buffers on the NVMe device."""
        with self._lock:
            if self.sock is None:
                return
            req = struct.pack("<IBBHQI", MAGIC, OP_FLUSH, 0, 0, 0, 0)
            self.sock.sendall(req)
            self._recv_exact(HEADER_SIZE)

    def stop_qemu(self) -> None:
        """Shuts down the guest daemon cleanly and terminates the QEMU process."""
        with self._lock:
            if self.sock is not None:
                try:
                    req = struct.pack("<IBBHQI", MAGIC, OP_SHUTDOWN, 0, 0, 0, 0)
                    self.sock.sendall(req)
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None
            if self._qemu_proc is not None:
                try:
                    self._qemu_proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._qemu_proc.kill()
                self._qemu_proc = None

    def close(self) -> None:
        """Alias for stop_qemu()."""
        self.stop_qemu()

    def reset_stats(self) -> None:
        """Resets all metrics and protocol timers to zero."""
        self.total_read_bytes = 0
        self.total_write_bytes = 0
        self.read_ops = 0
        self.write_ops = 0
        self.total_read_time_s = 0.0
        self.total_write_time_s = 0.0
        self.total_pack_time_s = 0.0
        self.total_send_time_s = 0.0
        self.total_wait_time_s = 0.0
        self.batch_count = 0
        self.batch_sizes.clear()
        self.topk_compute_ops = 0
        self.topk_internal_scanned_bytes = 0
        self.topk_query_transferred_bytes = 0
        self.topk_metadata_transferred_bytes = 0

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns hardware-level NVMe driver I/O telemetry."""
        avg_r_lat_us = (self.total_read_time_s / self.read_ops * 1e6) if self.read_ops > 0 else 0.0
        avg_w_lat_us = (self.total_write_time_s / self.write_ops * 1e6) if self.write_ops > 0 else 0.0
        total_time_s = self.total_read_time_s + self.total_write_time_s
        total_bytes = self.total_read_bytes + self.total_write_bytes
        throughput_mbs = (total_bytes / (1024 * 1024) / total_time_s) if total_time_s > 0 else 0.0

        return {
            "classification": "VIRTUAL-DEVICE",
            "nvme_read_ops": self.read_ops,
            "nvme_write_ops": self.write_ops,
            "nvme_read_bytes": self.total_read_bytes,
            "nvme_write_bytes": self.total_write_bytes,
            "total_bytes": total_bytes,
            "avg_read_latency_us": round(avg_r_lat_us, 2),
            "avg_write_latency_us": round(avg_w_lat_us, 2),
            "total_storage_time_s": round(total_time_s, 4),
            "storage_throughput_mbs": round(throughput_mbs, 2),
            "total_pack_time_s": round(self.total_pack_time_s, 4),
            "total_send_time_s": round(self.total_send_time_s, 4),
            "total_wait_time_s": round(self.total_wait_time_s, 4),
            "total_recv_time_s": round(self.total_recv_time_s, 4),
            "batch_count": self.batch_count,
            "min_batch_size": min(self.batch_sizes) if self.batch_sizes else 0,
            "max_batch_size": max(self.batch_sizes) if self.batch_sizes else 0,
            "avg_batch_size": round(sum(self.batch_sizes) / len(self.batch_sizes), 2) if self.batch_sizes else 0.0,
            "topk_compute_ops": self.topk_compute_ops,
            "topk_internal_scanned_bytes": self.topk_internal_scanned_bytes,
            "topk_query_transferred_bytes": self.topk_query_transferred_bytes,
            "topk_metadata_transferred_bytes": self.topk_metadata_transferred_bytes,
        }

    def __del__(self):
        self.stop_qemu()
