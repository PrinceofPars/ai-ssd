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
        self.total_read_bytes: int = 0
        self.total_write_bytes: int = 0
        self.read_ops: int = 0
        self.write_ops: int = 0
        self.total_read_time_s: float = 0.0
        self.total_write_time_s: float = 0.0

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
        if self.sock is None:
            self.connect()
        req = struct.pack("<IBBHQI", MAGIC, OP_READ, 0, 0, offset, length)
        t0 = time.perf_counter()
        self.sock.sendall(req)
        resp = self._recv_exact(HEADER_SIZE)
        magic, status, op, _, resp_offset, data_len = struct.unpack("<IBBHQI", resp)
        if magic != MAGIC or status != 0:
            raise IOError(f"NVMe read failed at offset {offset}, length {length}, status {status}")

        data = self._recv_exact(data_len)
        elapsed = time.perf_counter() - t0

        self.read_ops += 1
        self.total_read_bytes += data_len
        self.total_read_time_s += elapsed
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
        if self.sock is None:
            self.connect()

        num_items = len(requests)
        req_hdr = struct.pack("<IBBHQI", MAGIC, OP_BATCH_READ, 0, 0, 0, num_items)
        items_payload = bytearray()
        total_req_bytes = 0
        for offset, length, bid in requests:
            items_payload.extend(struct.pack("<QII", offset, length, bid))
            total_req_bytes += length

        t0 = time.perf_counter()
        self.sock.sendall(req_hdr + items_payload)
        resp = self._recv_exact(HEADER_SIZE)
        magic, status, op, _, _, resp_items = struct.unpack("<IBBHQI", resp)
        if magic != MAGIC or status != 0:
            raise IOError(f"NVMe batch read failed, status {status}")

        results: Dict[int, bytes] = {}
        for _ in range(resp_items):
            item_hdr = self._recv_exact(9)
            item_status, bid, item_len = struct.unpack("<BII", item_hdr)
            if item_status != 0:
                raise IOError(f"NVMe batch read item failed for block {bid}")
            data = self._recv_exact(item_len)
            results[bid] = data

        elapsed = time.perf_counter() - t0
        self.read_ops += num_items
        self.total_read_bytes += total_req_bytes
        self.total_read_time_s += elapsed
        return results

    def flush(self) -> None:
        """Flushes volatile write buffers on the NVMe device."""
        if self.sock is None:
            return
        req = struct.pack("<IBBHQI", MAGIC, OP_FLUSH, 0, 0, 0, 0)
        self.sock.sendall(req)
        self._recv_exact(HEADER_SIZE)

    def stop_qemu(self) -> None:
        """Shuts down the guest daemon cleanly and terminates the QEMU process."""
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
        }

    def __del__(self):
        self.stop_qemu()
