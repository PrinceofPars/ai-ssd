# Practical Linux I/O Mechanisms for AI-SSD V2 (P3 Research)

**Date**: 2026-10-02
**Agent**: P3 (System Integration)

## 1. Overview of Evaluated I/O Mechanisms

To connect KV offloading with real storage / NVMe / FEMU, the upper storage layer requires low overhead, high concurrency, and fine-grained latency profiling without OS buffering artifacts.

| Mechanism | Kernel Req | Pros | Cons | Verdict for V2 |
|---|---|---|---|---|
| **Synchronous POSIX (`read`/`pread`)** | Any Linux | Deterministic, simple, universal, zero dependency | Blocks calling thread, cannot overlap I/O easily without threads | Baseline engine for deterministic analytical timing |
| **POSIX Direct I/O (`O_DIRECT`)** | Any Linux | Bypasses Linux page cache, forces actual storage I/O, realistic latency | Alignment requirements (typically 512B or 4096B boundaries) | **Mandatory** for real NVMe/file bench to prevent page-cache masking |
| **POSIX AIO (`libaio`)** | Linux 2.6+ | Kernel-level async for block devices | Only truly async with `O_DIRECT`, awkward signal/eventfd interface, legacy | Usable, but surpassed by io_uring |
| **Threadpool Asynchronous (`concurrent.futures`)** | Any Python | Portable, works across all backends, allows clean task pipelining | Thread context switch overhead (~2-5 us) | **Primary user-space async** for portable mock & analytical backends |
| **`io_uring` (Linux 5.1+)** | Linux 5.1+ | Zero-copy submission/completion queues, lowest system call overhead, polling mode (`IORING_SETUP_SQPOLL`) | Requires external C-binding or Python wrapper (`liburing`), kernel version sensitivity | Ideal for future high-speed FEMU NVMe raw character/block device |

## 2. Selection Rationale for V2 Implementation

1. **Analytical / Mock Mode**:
   - Uses `StorageBackend` interface with simulated hardware physics (NAND channel contention, bus transfer delay, queue wait).
   - Python async wrapper via `concurrent.futures.ThreadPoolExecutor` provides non-blocking futures with deterministic simulated time stamps.

2. **File / RAM-Disk / Block Storage Mode**:
   - `FileStorageBackend` implements both buffered and `O_DIRECT` modes.
   - Using `O_DIRECT` ensures that read/write requests measure true disk I/O rather than RAM cache hits.

3. **FEMU / NVMe Device Mode**:
   - Direct I/O against `/dev/nvmeXnY` or file-backed image.
   - Decoupled via `StorageBackend` so switching to `io_uring` requires only a drop-in backend implementation without touching prefetch or experiment runner logic.
