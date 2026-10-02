# FEMU vs. QEMU Native Virtual NVMe: Architecture, Feasibility & Evaluation

**Author**: Person 2 (Storage & Systems Engineer)  
**Date**: 2026-10-02  
**Context**: AI-SSD V2 Virtual Storage Evaluation  

---

## 1. Background & Objectives

The primary mandate of Person 2 in AI-SSD V2 is to transition from the pure V1 Python analytical simulator into a verified 3-level storage evaluation methodology:

1. **Level 1**: Analytical / Cycle-Style Simulator (NAND geometry, contention physics, bus transfer).
2. **Level 2**: Trace-Driven Storage Evaluation (Replaying LLM KV access traces against FTL models, tracking queue depth, channel contention, and die utilization).
3. **Level 3**: Executable Virtual NVMe Storage Path (Operating an emulated NVMe controller under the Linux kernel NVMe driver, exercised via `fio` and `nvme-cli`).

This research document evaluates the practical implementation options for Level 3: **FEMU** vs. **QEMU Native Virtual NVMe**.

---

## 2. FEMU Investigation & Feasibility Analysis

### 2.1 What is FEMU?
FEMU (Fast, Exact, Flash Emulator) is an open-source NVMe SSD emulator developed by Huaicheng Li et al. (published in USENIX FAST 2018). It is implemented as a specialized patchset on top of QEMU (historically QEMU 2.9, 5.0, or 7.0).

FEMU implements an internal C-based flash translation layer (FTL) and delay emulation engine:
- **Black-box SSD (`bbssd`)**: Emulates page-level FTL with channel/chip/die delay queuing.
- **ZNS SSD (`zns`)**: Emulates Zoned Namespaces.
- **OC-SSD (`ocssd`)**: Emulates Open-Channel SSD 2.0 interface.
- **NoSSD (`nossd`)**: Pure NVMe pass-through with minimal delay emulation.

### 2.2 Feasibility & Constraints on the Current Host Machine
The evaluation environment is an AWS EC2 instance with:
- **8 vCPUs** (Shared across P1, P2, P3, and OS).
- **61 GiB RAM**.
- Target resource allocation for P2: **~2 vCPUs**.

#### Critical Limitations of In-Tree FEMU Compilation:
1. **Compilation Footprint & Host Disruption**:
   - Building FEMU requires cloning the full QEMU repository (~1.2 GB), configuring with Meson/Ninja, and compiling ~4,000 C/C++ source files.
   - A full QEMU compilation consumes 6–8 cores at 100% CPU for 20–35 minutes, generating 8–15 GB of build artifacts.
   - On a shared 8-vCPU instance, this creates severe resource contention, risking timeouts or crashes in P1 (LLM training/inference) and P3 (orchestration).
2. **Custom FTL Logic Inflexibility**:
   - FEMU's internal FTL is written in low-level C (`femu/bbssd/ftl.c`) and implements fixed sequential/greedy striping.
   - Injecting AI-SSD's tensor-aware placement algorithm into FEMU requires modifying QEMU C internals, risking emulator instability and kernel memory corruption inside the guest.
3. **Reproducibility**:
   - Compiling out-of-tree QEMU forks creates brittle toolchain dependencies across different environments.

---

## 3. The QEMU Native Virtual NVMe Path (Adopted Level 3 Path)

### 3.1 Architecture
The host environment already provides:
- **QEMU 6.2.0** (`/usr/bin/qemu-system-x86_64`) pre-installed.
- **KVM Acceleration** (`/dev/kvm` accessible with hardware virtualization).
- **In-Kernel Linux NVMe Driver** (`CONFIG_BLK_DEV_NVME=y` built into host kernel `6.5.0-1020-aws`).
- **Standard Storage Tooling**: `nvme-cli` (v1.16) and `fio` (v3.28).

### 3.2 Implemented Level 3 Storage Pipeline
To achieve a completely reproducible, zero-overhead virtual NVMe execution path:
1. **Virtual Controller**: QEMU PCI NVMe 1.4 controller (`-device nvme`).
   - Configured with `num_queues=8` (or `max_ioqpairs=8`) to model 8 independent host I/O queue pairs matching the 8 NAND channels.
   - Formatted with 4096-byte logical and physical block size (`logical_block_size=4096, physical_block_size=4096`), matching the 4 KB KV block / flash page unit.
   - Serial number: `v2-ai-ssd-001`.
2. **Storage Backend**: A sparse raw disk image (`/opt/ai-ssd-v2/images/v2_nvme.raw`, 1.0 GiB logical, 0 bytes initial physical).
3. **Lightweight Micro-Kernel Boot Environment**:
   - Direct kernel boot (`-kernel /opt/ai-ssd-v2/images/vmlinuz`) with hardware KVM acceleration.
   - Tailored micro-initramfs (`/opt/ai-ssd-v2/images/initramfs.cpio.gz`, 18 MB) containing static BusyBox, `nvme-cli`, and `fio` with dynamic libraries resolved.
   - Boots in **< 1.5 seconds**, executes automated device discovery, controller validation, namespace query, direct I/O read/write tests, and multi-depth `fio` benchmarks, then powers off cleanly.

### 3.3 Verification & Smoke Test Results (Measured 2026-10-02)
- **Controller Identified**: `QEMU NVMe Ctrl`, Serial: `v2-ai-ssd-001`, Firmware: `1.0`.
- **Namespace Verified**: `/dev/nvme0n1`, Size: `0x40000` blocks (1.0 GiB), `lbads: 12` (4096 B per LBA).
- **Direct I/O Sequential Write (4KB blocks)**: Passed (100 blocks written cleanly).
- **FIO Random Read (4KB, QD=8, libaio)**:
  - Throughput: **92.1 MiB/s** (96.6 MB/s).
  - IOPS: **23.6k IOPS**.
  - Average Latency: **338.57 $\mu$s** (min: 73 $\mu$s, 99th percentile: 379 $\mu$s).
- **FIO Random Write (4KB, QD=8, libaio)**:
  - Throughput: **98.0 MiB/s** (103 MB/s).
  - IOPS: **25.1k IOPS**.
  - Average Latency: **317.94 $\mu$s** (min: 64 $\mu$s, 99th percentile: 379 $\mu$s).

---

## 4. Key Limitations & Distinction Between Simulation Levels

As mandated by the V2 experimental protocol:
- **Level 3 (Virtual NVMe / QEMU)** proves full operating system and driver compatibility, DMA queue dispatch, and block-level I/O through the Linux kernel storage stack. However, the host backing file stores data in host OS page cache/file system blocks; it does *not* emulate internal NAND channel contention or $t_R$/$t_{\text{PROG}}$ physics.
- **Level 1 (Analytical Simulator)** and **Level 2 (Trace Replayer)** provide the cycle-accurate contention model for 8 channels, 4 dies/channel, and physical bus serialization.
- Level 3 results must **never** be conflated with Level 1/2 results. Both paths are maintained and reported with clear separation.