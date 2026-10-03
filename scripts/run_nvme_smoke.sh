#!/bin/bash
set -e

RAW_IMG="/opt/ai-ssd-v2/images/v2_nvme.raw"
INITRD="/opt/ai-ssd-v2/images/initramfs.cpio.gz"
KERNEL="/opt/ai-ssd-v2/images/vmlinuz"
LOG_FILE="/opt/ai-ssd-v2/logs/virtual_nvme_smoke_test.log"

echo "=== Starting Virtual NVMe Smoke Test ===" | tee "$LOG_FILE"
echo "Host Kernel: $(uname -a)" | tee -a "$LOG_FILE"
echo "QEMU Version: $(qemu-system-x86_64 --version | head -n 1)" | tee -a "$LOG_FILE"
echo "Storage Backend Image: $RAW_IMG ($(ls -lh $RAW_IMG | awk '{print $5}'))" | tee -a "$LOG_FILE"
echo "Initramfs: $INITRD ($(ls -lh $INITRD | awk '{print $5}'))" | tee -a "$LOG_FILE"
echo "Kernel: $KERNEL ($(ls -lh $KERNEL | awk '{print $5}'))" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Launch QEMU with KVM, 2 vCPUs, 2GB RAM, virtual NVMe device (num_queues=8, 4KB LBA)
qemu-system-x86_64 \
    -enable-kvm \
    -cpu host \
    -m 2048 \
    -smp 2 \
    -no-reboot \
    -kernel "$KERNEL" \
    -initrd "$INITRD" \
    -drive file="$RAW_IMG",format=raw,if=none,id=nvme0 \
    -device nvme,drive=nvme0,serial=v2-ai-ssd-001,num_queues=8,logical_block_size=4096,physical_block_size=4096 \
    -nographic \
    -append "console=ttyS0 panic=-1 quiet loglevel=3 bench=1" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Virtual NVMe Smoke Test Finished Cleanly ===" | tee -a "$LOG_FILE"