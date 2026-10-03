import os
import shutil
import subprocess

INITRAMFS_DIR = "/tmp/nvme_initramfs"
OUTPUT_CPIO = "/opt/ai-ssd-v2/images/initramfs.cpio.gz"

if os.path.exists(INITRAMFS_DIR):
    shutil.rmtree(INITRAMFS_DIR)

for d in ["bin", "sbin", "usr/bin", "usr/sbin", "lib", "lib64", "lib/x86_64-linux-gnu", "usr/lib/x86_64-linux-gnu", "proc", "sys", "dev", "etc", "mnt", "root"]:
    os.makedirs(os.path.join(INITRAMFS_DIR, d), exist_ok=True)

# Install static busybox
shutil.copy2("/usr/bin/busybox", os.path.join(INITRAMFS_DIR, "bin/busybox"))
shutil.copy2("/usr/bin/busybox", os.path.join(INITRAMFS_DIR, "usr/bin/busybox"))

tools = subprocess.check_output(["/usr/bin/busybox", "--list"], text=True).splitlines()
for tool in tools:
    tool = tool.strip()
    if tool:
        p1 = os.path.join(INITRAMFS_DIR, "bin", tool)
        if not os.path.exists(p1):
            os.symlink("busybox", p1)
        p2 = os.path.join(INITRAMFS_DIR, "usr/bin", tool)
        if not os.path.exists(p2):
            os.symlink("busybox", p2)

def copy_with_libs(bin_path):
    dest_path = os.path.join(INITRAMFS_DIR, bin_path.lstrip("/"))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(bin_path, dest_path)
    
    out = subprocess.check_output(["ldd", bin_path], text=True)
    for line in out.splitlines():
        line = line.strip()
        if "=>" in line:
            parts = line.split("=>")
            src = parts[1].split()[0]
            if os.path.exists(src):
                dest = os.path.join(INITRAMFS_DIR, src.lstrip("/"))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if not os.path.exists(dest):
                    shutil.copy2(src, dest)
        elif line.startswith("/"):
            src = line.split()[0]
            if os.path.exists(src):
                dest = os.path.join(INITRAMFS_DIR, src.lstrip("/"))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if not os.path.exists(dest):
                    shutil.copy2(src, dest)

copy_with_libs("/usr/sbin/nvme")
copy_with_libs("/usr/bin/fio")

# Compile static nvme_guest_daemon
daemon_src = os.path.join(os.path.dirname(__file__), "nvme_guest_daemon.c")
daemon_dest = os.path.join(INITRAMFS_DIR, "usr/bin/nvme_guest_daemon")
subprocess.run(["gcc", "-O3", "-mavx2", "-mfma", "-static", daemon_src, "-o", daemon_dest], check=True)

init_script = """#!/bin/sh
mount -t proc none /proc
mount -t sysfs none /sys
mount -t devtmpfs none /dev

echo "=================================================="
echo "AI-SSD V2 Virtual NVMe Guest Environment Booted"
echo "Classification: VIRTUAL-DEVICE"
echo "Kernel: $(uname -r)"
echo "=================================================="

for i in 1 2 3 4 5; do
    if [ -e /dev/nvme0n1 ]; then
        break
    fi
    sleep 1
done

if [ ! -e /dev/nvme0n1 ]; then
    echo "[ERROR] /dev/nvme0n1 not found!"
    poweroff -f
fi

echo "[SUCCESS] Virtual NVMe block device found: /dev/nvme0n1"
echo ""
echo "--- NVMe Controller Info ---"
nvme id-ctrl /dev/nvme0 2>/dev/null | grep -E "(sn|mn|fr|mdts|nn|sqes|cqes)"
echo ""
echo "--- NVMe Namespace Info ---"
nvme id-ns /dev/nvme0n1 2>/dev/null | grep -E "(nsze|ncap|nuse|lbaf)"
echo ""

CMDLINE=$(cat /proc/cmdline)

if echo "$CMDLINE" | grep -q "bench=1"; then
    echo "=== Running FIO Benchmarks ==="
    echo "=== FIO_START: seq_read_64k ==="
    fio --name=seq_read_64k --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=64k --ioengine=libaio --iodepth=4 --runtime=3 --time_based --group_reporting --output-format=json
    echo "=== FIO_END: seq_read_64k ==="

    echo "=== FIO_START: seq_write_64k ==="
    fio --name=seq_write_64k --filename=/dev/nvme0n1 --direct=1 --rw=write --bs=64k --ioengine=libaio --iodepth=4 --runtime=3 --time_based --group_reporting --output-format=json
    echo "=== FIO_END: seq_write_64k ==="

    echo "=== FIO_START: rand_read_4k ==="
    fio --name=rand_read_4k --filename=/dev/nvme0n1 --direct=1 --rw=randread --bs=4k --ioengine=libaio --iodepth=8 --runtime=3 --time_based --group_reporting --output-format=json
    echo "=== FIO_END: rand_read_4k ==="

    echo "=== FIO_START: rand_write_4k ==="
    fio --name=rand_write_4k --filename=/dev/nvme0n1 --direct=1 --rw=randwrite --bs=4k --ioengine=libaio --iodepth=8 --runtime=3 --time_based --group_reporting --output-format=json
    echo "=== FIO_END: rand_write_4k ==="

    echo "=== FIO_START: rand_read_8k ==="
    fio --name=rand_read_8k --filename=/dev/nvme0n1 --direct=1 --rw=randread --bs=8k --ioengine=libaio --iodepth=8 --runtime=3 --time_based --group_reporting --output-format=json
    echo "=== FIO_END: rand_read_8k ==="

    echo "[VIRTUAL NVME BENCHMARKS COMPLETE]"
fi

if echo "$CMDLINE" | grep -q "daemon=1"; then
    echo "=== Starting NVMe Guest Daemon on port 9999 ==="
    ifconfig lo 127.0.0.1 up 2>/dev/null
    ifconfig eth0 10.0.2.15 netmask 255.255.255.0 up 2>/dev/null
    route add default gw 10.0.2.2 eth0 2>/dev/null
    echo "[READY] NVMe Guest Daemon listening on port 9999 for live I/O"
    /usr/bin/nvme_guest_daemon /dev/nvme0n1 9999
fi

echo "=================================================="
echo "Halting virtual machine cleanly..."
poweroff -f
"""

init_path = os.path.join(INITRAMFS_DIR, "init")
with open(init_path, "w") as f:
    f.write(init_script)
os.chmod(init_path, 0o755)

cmd = f"cd {INITRAMFS_DIR} && find . -print0 | cpio --null -ov --format=newc | gzip -9 > {OUTPUT_CPIO}"
subprocess.run(cmd, shell=True, check=True)
print(f"Built {OUTPUT_CPIO}, size: {os.path.getsize(OUTPUT_CPIO)} bytes")