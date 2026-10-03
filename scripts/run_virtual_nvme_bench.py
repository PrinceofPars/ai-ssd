"""
Virtual NVMe Device Evaluation Harness (Level 3 - Person 2).

Executes QEMU virtual NVMe device under Linux KVM with in-kernel NVMe driver.
Benchmarks:
  1. Sequential Read (64 KiB, QD=4)
  2. Sequential Write (64 KiB, QD=4)
  3. Random Read (4 KiB, QD=8, libaio)
  4. Random Write (4 KiB, QD=8, libaio)
  5. Random Read (8 KiB, QD=8, libaio - matching combined KV block page)

Classification: VIRTUAL-DEVICE
Do not imply physical SSD hardware/firmware behavior.
"""

import sys
import os
import json
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_IMG = "/opt/ai-ssd-v2/images/v2_nvme.raw"
INITRD = "/opt/ai-ssd-v2/images/initramfs.cpio.gz"
KERNEL = "/opt/ai-ssd-v2/images/vmlinuz"
LOG_FILE = "/opt/ai-ssd-v2/logs/virtual_nvme_phase3_benchmarks.log"


def run_qemu_benchmarks():
    print("================================================================================")
    print("       AI-SSD V2 Virtual NVMe Benchmark (Level 3 - VIRTUAL-DEVICE)             ")
    print("================================================================================")

    # Rebuild initramfs with benchmark fio scripts
    print("[1/3] Building updated micro-initramfs...")
    subprocess.run(["python3", str(PROJECT_ROOT / "scripts" / "build_initramfs.py")], check=True)

    # Launch QEMU
    print("[2/3] Booting QEMU NVMe Guest with KVM...")
    qemu_cmd = [
        "qemu-system-x86_64",
        "-enable-kvm",
        "-cpu", "host",
        "-m", "2048",
        "-smp", "2",
        "-no-reboot",
        "-kernel", KERNEL,
        "-initrd", INITRD,
        "-drive", f"file={RAW_IMG},format=raw,if=none,id=nvme0",
        "-device", "nvme,drive=nvme0,serial=v2-ai-ssd-001,num_queues=8,logical_block_size=4096,physical_block_size=4096",
        "-nographic",
        "-append", "console=ttyS0 panic=-1 quiet loglevel=3",
    ]

    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as lf:
        proc = subprocess.run(qemu_cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)

    print(f"[QEMU] Guest exited with returncode {proc.returncode}.")

    # Read and parse log
    print("[3/3] Parsing FIO benchmark results...")
    with open(LOG_FILE, "r", encoding="utf-8") as lf:
        log_content = lf.read()

    # Extract JSON blocks
    pattern = re.compile(r"=== FIO_START: (?P<name>\w+) ===\s*(?P<json_data>\{.*?\})\s*=== FIO_END: (?P=name) ===", re.DOTALL)
    matches = pattern.findall(log_content)

    results = {}
    print("\n--- Virtual NVMe Benchmark Results (Classification: VIRTUAL-DEVICE) ---")
    print(f"{'Benchmark':22s} | {'IOPS':10s} | {'Bandwidth':14s} | {'Avg Latency':12s} | {'99th Latency':12s}")
    print("-" * 80)

    for name, json_str in matches:
        try:
            data = json.loads(json_str)
            job = data["jobs"][0]
            is_read = "read" in name
            io_stat = job["read"] if is_read else job["write"]

            iops = io_stat["iops"]
            bw_bytes = io_stat["bw_bytes"]
            bw_mbs = bw_bytes / (1024 * 1024)

            # Latency (convert to microseconds)
            lat_ns = io_stat["lat_ns"]["mean"]
            lat_us = lat_ns / 1000.0
            p99_ns = io_stat["clat_ns"]["percentile"].get("99.000000", 0)
            p99_us = p99_ns / 1000.0

            results[name] = {
                "benchmark": name,
                "classification": "VIRTUAL-DEVICE",
                "io_type": "read" if is_read else "write",
                "block_size": "64k" if "64k" in name else ("8k" if "8k" in name else "4k"),
                "iops": round(iops, 1),
                "bandwidth_mbs": round(bw_mbs, 2),
                "avg_latency_us": round(lat_us, 2),
                "p99_latency_us": round(p99_us, 2),
                "cpu_usr_pct": job["usr_cpu"],
                "cpu_sys_pct": job["sys_cpu"],
            }
            print(f"{name:22s} | {iops:9.1f}  | {bw_mbs:10.2f} MB/s | {lat_us:9.2f} us | {p99_us:9.2f} us")
        except Exception as e:
            print(f"Failed to parse {name}: {e}")

    # Save results
    p2_results_dir = Path("/opt/ai-ssd-v2/results/p2")
    p2_results_dir.mkdir(parents=True, exist_ok=True)
    out_file = p2_results_dir / "virtual_nvme_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    local_out = PROJECT_ROOT / "results" / "raw" / "virtual_nvme_benchmarks.json"
    local_out.parent.mkdir(parents=True, exist_ok=True)
    with open(local_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("-" * 80)
    print(f"[SUCCESS] Saved Virtual NVMe benchmark results to:")
    print(f"  - {out_file}")
    print(f"  - {local_out}")
    print("================================================================================")
    return results


if __name__ == "__main__":
    run_qemu_benchmarks()