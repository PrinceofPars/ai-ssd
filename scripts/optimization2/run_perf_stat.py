#!/usr/bin/env python3
"""
AI-SSD V2 — Run single rep for perf stat measurement using context_scaling_worker.py.
"""

import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON_BIN = "/home/ubuntu/ai-ssd-p1/.venv/bin/python3"
WORKER_SCRIPT = str(PROJECT_ROOT / "scripts" / "context_scaling_worker.py")
OUT_FILE = "/tmp/perf_stat_rep.json"


def main():
    if os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)

    cmd = [
        PYTHON_BIN, WORKER_SCRIPT,
        "--mode", "aissd",
        "--storage-mode", "nvme_qemu",
        "--enable-computational-storage",
        "--disable-prefetch",
        "--enable-async-pipeline",
        "--context", "4096",
        "--decode", "16",
        "--seed", "42",
        "--rep", "0",
        "--output-json", OUT_FILE,
    ]

    print("Executing single rep for perf stat...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    print("Repetition completed successfully.")


if __name__ == "__main__":
    main()
