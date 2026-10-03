"""
Environment Provenance & System Metadata Capture for V2 Experiments.
"""

from __future__ import annotations
import os
import sys
import socket
import platform
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, Any
from datetime import datetime


@dataclass
class EnvironmentProvenance:
    git_commit: str
    git_branch: str
    hostname: str
    cpu_model: str
    cpu_count: int
    ram_total_gb: float
    kernel: str
    os_name: str
    python_version: str
    dependency_versions: Dict[str, str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def capture(cls) -> EnvironmentProvenance:
        # Git commit & branch
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            branch = subprocess.check_output(
                ["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            commit = "unknown"
            branch = "unknown"

        # CPU model
        cpu_model = "unknown"
        if os.path.exists("/proc/cpuinfo"):
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_model = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass

        # RAM
        ram_gb = 0.0
        if os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "MemTotal" in line:
                            kb = int(line.split()[1])
                            ram_gb = round(kb / (1024 * 1024), 2)
                            break
            except Exception:
                pass

        # Dependencies
        deps = {}
        for mod_name in ["numpy", "pydantic", "pytest", "pandas", "yaml"]:
            try:
                mod = __import__(mod_name)
                deps[mod_name] = getattr(mod, "__version__", "installed")
            except ImportError:
                deps[mod_name] = "not_installed"

        return cls(
            git_commit=commit,
            git_branch=branch,
            hostname=socket.gethostname(),
            cpu_model=cpu_model,
            cpu_count=os.cpu_count() or 1,
            ram_total_gb=ram_gb,
            kernel=platform.release(),
            os_name=platform.platform(),
            python_version=sys.version.split()[0],
            dependency_versions=deps,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
