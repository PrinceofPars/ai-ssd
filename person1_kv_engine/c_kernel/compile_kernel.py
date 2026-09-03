"""Automated build script for the native in-storage attention C kernel.

Supports 64-bit MSVC BuildTools (freestanding mode) and GCC.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def compile_c_kernel() -> Path:
    kernel_dir = Path(__file__).parent.resolve()
    c_source = kernel_dir / "instorage_attention.c"
    dll_path = kernel_dir / "instorage_attention.dll"

    if not c_source.exists():
        raise FileNotFoundError(f"Source file {c_source} not found.")

    # Check for MSVC BuildTools vcvarsall.bat
    vcvars_paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
    ]

    for vcvars in vcvars_paths:
        if os.path.exists(vcvars):
            cmd = f'cmd.exe /c "call "{vcvars}" x64 && cl /O2 /LD /GS- "{c_source}" /Fe:"{dll_path}" /link /NOENTRY /NODEFAULTLIB"'
            print(f"Compiling with MSVC x64: {cmd}")
            res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and dll_path.exists():
                print(f"[SUCCESS] Native C kernel compiled to {dll_path}")
                return dll_path
            else:
                print(f"[WARNING] MSVC build failed: {res.stderr}")

    # Fallback to gcc
    gcc_path = shutil.which("gcc") or (r"C:\MinGW\bin\gcc.exe" if os.path.exists(r"C:\MinGW\bin\gcc.exe") else None)
    if gcc_path:
        cmd = [gcc_path, "-O3", "-shared", "-fPIC", "-std=c99", str(c_source), "-o", str(dll_path)]
        print(f"Compiling with GCC: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and dll_path.exists():
            print(f"[SUCCESS] Native C kernel compiled to {dll_path}")
            return dll_path

    print("[INFO] Using Python NumPy fallback engine.")
    return dll_path


if __name__ == "__main__":
    compile_c_kernel()
