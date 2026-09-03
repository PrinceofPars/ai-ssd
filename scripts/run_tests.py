"""
Zero-dependency test runner.
Discovers and executes all test functions across common and subsystem packages.
"""

import sys
import inspect
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common.tests.test_schemas as t_common
import person1_kv_engine.tests.test_p1_mock as t_p1
import person2_ssd.tests.test_p2_mock as t_p2
import person3_system.tests.test_p3_mock_pipeline as t_p3


def run_module_tests(module, module_name):
    print(f"[*] Testing {module_name}...")
    functions = [
        obj for name, obj in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("test_")
    ]
    passed = 0
    failed = 0
    for func in functions:
        try:
            func()
            print(f"    [PASS] {func.__name__}")
            passed += 1
        except Exception as e:
            print(f"    [FAIL] {func.__name__}: {e}")
            failed += 1
    return passed, failed


def main():
    print("==================================================")
    print("       AI-SSD Zero-Dependency Test Suite          ")
    print("==================================================")
    modules = [
        (t_common, "Common Schemas & Data Contracts"),
        (t_p1, "Person 1: KV Engine & MockSSD"),
        (t_p2, "Person 2: SSD / FTL & MockKVEngine"),
        (t_p3, "Person 3: Unified API & Mock Pipeline"),
    ]

    total_passed = 0
    total_failed = 0
    for mod, name in modules:
        p, f = run_module_tests(mod, name)
        total_passed += p
        total_failed += f

    print("==================================================")
    print(f"Results: {total_passed} Passed, {total_failed} Failed")
    print("==================================================")
    if total_failed > 0:
        sys.exit(1)
    else:
        print("[SUCCESS] All mock and contract tests passed cleanly!\n")


if __name__ == "__main__":
    main()
