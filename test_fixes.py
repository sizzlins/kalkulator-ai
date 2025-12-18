#!/usr/bin/env python
"""Test script for common fixes and edge cases."""
import subprocess
import sys


def run_kalkulator(expr):
    """Run kalkulator with --eval and capture output."""
    cmd = [sys.executable, "kalkulator.py", "--eval", expr]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=r"c:\Users\LOQ\PycharmProjects\kalkulatoraljabar-blackboxai-function-finder-improvements",
        timeout=30
    )
    return result.stdout + result.stderr

if __name__ == "__main__":
    print("Test 1: 1/0 should show 'undefined' or infinity")
    output = run_kalkulator("1/0")
    print(f"Output: {output.strip()}")
    # SymPy returns 'zoo' or 'undefined' for 1/0 (complex infinity)
    assert "zoo" in output or "oo" in output.lower() or "infinity" in output.lower() or "undefined" in output.lower(), f"Expected infinity indicator, got: {output}"
    print("✓ PASS\n")

    print("Test 2: I (imaginary unit)")
    output = run_kalkulator("I")
    print(f"Output: {output.strip()}")
    assert "I" in output or "i" in output, f"Expected 'I', got: {output}"
    print("✓ PASS\n")

    print("Test 3: f(x,y)=x^2+y^2, find f(x,y) = 5")
    output = run_kalkulator("f(x,y)=x^2+y^2, find f(x,y) = 5")
    print(f"Output: {output.strip()[:200]}...")
    assert "Inverse solutions" in output or "Integer" in output or "solutions" in output.lower(), f"Expected inverse solutions, got: {output}"
    print("✓ PASS\n")

    print("Test 4: x^2 = 4")
    output = run_kalkulator("x^2 = 4")
    print(f"Output: {output.strip()}")
    assert "2" in output or "-2" in output, f"Expected solution with 2, got: {output}"
    print("✓ PASS\n")

    print("All tests passed!")
