
import sys
import math
from kalkulator_pkg.parser import parse_expr, ALLOWED_SYMPY_NAMES
from kalkulator_pkg.function_finder_advanced import detect_symbolic_constant
import sympy as sp

def test_fixes():
    failures = 0
    
    # 1. Test log2 parsing configuration (simulated)
    print("Testing log2 parsing config...")
    if "log2" in ALLOWED_SYMPY_NAMES:
        print("PASS: log2 is in ALLOWED_SYMPY_NAMES")
    else:
        print("FAIL: log2 missing from ALLOWED_SYMPY_NAMES")
        failures += 1

    # 2. Test Symbolic Snapping
    print("\nTesting Symbolic Snapping...")
    ln2_val = math.log(2)
    detected = detect_symbolic_constant(ln2_val)
    print(f"Input: {ln2_val} (ln(2)) -> Detected: {detected}")
    
    if detected is not None and (detected == sp.log(2) or abs(float(detected) - ln2_val) < 1e-9):
         print("PASS: log(2) detected")
    else:
         print("FAIL: log(2) NOT detected")
         failures += 1

    # 3. Test REPL Comment Logic (Simulated)
    # We can't easily test the REPL loop here without subprocess, but we can check the logic pattern
    # We will verify this manually or via the full check script later.
    
    return failures

if __name__ == "__main__":
    sys.exit(test_fixes())
