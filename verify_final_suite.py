
import sys
import os
sys.path.append(os.getcwd())
from kalkulator_pkg.function_manager import find_function_from_data
import numpy as np

def run_test(name, data, params, expected_val):
    print(f"--- {name} ---")
    try:
        found, func, _, _ = find_function_from_data(data, params)
        print(f"Result: {func}")
        # Normalize spaces for comparison
        func_norm = func.replace(" ", "")
        expected_norm = expected_val.replace(" ", "")
        
        if expected_norm in func_norm:
            print("PASS")
        else:
            print(f"FAIL (Expected '{expected_val}' in '{func}')")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

tests = [
    ("LINE (x+10)", [([1], 11), ([2], 12), ([3], 13)], ['x'], "x+10"),
    ("QUAD (x^2+100)", [([1], 101), ([2], 104), ([3], 109)], ['x'], "x^2+100"),
    ("CAP (1-exp(-t))", [([0], 0), ([1], 0.63212), ([2], 0.86466), ([5], 0.99326)], ['t'], "1-exp(-t)"),
    ("GEO (sqrt(x*y))", [([1, 1], 1), ([4, 1], 2), ([4, 16], 8), ([9, 4], 6)], ['x', 'y'], "sqrt(x*y)"),
]

for t in tests:
    run_test(*t)
