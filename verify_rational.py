
import sys
import os
sys.path.append(os.getcwd())
try:
    from kalkulator_pkg.function_manager import find_function_from_data
    import numpy as np
except ImportError:
    # If installed in site-packages
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
    ("D1 (1/(x+1))", [([0], 1), ([1], 0.5), ([3], 0.25), ([9], 0.1)], ['x'], "1/(x+1)"),
    ("D2 (1-1/(x+1))", [([0], 0), ([1], 0.5), ([3], 0.75), ([9], 0.9)], ['x'], "1-1/(x+1)"),
    ("RATIO (x/z)", [([10, 2], 5), ([10, 5], 2), ([100, 10], 10), ([9, 3], 3)], ['x', 'z'], "x/z"),
    ("POT (1/sqrt(x^2+1))", [([0], 1), ([1], 0.707106), ([2], 0.447213), ([3], 0.316227)], ['x'], "1/sqrt(x^2+1)")
]

# Note: D2 might return "x/(x+1)" which is equivalent.
# RATIO might return "x*z^-1".

for t in tests:
    run_test(*t)
