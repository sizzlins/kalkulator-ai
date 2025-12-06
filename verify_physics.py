
import sys
import os
sys.path.append(os.getcwd())
try:
    from kalkulator_pkg.function_manager import find_function_from_data
    import numpy as np
except ImportError:
    from kalkulator_pkg.function_manager import find_function_from_data
    import numpy as np

def run_test(name, data, params, expected_val):
    print(f"--- {name} ---")
    try:
        found, func, _, _ = find_function_from_data(data, params)
        print(f"Result: {func}")
        func_norm = func.replace(" ", "")
        expected_norm = expected_val.replace(" ", "")
        
        # Check for key features
        if "sqrt" in expected_norm and "sqrt" not in func_norm:
             print(f"FAIL (Missing sqrt)")
        elif "/" in expected_norm and "/" not in func_norm and "^-1" not in func_norm:
             print(f"FAIL (Missing division)")
        elif expected_norm in func_norm or ("m1*m2" in func_norm and "r^2" in func_norm):
             print("PASS")
        else:
             # Relaxed check for Gravity (allow reordering)
             print(f"FAIL (Expected '{expected_val}', Got '{func}')")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

tests = [
    ("Lor (1/sqrt(1-v^2))", [([0], 1.0), ([0.5], 1.1547005), ([0.8], 1.666666), ([0.9], 2.294157)], ['v'], "1/sqrt(1-v^2)"),
    ("Grav (6.67*m1*m2/r^2)", [([10, 20, 2], 333.5), ([5, 5, 5], 6.67), ([100, 1, 10], 6.67), ([2, 2, 1], 26.68)], ['m1', 'm2', 'r'], "m1*m2/r^2")
]

for t in tests:
    run_test(*t)
