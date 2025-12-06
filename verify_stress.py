
import sys
import os
sys.path.append(os.getcwd())
try:
    from kalkulator_pkg.function_manager import find_function_from_data
    import numpy as np
except ImportError:
    from kalkulator_pkg.function_manager import find_function_from_data
    import numpy as np

def run_test(name, data, params, expected_vals):
    print(f"--- {name} ---")
    try:
        found, func, _, _ = find_function_from_data(data, params)
        print(f"Result: {func}")
        func_norm = func.replace(" ", "")
        passed = False
        for expected in expected_vals:
            expected_norm = expected.replace(" ", "")
            if expected_norm in func_norm:
                passed = True
                break
        
        if passed:
            print("PASS")
        else:
            print(f"FAIL (Expected one of {expected_vals})")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

tests = [
    # Hyp: sqrt(a^2 + b^2)
    ("Hyp (sqrt(a^2+b^2))", [([10, 0], 10), ([0, 10], 10), ([10, 10], 14.1421356), ([3, 4], 5)], ['a', 'b'], ["sqrt(a^2+b^2)", "sqrt(b^2+a^2)"]),
    
    # Field: q/r^2
    ("Field (q/r^2)", [([1, 1], 1), ([1, 0.1], 100), ([100, 1], 100), ([4, 2], 1)], ['q', 'r'], ["q/r^2"]),
    
    # Diff: a^2 - b^2
    ("Diff (a^2-b^2)", [([2, 1], 3), ([3, 2], 5), ([4, 1], 15), ([5, 5], 0)], ['a', 'b'], ["a^2-b^2"]),
    
    # Res: a*b/(a+b)
    ("Res (ab/(a+b))", [([2, 2], 1), ([6, 3], 2), ([10, 10], 5), ([12, 4], 3)], ['a', 'b'], ["a*b/(a+b)", "b*a/(a+b)", "a*b/(b+a)"])
]

for t in tests:
    run_test(*t)
