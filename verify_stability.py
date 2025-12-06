
import sys
import os
import numpy as np
import warnings

# Suppress warnings to check if we catch them cleanly or if they leak
# actually, better to let them print if we want to see if we silenced them?
# The user wants proof it's stable. If warnings appear, it's "ugly".
# But key is: NO CRASH (Exception).

sys.path.append(os.getcwd())
try:
    from kalkulator_pkg.function_manager import find_function_from_data
except ImportError:
    from kalkulator_pkg.function_manager import find_function_from_data

def run_test(name, data, params, expected_behavior="No Crash"):
    print(f"--- {name} ---")
    try:
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            found, func, _, _ = find_function_from_data(data, params)
            
            print(f"Result: {func}")
            if w:
                print(f"WARNINGS CAUGHT: {len(w)}")
                for warning in w:
                     print(f"  {warning.category.__name__}: {warning.message}")
            
            if found:
                print("Status: FOUND")
            else:
                print("Status: NOT FOUND (Safe Fail)")
                
    except Exception as e:
        print(f"STATUS: CRASH ({e})")
        import traceback
        traceback.print_exc()

tests = [
    # 1. Field with Zero Denominator (r=0)
    # This should be skipped by 'safe division' masking or result in valid masked data
    ("Field Crash Test (r=0)", [([1, 1], 1), ([2, 1], 0.25), ([1, 0], 999999)], ['m', 'r']),
    
    # 2. Resistors with Zero Sum (x+y=0)
    ("Res Crash Test (x=-y)", [([2, 2], 1), ([3, -3], 999)], ['a', 'b']),
    
    # 3. Geometric Mean with Negative Domain (sqrt(-1))
    ("Geo Crash Test (Negative)", [([4, 1], 2), ([1, -1], 999)], ['x', 'y']),
    
    # 4. Hyp at Zero (0,0) -> Term=0 check
    # Should still find function using other points if masking works
    ("Hyp Zero Test (0,0)", [([3, 4], 5), ([0, 0], 0), ([5, 12], 13)], ['a', 'b']),
    
    # 5. Log with Zero/Negative
    ("Log Crash Test", [([2.718], 1), ([0], -999), ([-1], -999)], ['x'])
]

for t in tests:
    run_test(*t)
