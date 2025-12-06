import os
import sys

sys.path.append(os.getcwd())
try:
    import numpy as np

    from kalkulator_pkg.function_manager import find_function_from_data
except ImportError:
    import numpy as np

    from kalkulator_pkg.function_manager import find_function_from_data


def run_test(name, data, params, expected_list):
    print(f"--- {name} ---")
    try:
        found, func, _, _ = find_function_from_data(data, params)
        print(f"Result: {func}")
        func_norm = func.replace(" ", "")
        passed = False
        for exp in expected_list:
            if exp.replace(" ", "") in func_norm:
                passed = True
                break

        if passed:
            print("PASS")
        else:
            print(f"FAIL (Expected one of {expected_list})")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()


tests = [
    # 1. Poly (x^2+1)
    (
        "Poly (x^2+1)",
        [([1], 2), ([2], 5), ([3], 10), ([0], 1)],
        ["x"],
        ["x^2+1", "1+x^2"],
    ),
    # 2. Hyp (sqrt(a^2+b^2))
    (
        "Hyp (sqrt(a^2+b^2))",
        [([3, 4], 5), ([5, 12], 13), ([0, 1], 1)],
        ["a", "b"],
        ["sqrt(a^2+b^2)", "sqrt(b^2+a^2)"],
    ),
    # 3. Grav (m/r^2) - Corrected data for m/r^2
    (
        "Grav (m/r^2)",
        [([1, 1], 1), ([2, 1], 2), ([1, 2], 0.25), ([4, 2], 1)],
        ["m", "r"],
        ["m/r^2", "m*r^-2"],
    ),
    # 4. Wave (sin(t))
    ("Wave (sin(t))", [([0], 0), ([1.570796], 1), ([3.14159], 0)], ["t"], ["sin(t)"]),
    # 5. Sat (1 - 1/(x+1))
    (
        "Sat (1-1/(x+1))",
        [([1], 0.5), ([3], 0.75), ([9], 0.9)],
        ["x"],
        ["1-1/(x+1)", "x/(x+1)"],
    ),
]

for t in tests:
    run_test(*t)
