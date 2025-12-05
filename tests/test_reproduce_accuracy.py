import os
import sys

# Create project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from kalkulator_pkg.function_manager import find_function_from_data


def test_ghost_noise():
    print("Testing Ghost Noise and Shrinkage...")

    # Data points from user request
    # d(t) = 4.903325 * t^2
    points = [([0], 0), ([1], 4.9033250000), ([2], 19.6133000000), ([3], 44.1299250000)]

    param_names = ["t"]

    result = find_function_from_data(points, param_names)
    print("\nResult:", result)

    success, func_str, _, _ = result

    if success:
        print(f"Found function: {func_str}")

        # Check for ghost noise
        if "sinh" in func_str or "cosh" in func_str:
            print("FAILURE: Ghost noise detected (hyperbolic functions found)")
        else:
            print("SUCCESS: No ghost noise detected")

        # Check for shrinkage/precision
        # Expected: 4.903325*t^2
        # Current Bad Output: 4.903*t^2 ...

        import re

        # Extract coefficient of t^2
        match = re.search(r"([\d\.]+)\*t\^2", func_str)
        if match:
            coeff = float(match.group(1))
            print(f"Coefficient t^2: {coeff}")
            expected = 4.903325
            if abs(coeff - expected) < 1e-4:
                print(f"SUCCESS: Coefficient accuracy good ({coeff})")
            else:
                print(
                    f"FAILURE: Coefficient accuracy poor. Got {coeff}, expected {expected}"
                )
        else:
            print("Could not parse t^2 coefficient")


if __name__ == "__main__":
    test_ghost_noise()
