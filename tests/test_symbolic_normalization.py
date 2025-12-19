import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.function_manager import find_function_from_data


def test_symbolic_norm():
    print("Testing symbolic normalization (x=1 assumed)...")

    # User Case: f(x)=0, f(x/2)=1, f(x/3)=0.866...
    # We want the engine to find f(t) = sin(pi*t) assuming x=1.

    data_points = [(["x"], "0"), (["x/2"], "1"), (["x/3"], "0.866025403784439")]
    param_names = ["x_arg"]  # "find f(x_arg)" where x_arg is the expression passed

    # Step 1: Run and expect the NEW behavior (return success + function + message)
    result = find_function_from_data(data_points, param_names)

    print("\nResult:")
    print(f"Success: {result[0]}")
    print(f"Function: {result[1]}")
    print(f"Message: {result[3]}")

    if result[0]:
        # Check if function contains sin/cos and likely pi/3.14
        if "sin" in result[1] and ("3.14" in result[1] or "pi" in result[1]):
            print("PASS: Discovered sin(pi*x) logic.")
        else:
            print("FAIL: Did not discover expected sine function.")
    else:
        print("FAIL: Returned failure instead of fallback result.")


if __name__ == "__main__":
    test_symbolic_norm()
