import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from kalkulator_pkg.function_manager import find_function_from_data


def test_blindfold_physics():
    print("--- 1. Mechanical Energy (Renamed A, B, C) ---")
    # U = 0.5*A*B^2 + 9.8*A*C
    # U(A,B,C)
    feature_names = ["A", "B", "C"]
    data_points = [
        ([1, 2, 0], 2.0),
        ([1, 0, 1], 9.8),
        ([2, 2, 1], 23.6),
        ([10, 10, 10], 1480.0),
        ([5, 4, 2], 138.0),
        ([0.5, 10, 5], 49.5),
        ([1, 1, 1], 10.3),
        ([2, 0, 10], 196.0),
    ]

    success, func_str, _, _ = find_function_from_data(data_points, feature_names)
    print(f"Result: {func_str}")

    # Expected: 0.5*A*B^2 + 9.8*A*C
    if "A*B^2" in func_str and "A*C" in func_str and "sin" not in func_str:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")

    print("\n--- 2. Sphere Volume (Renamed x) ---")
    # Q = 4/3*pi*x^3
    feature_names_q = ["x"]
    data_points_q = [
        ([1], 4.1887902048),
        ([3], 113.097335529),
        ([6], 904.778684234),
        ([2], 33.510321638),
        ([0.5], 0.5235987756),
        ([10], 4188.790204786),
    ]
    success_q, func_str_q, _, _ = find_function_from_data(
        data_points_q, feature_names_q
    )
    print(f"Result: {func_str_q}")
    if "4/3*pi*x^3" in func_str_q:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")


if __name__ == "__main__":
    test_blindfold_physics()
