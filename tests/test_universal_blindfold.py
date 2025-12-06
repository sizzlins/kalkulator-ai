import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from kalkulator_pkg.function_manager import find_function_from_data


def test_universal_blindfold():
    print("--- 1. Spacetime (W = -A^2 + B^2 + C^2 + D^2) ---")
    # W(A,B,C,D)
    feature_names_w = ["A", "B", "C", "D"]
    data_points_w = [
        ([10, 2, 2, 2], -88.0),
        ([0, 3, 4, 0], 25.0),
        ([1, 1, 1, 1], 2.0),
        ([5, 0, 0, 0], -25.0),
        ([2, 5, 5, 5], 71.0),
        ([4, 4, 4, 4], 32.0),
        ([3, 0, 0, 0], -9.0),
        ([6, 1, 1, 1], -33.0),
        ([2, 10, 0, 0], 96.0),
        ([5, 3, 4, 0], 0.0),
    ]
    _, func_str_w, _, _ = find_function_from_data(data_points_w, feature_names_w)
    print(f"Result: {func_str_w}")
    # Check for correct form: -A^2 (or -1*A^2) and no cubic terms
    if (
        ("-A^2" in func_str_w or "- A^2" in func_str_w or "-1*A^2" in func_str_w)
        and "B^2" in func_str_w
        and "C^2" in func_str_w
        and "D^2" in func_str_w
        and "^3" not in func_str_w
    ):
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")

    print("\n--- 2. Gas Law (P = A*B/C) ---")
    # P(A,B,C) - More orthogonal data to break collinearity
    feature_names_p = ["A", "B", "C"]
    data_points_p = [
        ([1, 100, 10], 10.0),
        ([2, 300, 20], 30.0),
        ([5, 50, 25], 10.0),
        ([1, 300, 1], 300.0),
        ([2, 50, 5], 20.0),
        ([10, 10, 1], 100.0),
        ([4, 25, 2], 50.0),
        ([3, 100, 300], 1.0),
        ([2, 400, 8], 100.0),
        # Additional orthogonal points
        ([1, 1, 1], 1.0),
        ([10, 100, 10], 100.0),
        ([5, 200, 100], 10.0),
        ([100, 1, 10], 10.0),
        ([2, 2, 2], 2.0),
        ([3, 3, 3], 3.0),
    ]
    _, func_str_p, _, _ = find_function_from_data(data_points_p, feature_names_p)
    print(f"Result: {func_str_p}")
    if ("A*B/C" in func_str_p or "B*A/C" in func_str_p) and "1/B^4" not in func_str_p:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")

    print("\n--- 3. Kinetic Energy (K = 0.5*A*B^2) ---")
    feature_names_k = ["A", "B"]
    data_points_k = [
        ([10, 2], 20.0),
        ([1, 10], 50.0),
        ([5, 4], 40.0),
        ([2, 5], 25.0),
        ([100, 1], 50.0),
        ([0.5, 2], 1.0),
        ([4, 3], 18.0),
        ([8, 0.5], 1.0),
    ]
    _, func_str_k, _, _ = find_function_from_data(data_points_k, feature_names_k)
    print(f"Result: {func_str_k}")
    if ("A*B^2" in func_str_k or "B^2*A" in func_str_k) and "0.5" in func_str_k.replace(
        "1/2", "0.5"
    ):
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")

    print("\n--- 4. Sphere Volume (Y = 4/3*pi*X^3) ---")
    feature_names_y = ["X"]
    data_points_y = [
        ([1], 4.1887902048),
        ([3], 113.097335529),
        ([6], 904.778684234),
        ([2], 33.510321638),
        ([0.5], 0.5235987756),
        ([10], 4188.790204786),
    ]
    _, func_str_y, _, _ = find_function_from_data(data_points_y, feature_names_y)
    print(f"Result: {func_str_y}")
    if "X^3" in func_str_y and ("4/3*pi" in func_str_y or "4.188" in func_str_y):
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")


if __name__ == "__main__":
    test_universal_blindfold()
