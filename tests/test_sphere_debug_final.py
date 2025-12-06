
import sys
import os
# Insert local project path at the BEGINNING of sys.path to override installed package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
print(f"DEBUG PATH: {sys.path}")

import math
import numpy as np
import kalkulator_pkg.regression_solver
print(f"Loaded regression_solver from: {kalkulator_pkg.regression_solver.__file__}")

from kalkulator_pkg.function_manager import find_function_from_data

def test_sphere_debug():
    print("\n--- Sphere Debug ---")
    # Sphere Volume: V = 4/3 * pi * r^3
    # 4/3 * pi approx 4.188790204786
    pi_val = math.pi
    coeff = 4/3 * pi_val
    features = ['r']
    # Use exact formula to generate data
    data_points = []
    for r in [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 10]:
        y = coeff * (r**3)
        data_points.append(([r], y))
    
    print(f"Generating data with coeff: {coeff}")
    
    # Run solver
    success, result, acc, _ = find_function_from_data(data_points, features)
    print(f"Result: {result}")
    
    expected_start = "4/3*pi*r^3"
    if result == expected_start:
        print("STATUS: PASS")
    else:
        print("STATUS: CHECK")

if __name__ == "__main__":
    test_sphere_debug()
