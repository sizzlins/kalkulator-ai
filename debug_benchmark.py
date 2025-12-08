
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from kalkulator_pkg.regression_solver import solve_regression_stage

def debug_high_frequency_sine():
    print("\n--- Testing High Frequency Sine (Debug) ---")
    X = np.linspace(0, 5, 50).reshape(-1, 1) # Need dense sampling for 10x
    y = np.sin(10*X.flatten()) + np.sin(11*X.flatten())
    data_points = [(list(map(str, row)), val) for row, val in zip(X, y)]
    
    success, func_str, _, _ = solve_regression_stage(X, y, data_points, ['t'], include_transcendentals=True)
    print(f"High Freq Result: {func_str}")
    
    expected = "sin(10*t)" in func_str or "sin(11*t)" in func_str
    if expected:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    debug_high_frequency_sine()
