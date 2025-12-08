
import unittest
import numpy as np
import sys
import os
from io import StringIO
from scipy.special import erf

# Ensure we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalkulator_pkg.regression_solver import solve_regression_stage

class TestKnowledgeExpansion(unittest.TestCase):
    
    def test_abs(self):
        # Case 1: Absolute Value |x|
        # Useful for mechanics, distance
        print("\n--- Testing Abs |x| ---")
        X = np.linspace(-5, 5, 15).reshape(-1, 1)
        y = np.abs(X.flatten())
        data_points = [(list(map(str, row)), val) for row, val in zip(X, y)]
        
        success, func_str, _, _ = solve_regression_stage(X, y, data_points, ['x'], include_transcendentals=True)
        print(f"Abs Result: {func_str}")
        self.assertTrue("abs(x)" in func_str or "|x|" in func_str, msg=f"Got: {func_str}")

    def test_relu(self):
        # Case 2: ReLU (Rectified Linear Unit) max(0, x)
        # Fundamental for AI
        print("\n--- Testing ReLU max(0, x) ---")
        X = np.linspace(-5, 5, 15).reshape(-1, 1)
        y = np.maximum(0, X.flatten())
        data_points = [(list(map(str, row)), val) for row, val in zip(X, y)]
        
        success, func_str, _, _ = solve_regression_stage(X, y, data_points, ['x'], include_transcendentals=True)
        print(f"ReLU Result: {func_str}")
        self.assertTrue("relu(x)" in func_str or "max(0, x)" in func_str or ("(x+|x|)/2" in func_str), msg=f"Got: {func_str}")

    def test_floor(self):
        # Case 3: Floor / Step Function
        # Quantization
        print("\n--- Testing Floor floor(x) ---")
        X = np.linspace(0, 5, 15).reshape(-1, 1) # Positive range to avoid -1 issues
        y = np.floor(X.flatten())
        data_points = [(list(map(str, row)), val) for row, val in zip(X, y)]
        
        success, func_str, _, _ = solve_regression_stage(X, y, data_points, ['x'], include_transcendentals=True)
        print(f"Floor Result: {func_str}")
        self.assertTrue("floor(x)" in func_str, msg=f"Got: {func_str}")

    def test_tan(self):
        # Case 4: Tangent tan(x)
        print("\n--- Testing Tan tan(x) ---")
        X = np.linspace(-1, 1, 15).reshape(-1, 1) # Avoid asymptotes at pi/2
        y = np.tan(X.flatten())
        data_points = [(list(map(str, row)), val) for row, val in zip(X, y)]
        
        success, func_str, _, _ = solve_regression_stage(X, y, data_points, ['x'], include_transcendentals=True)
        print(f"Tan Result: {func_str}")
        self.assertTrue("tan(x)" in func_str or "sin(x)/cos(x)" in func_str, msg=f"Got: {func_str}")

    def test_arcsin(self):
        # Case 5: Arcsin (Inverse Sine)
        print("\n--- Testing Arcsin asin(x) ---")
        X = np.linspace(-0.9, 0.9, 15).reshape(-1, 1) # Avoid domain edges
        y = np.arcsin(X.flatten())
        data_points = [(list(map(str, row)), val) for row, val in zip(X, y)]
        
        success, func_str, _, _ = solve_regression_stage(X, y, data_points, ['x'], include_transcendentals=True)
        print(f"Arcsin Result: {func_str}")
        self.assertTrue("asin(x)" in func_str or "arcsin(x)" in func_str, msg=f"Got: {func_str}")

    def test_erf(self):
        # Case 6: Error Function erf(x)
        # Probability
        print("\n--- Testing Erf erf(x) ---")
        X = np.linspace(-2, 2, 15).reshape(-1, 1)
        y = erf(X.flatten())
        data_points = [(list(map(str, row)), val) for row, val in zip(X, y)]
        
        success, func_str, _, _ = solve_regression_stage(X, y, data_points, ['x'], include_transcendentals=True)
        print(f"Erf Result: {func_str}")
        self.assertTrue("erf(x)" in func_str, msg=f"Got: {func_str}")

if __name__ == "__main__":
    unittest.main()
