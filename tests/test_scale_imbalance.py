
import unittest
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from kalkulator_pkg.function_manager import find_function_from_data

class TestScaleImbalance(unittest.TestCase):
    def test_giants_vs_ants(self):
        """Test f(x) = x + sin(x) with exp(x) in search space."""
        # Data points from user
        # f(0)=0, f(1.57)=2.57, f(3.14)=3.14, f(4.71)=3.71, f(6.28)=6.28
        data = [
            ([0], 0),
            ([1.57], 2.57),
            ([3.14], 3.14),
            ([4.71], 3.71),
            ([6.28], 6.28)
        ]
        
        success, func_str, _, error = find_function_from_data(data, ["x"])
        print(f"\nFound function: {func_str}")
        
        # We expect x + sin(x)
        # Or 1.0*x + 1.0*sin(x)
        self.assertTrue("x" in func_str, "Missing linear term x")
        self.assertTrue("sin(x)" in func_str, "Missing sine term sin(x)")
        self.assertFalse("exp(x)" in func_str, f"Should not contain exp(x), but got: {func_str}")
        self.assertFalse("sin(2*x)" in func_str, f"Should not contain sin(2*x), but got: {func_str}")

if __name__ == "__main__":
    unittest.main()
