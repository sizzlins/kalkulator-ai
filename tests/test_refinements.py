import unittest
import numpy as np
import sys
import os

# Ensure we can import the package
sys.path.append(os.getcwd())

from kalkulator_pkg.regression_solver import solve_regression_stage
from kalkulator_pkg.dynamics_discovery import SINDy, SINDyConfig


class TestRefinements(unittest.TestCase):
    def test_robust_regression_outlier(self):
        # Case 1: f(x) = 2x with one massive outlier
        # Increase points to 10 to ensure RANSAC stability
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([2.0, 4.0, 100.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        data_points = [
            (["1"], 2),
            (["2"], 4),
            (["3"], 100),
            (["4"], 8),
            (["5"], 10),
            (["6"], 12),
            (["7"], 14),
            (["8"], 16),
            (["9"], 18),
            (["10"], 20),
        ]

        success, func_str, _, _ = solve_regression_stage(X, y, data_points, ["x"])

        print(f"\nRobust Regression Result: {func_str}")

        # It should be 2*x (or very close), NOT containing 100 or x^3
        self.assertTrue("2*x" in func_str or "2.0*x" in func_str)
        self.assertFalse("x^3" in func_str)
        self.assertFalse("x^4" in func_str)

    def test_interaction_discovery(self):
        # Case 2: f(x) = x * exp(x)
        X = np.linspace(0.1, 2.0, 10).reshape(-1, 1)
        # f(x) = x * exp(x)
        y = X.flatten() * np.exp(X.flatten())

        data_points = list(zip([list(row) for row in X], y))

        success, func_str, _, _ = solve_regression_stage(
            X, y, data_points, ["x"], include_transcendentals=True
        )

        print(f"\nInteraction Discovery Result: {func_str}")

        # Should contain x*exp(x)
        self.assertTrue("x*exp(x)" in func_str)
        self.assertFalse(
            "x^2" in func_str and "x^3" in func_str
        )  # Should not be polynomial approx

    def test_sindy_small_data(self):
        # Case 3: Simple decay dx/dt = -x, small noisy data
        # Use 8 points to ensure derivative is somewhat stable but still trigger N<10 logic
        t = np.linspace(0, 3, 8)
        x = np.exp(-t)
        X = x.reshape(-1, 1)

        # Manually invoke logic matching CLI for N<10 (poly_order=1, threshold=0.5)
        config = SINDyConfig(threshold=0.5, poly_order=1)
        sindy = SINDy(config)
        sindy.fit(X, t, variable_names=["x"])

        eqs = sindy.equations
        print(f"\nSINDy Result: {eqs}")

        # Should find dx/dt = -x (approx -1*x)
        # Should NOT have x^2 if threshold works
        dx_dt = eqs.get("dx/dt", "")
        self.assertTrue(
            "-x" in dx_dt or "-1*x" in dx_dt or "-0.99" in dx_dt or "-0.9" in dx_dt
        )
        self.assertFalse("x^2" in dx_dt)


if __name__ == "__main__":
    unittest.main()
