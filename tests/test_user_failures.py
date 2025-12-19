import os
import sys
import unittest

import numpy as np

sys.path.append(os.getcwd())

from kalkulator_pkg.regression_solver import solve_regression_stage


class TestUserReportedFailures(unittest.TestCase):
    def test_linear_plane(self):
        # Case 3: 3x + 2z - 5
        # y(1, 1)=0, y(2, 0)=1, y(0, 2.5)=0, y(10, 5)=35, y(2, 2)=5
        # y(5, 1)=12, y(0, 0)=-5, y(3, 3)=10, y(1, 4)=6
        # y(0.5, 0.5)=-2.5, y(100, 1)=297, y(-1, -1)=-10
        raw_data = [
            ([1, 1], 0),
            ([2, 0], 1),
            ([0, 2.5], 0),
            ([10, 5], 35),
            ([2, 2], 5),
            ([5, 1], 12),
            ([0, 0], -5),
            ([3, 3], 10),
            ([1, 4], 6),
            ([0.5, 0.5], -2.5),
            ([100, 1], 297),
            ([-1, -1], -10),
        ]
        X = np.array([d[0] for d in raw_data])
        y = np.array([float(d[1]) for d in raw_data])
        data_points = [(list(map(str, d[0])), d[1]) for d in raw_data]

        success, func_str, _, _ = solve_regression_stage(
            X, y, data_points, ["x", "z"], include_transcendentals=True
        )
        print(f"\nLinear Plane Result: {func_str}")

        # Expect "3*x + 2*z - 5"
        self.assertTrue("3*x" in func_str, msg=f"Got: {func_str}")
        self.assertTrue("2*z" in func_str, msg=f"Got: {func_str}")
        self.assertTrue("- 5" in func_str or "-5" in func_str, msg=f"Got: {func_str}")
        self.assertFalse("x^2" in func_str, msg=f"Got: {func_str}")

    def test_sqrt_interaction(self):
        # Case 2: sqrt(x * y)
        # g(1, 1)=1, g(4, 9)=6, g(2, 8)=4, g(100, 1)=10, g(0, 100)=0
        # g(3, 12)=6, g(16, 4)=8, g(25, 4)=10, g(5, 20)=10
        raw_data = [
            ([1, 1], 1),
            ([4, 9], 6),
            ([2, 8], 4),
            ([100, 1], 10),
            ([0, 100], 0),
            ([3, 12], 6),
            ([16, 4], 8),
            ([25, 4], 10),
            ([5, 20], 10),
        ]
        X = np.array([d[0] for d in raw_data])
        y = np.array([float(d[1]) for d in raw_data])
        data_points = [(list(map(str, d[0])), d[1]) for d in raw_data]

        success, func_str, _, _ = solve_regression_stage(
            X, y, data_points, ["x", "y"], include_transcendentals=True
        )
        print(f"\nSqrt Interaction Result: {func_str}")

        self.assertTrue(
            "sqrt(x*y)" in func_str
            or "x^0.5*y^0.5" in func_str
            or "(x*y)^0.5" in func_str,
            msg=f"Got: {func_str}",
        )

    def test_exponential_base_2(self):
        # Case 1: 2^x
        X = np.array([[0], [1], [2], [3], [4], [5]])
        y = np.array([1, 2, 4, 8, 16, 32])
        data_points = [(list(map(str, row)), val) for row, val in zip(X, y)]

        success, func_str, _, _ = solve_regression_stage(
            X, y, data_points, ["x"], include_transcendentals=True
        )
        print(f"\nExponential Result: {func_str}")

        # We want "2^x" or "2.0^x"
        self.assertTrue(
            "2^x" in func_str or "2.0^x" in func_str or "2.00^x" in func_str,
            msg=f"Got: {func_str}",
        )

    def test_sigmoid(self):
        # Case 4: Sigmoid 1/(1+exp(-x))
        # User reported this works.
        X = np.linspace(-5, 5, 20).reshape(-1, 1)
        y = 1.0 / (1.0 + np.exp(-X.flatten()))
        data_points = [(list(map(str, row)), val) for row, val in zip(X, y)]

        success, func_str, _, _ = solve_regression_stage(
            X, y, data_points, ["x"], include_transcendentals=True
        )
        print(f"\nSigmoid Result: {func_str}")

        # Verify if it finds "1/(1+exp(-x))" or equivalent "exp(x)/(1+exp(x))"
        self.assertTrue(
            "1/(1+exp(-x))" in func_str or "exp(x)/(1+exp(x))" in func_str,
            msg=f"Got: {func_str}",
        )

    def test_softplus(self):
        # Case 5: Softplus ln(1+exp(x))
        # User reported this FAILS.
        X = np.linspace(-2, 4, 20).reshape(-1, 1)
        # f(x) = log(1+exp(x))
        y = np.log(1.0 + np.exp(X.flatten()))
        data_points = [(list(map(str, row)), val) for row, val in zip(X, y)]

        success, func_str, _, _ = solve_regression_stage(
            X, y, data_points, ["x"], include_transcendentals=True
        )
        print(f"\nSoftplus Result: {func_str}")

        self.assertTrue(
            "log(1+exp(x))" in func_str or "ln(1+exp(x))" in func_str,
            msg=f"Got: {func_str}",
        )


if __name__ == "__main__":
    unittest.main()
