"""Unit tests for calculus operations."""

import unittest

from kalkulator_pkg.calculus import differentiate, integrate, matrix_determinant


class TestDifferentiation(unittest.TestCase):
    """Test differentiation functions."""

    def test_basic_differentiation(self):
        result = differentiate("x^2")
        self.assertTrue(result.ok)
        self.assertIn("2*x", result.result)

    def test_differentiation_with_variable(self):
        result = differentiate("y^3", variable="y")
        self.assertTrue(result.ok)

    def test_trig_differentiation(self):
        result = differentiate("sin(x)")
        self.assertTrue(result.ok)
        self.assertIn("cos", result.result)


class TestIntegration(unittest.TestCase):
    """Test integration functions."""

    def test_basic_integration(self):
        result = integrate("x")
        self.assertTrue(result.ok)

    def test_trig_integration(self):
        result = integrate("cos(x)")
        self.assertTrue(result.ok)
        self.assertIn("sin", result.result)


class TestMatrixDeterminant(unittest.TestCase):
    """Test matrix determinant calculation."""

    def test_2x2_determinant(self):
        result = matrix_determinant("Matrix([[1,2],[3,4]])")
        self.assertTrue(result.ok)
        # Determinant of [[1,2],[3,4]] = 1*4 - 2*3 = -2
        self.assertIn("-2", result.result)


if __name__ == "__main__":
    unittest.main()
