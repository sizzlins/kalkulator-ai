
import unittest
import math
from kalkulator_pkg.function_manager import find_function_from_data

class TestTranscendentalFunctions(unittest.TestCase):
    """Test finding transcendental functions (sin, cos, exp, log) and rational functions."""

    def test_sine_function(self):
        """Test finding f(x) = sin(x)."""
        # f(0)=0, f(pi/2)=1, f(pi)=0, f(3pi/2)=-1
        data = [
            ([0], 0),
            ([math.pi/2], 1),
            ([math.pi], 0),
            ([3*math.pi/2], -1)
        ]
        success, func_str, _, error = find_function_from_data(data, ["x"])
        self.assertTrue(success, f"Failed to find sin(x): {error}")
        self.assertIn("sin(x)", func_str)
        print(f"\nFound sine: {func_str}")

    def test_cosine_function(self):
        """Test finding f(x) = cos(x)."""
        # f(0)=1, f(pi/2)=0, f(pi)=-1
        data = [
            ([0], 1),
            ([math.pi/2], 0),
            ([math.pi], -1)
        ]
        success, func_str, _, error = find_function_from_data(data, ["x"])
        self.assertTrue(success, f"Failed to find cos(x): {error}")
        self.assertIn("cos(x)", func_str)
        print(f"\nFound cosine: {func_str}")

    def test_exponential_function(self):
        """Test finding f(x) = exp(x)."""
        # f(0)=1, f(1)=e, f(2)=e^2
        data = [
            ([0], 1),
            ([1], math.e),
            ([2], math.e**2)
        ]
        success, func_str, _, error = find_function_from_data(data, ["x"])
        self.assertTrue(success, f"Failed to find exp(x): {error}")
        # Could be exp(x) or e^x
        self.assertTrue("exp(x)" in func_str or "e^x" in func_str or "E^x" in func_str, f"Got: {func_str}")
        print(f"\nFound exponential: {func_str}")

    def test_log_function(self):
        """Test finding f(x) = log(x)."""
        # f(1)=0, f(e)=1, f(e^2)=2
        data = [
            ([1], 0),
            ([math.e], 1),
            ([math.e**2], 2)
        ]
        success, func_str, _, error = find_function_from_data(data, ["x"])
        self.assertTrue(success, f"Failed to find log(x): {error}")
        self.assertIn("log(x)", func_str)
        print(f"\nFound log: {func_str}")

    def test_inverse_function(self):
        """Test finding f(x) = 1/x."""
        # f(1)=1, f(2)=0.5, f(4)=0.25
        data = [
            ([1], 1),
            ([2], 0.5),
            ([4], 0.25)
        ]
        success, func_str, _, error = find_function_from_data(data, ["x"])
        self.assertTrue(success, f"Failed to find 1/x: {error}")
        self.assertTrue("1/x" in func_str or "x^-1" in func_str, f"Got: {func_str}")
        print(f"\nFound inverse: {func_str}")

    def test_composite_function(self):
        """Test finding f(x) = sin(x) + 1."""
        # f(0)=1, f(pi/2)=2, f(pi)=1
        data = [
            ([0], 1),
            ([math.pi/2], 2),
            ([math.pi], 1)
        ]
        success, func_str, _, error = find_function_from_data(data, ["x"])
        self.assertTrue(success, f"Failed to find sin(x)+1: {error}")
        self.assertTrue("sin(x)" in func_str and "1" in func_str, f"Got: {func_str}")
        print(f"\nFound composite: {func_str}")

    def test_power_law(self):
        """Test finding f(x) = x^2.5 (power law)."""
        # f(1)=1, f(4)=32, f(9)=243
        data = [
            ([1], 1),
            ([4], 32),
            ([9], 243)
        ]
        success, func_str, _, error = find_function_from_data(data, ["x"])
        self.assertTrue(success, f"Failed to find x^2.5: {error}")
        self.assertIn("x^2.5", func_str)
        print(f"\nFound power law: {func_str}")

if __name__ == "__main__":
    unittest.main()
