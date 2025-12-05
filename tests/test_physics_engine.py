import unittest
import math
from kalkulator_pkg.function_manager import find_function_from_data
from kalkulator_pkg.worker import evaluate_safely


class TestPhysicsEngine(unittest.TestCase):
    def test_find_frequency_scaled_sine(self):
        """Test finding sin(2x) which requires argument scaling."""
        # Data for y = sin(2x)
        # x=0 -> 0
        # x=pi/4 -> sin(pi/2) = 1
        # x=pi/2 -> sin(pi) = 0
        # x=3pi/4 -> sin(3pi/2) = -1
        data = [
            ([0], 0),
            ([math.pi / 4], 1),
            ([math.pi / 2], 0),
            ([3 * math.pi / 4], -1),
        ]
        success, func_str, _, error = find_function_from_data(data, ["x"])
        self.assertTrue(success, f"Failed to find sin(2x): {error}")
        print(f"\nFound scaled sine: {func_str}")
        self.assertTrue(
            "sin(2*x)" in func_str or "sin(2.0*x)" in func_str, f"Got: {func_str}"
        )

    def test_calculus_integration(self):
        """Test the 'Physics Engine' flow: find -> diff -> diff."""
        # 1. Find Position p(t) = sin(t)
        data = [([0], 0), ([math.pi / 2], 1), ([math.pi], 0), ([3 * math.pi / 2], -1)]
        success, func_str, _, error = find_function_from_data(data, ["t"])
        self.assertTrue(success, "Failed to find p(t)")
        self.assertTrue("sin(t)" in func_str, f"Expected sin(t), got {func_str}")

        # 2. Find Velocity: diff(sin(t), t)
        # We use evaluate_safely to simulate the REPL command
        expr = f"diff({func_str}, t)"
        result_dict = evaluate_safely(expr)
        result = result_dict["result"]
        print(f"\nDerivative of {func_str}: {result}")
        self.assertTrue("cos(t)" in str(result), f"Expected cos(t), got {result}")

        # 3. Find Acceleration: diff(cos(t), t)
        expr2 = f"diff({result}, t)"
        result2_dict = evaluate_safely(expr2)
        result2 = result2_dict["result"]
        print(f"Derivative of {result}: {result2}")
        self.assertTrue("-sin(t)" in str(result2), f"Expected -sin(t), got {result2}")


if __name__ == "__main__":
    unittest.main()
