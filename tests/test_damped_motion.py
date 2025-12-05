import unittest
import math
from kalkulator_pkg.function_manager import find_function_from_data


class TestDampedMotion(unittest.TestCase):
    def test_boss_level_damped_sine(self):
        """Test finding f(t) = exp(-t) * sin(t)."""
        # Data points provided by user
        # f(t) = exp(-t) * sin(t)
        # f(0) = 1 * 0 = 0
        # f(1) = 0.3678 * 0.8414 = 0.30955... -> 0.3096
        # f(2) = 0.1353 * 0.9092 = 0.1230... -> 0.1231
        # f(3) = 0.0497 * 0.1411 = 0.0070... -> 0.0070

        data = [([0], 0), ([1], 0.3096), ([2], 0.1231), ([3], 0.0070)]

        success, func_str, _, error = find_function_from_data(data, ["t"])
        self.assertTrue(success, f"Failed to find function: {error}")
        print(f"\nFound function: {func_str}")

        # Check for expected terms
        # Should contain exp(-t) and sin(t) combined
        self.assertTrue(
            "exp(-t)" in func_str or "exp(-1.0*t)" in func_str,
            f"Missing decay term: {func_str}",
        )
        self.assertTrue(
            "sin(t)" in func_str or "sin(1.0*t)" in func_str,
            f"Missing sine term: {func_str}",
        )

        # Ideally it should be exactly exp(-t)*sin(t)
        # But coefficients might be slightly off due to limited precision in data
        # e.g. 1.0001*exp(-t)*sin(t)


if __name__ == "__main__":
    unittest.main()
