"""Integration tests for end-to-end worker + main process workflows."""

import unittest

from kalkulator_pkg.parser import preprocess
from kalkulator_pkg.solver import solve_single_equation
from kalkulator_pkg.worker import evaluate_safely


class TestWorkerIntegration(unittest.TestCase):
    """Test worker evaluation end-to-end."""

    def test_simple_evaluation(self):
        result = evaluate_safely("2+2")
        self.assertTrue(result.get("ok"))
        self.assertIn("result", result)
        self.assertEqual(result.get("result"), "4")

    def test_variable_evaluation(self):
        result = evaluate_safely("x+1")
        self.assertTrue(result.get("ok"))
        # Should have x as free symbol
        self.assertIn("x", result.get("free_symbols", []))

    def test_function_evaluation(self):
        result = evaluate_safely("sin(0)")
        self.assertTrue(result.get("ok"))
        # Should evaluate to 0
        self.assertEqual(result.get("result"), "0")


class TestSolverIntegration(unittest.TestCase):
    """Test equation solving end-to-end."""

    def test_linear_solve(self):
        result = solve_single_equation("x + 1 = 0")
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("type"), "equation")
        self.assertIsNotNone(result.get("exact"))
        # Verify the actual solution
        exact_solutions = result.get("exact", [])
        self.assertIn("-1", exact_solutions)


class TestPreprocessingIntegration(unittest.TestCase):
    """Test preprocessing pipeline."""

    def test_complex_preprocessing(self):
        # Test multiple transformations together
        result = preprocess("2² + 50% * √(4)")
        self.assertIn("**", result)
        # Percentage conversion may use /100 or (50)/(100) format
        self.assertTrue(
            "/100" in result or "(50)/(100)" in result or "(50/100)" in result
        )
        self.assertIn("sqrt", result)


if __name__ == "__main__":
    unittest.main()
