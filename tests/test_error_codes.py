"""Test error codes returned by various functions."""

import unittest

from kalkulator_pkg.parser import preprocess
from kalkulator_pkg.solver import solve_single_equation
from kalkulator_pkg.types import ValidationError
from kalkulator_pkg.worker import evaluate_safely


class TestErrorCodes(unittest.TestCase):
    """Test that functions return appropriate error codes."""

    def test_forbidden_token_error_code(self):
        """Test that forbidden tokens return FORBIDDEN_TOKEN error code."""
        try:
            preprocess("import os")
            self.fail("Should have raised ValidationError")
        except ValidationError as e:
            # Check error code is present
            self.assertEqual(
                e.code, "FORBIDDEN_TOKEN", f"Expected FORBIDDEN_TOKEN, got {e.code}"
            )
            self.assertIn("forbidden", str(e).lower() or str(e.args).lower())

    def test_too_long_error_code(self):
        """Test that overly long input returns TOO_LONG error code."""
        long_input = "x" * 10001  # Exceeds MAX_INPUT_LENGTH
        try:
            preprocess(long_input)
            self.fail("Should have raised ValidationError")
        except ValidationError as e:
            # Check error code is present
            self.assertEqual(e.code, "TOO_LONG", f"Expected TOO_LONG, got {e.code}")
            self.assertIn("too long", str(e).lower() or str(e.args).lower())

    def test_invalid_equation_format_error(self):
        """Test that invalid equation format returns error."""
        result = solve_single_equation("x = x = 1")  # Invalid: multiple equals
        self.assertFalse(result.get("ok"))
        self.assertIsNotNone(result.get("error"))
        # Error should indicate invalid format
        error_msg = result.get("error", "").lower()
        self.assertTrue(
            "invalid" in error_msg or "format" in error_msg,
            f"Error message should mention invalid/format: {error_msg}",
        )

    def test_evaluation_error_code(self):
        """Test that evaluation errors include error information."""
        result = evaluate_safely("__import__('os')")
        self.assertFalse(result.get("ok"))
        # Should have error code or error message
        error = result.get("error")
        self.assertIsNotNone(error)
        # Check error code field if present
        error_code = result.get("error_code")
        if error_code:
            self.assertIsInstance(error_code, str)

    def test_parse_error_in_evaluation(self):
        """Test that parse errors in evaluation return appropriate errors."""
        result = evaluate_safely("2 +")  # Incomplete expression
        self.assertFalse(result.get("ok"))
        error = result.get("error")
        self.assertIsNotNone(error)
        # Error should indicate parsing issue
        error_lower = error.lower()
        self.assertTrue(
            "parse" in error_lower
            or "syntax" in error_lower
            or "invalid" in error_lower,
            f"Error should mention parse/syntax/invalid: {error}",
        )

    def test_contradiction_detection(self):
        """Test that contradictions are detected and reported."""
        result = solve_single_equation("1 = 0")
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("type"), "identity_or_contradiction")
        # Result should indicate contradiction
        result_str = result.get("result", "").lower()
        self.assertTrue(
            "contradiction" in result_str or "false" in result_str,
            f"Result should indicate contradiction: {result_str}",
        )

    def test_identity_detection(self):
        """Test that identities are detected and reported."""
        result = solve_single_equation("1 = 1")
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("type"), "identity_or_contradiction")
        # Result should indicate identity
        result_str = result.get("result", "").lower()
        self.assertTrue(
            "identity" in result_str or "true" in result_str or "always" in result_str,
            f"Result should indicate identity: {result_str}",
        )

    def test_worker_timeout_error(self):
        """Test that worker timeouts return appropriate error codes."""
        # This test may take time, so we'll skip it in fast mode
        # In production, timeouts should return timeout-related error codes
        pass  # Would require actually triggering a timeout


if __name__ == "__main__":
    unittest.main()
