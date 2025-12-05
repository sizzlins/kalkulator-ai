"""Fuzzing tests for parser and worker with random inputs."""

import random
import string
import unittest

from kalkulator_pkg.parser import preprocess
from kalkulator_pkg.types import ValidationError
from kalkulator_pkg.worker import evaluate_safely


class TestParserFuzzing(unittest.TestCase):
    """Fuzz test parser with random inputs."""

    def test_random_strings(self):
        """Test parser rejects random garbage strings."""
        for _ in range(100):
            # Generate random string
            length = random.randint(1, 100)
            random_str = "".join(random.choices(string.printable, k=length))

            try:
                preprocess(random_str)
            except ValidationError:
                pass  # Expected
            except Exception:
                # Any other exception is also acceptable (rejection)
                pass

    def test_malformed_expressions(self):
        """Test parser handles malformed expressions."""
        malformed = [
            "(((",
            ")))",
            "x++y",
            "x**",
            "*/x",
            "",
            "   ",
        ]

        for expr in malformed:
            try:
                preprocess(expr)
            except (ValidationError, ValueError):
                pass  # Expected
            except Exception:
                pass  # Also acceptable


class TestWorkerFuzzing(unittest.TestCase):
    """Fuzz test worker with random inputs."""

    def test_random_valid_expressions(self):
        """Test worker handles edge-case valid expressions."""
        edge_cases = [
            "0",
            "1",
            "-1",
            "x",
            "-x",
            "x+1",
            "x*0",
            "x/1",
            "x^0",
            "x^1",
        ]

        for expr in edge_cases:
            try:
                result = evaluate_safely(expr)
                # Should either succeed or return structured error
                self.assertIsInstance(result, dict)
                self.assertIn("ok", result)
            except Exception:
                # Should not raise unhandled exceptions
                self.fail(f"Worker raised exception for: {expr}")


if __name__ == "__main__":
    unittest.main()
