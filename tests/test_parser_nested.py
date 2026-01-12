import unittest

from kalkulator_pkg.parser import preprocess


class TestParserNested(unittest.TestCase):
    def test_diff_nested(self):
        """Test diff with nested function calls."""
        input_str = "diff(f(x), x)"
        # Should be protected
        processed = preprocess(input_str)
        print(f"Input: {input_str}")
        print(f"Processed: {processed}")
        self.assertIn("__COMMA_SEP_", processed)
        self.assertIn("f(x)", processed)

    def test_integrate_nested(self):
        """Test integrate with nested function calls."""
        input_str = "integrate(sin(x), x)"
        processed = preprocess(input_str)
        print(f"Input: {input_str}")
        print(f"Processed: {processed}")
        self.assertIn("__COMMA_SEP_", processed)

    def test_double_nested(self):
        """Test doubly nested calls."""
        input_str = "diff(diff(x, x), x)"
        processed = preprocess(input_str)
        print(f"Input: {input_str}")
        print(f"Processed: {processed}")
        # The outer diff should be protected
        self.assertIn("__COMMA_SEP_", processed)

    def test_no_comma(self):
        """Test function call without comma (should not be protected)."""
        input_str = "diff(x)"
        processed = preprocess(input_str)
        print(f"Input: {input_str}")
        print(f"Processed: {processed}")
        # Should NOT have marker if no top-level comma
        self.assertNotIn("__COMMA_SEP_", processed)


if __name__ == "__main__":
    unittest.main()
