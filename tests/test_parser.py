"""Unit tests for parser module."""

import unittest

from kalkulator_pkg.parser import format_number
from kalkulator_pkg.parser import format_superscript
from kalkulator_pkg.parser import is_balanced
from kalkulator_pkg.parser import parse_preprocessed
from kalkulator_pkg.parser import preprocess
from kalkulator_pkg.parser import split_top_level_commas
from kalkulator_pkg.types import ValidationError


class TestPreprocess(unittest.TestCase):
    """Test preprocessing functions."""

    def test_basic_arithmetic(self):
        self.assertEqual(preprocess("2+2"), "2+2")
        self.assertEqual(preprocess("2*3"), "2*3")
        self.assertEqual(preprocess("10/5"), "10/5")

    def test_exponent_conversion(self):
        self.assertEqual(preprocess("2^3"), "2**3")
        self.assertEqual(preprocess("x^2"), "x**2")

    def test_superscript_conversion(self):
        result = preprocess("2²")
        self.assertIn("**", result)
        self.assertIn("2", result)

    def test_percent_conversion(self):
        result = preprocess("50%")
        self.assertTrue("/100" in result or "((50)/(100))" in result)

    def test_sqrt_unicode(self):
        result = preprocess("√(4)")
        self.assertIn("sqrt", result)

    def test_implicit_multiplication(self):
        result = preprocess("2x")
        self.assertIn("2*x", result)

    def test_forbidden_tokens(self):
        with self.assertRaises(ValidationError):
            preprocess("__import__('os')")
        with self.assertRaises(ValidationError):
            preprocess("import sys")

    def test_input_length_limit(self):
        from kalkulator_pkg.config import MAX_INPUT_LENGTH

        long_input = "x" * (MAX_INPUT_LENGTH + 1)
        with self.assertRaises(ValidationError):
            preprocess(long_input)

    def test_parentheses_balancing(self):
        balanced, _ = is_balanced("(1+2)")
        self.assertTrue(balanced)
        balanced, _ = is_balanced("((1+2)*3)")
        self.assertTrue(balanced)
        balanced, _ = is_balanced("(1+2")
        self.assertFalse(balanced)
        balanced, _ = is_balanced("1+2)")
        self.assertFalse(balanced)


class TestParse(unittest.TestCase):
    """Test parsing functions."""

    def test_basic_parse(self):
        expr = parse_preprocessed("2+2")
        self.assertIsNotNone(expr)

    def test_variable_parse(self):
        expr = parse_preprocessed("x+1")
        self.assertIsNotNone(expr)


class TestFormatting(unittest.TestCase):
    """Test formatting functions."""

    def test_format_superscript(self):
        result = format_superscript("x**2")
        self.assertIn("²", result)

    def test_format_number(self):
        self.assertEqual(format_number(3.14159), "3.14159")
        self.assertEqual(format_number(0.0), "0")


class TestSplitCommas(unittest.TestCase):
    """Test comma splitting."""

    def test_matrix_splitting(self):
        parts = split_top_level_commas("Matrix([[1,2],[3,4]])")
        self.assertEqual(len(parts), 1)
        self.assertIn("Matrix", parts[0])

    def test_multiple_expressions(self):
        parts = split_top_level_commas("x=1, y=2")
        self.assertEqual(len(parts), 2)

    def test_nested_comma_preservation(self):
        parts = split_top_level_commas("det(Matrix([[1,2],[3,4]]))")
        self.assertEqual(len(parts), 1)


if __name__ == "__main__":
    unittest.main()
