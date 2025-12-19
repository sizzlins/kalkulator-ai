import unittest

import sympy as sp

from kalkulator_pkg.utils.formatting import format_solution, simplify_exponential_bases


class TestExponentialSimplification(unittest.TestCase):
    def test_basic_base_2(self):
        # exp(ln(2)*x) -> 2^x
        x = sp.Symbol("x")
        val = sp.log(2)
        expr = sp.exp(val * x)
        simplified = simplify_exponential_bases(expr)
        self.assertEqual(simplified, sp.Pow(2, x))

    def test_approximate_base_2(self):
        # exp(0.69314718056 * x) -> 2^x
        x = sp.Symbol("x")
        expr = sp.exp(0.69314718056 * x)
        simplified = simplify_exponential_bases(expr)
        # Should be exactly 2^x (integer base)
        self.assertEqual(simplified, sp.Pow(2, x))

    def test_base_3(self):
        # exp(ln(3)*x) -> 3^x
        x = sp.Symbol("x")
        expr = sp.exp(sp.log(3) * x)
        simplified = simplify_exponential_bases(expr)
        self.assertEqual(simplified, sp.Pow(3, x))

    def test_fractional_base(self):
        # exp(ln(0.5)*x) -> (1/2)^x
        x = sp.Symbol("x")
        expr = sp.exp(sp.log(0.5) * x)
        simplified = simplify_exponential_bases(expr)
        # Should likely become 2^(-x) or (1/2)^x depending on SymPy preference
        # Our function explicitly returns sp.Pow(sp.Rational(1, 2), x)
        expected = sp.Pow(sp.Rational(1, 2), x)
        self.assertEqual(simplified, expected)

    def test_non_simplifiable(self):
        # exp(x) -> exp(x) (base e)
        x = sp.Symbol("x")
        expr = sp.exp(x)
        simplified = simplify_exponential_bases(expr)
        self.assertEqual(simplified, expr)

        # exp(2.5*x) -> exp(2.5*x) (base e^2.5 ~= 12.18, not integer)
        expr2 = sp.exp(2.5 * x)
        simplified2 = simplify_exponential_bases(expr2)
        self.assertEqual(simplified2, expr2)

    def test_nested_expression(self):
        # 5 + exp(ln(2)*x) -> 5 + 2^x
        x = sp.Symbol("x")
        expr = 5 + sp.exp(sp.log(2) * x)
        simplified = simplify_exponential_bases(expr)
        self.assertEqual(simplified, 5 + sp.Pow(2, x))

    def test_formatting_integration(self):
        # implicit verification of format_solution string output
        x = sp.Symbol("x")
        expr = sp.exp(sp.log(2) * x)
        fmt = format_solution(expr)
        # Should be "2^x", not "exp(0.693...*x)" or "2**x"
        self.assertEqual(fmt, "2^x")


if __name__ == "__main__":
    unittest.main()
