"""Unit tests for solver handler functions."""

import sympy as sp

from kalkulator_pkg.solver import _solve_linear_equation
from kalkulator_pkg.solver import _solve_polynomial_equation
from kalkulator_pkg.solver import _solve_quadratic_equation
from kalkulator_pkg.solver import solve_single_equation


class TestLinearSolver:
    """Test linear equation solving."""

    def test_simple_linear(self):
        """Test simple linear equation."""
        x = sp.symbols("x")
        eq = sp.Eq(x + 1, 0)
        solutions = _solve_linear_equation(eq, x)
        assert len(solutions) > 0
        assert any(sp.simplify(sol + 1) == 0 for sol in solutions)

    def test_linear_with_coefficient(self):
        """Test linear equation with coefficient."""
        x = sp.symbols("x")
        eq = sp.Eq(2 * x + 3, 0)
        solutions = _solve_linear_equation(eq, x)
        assert len(solutions) > 0

    def test_linear_fractional(self):
        """Test linear equation with fractional solution."""
        x = sp.symbols("x")
        eq = sp.Eq(2 * x, 1)
        solutions = _solve_linear_equation(eq, x)
        assert len(solutions) > 0


class TestQuadraticSolver:
    """Test quadratic equation solving."""

    def test_simple_quadratic(self):
        """Test simple quadratic equation."""
        x = sp.symbols("x")
        eq = sp.Eq(x**2 - 1, 0)
        solutions = _solve_quadratic_equation(eq, x)
        assert len(solutions) >= 2

    def test_quadratic_no_real_roots(self):
        """Test quadratic with no real roots."""
        x = sp.symbols("x")
        eq = sp.Eq(x**2 + 1, 0)
        solutions = _solve_quadratic_equation(eq, x)
        # May return complex or empty
        assert isinstance(solutions, list)

    def test_quadratic_perfect_square(self):
        """Test perfect square quadratic."""
        x = sp.symbols("x")
        eq = sp.Eq((x - 2) ** 2, 0)
        solutions = _solve_quadratic_equation(eq, x)
        assert len(solutions) > 0


class TestPolynomialSolver:
    """Test polynomial equation solving."""

    def test_cubic_equation(self):
        """Test cubic equation."""
        x = sp.symbols("x")
        eq = sp.Eq(x**3 - 1, 0)
        solutions = _solve_polynomial_equation(eq, x)
        assert len(solutions) > 0

    def test_quartic_equation(self):
        """Test quartic equation."""
        x = sp.symbols("x")
        eq = sp.Eq(x**4 - 1, 0)
        solutions = _solve_polynomial_equation(eq, x)
        assert len(solutions) > 0


class TestSolverIntegration:
    """Test integration of solver handlers."""

    def test_linear_via_solve_single_equation(self):
        """Test linear equation via main solver."""
        result = solve_single_equation("x + 1 = 0")
        assert result["ok"] is True
        assert result["type"] == "equation"
        assert len(result.get("exact", [])) > 0

    def test_quadratic_via_solve_single_equation(self):
        """Test quadratic equation via main solver."""
        result = solve_single_equation("x^2 - 1 = 0")
        assert result["ok"] is True
        assert result["type"] == "equation"
        assert len(result.get("exact", [])) >= 2

    def test_polynomial_via_solve_single_equation(self):
        """Test polynomial equation via main solver."""
        result = solve_single_equation("x^3 - 8 = 0")
        assert result["ok"] is True
        assert result["type"] == "equation"
        assert len(result.get("exact", [])) > 0
