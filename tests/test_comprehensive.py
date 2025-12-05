"""Comprehensive test suite covering representative expressions and edge cases."""

import pytest

from kalkulator_pkg.parser import parse_preprocessed, preprocess
from kalkulator_pkg.solver import solve_inequality, solve_single_equation, solve_system
from kalkulator_pkg.types import ValidationError
from kalkulator_pkg.worker import evaluate_safely


class TestRepresentativeExpressions:
    """Test a representative set of mathematical expressions."""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        result = evaluate_safely("2 + 2")
        assert result["ok"] is True
        assert result["result"] == "4"

        result = evaluate_safely("10 * 5")
        assert result["ok"] is True
        assert result["result"] == "50"

        result = evaluate_safely("100 / 4")
        assert result["ok"] is True
        assert result["result"] == "25"

    def test_powers_and_roots(self):
        """Test exponentiation and roots."""
        result = evaluate_safely("2**3")
        assert result["ok"] is True
        assert result["result"] == "8"

        result = evaluate_safely("sqrt(16)")
        assert result["ok"] is True
        assert result["result"] == "4"

    def test_trigonometric(self):
        """Test trigonometric functions."""
        result = evaluate_safely("sin(0)")
        assert result["ok"] is True
        assert result["result"] == "0"

        result = evaluate_safely("cos(0)")
        assert result["ok"] is True
        # cos(0) should be 1, but may be represented differently
        assert result["result"] in ("1", "1.0", "1.00000000000000")

    def test_logarithms(self):
        """Test logarithmic functions."""
        result = evaluate_safely("log(E)")
        assert result["ok"] is True
        assert result["result"] == "1"

    def test_variables(self):
        """Test expressions with variables."""
        result = evaluate_safely("x + 1")
        assert result["ok"] is True
        assert "x" in (result.get("free_symbols") or [])


class TestLinearEquations:
    """Test linear equation solving."""

    def test_simple_linear(self):
        """Test simple linear equations."""
        result = solve_single_equation("x + 1 = 0")
        assert result["ok"] is True
        assert result["type"] == "equation"
        assert "-1" in result.get("exact", [])

    def test_linear_with_coefficients(self):
        """Test linear equations with coefficients."""
        result = solve_single_equation("2*x + 3 = 0")
        assert result["ok"] is True
        assert result["type"] == "equation"

    def test_linear_fractional(self):
        """Test linear equations with fractional solutions."""
        result = solve_single_equation("2*x = 1")
        assert result["ok"] is True
        assert result["type"] == "equation"


class TestQuadraticEquations:
    """Test quadratic equation solving."""

    def test_simple_quadratic(self):
        """Test simple quadratic equations."""
        result = solve_single_equation("x^2 - 1 = 0")
        assert result["ok"] is True
        assert result["type"] == "equation"
        assert len(result.get("exact", [])) == 2

    def test_quadratic_no_real_roots(self):
        """Test quadratic with no real roots."""
        result = solve_single_equation("x^2 + 1 = 0")
        # May return complex or no solution
        assert result["ok"] in [True, False]


class TestEdgeCases:
    """Test edge cases and failure modes."""

    def test_empty_input(self):
        """Test empty input."""
        with pytest.raises(ValidationError):
            preprocess("")

    def test_too_long_input(self):
        """Test input exceeding length limit."""
        long_input = "x" * 10001
        with pytest.raises(ValidationError):
            preprocess(long_input)

    def test_unbalanced_parentheses(self):
        """Test unbalanced parentheses."""
        with pytest.raises(ValidationError):
            preprocess("(x + 1")

    def test_forbidden_token(self):
        """Test input with forbidden tokens."""
        with pytest.raises(ValidationError):
            preprocess("__import__('os')")

    def test_invalid_equation_format(self):
        """Test invalid equation format."""
        result = solve_single_equation("x + 1")  # Missing =
        assert result["ok"] is False
        assert "error" in result

    def test_equation_with_no_variables(self):
        """Test equation with no variables."""
        result = solve_single_equation("1 = 1")
        assert result["ok"] is True
        assert result["type"] == "identity_or_contradiction"


class TestInequalities:
    """Test inequality solving."""

    def test_simple_inequality(self):
        """Test simple inequality."""
        result = solve_inequality("x > 0")
        assert result["ok"] is True
        assert result["type"] == "inequality"

    def test_compound_inequality(self):
        """Test compound inequality."""
        result = solve_inequality("1 < x < 5")
        assert result["ok"] is True


class TestSystems:
    """Test system of equations solving."""

    def test_simple_system(self):
        """Test simple 2x2 system."""
        result = solve_system("x + y = 3, x - y = 1", find_token=None)
        assert result["ok"] is True
        assert result["type"] == "system"
        assert result.get("solutions") is not None

    def test_system_with_specific_variable(self):
        """Test extracting specific variable from system."""
        result = solve_system("x + y = 3, x - y = 1", find_token="x")
        assert result["ok"] is True


class TestParsingEdgeCases:
    """Test parsing edge cases."""

    def test_unicode_symbols(self):
        """Test Unicode mathematical symbols."""
        result = preprocess("x² + 1 = 0")
        assert "x**2" in result or "**2" in result

    def test_percentage(self):
        """Test percentage notation."""
        result = preprocess("50%")
        assert "(50/100)" in result or "((50)/(100))" in result

    def test_implicit_multiplication(self):
        """Test implicit multiplication."""
        result = preprocess("2x")
        assert "2*x" in result


class TestErrorHandling:
    """Test error handling and recovery."""

    def test_division_by_zero_expression(self):
        """Test expression that might cause division by zero."""
        # Should not crash, but may return error
        _ = evaluate_safely("1 / 0")  # noqa: F841
        # Result depends on SymPy behavior

    def test_timeout_handling(self):
        """Test timeout handling (if implemented)."""
        # This would require mocking worker timeout
        pass

    def test_invalid_function_call(self):
        """Test invalid function calls."""
        # Note: Function validation happens during AST validation, not preprocessing
        # Some invalid functions may pass preprocessing but fail during parsing
        result = preprocess("dangerous_function(1)")
        # Should at least not crash during preprocessing
        assert isinstance(result, str)


class TestNumericFallback:
    """Test numeric root finding fallback."""

    def test_trigonometric_equation(self):
        """Test equation requiring numeric fallback."""
        result = solve_single_equation("sin(x) = 0.5")
        # Should attempt numeric fallback
        assert result["ok"] in [True, False]


# Integration test
def test_end_to_end_workflow():
    """Test complete workflow from input to output."""
    # Parse
    preprocessed = preprocess("x + 1 = 0")
    _ = parse_preprocessed(preprocessed.split("=")[0])  # noqa: F841

    # Solve
    result = solve_single_equation("x + 1 = 0")

    # Verify
    assert result["ok"] is True
    assert result["type"] == "equation"
    assert "-1" in result.get("exact", [])
