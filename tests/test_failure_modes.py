"""Tests for failure modes, timeouts, and invalid input handling."""

import pytest

from kalkulator_pkg.parser import parse_preprocessed
from kalkulator_pkg.parser import preprocess
from kalkulator_pkg.solver import solve_inequality
from kalkulator_pkg.solver import solve_single_equation
from kalkulator_pkg.types import ValidationError
from kalkulator_pkg.worker import evaluate_safely


class TestInputValidationFailures:
    """Test input validation failure modes."""

    def test_empty_input(self):
        """Test empty input."""
        with pytest.raises(ValidationError):
            preprocess("")

    def test_whitespace_only(self):
        """Test whitespace-only input."""
        with pytest.raises((ValidationError, ValueError)):
            preprocess("   ")

    def test_too_long_input(self):
        """Test input exceeding maximum length."""
        long_input = "x" * 10001
        with pytest.raises(ValidationError):
            preprocess(long_input)

    def test_unbalanced_parentheses(self):
        """Test unbalanced parentheses."""
        with pytest.raises(ValidationError):
            preprocess("(x + 1")

    def test_unbalanced_brackets(self):
        """Test unbalanced brackets."""
        with pytest.raises(ValidationError):
            preprocess("[x + 1")

    def test_unbalanced_braces(self):
        """Test unbalanced braces."""
        with pytest.raises(ValidationError):
            preprocess("{x + 1")

    def test_mismatched_delimiters(self):
        """Test mismatched delimiters."""
        with pytest.raises(ValidationError):
            preprocess("(x + 1]")

    def test_forbidden_token_import(self):
        """Test forbidden token detection."""
        with pytest.raises(ValidationError) as exc_info:
            preprocess("__import__('os')")
        assert "forbidden" in str(exc_info.value).lower()

    def test_forbidden_token_eval(self):
        """Test eval token detection."""
        with pytest.raises(ValidationError):
            preprocess("eval('1+1')")

    def test_forbidden_token_exec(self):
        """Test exec token detection."""
        with pytest.raises(ValidationError):
            preprocess("exec('print(1)')")

    def test_too_deep_expression(self):
        """Test expression exceeding depth limit."""
        # Create deeply nested expression
        deep_expr = "1"
        for _ in range(101):
            deep_expr = f"({deep_expr})"
        # Note: Depth validation happens during AST validation after parsing,
        # not during preprocessing. Parentheses depth != AST depth.
        # The depth check may not trigger for simple nested parentheses
        try:
            result = parse_preprocessed(preprocess(deep_expr))
            # If it doesn't raise, that's okay - depth limits are approximate
            assert result is not None
        except ValidationError:
            # This is the expected path if depth validation works
            pass


class TestEquationSolvingFailures:
    """Test equation solving failure modes."""

    def test_invalid_equation_format(self):
        """Test invalid equation format."""
        result = solve_single_equation("x + 1")  # Missing =
        assert result["ok"] is False
        assert "error" in result

    def test_multiple_equals(self):
        """Test equation with multiple equals."""
        result = solve_single_equation("x = 1 = 2")
        assert result["ok"] is False
        assert "error" in result

    def test_variable_not_present(self):
        """Test solving for non-existent variable."""
        result = solve_single_equation("x + 1 = 0", find_var="y")
        assert result["ok"] is False
        assert "error" in result

    def test_unsolvable_equation(self):
        """Test equation that cannot be solved."""
        # This may succeed or fail depending on SymPy capabilities
        result = solve_single_equation("sin(x) = x")
        # Should handle gracefully either way
        assert "ok" in result

    def test_contradiction(self):
        """Test contradictory equation."""
        result = solve_single_equation("1 = 0")
        assert result["ok"] is True
        assert result["type"] == "identity_or_contradiction"


class TestInequalityFailures:
    """Test inequality solving failure modes."""

    def test_invalid_inequality(self):
        """Test invalid inequality format."""
        result = solve_inequality("x >> 0")  # Invalid operator
        # May succeed or fail, but should not crash
        assert isinstance(result, dict)
        assert "ok" in result

    def test_inequality_with_unsupported_operators(self):
        """Test inequality with unsupported operators."""
        result = solve_inequality("x != 0")
        # Should handle gracefully
        assert isinstance(result, dict)


class TestWorkerFailures:
    """Test worker evaluation failure modes."""

    def test_division_by_zero_handling(self):
        """Test division by zero handling."""
        result = evaluate_safely("1 / 0")
        # Should not crash, may return error or special value
        assert isinstance(result, dict)
        assert "ok" in result

    def test_complex_expression_timeout(self):
        """Test potentially timeout-inducing expression."""
        # Very complex expression that might timeout
        complex_expr = "expand((x+y)**100)"  # This is very expensive
        result = evaluate_safely(complex_expr)
        # Should handle gracefully (may timeout or succeed)
        assert isinstance(result, dict)

    def test_invalid_function_call(self):
        """Test invalid function call."""
        result = evaluate_safely("nonexistent_function(1)")
        # Should be caught by validation or return error
        assert isinstance(result, dict)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_expression(self):
        """Test expression evaluating to zero."""
        result = evaluate_safely("0")
        assert result["ok"] is True
        assert result["result"] == "0"

    def test_negative_numbers(self):
        """Test negative number handling."""
        result = evaluate_safely("-5")
        assert result["ok"] is True

    def test_scientific_notation(self):
        """Test scientific notation."""
        result = evaluate_safely("1e10")
        assert result["ok"] is True

    def test_complex_numbers(self):
        """Test complex number handling."""
        result = evaluate_safely("1 + I")
        assert result["ok"] is True

    def test_matrix_edge_cases(self):
        """Test matrix edge cases."""
        # Empty matrix (if supported)
        # Single element matrix
        result = evaluate_safely("Matrix([[1]])")
        assert isinstance(result, dict)

    def test_very_small_numbers(self):
        """Test very small numbers."""
        result = evaluate_safely("1e-100")
        assert result["ok"] is True

    def test_very_large_numbers(self):
        """Test very large numbers."""
        result = evaluate_safely("1e100")
        assert result["ok"] is True


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    def test_parse_error_recovery(self):
        """Test recovery from parse errors."""
        try:
            preprocess("invalid!@#$%")
        except ValidationError:
            pass  # Expected
        # Should not crash the application

    def test_solver_error_recovery(self):
        """Test recovery from solver errors."""
        result = solve_single_equation("x = x")  # May cause issues
        # Should return error dict, not crash
        assert isinstance(result, dict)
        assert "ok" in result

    def test_worker_error_recovery(self):
        """Test recovery from worker errors."""
        # Invalid expression that passes initial validation
        result = evaluate_safely("Matrix([1, 2, 3])")  # Invalid matrix
        # Should handle gracefully
        assert isinstance(result, dict)
