"""Test that API functions return typed dataclasses."""

from kalkulator_pkg.api import det
from kalkulator_pkg.api import diff
from kalkulator_pkg.api import evaluate
from kalkulator_pkg.api import integrate_expr
from kalkulator_pkg.api import solve_equation
from kalkulator_pkg.api import solve_inequality
from kalkulator_pkg.api import solve_system
from kalkulator_pkg.api import validate_expression
from kalkulator_pkg.types import EvalResult
from kalkulator_pkg.types import InequalityResult
from kalkulator_pkg.types import SolveResult


class TestAPITypedReturns:
    """Test that all API functions return typed dataclasses."""

    def test_evaluate_returns_eval_result(self):
        """Test that evaluate() returns EvalResult."""
        result = evaluate("2 + 2")
        assert isinstance(result, EvalResult)
        assert result.ok is True
        assert result.result == "4"

    def test_evaluate_error_returns_eval_result(self):
        """Test that evaluate() errors return EvalResult."""
        result = evaluate("__import__('os')")
        assert isinstance(result, EvalResult)
        assert result.ok is False
        assert result.error is not None

    def test_solve_equation_returns_solve_result(self):
        """Test that solve_equation() returns SolveResult."""
        result = solve_equation("x + 1 = 0")
        assert isinstance(result, SolveResult)
        assert result.ok is True
        assert result.result_type == "equation"
        assert result.exact is not None

    def test_solve_equation_error_returns_solve_result(self):
        """Test that solve_equation() errors return SolveResult."""
        result = solve_equation("x = x = 1")  # Invalid format
        assert isinstance(result, SolveResult)
        assert result.ok is False
        assert result.error is not None

    def test_solve_inequality_returns_inequality_result(self):
        """Test that solve_inequality() returns InequalityResult."""
        result = solve_inequality("x > 0")
        assert isinstance(result, InequalityResult)
        assert result.ok is True
        assert result.result_type == "inequality"

    def test_solve_inequality_error_returns_inequality_result(self):
        """Test that solve_inequality() errors return InequalityResult."""
        result = solve_inequality("x >> 0")  # Invalid
        assert isinstance(result, InequalityResult)
        # May succeed or fail, but always returns InequalityResult

    def test_solve_system_returns_solve_result(self):
        """Test that solve_system() returns SolveResult."""
        result = solve_system("x + y = 3, x - y = 1")
        assert isinstance(result, SolveResult)
        assert result.ok is True
        assert result.result_type in ["system", "equation"]

    def test_solve_system_error_returns_solve_result(self):
        """Test that solve_system() errors return SolveResult."""
        result = solve_system("invalid")
        assert isinstance(result, SolveResult)
        # May succeed or fail, but always returns SolveResult

    def test_validate_expression_returns_tuple(self):
        """Test that validate_expression() returns tuple."""
        is_valid, error = validate_expression("x + 1")
        assert isinstance(is_valid, bool)
        assert error is None or isinstance(error, str)

    def test_diff_returns_eval_result(self):
        """Test that diff() returns EvalResult."""
        result = diff("x^2")
        assert isinstance(result, EvalResult)
        assert result.ok is True
        assert result.result is not None

    def test_integrate_expr_returns_eval_result(self):
        """Test that integrate_expr() returns EvalResult."""
        result = integrate_expr("x")
        assert isinstance(result, EvalResult)
        assert result.ok is True
        assert result.result is not None

    def test_det_returns_eval_result(self):
        """Test that det() returns EvalResult."""
        result = det("Matrix([[1,2],[3,4]])")
        assert isinstance(result, EvalResult)
        # May succeed or fail, but always returns EvalResult

    def test_result_has_repr(self):
        """Test that all result types have __repr__."""
        eval_result = evaluate("2+2")
        assert hasattr(eval_result, "__repr__")
        repr_str = repr(eval_result)
        assert isinstance(repr_str, str)
        assert "EvalResult" in repr_str

    def test_result_has_to_dict(self):
        """Test that all result types have to_dict()."""
        eval_result = evaluate("2+2")
        assert hasattr(eval_result, "to_dict")
        result_dict = eval_result.to_dict()
        assert isinstance(result_dict, dict)
        assert "ok" in result_dict

    def test_solve_result_repr(self):
        """Test SolveResult __repr__."""
        result = solve_equation("x + 1 = 0")
        assert hasattr(result, "__repr__")
        repr_str = repr(result)
        assert isinstance(repr_str, str)
        assert "SolveResult" in repr_str

    def test_inequality_result_repr(self):
        """Test InequalityResult __repr__."""
        result = solve_inequality("x > 0")
        assert hasattr(result, "__repr__")
        repr_str = repr(result)
        assert isinstance(repr_str, str)
        assert "InequalityResult" in repr_str
