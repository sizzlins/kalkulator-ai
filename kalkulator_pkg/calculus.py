"""Dedicated calculus operations module."""

from __future__ import annotations

import sympy as sp

from .parser import parse_preprocessed
from .types import EvalResult
from .worker import evaluate_safely


def differentiate(expression: str, variable: str | None = None) -> EvalResult:
    """Differentiate an expression with respect to a variable.

    Args:
        expression: Expression string (e.g., "x^3")
        variable: Variable to differentiate with respect to (default: first variable found)

    Returns:
        EvalResult with derivative as result
    """
    try:
        eval_result = evaluate_safely(expression)
        if not eval_result.get("ok"):
            return EvalResult(ok=False, error=eval_result.get("error"))

        expr = parse_preprocessed(eval_result["result"])
        free_syms = list(expr.free_symbols)

        if not free_syms:
            return EvalResult(ok=False, error="No variables found in expression")

        if variable:
            var_sym = sp.symbols(variable)
            if var_sym not in free_syms:
                return EvalResult(
                    ok=False, error=f"Variable '{variable}' not found in expression"
                )
            diff_expr = sp.diff(expr, var_sym)
        else:
            # Use first variable found
            diff_expr = sp.diff(expr, free_syms[0])

        return EvalResult(ok=True, result=str(diff_expr))
    except (ValueError, TypeError, AttributeError) as e:
        return EvalResult(ok=False, error=f"Differentiation error: {e}")
    except NotImplementedError as e:
        return EvalResult(ok=False, error=f"Differentiation not implemented: {e}")
    except Exception as e:
        try:
            from .logging_config import get_logger

            logger = get_logger("calculus")
            logger.error(f"Unexpected differentiation error: {e}", exc_info=True)
        except ImportError:
            pass
        return EvalResult(ok=False, error="Differentiation failed unexpectedly")


def integrate(expression: str, variable: str | None = None) -> EvalResult:
    """Integrate an expression with respect to a variable.

    Args:
        expression: Expression string (e.g., "sin(x)")
        variable: Variable to integrate with respect to (default: first variable found)

    Returns:
        EvalResult with integral as result
    """
    try:
        eval_result = evaluate_safely(expression)
        if not eval_result.get("ok"):
            return EvalResult(ok=False, error=eval_result.get("error"))

        expr = parse_preprocessed(eval_result["result"])
        free_syms = list(expr.free_symbols)

        if not free_syms:
            return EvalResult(ok=False, error="No variables found in expression")

        if variable:
            var_sym = sp.symbols(variable)
            if var_sym not in free_syms:
                return EvalResult(
                    ok=False, error=f"Variable '{variable}' not found in expression"
                )
            int_expr = sp.integrate(expr, var_sym)
        else:
            # Use first variable found
            int_expr = sp.integrate(expr, free_syms[0])

        return EvalResult(ok=True, result=str(int_expr))
    except (ValueError, TypeError, AttributeError) as e:
        return EvalResult(ok=False, error=f"Integration error: {e}")
    except NotImplementedError as e:
        return EvalResult(ok=False, error=f"Integration not implemented: {e}")
    except Exception as e:
        try:
            from .logging_config import get_logger

            logger = get_logger("calculus")
            logger.error(f"Unexpected integration error: {e}", exc_info=True)
        except ImportError:
            pass
        return EvalResult(ok=False, error="Integration failed unexpectedly")


def matrix_determinant(matrix_str: str) -> EvalResult:
    """Calculate the determinant of a matrix.

    Args:
        matrix_str: Matrix string (e.g., "Matrix([[1,2],[3,4]])")

    Returns:
        EvalResult with determinant as result
    """
    try:
        eval_result = evaluate_safely(matrix_str)
        if not eval_result.get("ok"):
            return EvalResult(ok=False, error=eval_result.get("error"))

        expr = parse_preprocessed(eval_result["result"])
        if not isinstance(expr, sp.Matrix):
            return EvalResult(ok=False, error="Expression is not a matrix")

        det = sp.det(expr)
        return EvalResult(ok=True, result=str(det))
    except (ValueError, TypeError, AttributeError) as e:
        return EvalResult(ok=False, error=f"Determinant calculation error: {e}")
    except NotImplementedError as e:
        return EvalResult(
            ok=False, error=f"Determinant calculation not implemented: {e}"
        )
    except Exception as e:
        try:
            from .logging_config import get_logger

            logger = get_logger("calculus")
            logger.error(f"Unexpected determinant error: {e}", exc_info=True)
        except ImportError:
            pass
        return EvalResult(ok=False, error="Determinant calculation failed unexpectedly")
