"""Public API for Kalkulator - returns structured objects without side effects."""

from __future__ import annotations

# Type annotations use modern | syntax (no import needed)
from .calculus import differentiate, integrate, matrix_determinant
from .parser import preprocess
from .plotting import plot_function
from .solver import solve_inequality as _solve_inequality
from .solver import solve_single_equation
from .solver import solve_system as _solve_system
from .types import EvalResult, InequalityResult, SolveResult, ValidationError
from .worker import evaluate_safely


def evaluate(expression: str) -> EvalResult:
    """Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression string (e.g., "2+2", "sin(pi/2)")

    Returns:
        EvalResult with result, approximation, and free symbols
    """
    data = evaluate_safely(expression)
    if not data.get("ok"):
        return EvalResult(ok=False, error=data.get("error") or "Unknown error")
    return EvalResult(
        ok=True,
        result=data.get("result"),
        approx=data.get("approx"),
        free_symbols=data.get("free_symbols"),
    )


def solve_equation(equation: str, find_var: str | None = None) -> SolveResult:
    """Solve a single equation.

    Args:
        equation: Equation string (e.g., "x+1=0", "x^2-1=0")
        find_var: Optional variable to solve for (e.g., "x")

    Returns:
        SolveResult with solutions

    Example:
        >>> from kalkulator_pkg.api import solve_equation
        >>> result = solve_equation("x + 1 = 0")
        >>> print(result.exact)
        ['-1']
        >>> result = solve_equation("x^2 - 4 = 0")
        >>> print(result.exact)
        ['-2', '2']
    """
    data = solve_single_equation(equation, find_var)
    if not data.get("ok"):
        return SolveResult(ok=False, result_type="equation", error=data.get("error"))

    result_type = data.get("type", "equation")
    if result_type == "pell":
        return SolveResult(ok=True, result_type="pell", solution=data.get("solution"))
    elif result_type == "identity_or_contradiction":
        result_str = data.get("result", "")
        return SolveResult(
            ok=True, result_type="identity_or_contradiction", error=result_str
        )
    elif result_type == "multi_isolate":
        return SolveResult(
            ok=True,
            result_type="multi_isolate",
            solutions=data.get("solutions"),
            exact=None,  # multi_isolate uses solutions dict
            approx=data.get("approx"),
        )
    else:  # equation
        return SolveResult(
            ok=True,
            result_type="equation",
            exact=data.get("exact"),
            approx=data.get("approx"),
        )


def solve_inequality(inequality: str, find_var: str | None = None) -> InequalityResult:
    """Solve an inequality.

    Args:
        inequality: Inequality string (e.g., "x > 0", "1 < x < 5")
        find_var: Optional variable to solve for

    Returns:
        InequalityResult with solutions

    Example:
        >>> from kalkulator_pkg.api import solve_inequality
        >>> result = solve_inequality("x > 0")
        >>> print(result.solutions)
        {'x': 'x > 0'}
        >>> result = solve_inequality("1 < x < 5")
        >>> print(result.solutions)
        {'x': '1 < x < 5'}
    """
    data = _solve_inequality(inequality, find_var)
    if not data.get("ok"):
        return InequalityResult(ok=False, error=data.get("error"))
    return InequalityResult(ok=True, solutions=data.get("solutions"))


def solve_system(equations: str, find_var: str | None = None) -> SolveResult:
    """Solve a system of equations.

    Args:
        equations: Comma-separated equations (e.g., "x+y=3, x-y=1")
        find_var: Optional variable to find in solutions

    Returns:
        SolveResult with system solutions

    Example:
        >>> from kalkulator_pkg.api import solve_system
        >>> result = solve_system("x+y=3, x-y=1")
        >>> print(result.system_solutions)
        [{'x': '2', 'y': '1'}]
        >>> result = solve_system("x+y=3, x-y=1", find_var="x")
        >>> print(result.exact)
        ['2']
    """
    data = _solve_system(equations, find_var)
    if not data.get("ok"):
        return SolveResult(ok=False, result_type="system", error=data.get("error"))

    result_type = data.get("type", "system")
    if result_type == "system":
        return SolveResult(
            ok=True, result_type="system", system_solutions=data.get("solutions")
        )
    else:  # system_var
        return SolveResult(
            ok=True,
            result_type="equation",  # single variable result
            exact=data.get("exact"),
            approx=data.get("approx"),
        )


def validate_expression(expression: str) -> tuple[bool, str | None]:
    """Validate an expression without evaluating it.

    Args:
        expression: Expression string to validate

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> from kalkulator_pkg.api import validate_expression
        >>> is_valid, error = validate_expression("2 + 2")
        >>> print(is_valid)
        True
        >>> is_valid, error = validate_expression("import os")
        >>> print(is_valid)
        False
        >>> print(error)
        'Input contains forbidden token: import'
    """
    try:
        preprocess(expression)
        return True, None
    except (ValueError, ValidationError) as e:
        return False, str(e)
    except (TypeError, AttributeError) as e:
        return False, f"Validation error: {e}"
    except Exception as e:
        # Log unexpected errors for debugging
        try:
            from .logging_config import get_logger

            logger = get_logger("api")
            logger.warning(f"Unexpected validation error: {e}", exc_info=True)
        except ImportError:
            pass
        return False, "Unexpected validation error"


def diff(expression: str, variable: str | None = None) -> EvalResult:
    """Differentiate an expression.

    Args:
        expression: Expression to differentiate (e.g., "x^3")
        variable: Variable to differentiate with respect to (optional)

    Returns:
        EvalResult with derivative

    Example:
        >>> from kalkulator_pkg.api import diff
        >>> result = diff("x^3")
        >>> print(result.result)
        '3*x**2'
        >>> result = diff("sin(x)", "x")
        >>> print(result.result)
        'cos(x)'
    """
    return differentiate(expression, variable)


def integrate_expr(expression: str, variable: str | None = None) -> EvalResult:
    """Integrate an expression.

    Args:
        expression: Expression to integrate (e.g., "sin(x)")
        variable: Variable to integrate with respect to (optional)

    Returns:
        EvalResult with integral

    Example:
        >>> from kalkulator_pkg.api import integrate_expr
        >>> result = integrate_expr("x")
        >>> print(result.result)
        'x**2/2'
        >>> result = integrate_expr("sin(x)", "x")
        >>> print(result.result)
        '-cos(x)'
    """
    return integrate(expression, variable)


def det(matrix_str: str) -> EvalResult:
    """Calculate matrix determinant.

    Args:
        matrix_str: Matrix expression (e.g., "Matrix([[1,2],[3,4]])")

    Returns:
        EvalResult with determinant

    Example:
        >>> from kalkulator_pkg.api import det
        >>> result = det("Matrix([[1,2],[3,4]])")
        >>> print(result.result)
        '-2'
    """
    return matrix_determinant(matrix_str)


def plot(
    expression: str,
    variable: str = "x",
    x_min: float = -10,
    x_max: float = 10,
    ascii: bool = False,
) -> EvalResult:
    """Plot a single-variable function.

    Args:
        expression: Function expression
        variable: Variable name
        x_min: Minimum x value
        x_max: Maximum x value
        ascii: Return ASCII plot instead of opening window

    Returns:
        EvalResult with plot data

    Example:
        >>> from kalkulator_pkg.api import plot
        >>> result = plot("x^2", x_min=-5, x_max=5, ascii=True)
        >>> print(result.ok)  # True if successful
        True
        >>> # For GUI plot, set ascii=False (requires matplotlib)
    """
    return plot_function(expression, variable, x_min, x_max, ascii=ascii)
