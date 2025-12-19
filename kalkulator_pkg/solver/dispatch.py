from __future__ import annotations

from typing import Any

import sympy as sp

from ..config import NUMERIC_FALLBACK_ENABLED
from ..config import NUMERIC_TOLERANCE
from ..parser import parse_preprocessed
from ..types import ParseError
from ..types import ValidationError
from ..worker import evaluate_safely
from .algebraic import _solve_linear_equation
from .algebraic import _solve_polynomial_equation
from .algebraic import _solve_quadratic_equation
from .algebraic import is_pell_equation_from_eq
from .algebraic import solve_pell_equation_from_eq
from .modular import _solve_modulo_equation
from .numeric import _numeric_roots_for_single_var

try:
    from ..logging_config import get_logger

    logger = get_logger("solver.dispatch")
except ImportError:
    import logging

    logger = logging.getLogger("solver.dispatch")


def solve_single_equation(
    eq_str: str,
    find_var: str | None = None,
    allowed_functions: frozenset[str] | None = None,
) -> dict[str, Any]:
    """
    Solve a single equation.

    Args:
        eq_str: Equation string (e.g., "x+1=0", "x^2-1=0")
        find_var: Optional variable to solve for (e.g., "x")

    Returns:
        Dictionary with keys:
            - ok: Boolean indicating success
            - type: Result type ("equation", "pell", "identity_or_contradiction", "multi_isolate")
            - exact: List of exact solutions (strings)
            - approx: List of approximate solutions (strings or None)
            - error: Error message if ok is False
    """
    parts = eq_str.split("=", 1)
    if len(parts) != 2:
        return {
            "ok": False,
            "error": "Invalid equation format: Expected exactly one '='. Use format like 'x+1=0' or 'x^2=4'.",
            "error_code": "INVALID_FORMAT",
        }
    lhs_s, rhs_s = parts[0].strip(), parts[1].strip()

    # Handle empty RHS: treat as evaluation of LHS
    if not rhs_s:
        # If RHS is empty, evaluate the LHS expression
        lhs_eval = evaluate_safely(lhs_s, allowed_functions=allowed_functions)
        if lhs_eval.get("ok"):
            return {
                "ok": True,
                "type": "evaluation",
                "exact": [lhs_eval.get("result") or ""],
                "approx": [lhs_eval.get("approx")],
            }
        else:
            return {
                "ok": False,
                "error": f"Failed to evaluate '{lhs_s}': {lhs_eval.get('error')}",
                "error_code": lhs_eval.get("error_code", "EVAL_ERROR"),
            }

    rhs_s = rhs_s or "0"
    # Clear cache hits at start to track fresh hits for this equation
    try:
        from ..cache_manager import clear_cache_hits

        clear_cache_hits()
    except ImportError:
        pass
    lhs = evaluate_safely(lhs_s, allowed_functions=allowed_functions)
    # Capture cache hits from LHS evaluation
    cache_hits: list[tuple[str, str]] = []
    cache_hits.extend(lhs.get("cache_hits", []))
    if not lhs.get("ok"):
        error_msg = f"Failed to parse left-hand side '{lhs_s}': {lhs.get('error')}"
        error_code = lhs.get("error_code", "PARSE_ERROR")

        # Provide helpful hints for common syntax errors
        if error_code == "UNBALANCED_PARENS":
            error_msg += ". Hint: Make sure your equation uses the format 'expression1 = expression2' (with spaces around =). For example, use 'x^x = log(25^x)/x' instead of '((x^x)=log(25^x))/(x)'."
        elif error_code == "UNMATCHED_QUOTES":
            error_msg += ". Check that all quotes are properly matched."
        elif error_code == "SYNTAX_ERROR":
            # Additional hints may already be in the error message
            pass
        elif "cannot assign" in str(lhs.get("error", "")).lower():
            error_msg += ". Hint: The left-hand side appears to contain an assignment '='. If you're trying to solve an equation, use '==' for comparison, not '='. For example: 'x == 5' not 'x = 5'."
        else:
            error_msg += ". Please check your input syntax."

        return {
            "ok": False,
            "error": error_msg,
            "error_code": error_code,
        }
    rhs = evaluate_safely(rhs_s, allowed_functions=allowed_functions)
    # Capture cache hits from RHS evaluation
    cache_hits.extend(rhs.get("cache_hits", []))
    if not rhs.get("ok"):
        error_msg = f"Failed to parse right-hand side '{rhs_s}': {rhs.get('error')}"
        error_code = rhs.get("error_code", "PARSE_ERROR")

        # Provide helpful hints for common syntax errors
        if error_code == "UNBALANCED_PARENS":
            error_msg += ". Hint: Make sure your equation uses the format 'expression1 = expression2' (with spaces around =). For example, use 'x^x = log(25^x)/x' instead of '((x^x)=log(25^x))/(x)'."
        elif error_code == "UNMATCHED_QUOTES":
            error_msg += ". Check that all quotes are properly matched."
        elif error_code == "SYNTAX_ERROR":
            # Additional hints may already be in the error message
            pass
        elif "cannot assign" in str(rhs.get("error", "")).lower():
            error_msg += ". Hint: The right-hand side contains an equation '= 0'. If you're assigning a variable to an equation result, you cannot nest equations inside assignments. Try: 'a = expression' then solve 'expression = 0' separately."
        else:
            # Don't add duplicate "Please check your input syntax" if error message already contains it
            # or if it already ends with a period (user-friendly messages are usually complete)
            if (
                "Please check your input syntax" not in error_msg
                and not error_msg.endswith(".")
            ):
                error_msg += ". Please check your input syntax."

        return {
            "ok": False,
            "error": error_msg,
            "error_code": error_code,
        }
    try:
        left_expr = parse_preprocessed(
            lhs["result"], allowed_functions=allowed_functions
        )
        right_expr = parse_preprocessed(
            rhs["result"], allowed_functions=allowed_functions
        )
    except (ParseError, ValidationError) as e:
        logger.warning("Parse error assembling SymPy expressions", exc_info=True)
        return {"ok": False, "error": f"Parse error: {e}", "error_code": "PARSE_ERROR"}
    except (ValueError, TypeError) as e:
        logger.warning("Type error assembling SymPy expressions", exc_info=True)
        return {"ok": False, "error": f"Type error: {e}", "error_code": "TYPE_ERROR"}
    equation = sp.Eq(left_expr, right_expr)
    if is_pell_equation_from_eq(equation):
        try:
            pell_str = solve_pell_equation_from_eq(equation)
            # Don't prettify Pell solutions to avoid Unicode issues on Windows
            return {
                "ok": True,
                "type": "pell",
                "solution": pell_str,
                "cache_hits": cache_hits,
            }
        except ValueError as e:
            logger.warning("Pell solver error: invalid equation", exc_info=True)
            return {
                "ok": False,
                "error": f"Pell solver error: {e}",
                "error_code": "PELL_SOLVER_ERROR",
            }
        except Exception as e:
            logger.error("Unexpected error in Pell solver", exc_info=True)
            return {
                "ok": False,
                "error": f"Pell solver error: {e}",
                "error_code": "PELL_SOLVER_ERROR",
            }
    symbols = list(equation.free_symbols)
    if not symbols:
        try:
            simp = sp.simplify(left_expr - right_expr)
            if simp == 0:
                return {
                    "ok": True,
                    "type": "identity_or_contradiction",
                    "result": "Identity",
                    "cache_hits": cache_hits,
                }
        except (TypeError, ValueError, AttributeError, NotImplementedError):
            # These are expected for some expressions that can't be simplified
            simp = None
        except Exception as e:
            # Unexpected error - log it
            logger.debug(f"Unexpected error in simplify check: {e}", exc_info=True)
            simp = None
        try:
            # First, try SymPy's equals() method which handles symbolic equality well
            # This can catch cases like pi expressions that are symbolically equal
            try:
                if left_expr.equals(right_expr):
                    return {
                        "ok": True,
                        "type": "identity_or_contradiction",
                        "result": "Identity",
                        "cache_hits": cache_hits,
                    }
            except Exception:
                # equals() might fail for some expressions, continue to numeric check
                pass

            # Numeric comparison with relative tolerance
            # Use higher precision for evaluation
            diff = sp.N(left_expr - right_expr, 60)
            re_diff = sp.re(diff)
            im_diff = sp.im(diff)

            # Calculate relative tolerance based on the magnitude of the expressions
            # This handles both small and large numbers better
            try:
                left_val = abs(float(sp.N(left_expr, 30)))
                right_val = abs(float(sp.N(right_expr, 30)))
                max_magnitude = max(
                    left_val, right_val, 1.0
                )  # At least 1.0 to avoid division by very small numbers
                # Use relative tolerance: 1e-10 relative to the magnitude
                rel_tol = max_magnitude * 1e-10
                # Also use absolute tolerance for very small numbers
                abs_tol = 1e-12
                tol = max(rel_tol, abs_tol)
            except (TypeError, ValueError, OverflowError):
                # Fallback to absolute tolerance if relative calculation fails
                tol = 1e-12

            if abs(re_diff) < tol and abs(im_diff) < tol:
                return {
                    "ok": True,
                    "type": "identity_or_contradiction",
                    "result": "Identity (numeric)",
                    "cache_hits": cache_hits,
                }
            else:
                return {
                    "ok": True,
                    "type": "identity_or_contradiction",
                    "result": "Contradiction (numeric)",
                    "cache_hits": cache_hits,
                }
        except (TypeError, ValueError, AttributeError):
            # Expected for some expressions that can't be evaluated numerically
            return {
                "ok": True,
                "type": "identity_or_contradiction",
                "result": "Contradiction (unable to confirm identity symbolically or numerically)",
                "cache_hits": cache_hits,
            }
        except Exception as e:
            # Unexpected error - log it but still return reasonable result
            logger.debug(
                f"Unexpected error in numeric identity check: {e}", exc_info=True
            )
            return {
                "ok": True,
                "type": "identity_or_contradiction",
                "result": "Contradiction (unable to confirm identity symbolically or numerically)",
                "cache_hits": cache_hits,
            }

    # Use module-level _numeric_roots_for_single_var function (defined above)
    try:
        if find_var:
            sym = sp.symbols(find_var)
            if sym not in symbols:
                return {
                    "ok": False,
                    "error": f"Variable '{find_var}' not present.",
                    "error_code": "VARIABLE_NOT_FOUND",
                }
            sols = sp.solve(equation, sym)
            # Filter to only real solutions
            real_sols = []
            real_approx = []
            for solution in sols if isinstance(sols, (list, tuple)) else [sols]:
                try:
                    num_val = sp.N(solution)
                    # Check if solution is real (imaginary part is negligible)
                    if abs(sp.im(num_val)) < NUMERIC_TOLERANCE:
                        real_sols.append(solution)
                        real_approx.append(str(sp.re(num_val)))
                except (ValueError, TypeError, OverflowError, ArithmeticError):
                    # Can't evaluate numerically - check if it's obviously complex
                    sol_str = str(solution)
                    if "I" not in sol_str:
                        # Might be real but can't evaluate - include it
                        real_sols.append(solution)
                        real_approx.append(None)
                    # Otherwise skip complex solutions

            if not real_sols:
                # Check if equation is impossible (e.g., sin(x) = pi)
                error_hint = None
                if equation.has(sp.sin):
                    try:
                        if equation.lhs.has(sp.sin) and not equation.rhs.has(sp.sin):
                            rhs_val = float(sp.N(equation.rhs))
                            if abs(rhs_val) > 1:
                                error_hint = f"sin({find_var}) cannot equal {rhs_val} (|sin({find_var})| <= 1)"
                        elif equation.rhs.has(sp.sin) and not equation.lhs.has(sp.sin):
                            lhs_val = float(sp.N(equation.lhs))
                            if abs(lhs_val) > 1:
                                error_hint = f"sin({find_var}) cannot equal {lhs_val} (|sin({find_var})| <= 1)"
                    except (ValueError, TypeError, AttributeError):
                        pass
                if equation.has(sp.cos) and not error_hint:
                    try:
                        if equation.lhs.has(sp.cos) and not equation.rhs.has(sp.cos):
                            rhs_val = float(sp.N(equation.rhs))
                            if abs(rhs_val) > 1:
                                error_hint = f"cos({find_var}) cannot equal {rhs_val} (|cos({find_var})| <= 1)"
                        elif equation.rhs.has(sp.cos) and not equation.lhs.has(sp.cos):
                            lhs_val = float(sp.N(equation.lhs))
                            if abs(lhs_val) > 1:
                                error_hint = f"cos({find_var}) cannot equal {lhs_val} (|cos({find_var})| <= 1)"
                    except (ValueError, TypeError, AttributeError):
                        pass

                if error_hint:
                    return {
                        "ok": False,
                        "error": f"This equation has no real solutions: {error_hint}.",
                        "error_code": "NO_REAL_SOLUTIONS",
                    }
                else:
                    return {
                        "ok": False,
                        "error": "This equation has no real solutions (only complex solutions exist).",
                        "error_code": "NO_REAL_SOLUTIONS",
                    }

            exacts = [str(s) for s in real_sols]
            return {
                "ok": True,
                "type": "equation",
                "exact": exacts,
                "approx": real_approx,
            }
        if len(symbols) == 1:
            sym = symbols[0]
            # Check if equation contains trigonometric functions - use numeric fallback directly
            if NUMERIC_FALLBACK_ENABLED and equation.has(sp.sin, sp.cos, sp.tan):
                equation_expr = left_expr - right_expr
                numeric_roots = _numeric_roots_for_single_var(
                    equation_expr, sym, interval=(-4 * sp.pi, 4 * sp.pi)
                )
                if numeric_roots:
                    exacts = [str(r) for r in numeric_roots]
                    approx = [str(sp.N(r)) for r in numeric_roots]
                    return {
                        "ok": True,
                        "type": "equation",
                        "exact": exacts,
                        "approx": approx,
                    }
                # No numeric roots found - equation may be unsolvable or have no real solutions
                # For equations like sin(x)=pi/2 (which has no real solutions since |sin(x)| <= 1)
                # Try sp.solve() for exact symbolic solution, but catch generator errors gracefully
                try:
                    sols = sp.solve(equation, sym)
                    if sols:
                        # Filter to only real solutions
                        real_sols = []
                        real_approx = []
                        for s in sols if isinstance(sols, (list, tuple)) else [sols]:
                            try:
                                num_val = sp.N(s)
                                # Check if solution is real (imaginary part is negligible)
                                if abs(sp.im(num_val)) < NUMERIC_TOLERANCE:
                                    real_sols.append(s)
                                    real_approx.append(str(sp.re(num_val)))
                                else:
                                    # Solution is complex - check if it's from an impossible trig equation
                                    s_str = str(s)
                                    # Check for asin/acos with argument > 1
                                    if "asin" in s_str.lower():
                                        import re

                                        match = re.search(
                                            r"asin\(([^)]+)\)", s_str, re.IGNORECASE
                                        )
                                        if match:
                                            try:
                                                inner_val = float(sp.N(match.group(1)))
                                                if abs(inner_val) > 1:
                                                    # This is from an impossible equation (e.g., sin(x) = pi where pi > 1)
                                                    pass  # Will be handled below
                                            except (
                                                ValueError,
                                                TypeError,
                                                AttributeError,
                                            ):
                                                pass
                            except (
                                ValueError,
                                TypeError,
                                OverflowError,
                                ArithmeticError,
                            ):
                                # Can't evaluate - might be symbolic, skip for now
                                pass

                        # If we found real solutions, return them
                        if real_sols:
                            exacts = [str(s) for s in real_sols]
                            return {
                                "ok": True,
                                "type": "equation",
                                "exact": exacts,
                                "approx": real_approx,
                            }

                        # No real solutions found - check if equation is impossible
                        # Check if sin(x) = k or cos(x) = k where |k| > 1
                        error_hint = None
                        if equation.has(sp.sin):
                            try:
                                if equation.lhs.has(sp.sin) and not equation.rhs.has(
                                    sp.sin
                                ):
                                    rhs_val = float(sp.N(equation.rhs))
                                    if abs(rhs_val) > 1:
                                        error_hint = f"sin(x) cannot equal {rhs_val} (|sin(x)| <= 1)"
                                elif equation.rhs.has(sp.sin) and not equation.lhs.has(
                                    sp.sin
                                ):
                                    lhs_val = float(sp.N(equation.lhs))
                                    if abs(lhs_val) > 1:
                                        error_hint = f"sin(x) cannot equal {lhs_val} (|sin(x)| <= 1)"
                            except (ValueError, TypeError, AttributeError):
                                pass
                        if equation.has(sp.cos) and not error_hint:
                            try:
                                if equation.lhs.has(sp.cos) and not equation.rhs.has(
                                    sp.cos
                                ):
                                    rhs_val = float(sp.N(equation.rhs))
                                    if abs(rhs_val) > 1:
                                        error_hint = f"cos(x) cannot equal {rhs_val} (|cos(x)| <= 1)"
                                elif equation.rhs.has(sp.cos) and not equation.lhs.has(
                                    sp.cos
                                ):
                                    lhs_val = float(sp.N(equation.lhs))
                                    if abs(lhs_val) > 1:
                                        error_hint = f"cos(x) cannot equal {lhs_val} (|cos(x)| <= 1)"
                            except (ValueError, TypeError, AttributeError):
                                pass

                        if error_hint:
                            return {
                                "ok": False,
                                "error": f"This trigonometric equation has no real solutions: {error_hint}.",
                                "error_code": "NO_REAL_SOLUTIONS",
                            }
                        else:
                            return {
                                "ok": False,
                                "error": "This trigonometric equation has no real solutions (only complex solutions exist).",
                                "error_code": "NO_REAL_SOLUTIONS",
                            }
                    # If sp.solve() returns empty list, check for impossible inverse trig equations
                    # Check for asin/acos/atan with impossible target values
                    error_hint = None
                    if equation.has(sp.asin):
                        try:
                            # asin(x) = k: range of asin is [-pi/2, pi/2]
                            if equation.lhs.has(sp.asin) and not equation.rhs.has(
                                sp.asin
                            ):
                                rhs_val = float(sp.N(equation.rhs))
                                if abs(rhs_val) > sp.pi / 2:
                                    error_hint = f"asin(x) cannot equal {rhs_val} (range of asin is [-pi/2, pi/2])"
                            elif equation.rhs.has(sp.asin) and not equation.lhs.has(
                                sp.asin
                            ):
                                lhs_val = float(sp.N(equation.lhs))
                                if abs(lhs_val) > sp.pi / 2:
                                    error_hint = f"asin(x) cannot equal {lhs_val} (range of asin is [-pi/2, pi/2])"
                        except (ValueError, TypeError, AttributeError):
                            pass
                    if equation.has(sp.acos) and not error_hint:
                        try:
                            # acos(x) = k: range of acos is [0, pi]
                            if equation.lhs.has(sp.acos) and not equation.rhs.has(
                                sp.acos
                            ):
                                rhs_val = float(sp.N(equation.rhs))
                                if rhs_val < 0 or rhs_val > sp.pi:
                                    error_hint = f"acos(x) cannot equal {rhs_val} (range of acos is [0, pi])"
                            elif equation.rhs.has(sp.acos) and not equation.lhs.has(
                                sp.acos
                            ):
                                lhs_val = float(sp.N(equation.lhs))
                                if lhs_val < 0 or lhs_val > sp.pi:
                                    error_hint = f"acos(x) cannot equal {lhs_val} (range of acos is [0, pi])"
                        except (ValueError, TypeError, AttributeError):
                            pass
                    if equation.has(sp.atan) and not error_hint:
                        try:
                            # atan(x) = k: range of atan is (-pi/2, pi/2), but we'll be lenient
                            # atan can actually approach but never equal pi/2 or -pi/2
                            if equation.lhs.has(sp.atan) and not equation.rhs.has(
                                sp.atan
                            ):
                                rhs_val = float(sp.N(equation.rhs))
                                if abs(rhs_val) >= sp.pi / 2:
                                    error_hint = f"atan(x) cannot equal {rhs_val} (range of atan is (-pi/2, pi/2))"
                            elif equation.rhs.has(sp.atan) and not equation.lhs.has(
                                sp.atan
                            ):
                                lhs_val = float(sp.N(equation.lhs))
                                if abs(lhs_val) >= sp.pi / 2:
                                    error_hint = f"atan(x) cannot equal {lhs_val} (range of atan is (-pi/2, pi/2))"
                        except (ValueError, TypeError, AttributeError):
                            pass

                    if error_hint:
                        return {
                            "ok": False,
                            "error": f"This equation has no real solutions: {error_hint}.",
                            "error_code": "NO_REAL_SOLUTIONS",
                        }

                    return {
                        "ok": False,
                        "error": "No real solutions found for this trigonometric equation.",
                        "error_code": "NO_REAL_SOLUTIONS",
                    }
                except Exception as solve_err:
                    error_msg = str(solve_err).lower()
                    # Check for the specific generator error that occurs with trigonometric functions
                    # This can be ValueError, TypeError, or other SymPy-specific exceptions
                    if "generators" in error_msg or "contains an element" in error_msg:
                        return {
                            "ok": False,
                            "error": "This trigonometric equation cannot be solved symbolically. No real solutions found in the search interval.",
                            "error_code": "NO_REAL_SOLUTIONS",
                        }
                    # For other errors, return appropriate error message
                    return {
                        "ok": False,
                        "error": f"Solving error: {solve_err}",
                        "error_code": "SOLVER_ERROR",
                    }
            # Try specialized handlers first for polynomial equations
            try:
                # Check for modulo equations first (before polynomial detection)
                if equation.has(sp.Mod):
                    handler_sols = _solve_modulo_equation(equation, sym)
                    if handler_sols:
                        # Format modulo solutions nicely
                        formatted_sols = []
                        for sol in handler_sols:
                            sol_str = str(sp.simplify(sol))
                            formatted_sols.append(sol_str)
                        exact_sols_list = formatted_sols
                        # Also compute some numeric examples
                        approx_sols_list = []
                        try:
                            # Extract n and k from Mod(x, n) = k
                            if isinstance(equation.lhs, sp.Mod):
                                n = float(sp.N(equation.lhs.args[1]))
                                k = float(sp.N(equation.rhs))
                                # Generate some example solutions
                                example_sols = []
                                for t_val in range(-3, 4):
                                    x_val = k + n * t_val
                                    example_sols.append(x_val)
                                approx_sols_list = [
                                    str(int(x)) if x == int(x) else str(x)
                                    for x in example_sols
                                ]
                        except (ValueError, TypeError, AttributeError):
                            pass
                        # Continue to format and return results
                        if exact_sols_list or approx_sols_list:
                            return {
                                "ok": True,
                                "type": "equation",
                                "exact": exact_sols_list,
                                "approx": approx_sols_list,
                                "cache_hits": cache_hits,
                            }

                # Detect equation type and route to appropriate handler
                poly = sp.Poly(equation.lhs - equation.rhs, sym)
                if poly is not None and poly.degree() > 0:
                    if poly.degree() == 1:
                        # Linear equation: use specialized handler
                        handler_sols = _solve_linear_equation(equation, sym)
                        if handler_sols:
                            sols = handler_sols
                        else:
                            # Fallback to general solve
                            sols = sp.solve(equation, sym)
                    elif poly.degree() == 2:
                        # Quadratic equation: use specialized handler
                        handler_sols = _solve_quadratic_equation(equation, sym)
                        if handler_sols:
                            sols = handler_sols
                        else:
                            # Fallback to general solve
                            sols = sp.solve(equation, sym)
                    elif poly.degree() > 2:
                        # Higher-degree polynomial: use specialized handler
                        handler_sols = _solve_polynomial_equation(equation, sym)
                        if handler_sols:
                            sols = handler_sols
                        else:
                            # Fallback to general solve
                            sols = sp.solve(equation, sym)
                    else:
                        # Not a polynomial or degree 0, use general solve
                        sols = sp.solve(equation, sym)
                else:
                    # Not a polynomial, use general solve
                    sols = sp.solve(equation, sym)
            except (ValueError, TypeError, AttributeError):
                # Poly construction failed (non-polynomial equation), try general solve
                poly = None
            except sp.polys.polyerrors.PolynomialError:
                # Explicitly caught PolynomialError - expression is not a polynomial
                # This happens when trying to construct Poly from expressions with
                # exponentials, logarithms, or other non-polynomial terms
                # Fall through to general solve
                poly = None
            # If Poly construction failed, try general solve
            if "poly" not in locals() or poly is None:
                # Check for trig functions first before attempting sp.solve()
                if NUMERIC_FALLBACK_ENABLED and equation.has(sp.sin, sp.cos, sp.tan):
                    equation_expr = left_expr - right_expr
                    numeric_roots = _numeric_roots_for_single_var(
                        equation_expr, sym, interval=(-4 * sp.pi, 4 * sp.pi)
                    )
                    if numeric_roots:
                        exacts = [str(r) for r in numeric_roots]
                        approx = [str(sp.N(r)) for r in numeric_roots]
                        return {
                            "ok": True,
                            "type": "equation",
                            "exact": exacts,
                            "approx": approx,
                        }
                # Try general solve for non-polynomial, non-trig equations
                try:
                    sols = sp.solve(equation, sym)
                    # Check if solution list is empty and equation is impossible
                    if not sols:
                        # Check for impossible inverse trig equations
                        error_hint = None
                        if equation.has(sp.asin):
                            try:
                                if equation.lhs.has(sp.asin) and not equation.rhs.has(
                                    sp.asin
                                ):
                                    rhs_val = float(sp.N(equation.rhs))
                                    if abs(rhs_val) > sp.pi / 2:
                                        error_hint = f"asin(x) cannot equal {rhs_val} (range of asin is [-pi/2, pi/2])"
                                elif equation.rhs.has(sp.asin) and not equation.lhs.has(
                                    sp.asin
                                ):
                                    lhs_val = float(sp.N(equation.lhs))
                                    if abs(lhs_val) > sp.pi / 2:
                                        error_hint = f"asin(x) cannot equal {lhs_val} (range of asin is [-pi/2, pi/2])"
                            except (ValueError, TypeError, AttributeError):
                                pass
                        if equation.has(sp.acos) and not error_hint:
                            try:
                                if equation.lhs.has(sp.acos) and not equation.rhs.has(
                                    sp.acos
                                ):
                                    rhs_val = float(sp.N(equation.rhs))
                                    if rhs_val < 0 or rhs_val > sp.pi:
                                        error_hint = f"acos(x) cannot equal {rhs_val} (range of acos is [0, pi])"
                                elif equation.rhs.has(sp.acos) and not equation.lhs.has(
                                    sp.acos
                                ):
                                    lhs_val = float(sp.N(equation.lhs))
                                    if lhs_val < 0 or lhs_val > sp.pi:
                                        error_hint = f"acos(x) cannot equal {lhs_val} (range of acos is [0, pi])"
                            except (ValueError, TypeError, AttributeError):
                                pass
                        if equation.has(sp.atan) and not error_hint:
                            try:
                                if equation.lhs.has(sp.atan) and not equation.rhs.has(
                                    sp.atan
                                ):
                                    rhs_val = float(sp.N(equation.rhs))
                                    if abs(rhs_val) >= sp.pi / 2:
                                        error_hint = f"atan(x) cannot equal {rhs_val} (range of atan is (-pi/2, pi/2))"
                                elif equation.rhs.has(sp.atan) and not equation.lhs.has(
                                    sp.atan
                                ):
                                    lhs_val = float(sp.N(equation.lhs))
                                    if abs(lhs_val) >= sp.pi / 2:
                                        error_hint = f"atan(x) cannot equal {lhs_val} (range of atan is (-pi/2, pi/2))"
                            except (ValueError, TypeError, AttributeError):
                                pass

                        if error_hint:
                            return {
                                "ok": False,
                                "error": f"This equation has no real solutions: {error_hint}.",
                                "error_code": "NO_REAL_SOLUTIONS",
                            }
                        # If no error hint found but sols is empty, return appropriate error
                        return {
                            "ok": False,
                            "error": "No real solutions found for this equation.",
                            "error_code": "NO_REAL_SOLUTIONS",
                        }
                except NotImplementedError as e:
                    logger.debug(
                        f"Symbolic solve not implemented, trying numeric fallback: {e}"
                    )
                    if NUMERIC_FALLBACK_ENABLED:
                        equation_expr = left_expr - right_expr
                        # Use wider interval for complex equations (e.g., with exponentials, mixed terms)
                        # Try positive domain first (many exponential equations require x > 0)
                        numeric_roots = _numeric_roots_for_single_var(
                            equation_expr, sym, interval=(0.01, 50)
                        )
                        if not numeric_roots:
                            # If no roots in positive domain, try full range
                            numeric_roots = _numeric_roots_for_single_var(
                                equation_expr, sym, interval=(-20, 20)
                            )
                        if numeric_roots:
                            exacts = [str(r) for r in numeric_roots]
                            approx = [str(sp.N(r)) for r in numeric_roots]
                            return {
                                "ok": True,
                                "type": "equation",
                                "exact": exacts,
                                "approx": approx,
                            }
                    return {
                        "ok": False,
                        "error": "This equation cannot be solved symbolically. Numeric root finding found no real solutions in the search interval. The equation may have no real solutions, or solutions may be outside the search range.",
                        "error_code": "NO_REAL_SOLUTIONS",
                    }
                except (ValueError, TypeError) as solve_error:
                    # sp.solve() failed - try numeric fallback
                    if NUMERIC_FALLBACK_ENABLED:
                        equation_expr = left_expr - right_expr
                        numeric_roots = _numeric_roots_for_single_var(
                            equation_expr, sym, interval=(-4 * sp.pi, 4 * sp.pi)
                        )
                        if numeric_roots:
                            exacts = [str(r) for r in numeric_roots]
                            approx = [str(sp.N(r)) for r in numeric_roots]
                            return {
                                "ok": True,
                                "type": "equation",
                                "exact": exacts,
                                "approx": approx,
                            }
                    logger.warning(
                        f"Error in symbolic solve, trying numeric fallback: {solve_error}",
                        exc_info=True,
                    )
                    return {
                        "ok": False,
                        "error": f"Solving error: {solve_error}",
                        "error_code": "SOLVER_ERROR",
                    }
                except Exception as solve_error:
                    # Catch other exceptions from sp.solve() (e.g., generator errors)
                    error_msg = str(solve_error).lower()
                    # Check for the specific generator error that can occur with various function types
                    if "generators" in error_msg or "contains an element" in error_msg:
                        # Try numeric fallback for equations that can't be solved symbolically
                        if NUMERIC_FALLBACK_ENABLED:
                            equation_expr = left_expr - right_expr
                            numeric_roots = _numeric_roots_for_single_var(
                                equation_expr,
                                sym,
                                interval=(
                                    -20,
                                    20,
                                ),  # Wider interval for exponential equations
                            )
                            if numeric_roots:
                                exacts = [str(r) for r in numeric_roots]
                                approx = [str(sp.N(r)) for r in numeric_roots]
                                return {
                                    "ok": True,
                                    "type": "equation",
                                    "exact": exacts,
                                    "approx": approx,
                                }
                        return {
                            "ok": False,
                            "error": "This equation cannot be solved symbolically. No real solutions found in the search interval.",
                            "error_code": "NO_REAL_SOLUTIONS",
                        }
                    # For other errors, try numeric fallback if enabled
                    if NUMERIC_FALLBACK_ENABLED:
                        equation_expr = left_expr - right_expr
                        numeric_roots = _numeric_roots_for_single_var(
                            equation_expr, sym, interval=(-20, 20)
                        )
                        if numeric_roots:
                            exacts = [str(r) for r in numeric_roots]
                            approx = [str(sp.N(r)) for r in numeric_roots]
                            return {
                                "ok": True,
                                "type": "equation",
                                "exact": exacts,
                                "approx": approx,
                            }
                    logger.error("Unexpected error in symbolic solve", exc_info=True)
                    return {
                        "ok": False,
                        "error": f"Unexpected solving error: {solve_error}",
                        "error_code": "SOLVER_ERROR",
                    }
            # Check if sols is empty before processing
            if not sols:
                return {
                    "ok": False,
                    "error": "No real solutions found for this equation.",
                    "error_code": "NO_REAL_SOLUTIONS",
                }
            exacts = (
                [str(solution) for solution in sols]
                if isinstance(sols, (list, tuple))
                else [str(sols)]
            )
            approx = []
            for solution in sols if isinstance(sols, (list, tuple)) else [sols]:
                try:
                    approx.append(str(sp.N(solution)))
                except (ValueError, TypeError, OverflowError, ArithmeticError):
                    # Expected for some symbolic solutions
                    approx.append(None)
            return {
                "ok": True,
                "type": "equation",
                "exact": exacts,
                "approx": approx,
                "cache_hits": cache_hits,
            }
        multi_solutions: dict[str, list[str]] = {}
        multi_approx: dict[str, list[str | None]] = {}
        for sym in symbols:
            try:
                sols_for_sym = sp.solve(equation, sym)
                if isinstance(sols_for_sym, dict):
                    sols_list = [str(v) for v in sols_for_sym.values()]
                    sols_exprs = list(sols_for_sym.values())
                elif isinstance(sols_for_sym, (list, tuple)):
                    sols_list = [str(s) for s in sols_for_sym]
                    sols_exprs = list(sols_for_sym)
                else:
                    sols_list = [str(sols_for_sym)]
                    sols_exprs = [sols_for_sym]
                multi_solutions[str(sym)] = sols_list
                approx_list: list[str | None] = []
                for expr in sols_exprs:
                    try:
                        approx_list.append(str(sp.N(expr)))
                    except (ValueError, TypeError, OverflowError, ArithmeticError):
                        # Expected for some symbolic solutions
                        approx_list.append(None)
                multi_approx[str(sym)] = approx_list
            except NotImplementedError as e:
                logger.info(f"Solving for {sym} not implemented: {e}")
                multi_solutions[str(sym)] = [
                    "Solving not implemented for this variable"
                ]
                multi_approx[str(sym)] = [None]
            except (ValueError, TypeError) as e:
                logger.warning(f"Error solving for {sym}", exc_info=True)
                multi_solutions[str(sym)] = [f"Error: {e}"]
                multi_approx[str(sym)] = [None]
    except Exception as e:
        logger.error(f"Unexpected error solving for {sym}", exc_info=True)
        multi_solutions[str(sym)] = [f"Unexpected error: {e}"]
        multi_approx[str(sym)] = [None]
    return {
        "ok": True,
        "type": "multi_isolate",
        "solutions": multi_solutions,
        "approx": multi_approx,
    }
