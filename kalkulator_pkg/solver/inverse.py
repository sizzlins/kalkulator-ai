from typing import Any

import sympy as sp

from ..parser import parse_preprocessed

try:
    from ..logging_config import get_logger

    logger = get_logger("solver.inverse")
except ImportError:
    import logging

    logger = logging.getLogger("solver.inverse")


def solve_inverse_function(
    func_name: str, target_value: str, param_names: list[str]
) -> dict[str, Any]:
    """Solve f(x, y, ...) = value for integer, rational, real, and complex solutions.

    Args:
        func_name: Name of the function (must be defined)
        target_value: Target value as string
        param_names: List of parameter names

    Returns:
        Dictionary with solutions for different domains, properly classified
    """
    try:
        from ..function_manager import _function_registry

        if func_name not in _function_registry:
            return {
                "ok": False,
                "error": f"Function '{func_name}' is not defined",
                "error_code": "FUNC_NOT_DEFINED",
            }

        stored_params, body = _function_registry[func_name]

        # Parse target value
        try:
            target = parse_preprocessed(target_value)
        except Exception as e:
            return {
                "ok": False,
                "error": f"Invalid target value: {e}",
                "error_code": "INVALID_TARGET",
            }

        # Create equation: body = target
        eq = sp.Eq(body, target)
        syms = list(body.free_symbols)
        sorted_syms = sorted(syms, key=lambda s: s.name)

        results: dict[str, Any] = {
            "ok": True,
            "type": "inverse_function",
            "func_name": func_name,
            "target": str(target),
        }
        domains: dict[str, Any] = {}
        results["domains"] = domains

        def is_truly_integer(val):
            """Check if a value is truly an integer (not symbolic like pi/2)."""
            try:
                if val.is_Integer:
                    return True
                # Check for irrational constants - these are NOT integers
                if val.has(sp.pi, sp.E, sp.I):
                    return False
                # Check for sqrt of non-perfect squares
                for atom in val.atoms(sp.Pow):
                    if atom.exp == sp.Rational(1, 2) or (
                        hasattr(atom.exp, "q") and atom.exp.q == 2
                    ):
                        base = atom.base
                        if not (base.is_Integer and sp.sqrt(base).is_Integer):
                            return False
                simplified = sp.simplify(val)
                return simplified.is_Integer
            except Exception:
                return False

        def is_truly_rational(val):
            """Check if a value is truly rational (not containing pi, e, sqrt of non-squares)."""
            try:
                if val.is_Rational:
                    return True
                # Check for irrational constants
                if val.has(sp.pi, sp.E, sp.I):
                    return False
                # Check for square roots of non-perfect squares
                for atom in val.atoms(sp.Pow):
                    if (
                        hasattr(atom.exp, "q") and atom.exp.q == 2
                    ):  # Fractional exponent like 1/2
                        base = atom.base
                        if base.is_Integer:
                            sqrt_val = sp.sqrt(base)
                            if not sqrt_val.is_Integer:
                                return False
                        else:
                            return False
                simplified = sp.simplify(val)
                return simplified.is_Rational
            except Exception:
                return False

        def classify_solution(sol):
            """Classify a solution: 'integer', 'rational', 'real', or 'complex'."""
            try:
                if is_truly_integer(sol):
                    return "integer"
                if is_truly_rational(sol):
                    return "rational"
                # Check if complex (has imaginary unit)
                if sol.has(sp.I):
                    return "complex"
                # Otherwise real (may be irrational)
                return "real"
            except Exception:
                return "complex"

        # 1. Integer Solutions - brute-force search
        integer_solutions: list[tuple[int, int] | tuple[int]] = []

        try:
            if len(sorted_syms) == 2:
                x_sym, y_sym = sorted_syms[0], sorted_syms[1]

                try:
                    target_num = int(float(sp.N(target)))
                    import math

                    search_range = max(int(math.sqrt(abs(target_num)) + 2), 10)
                    search_range = min(search_range, 100)
                except (ValueError, TypeError):
                    search_range = 20

                equation_expr = body - target

                for x_val in range(-search_range, search_range + 1):
                    for y_val in range(-search_range, search_range + 1):
                        try:
                            result = equation_expr.subs(
                                [(x_sym, x_val), (y_sym, y_val)]
                            )
                            if result == 0:
                                integer_solutions.append((x_val, y_val))
                        except Exception:
                            pass

                integer_solutions.sort(
                    key=lambda p: (
                        (abs(p[0]), p[0], abs(p[1]), p[1]) if len(p) == 2 else 0
                    )
                )

            elif len(sorted_syms) == 1:
                sym = sorted_syms[0]
                for val in range(-100, 101):
                    try:
                        if body.subs(sym, val) == target:
                            integer_solutions.append((val,))
                    except Exception:
                        pass
        except Exception:
            pass

        if integer_solutions:
            if len(sorted_syms) == 2:
                domains["integers"] = {
                    "count": len(integer_solutions),
                    "solutions": [
                        {"x": s[0], "y": s[1]}
                        for s in integer_solutions[:20]
                        if len(s) == 2
                    ],
                }
            else:
                domains["integers"] = {
                    "count": len(integer_solutions),
                    "solutions": [{"x": s[0]} for s in integer_solutions[:20]],
                }
        else:
            domains["integers"] = None

        # 2. Get symbolic solutions and classify them
        rational_solutions = []
        real_solutions = []
        complex_solutions = []

        if len(sorted_syms) >= 1:
            try:
                solve_var = sorted_syms[0]
                all_sols = sp.solve(eq, solve_var)

                for sol in all_sols:
                    domain = classify_solution(sol)
                    sol_str = str(sol)

                    try:
                        numeric_val = complex(sp.N(sol))
                        if abs(numeric_val.imag) < 1e-10:
                            numeric_str = f"{numeric_val.real:.10g}"
                        else:
                            numeric_str = (
                                f"{numeric_val.real:.6g} + {numeric_val.imag:.6g}i"
                            )
                    except Exception:
                        numeric_str = None

                    sol_entry = {"exact": sol_str, "numeric": numeric_str}

                    if domain == "rational":
                        rational_solutions.append(sol_entry)
                    elif domain == "real":
                        real_solutions.append(sol_entry)
                    elif domain == "complex":
                        complex_solutions.append(sol_entry)
                    # Skip 'integer' - already handled via brute force

            except Exception as e:
                logger.debug(f"Error solving equation: {e}")

        results["domains"]["rationals"] = (
            rational_solutions if rational_solutions else None
        )
        results["domains"]["reals"] = real_solutions if real_solutions else None
        results["domains"]["complex"] = complex_solutions if complex_solutions else None

        # 3. Parametric form for 2-variable
        if len(sorted_syms) == 2:
            x_sym, y_sym = sorted_syms[0], sorted_syms[1]
            try:
                if body.equals(x_sym**2 + y_sym**2) and target.is_positive:
                    r = sp.sqrt(target)
                    results["domains"]["parametric"] = {
                        "form": f"{x_sym} = {r}*cos(t), {y_sym} = {r}*sin(t)",
                        "parameter": "t ∈ ℝ",
                    }
            except Exception:
                pass

            try:
                x_sols = sp.solve(eq, x_sym)
                if x_sols:
                    general_forms = [f"{x_sym} = {sol}" for sol in x_sols]
                    results["domains"]["general"] = {
                        "forms": general_forms,
                        "note": "principal branch",
                    }
            except Exception:
                pass

        return results

    except Exception as e:
        logger.exception("Error in solve_inverse_function")
        return {
            "ok": False,
            "error": f"Solver error: {e}",
            "error_code": "SOLVER_ERROR",
        }
