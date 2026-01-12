from __future__ import annotations

import math
import sympy as sp

from ..config import COARSE_GRID_MIN_SIZE
from ..config import MAX_NSOLVE_GUESSES
from ..config import MAX_NSOLVE_STEPS
from ..config import NUMERIC_TOLERANCE
from ..config import ROOT_DEDUP_TOLERANCE
from ..config import ROOT_SEARCH_TOLERANCE


def _solve_simple_trig_equation(
    expr: sp.Basic,
    variable: sp.Symbol,
    interval: tuple[float, float] = (-4 * sp.pi, 4 * sp.pi),
) -> list[sp.Number] | None:
    """Solve simple trig equations using direct inverse calculation.
    
    For patterns like sin(x) = c, cos(x) = c, tan(x) = c:
    - Uses direct inverse function (arcsin, arccos, arctan) for maximum precision
    - Expands to all solutions within the interval using periodicity
    
    This "Inverse & Expand" algorithm is faster and more precise than
    iterative root-finding methods which introduce floating-point noise.
    
    Args:
        expr: Expression of the form trig(x) - c = 0
        variable: Symbol to solve for
        interval: Search interval (min, max)
        
    Returns:
        List of solutions if pattern matches, None otherwise (fall back to general solver)
    """
    interval_min, interval_max = float(interval[0]), float(interval[1])
    
    # Try to extract pattern: trig(x) - c = 0 or trig(x) + c = 0
    # We need to identify: sin(x) = c, cos(x) = c, or tan(x) = c
    
    # Expand expression to canonical form
    try:
        expr_expanded = sp.expand(expr)
    except Exception:
        return None
    
    # Check if expression is of form f(x) - constant or f(x) + constant
    # where f(x) is a trig function of just the variable
    
    if not isinstance(expr_expanded, sp.Add):
        # Could be just sin(x) = 0 (no constant part)
        if expr_expanded.func in (sp.sin, sp.cos, sp.tan):
            # Check if arg is exactly the variable
            if expr_expanded.args[0] == variable:
                # trig(x) = 0
                trig_func = expr_expanded.func
                target_value = 0.0
            else:
                return None
        else:
            return None
    else:
        # Expression is a sum: look for trig(x) + constant
        trig_part = None
        const_part = sp.S.Zero
        
        for term in expr_expanded.args:
            if term.is_number:
                const_part += term
            elif term.func in (sp.sin, sp.cos, sp.tan):
                if term.args[0] == variable:
                    if trig_part is not None:
                        return None  # Multiple trig functions (e.g., sin(x)-cos(x)) - not simple
                    trig_part = term
                else:
                    return None  # Complex argument like sin(2*x)
            elif isinstance(term, sp.Mul) and len(term.args) == 2:
                # Could be -sin(x) which is Mul(-1, sin(x))
                coeff, func = term.args
                if coeff == -1 and func.func in (sp.sin, sp.cos, sp.tan):
                    if func.args[0] == variable:
                        if trig_part is not None:
                            return None  # Multiple trig functions - not simple
                        trig_part = sp.Mul(-1, func)
                    else:
                        return None
                else:
                    return None  # Something like 2*sin(x) - not simple
            else:
                return None  # Complex expression
        
        if trig_part is None:
            return None
        
        # Now we have trig_part + const_part = 0
        # So trig(x) = -const_part
        target_value = float(-const_part.evalf())
        
        # Handle -sin(x) case: -sin(x) = c means sin(x) = -c
        if isinstance(trig_part, sp.Mul) and trig_part.args[0] == -1:
            trig_func = trig_part.args[1].func
            target_value = -target_value
        else:
            trig_func = trig_part.func
    
    # Now solve using direct inverse calculation
    roots: list[float] = []
    pi_val = math.pi
    
    if trig_func == sp.sin:
        # sin(x) = c where |c| <= 1
        if abs(target_value) > 1:
            return []  # No real solutions
        
        # Principal root: x0 = arcsin(c)
        x0 = math.asin(target_value)
        # Secondary root: x1 = π - arcsin(c)
        x1 = pi_val - x0
        
        # Expand with periodicity: x0 + 2πk, x1 + 2πk
        for k in range(-10, 11):  # Generous range to cover typical intervals
            root1 = x0 + 2 * pi_val * k
            root2 = x1 + 2 * pi_val * k
            if interval_min <= root1 <= interval_max:
                roots.append(root1)
            if interval_min <= root2 <= interval_max:
                if abs(root2 - root1) > 1e-10:  # Avoid duplicates at π/2
                    roots.append(root2)
    
    elif trig_func == sp.cos:
        # cos(x) = c where |c| <= 1
        if abs(target_value) > 1:
            return []  # No real solutions
        
        # Principal root: x0 = arccos(c)
        x0 = math.acos(target_value)
        # Secondary root: x1 = -arccos(c) = 2π - arccos(c) (mod 2π)
        x1 = -x0
        
        # Expand with periodicity: x0 + 2πk, x1 + 2πk
        for k in range(-10, 11):
            root1 = x0 + 2 * pi_val * k
            root2 = x1 + 2 * pi_val * k
            if interval_min <= root1 <= interval_max:
                roots.append(root1)
            if interval_min <= root2 <= interval_max:
                if abs(root2 - root1) > 1e-10:  # Avoid duplicates at 0, π
                    roots.append(root2)
    
    elif trig_func == sp.tan:
        # tan(x) = c (always has solutions)
        # Principal root: x0 = arctan(c)
        x0 = math.atan(target_value)
        
        # Expand with periodicity: x0 + πk (period is π for tan)
        for k in range(-20, 21):  # More points since period is smaller
            root = x0 + pi_val * k
            if interval_min <= root <= interval_max:
                roots.append(root)
    
    else:
        return None  # Not a recognized trig function
    
    # Sort and deduplicate (use 14 decimals to preserve precision while still deduping)
    safe_roots = []
    for r in roots:
        try:
            safe_roots.append(round(float(r), 14))
        except (TypeError, ValueError):
            safe_roots.append(r)
    roots = sorted(set(safe_roots), key=lambda x: abs(x))
    
    return [sp.N(r, 15) for r in roots]  # Return with high precision


def _numeric_roots_for_single_var(
    expr: sp.Basic,
    variable: sp.Symbol,
    interval: tuple[float, float] = (-4 * sp.pi, 4 * sp.pi),
    max_guesses: int | None = None,
) -> list[sp.Number]:
    """Find numeric roots of expression using multiple strategies.

    Args:
        expr: SymPy expression to find roots of (set to zero)
        variable: Symbol to solve for
        interval: Search interval (min, max)
        max_guesses: Maximum number of guess points (default: MAX_NSOLVE_GUESSES)

    Returns:
        List of numeric roots found
    """
    if max_guesses is None:
        max_guesses = MAX_NSOLVE_GUESSES
    roots: list[float] = []
    interval_min, interval_max = float(interval[0]), float(interval[1])

    # Strategy 0: Try direct inverse for simple trig equations (sin(x)=c, cos(x)=c, tan(x)=c)
    # This is the most precise method - uses exact arcsin/arccos/arctan + periodicity
    try:
        trig_roots = _solve_simple_trig_equation(expr, variable, interval)
        if trig_roots is not None:
            return trig_roots  # Direct inverse succeeded - no need for iterative methods
    except Exception:
        pass  # Fall through to other strategies

    # Strategy 1: Try solveset over the reals within interval
    try:
        from sympy import solveset

        solset = solveset(
            sp.Eq(expr, 0), variable, domain=sp.Interval(interval_min, interval_max)
        )
        finite_values = []
        for solution in solset:
            try:
                solution_value = float(sp.N(solution))
                finite_values.append(solution_value)
            except (ValueError, TypeError):
                continue
        if finite_values:
            unique_values_list = []
            for x_val in finite_values:
                try:
                    unique_values_list.append(round(x_val, 12))
                except (TypeError, ValueError):
                    pass
            unique_values = sorted(set(unique_values_list))
            return [sp.N(root_val) for root_val in unique_values]
    except (ValueError, TypeError, NotImplementedError):
        pass

    # Strategy 2: Try polynomial root finding
    try:
        poly = sp.Poly(expr, variable)
        if poly is not None and poly.total_degree() > 0:
            for root in poly.nroots():
                if abs(sp.im(root)) < NUMERIC_TOLERANCE:
                    roots.append(float(sp.re(root)))
            if roots:
                unique_roots_list = []
                for x_val in roots:
                    try:
                        unique_roots_list.append(round(x_val, 12))
                    except (TypeError, ValueError):
                         pass
                unique_roots = sorted(set(unique_roots_list))
                return [sp.N(root_val) for root_val in unique_roots]
    except (ValueError, TypeError):
        pass
    except sp.polys.polyerrors.PolynomialError:
        # Expression is not a polynomial (e.g., contains trig functions, exponentials, etc.)
        # Skip this strategy and continue to Strategy 3
        pass

    # Strategy 3: Detect sign changes and use nsolve
    interval_min, interval_max = interval_min, interval_max
    coarse_grid_size = max(COARSE_GRID_MIN_SIZE, max_guesses // 3)
    sample_points = [
        interval_min + (interval_max - interval_min) * idx / coarse_grid_size
        for idx in range(coarse_grid_size + 1)
    ]
    candidate_points = []
    previous_value = None
    for sample_point in sample_points:
        try:
            current_value = float(sp.N(expr.subs({variable: sample_point})))
            if (
                previous_value is not None
                and current_value == current_value
                and previous_value == previous_value
                and previous_value * current_value <= 0
            ):
                candidate_points.append(sample_point)
            previous_value = current_value
        except (ValueError, TypeError):
            previous_value = None

    candidate_points_safe = []
    for candidate in candidate_points:
         try:
             candidate_points_safe.append(round(candidate, 8))
         except (TypeError, ValueError):
             pass
    
    candidate_points = sorted(set(candidate_points_safe))[:COARSE_GRID_MIN_SIZE]
    for guess in candidate_points:
        try:
            root = sp.nsolve(
                expr,
                variable,
                guess,
                tol=ROOT_SEARCH_TOLERANCE,
                maxsteps=MAX_NSOLVE_STEPS,
            )
            if abs(sp.im(root)) > NUMERIC_TOLERANCE:
                continue
            root_real = float(sp.re(root))
            if not any(
                abs(existing - root_real) < ROOT_DEDUP_TOLERANCE for existing in roots
            ):
                roots.append(root_real)
        except (ValueError, TypeError, NotImplementedError):
            continue

    sorted_roots = sorted(roots)
    return [sp.N(root_val) for root_val in sorted_roots]
