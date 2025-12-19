from __future__ import annotations

import sympy as sp

from ..config import COARSE_GRID_MIN_SIZE
from ..config import MAX_NSOLVE_GUESSES
from ..config import MAX_NSOLVE_STEPS
from ..config import NUMERIC_TOLERANCE
from ..config import ROOT_DEDUP_TOLERANCE
from ..config import ROOT_SEARCH_TOLERANCE


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
            unique_values = sorted({round(x_val, 12) for x_val in finite_values})
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
                unique_roots = sorted({round(x_val, 12) for x_val in roots})
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

    # De-duplicate and limit candidate points
    candidate_points = sorted({round(candidate, 8) for candidate in candidate_points})[
        :COARSE_GRID_MIN_SIZE
    ]
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
