"""Advanced function finding capabilities.

This module implements:
1. Constant detection (PSLQ/nsimplify) for symbolic recognition
2. High-precision parsing (Decimal/mpmath)
3. Sparse regression (LASSO/OMP)
4. Model selection (AIC/BIC)
5. Tolerance and validation utilities
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Any

import numpy as np
import sympy as sp

try:
    import mpmath  # noqa: F401

    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

from .config import (
    ABSOLUTE_TOLERANCE,
    CONSTANT_DETECTION_TOLERANCE,
    LASSO_LAMBDA,
    OMP_MAX_ITERATIONS,
    RELATIVE_TOLERANCE,
)

# Set high precision for Decimal
getcontext().prec = 50

# Library of known constants for detection 
# (Restricted to common physics constants to avoid false positives like EulerGamma)
KNOWN_CONSTANTS = {
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E, 
    "sqrt(2)": sp.sqrt(2),
    "sqrt(3)": sp.sqrt(3),
    # Removed rarer constants to prevent hallucinations in regression
}


def detect_symbolic_constant(
    value: float | Fraction | Decimal, tolerance: float = CONSTANT_DETECTION_TOLERANCE
) -> sp.Basic | None:
    """Detect if a numeric value is close to a known symbolic constant.

    Uses SymPy's nsimplify and direct comparison with known constants.

    Args:
        value: Numeric value to check
        tolerance: Relative tolerance for matching

    Returns:
        Symbolic constant if detected, None otherwise
    """
    if isinstance(value, Fraction):
        float_val = float(value)
    elif isinstance(value, Decimal):
        float_val = float(value)
    else:
        float_val = float(value)

    # Direct comparison with known constants
    for _const_name, const_symbol in KNOWN_CONSTANTS.items():
        try:
            const_val = float(sp.N(const_symbol))
            # 1. Check if value is close to constant directly
            if abs(float_val - const_val) / (abs(const_val) + 1e-10) < tolerance:
                return const_symbol

            # 2. Check if value is a rational multiple (e.g. 4/3 * pi)
            # Avoid division by zero
            if abs(const_val) > 1e-9:
                ratio = float_val / const_val
                # Check if ratio is close to a SIMPLE rational
                try:
                    # RESTRICT DENOMINATOR: We only want nice multiples like pi/2, 4/3*pi
                    # NOT 209/67 * pi.
                    frac = Fraction(ratio).limit_denominator(12) 
                    if abs(ratio - float(frac)) < tolerance:
                        # Success! Return fraction * constant
                        if frac == 1:
                            return const_symbol
                        return frac * const_symbol
                except (ValueError, TypeError):
                    pass
        except (ValueError, TypeError):
            continue

    # 3. Fallback to nsimplify (it handles some cases but not others)
    try:
        # Try to simplify to a known constant using sympy's nsimplify
        # We need to provide constants explicitly for best results
        constants_list = [sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)]
        simplified = sp.nsimplify(
            float_val,
            tolerance=tolerance,
            constants=constants_list,
            rational=True
        )

        # Check if simplified result actually contains our constants
        # (nsimplify sometimes just returns a fraction)
        if simplified.has(sp.pi) or simplified.has(sp.E) or simplified.has(sp.sqrt):
             # Double check numerical accuracy
             if abs(float_val - float(simplified.evalf())) < tolerance:
                 # Extra check: Is it simple?
                 den = sp.denom(simplified)
                 if abs(den) <= 24: # Reject weird denominators
                     return simplified

    except (ValueError, TypeError, AttributeError):
        pass

    return None


def parse_with_precision(
    val: str | float | int | Fraction | Decimal,
) -> Fraction | Decimal:
    """Parse input with appropriate precision based on format.

    Strategy:
    - Integers → Fraction
    - Fractions (a/b) → Fraction
    - Decimals with few places → Fraction (if exact)
    - High-precision decimals → Decimal
    - Very high precision → mpmath (if available)

    Args:
        val: Input value in various formats

    Returns:
        Fraction or Decimal with appropriate precision
    """
    # Already a Fraction
    if isinstance(val, Fraction):
        return val

    # Already a Decimal
    if isinstance(val, Decimal):
        return val

    # Integer
    if isinstance(val, int):
        return Fraction(val)

    # String parsing
    if isinstance(val, str):
        val = val.strip()

        # Try fraction format
        if "/" in val:
            try:
                parts = val.split("/")
                if len(parts) == 2:
                    num = int(parts[0].strip())
                    den = int(parts[1].strip())
                    return Fraction(num, den)
            except (ValueError, TypeError):
                pass

        # Try decimal format
        try:
            # Count decimal places
            if "." in val:
                decimal_places = len(val.split(".")[1])
                # If few decimal places, try Fraction first
                if decimal_places <= 6:
                    try:
                        return Fraction(val)
                    except (ValueError, TypeError):
                        pass
                # Otherwise use Decimal for high precision
                return Decimal(val)
            else:
                # Integer string
                return Fraction(int(val))
        except (ValueError, TypeError):
            pass

    # Float - try to convert to Fraction if it's a simple decimal
    if isinstance(val, float):
        # Check if it's close to a simple fraction
        try:
            frac = Fraction(val).limit_denominator(10000)
            if abs(float(frac) - val) < 1e-10:
                return frac
        except (ValueError, OverflowError):
            pass

        # Otherwise use Decimal for precision
        return Decimal(str(val))

    raise ValueError(f"Cannot parse {val} with precision")


def is_exact_fit(
    computed: float | Fraction | Decimal,
    expected: float | Fraction | Decimal,
    abs_tol: float = ABSOLUTE_TOLERANCE,
    rel_tol: float = RELATIVE_TOLERANCE,
) -> bool:
    """Check if a computed value matches expected value within tolerances.

    Uses both absolute and relative tolerance checks.

    Args:
        computed: Computed value
        expected: Expected value
        abs_tol: Absolute tolerance
        rel_tol: Relative tolerance

    Returns:
        True if match within tolerances, False otherwise
    """
    # Convert to float for comparison
    comp_float = float(computed)
    exp_float = float(expected)

    # Absolute difference
    abs_diff = abs(comp_float - exp_float)

    # Relative difference
    rel_diff = abs_diff / (abs(exp_float) + 1e-10)

    # Check both tolerances
    return abs_diff < abs_tol or rel_diff < rel_tol


def calculate_residuals(
    computed_values: list[float | Fraction],
    expected_values: list[float | Fraction],
) -> tuple[list[float], float, float]:
    """Calculate residuals and statistics.

    Args:
        computed_values: List of computed function values
        expected_values: List of expected values

    Returns:
        Tuple of (residuals, max_residual, mean_squared_error)
    """
    residuals = []
    for comp, exp in zip(computed_values, expected_values):
        residuals.append(float(comp) - float(exp))

    max_residual = max(abs(r) for r in residuals)
    mse = sum(r * r for r in residuals) / len(residuals) if residuals else 0.0

    return residuals, max_residual, mse


def calculate_aic(n_params: int, n_samples: int, mse: float) -> float:
    """Calculate Akaike Information Criterion (AIC).

    AIC = 2k - 2*ln(L) where k = number of parameters, L = likelihood.
    For least squares: AIC = n*ln(MSE) + 2k

    Args:
        n_params: Number of parameters in the model
        n_samples: Number of data points
        mse: Mean squared error

    Returns:
        AIC value (lower is better)
    """
    if mse <= 0:
        return float("inf")
    return n_samples * math.log(mse) + 2 * n_params


def calculate_bic(n_params: int, n_samples: int, mse: float) -> float:
    """Calculate Bayesian Information Criterion (BIC).

    BIC = k*ln(n) - 2*ln(L) where k = parameters, n = samples, L = likelihood.
    For least squares: BIC = n*ln(MSE) + k*ln(n)

    Args:
        n_params: Number of parameters in the model
        n_samples: Number of data points
        mse: Mean squared error

    Returns:
        BIC value (lower is better)
    """
    if mse <= 0:
        return float("inf")
    return n_samples * math.log(mse) + n_params * math.log(n_samples)


def orthogonal_matching_pursuit(
    A: list[list[float | Fraction]],
    b: list[float | Fraction],
    max_nonzero: int | None = None,
    max_iterations: int = OMP_MAX_ITERATIONS,
) -> tuple[list[float], list[int]]:
    """Orthogonal Matching Pursuit for sparse regression.

    Greedy algorithm that iteratively selects the column of A that best
    matches the residual.

    Args:
        A: Design matrix (list of rows, each row is a list)
        b: Target vector
        max_nonzero: Maximum number of non-zero coefficients (default: min(n, m))
        max_iterations: Maximum iterations

    Returns:
        Tuple of (coefficients, selected_indices)
    """
    import numpy as np

    # Convert to numpy arrays
    A_arr = np.array([[float(x) for x in row] for row in A])
    b_arr = np.array([float(x) for x in b])

    n_samples, n_features = A_arr.shape

    if max_nonzero is None:
        max_nonzero = min(n_samples, n_features)

    # Initialize
    residual = b_arr.copy()
    selected: list[int] = []
    coefficients = np.zeros(n_features)

    for _iteration in range(min(max_nonzero, max_iterations)):
        # Find column with maximum correlation with residual
        correlations = np.abs(A_arr.T @ residual)
        correlations[selected] = -np.inf  # Don't reselect

        if np.max(correlations) < 1e-10:
            break  # No significant correlation

        new_idx = np.argmax(correlations)
        selected.append(new_idx)

        # Solve least squares with selected columns
        A_selected = A_arr[:, selected]
        coeffs_selected = np.linalg.lstsq(A_selected, b_arr, rcond=None)[0]

        # Update residual
        residual = b_arr - A_selected @ coeffs_selected

        # Check convergence
        if np.linalg.norm(residual) < 1e-10:
            break

    # Set coefficients
    for i, idx in enumerate(selected):
        coefficients[idx] = coeffs_selected[i]

    return coefficients.tolist(), selected


def lasso_regression(
    A: list[list[float | Fraction]],
    b: list[float | Fraction],
    lambda_reg: float = LASSO_LAMBDA,
    max_iterations: int = 1000,
) -> list[float]:
    """L1-regularized (LASSO) regression using coordinate descent.

    Minimizes: ||Ax - b||² + λ||x||₁

    Args:
        A: Design matrix
        b: Target vector
        lambda_reg: Regularization parameter
        max_iterations: Maximum iterations

    Returns:
        Coefficient vector
    """
    try:
        # Convert to numpy arrays
        import numpy as np
        from sklearn.linear_model import Lasso

        A_arr = np.array([[float(x) for x in row] for row in A])
        b_arr = np.array([float(x) for x in b])

        # Use sklearn's LASSO
        lasso = Lasso(alpha=lambda_reg, max_iter=max_iterations, fit_intercept=False)
        lasso.fit(A_arr, b_arr)
        return lasso.coef_.tolist()
    except ImportError:
        # Fallback: simple coordinate descent implementation
        import numpy as np

        A_arr = np.array([[float(x) for x in row] for row in A])
        b_arr = np.array([float(x) for x in b])

        n_samples, n_features = A_arr.shape
        coefficients = np.zeros(n_features)

        for _iteration in range(max_iterations):
            old_coeffs = coefficients.copy()

            for j in range(n_features):
                # Coordinate descent update
                r_j = b_arr - A_arr @ coefficients + A_arr[:, j] * coefficients[j]
                a_j = A_arr[:, j]

                # Soft thresholding
                numerator = a_j @ r_j
                denominator = a_j @ a_j

                if denominator > 1e-10:
                    z_j = numerator / denominator
                    coefficients[j] = np.sign(z_j) * max(
                        0, abs(z_j) - lambda_reg / (2 * denominator)
                    )

            # Check convergence
            if np.linalg.norm(coefficients - old_coeffs) < 1e-6:
                break

        return coefficients.tolist()


def lasso_cv_regression(
    A: list[list[float | Fraction]],
    b: list[float | Fraction],
    max_iterations: int = 10000,
) -> list[float]:
    """Lasso regression with automatic cross-validation for alpha.

    Args:
        A: Design matrix
        b: Target vector
        max_iterations: Maximum iterations

    Returns:
        Coefficient vector
    """
    try:
        import numpy as np
        from sklearn.linear_model import lasso_path

        A_arr = np.array([[float(x) for x in row] for row in A])
        b_arr = np.array([float(x) for x in b])
        n_samples = A_arr.shape[0]

        # Compute Lasso path
        # eps=1e-3 is default, maybe smaller for better path?
        alphas, coefs, _ = lasso_path(A_arr, b_arr, eps=1e-4)

        # coefs shape: (n_features, n_alphas)
        # alphas shape: (n_alphas,)

        best_bic = float("inf")
        best_coef = None

        for i in range(len(alphas)):
            coef = coefs[:, i]
            # Count non-zero coefficients
            k = np.sum(np.abs(coef) > 1e-5)

            # Calculate RSS
            residuals = b_arr - A_arr @ coef
            rss = np.sum(residuals**2)

            # Calculate BIC
            # BIC = n * log(RSS/n) + k * log(n)
            # Add small epsilon to RSS to avoid log(0)
            if rss < 1e-10:
                rss = 1e-10

            bic = n_samples * np.log(rss / n_samples) + k * np.log(n_samples)

            if bic < best_bic:
                best_bic = bic
                best_coef = coef

        if best_coef is None:
            # Fallback to last (most dense) or first (most sparse)?
            # Usually last is OLS. First is all zeros.
            # If everything failed, return zeros
            return [0.0] * A_arr.shape[1]

        return best_coef.tolist()

    except ImportError:
        # Fallback to fixed alpha Lasso if sklearn missing
        return lasso_regression(A, b, lambda_reg=0.001, max_iterations=max_iterations)


def detect_power_laws(x_col: np.ndarray, y_col: np.ndarray) -> list[float]:
    """Dynamically detect candidate power law exponents from data."""
    try:
        import numpy as np

        # 1. Filter valid log-log domain
        mask = (np.abs(x_col) > 1e-9) & (np.abs(y_col) > 1e-9)
        if np.sum(mask) < 4:
            return []

        lx = np.log(np.abs(x_col[mask]))
        ly = np.log(np.abs(y_col[mask]))

        # Sort by x
        sort_idx = np.argsort(lx)
        lx = lx[sort_idx]
        ly = ly[sort_idx]

        candidates = set()

        # Global fit
        try:
            coeffs = np.polyfit(lx, ly, 1)
            candidates.add(round(coeffs[0] * 2) / 2)
        except:
            pass

        # Local slopes
        dx = np.diff(lx)
        dy = np.diff(ly)

        valid_slope_mask = dx > 1e-3
        slopes = dy[valid_slope_mask] / dx[valid_slope_mask]

        # Cluster slopes
        if len(slopes) > 0:
            rounded_slopes = np.round(slopes * 2) / 2
            unique, counts = np.unique(rounded_slopes, return_counts=True)

            # Reduce threshold for small datasets
            threshold = 1 if len(slopes) < 10 else max(2, len(slopes) * 0.15)

            # print(f"DEBUG SLOPES: {rounded_slopes}, T={threshold}", flush=True)

            for s, c in zip(unique, counts):
                if c >= threshold:
                    candidates.add(s)

        # Curiosity Expansion: If we see exponent e, try 2e and e/2
        # (e.g. found r^-6, try r^-12. Found t^2, try t^1 and t^4)
        expansion = set()
        for e in candidates:
            expansion.add(e * 2)
            expansion.add(e / 2)

        candidates.update(expansion)

        res = sorted([e for e in candidates if 0.5 <= abs(e) <= 100])
        # print(f"DEBUG DETECTED: {res}", flush=True)
        return res
    except Exception:
        return []


def detect_frequency(x_col: np.ndarray, y_col: np.ndarray) -> list[float]:
    """Dynamically detect candidate frequencies in periodic data.

    Uses Zero-Crossing Rate and Peak-to-Peak analysis.
    Returns candidate k values for sin(k*x) or cos(k*x).
    """
    try:
        import numpy as np

        # 1. Need sorted data by x
        sort_idx = np.argsort(x_col)
        x = x_col[sort_idx]
        y = y_col[sort_idx]

        # Need at least 10 points for frequency detection
        if len(x) < 10:
            return []

        # 2. Remove trend (Detrend) - subtract linear fit
        try:
            trend = np.polyval(np.polyfit(x, y, 1), x)
            y_detrended = y - trend
        except:
            y_detrended = y

        # 3. Zero-Crossing Analysis
        # Count how many times y crosses zero
        signs = np.sign(y_detrended)
        crossings = np.where(np.diff(signs) != 0)[0]

        candidates = set()

        if len(crossings) >= 2:
            # Estimate periods from crossing intervals
            crossing_xs = x[crossings]
            intervals = np.diff(crossing_xs)

            # Period is roughly 2 * average interval (zero to zero is half period)
            valid_intervals = intervals[intervals > 1e-6]
            if len(valid_intervals) > 0:
                avg_half_period = np.median(valid_intervals)
                period = 2 * avg_half_period

                if period > 1e-6:
                    freq = 2 * np.pi / period  # Angular frequency
                    # Round to nearest 0.5
                    freq_rounded = round(freq * 2) / 2
                    if 0.5 <= freq_rounded <= 200:
                        candidates.add(freq_rounded)
                        # Curiosity: try integer versions
                        candidates.add(round(freq_rounded))

        # 4. Peak-to-Peak Analysis (Alternative)
        # Find local maxima/minima
        peaks = []
        for i in range(1, len(y_detrended) - 1):
            if (
                y_detrended[i] > y_detrended[i - 1]
                and y_detrended[i] > y_detrended[i + 1]
            ):
                peaks.append(x[i])
            if (
                y_detrended[i] < y_detrended[i - 1]
                and y_detrended[i] < y_detrended[i + 1]
            ):
                peaks.append(x[i])

        if len(peaks) >= 2:
            peak_intervals = np.diff(sorted(peaks))
            valid_peak_intervals = peak_intervals[peak_intervals > 1e-6]
            if len(valid_peak_intervals) > 0:
                # Peak to next peak of same type is full period
                # But we're measuring all extrema, so half-period
                avg_period = np.median(valid_peak_intervals) * 2
                if avg_period > 1e-6:
                    freq = 2 * np.pi / avg_period
                    freq_rounded = round(freq * 2) / 2
                    if 0.5 <= freq_rounded <= 200:
                        candidates.add(freq_rounded)
                        candidates.add(round(freq_rounded))

        # Harmonic Expansion: If we found k, try 2k, k/2
        expansion = set()
        for k in candidates:
            expansion.add(k * 2)
            if k / 2 >= 0.5:
                expansion.add(k / 2)

        candidates.update(expansion)

        res = sorted([k for k in candidates if 0.5 <= k <= 500])
        # print(f"DEBUG FREQ DETECT: crossings={len(crossings)}, peaks={len(peaks)}, candidates={res}", flush=True)
        return res

    except Exception:
        return []


def detect_curvature(x_col: np.ndarray, y_col: np.ndarray) -> dict:
    """Detect curvature patterns to suggest exp, log, or polynomial.

    Returns dict with suggested feature types based on second derivative analysis.
    """
    try:
        import numpy as np

        # Sort by x
        sort_idx = np.argsort(x_col)
        x = x_col[sort_idx]
        y = y_col[sort_idx]

        if len(x) < 5:
            return {}

        # Compute first and second derivatives (finite differences)
        dx = np.diff(x)
        dy = np.diff(y)

        # Filter near-zero dx
        valid = dx > 1e-9
        if np.sum(valid) < 3:
            return {}

        dy_dx = dy[valid] / dx[valid]

        # Second derivative
        if len(dy_dx) < 2:
            return {}

        x_mid = x[:-1][valid]
        dx2 = np.diff(x_mid)
        d2y = np.diff(dy_dx)

        valid2 = dx2 > 1e-9
        if np.sum(valid2) < 2:
            return {}

        d2y_dx2 = d2y[valid2] / dx2[valid2]

        suggestions = {}

        # Check for exponential: d²y/dx² / (dy/dx) ≈ constant
        dy_dx_mid = dy_dx[:-1][valid2]
        nonzero_dy = np.abs(dy_dx_mid) > 1e-9
        if np.sum(nonzero_dy) > 2:
            ratio = d2y_dx2[nonzero_dy] / dy_dx_mid[nonzero_dy]
            if np.std(ratio) < 0.3 * np.abs(np.mean(ratio)):
                k = np.mean(ratio)
                if np.abs(k) > 0.1:
                    suggestions["exp"] = k  # Suggests exp(k*x)

        # Check for polynomial: constant second derivative means parabola
        if np.std(d2y_dx2) < 0.2 * np.abs(np.mean(d2y_dx2)):
            if np.abs(np.mean(d2y_dx2)) > 1e-6:
                suggestions["poly"] = 2  # Suggests x^2

        # Check for logarithm: d²y/dx² * x ≈ constant (and negative)
        x_for_d2 = x_mid[:-1][valid2]
        if len(x_for_d2) > 2:
            product = d2y_dx2 * x_for_d2
            if np.std(product) < 0.3 * np.abs(np.mean(product)):
                if np.mean(product) < -1e-6:
                    suggestions["log"] = True

        return suggestions

    except Exception:
        return {}


def detect_saturation(x_col: np.ndarray, y_col: np.ndarray) -> dict:
    """Detect saturation/asymptotic behavior to suggest sigmoid-family.

    Returns dict with suggested feature types based on saturation analysis.
    """
    try:
        import numpy as np

        # Sort by x
        sort_idx = np.argsort(x_col)
        x = x_col[sort_idx]
        y = y_col[sort_idx]

        if len(x) < 8:
            return {}

        suggestions = {}

        # Split into thirds: left, middle, right
        n = len(x)
        left_y = y[: n // 3]
        right_y = y[2 * n // 3 :]

        # Check for saturation: variance at edges << variance in middle
        left_var = np.var(left_y) if len(left_y) > 1 else 0
        right_var = np.var(right_y) if len(right_y) > 1 else 0
        total_var = np.var(y)

        # Check monotonicity
        is_monotonic_increasing = np.all(np.diff(y) >= -1e-9 * np.abs(y[:-1]))
        is_monotonic_decreasing = np.all(np.diff(y) <= 1e-9 * np.abs(y[:-1]))
        is_monotonic = is_monotonic_increasing or is_monotonic_decreasing

        if total_var > 1e-9:
            # Saturation on right side (like sigmoid, tanh, softplus)
            if right_var / total_var < 0.1 and is_monotonic:
                suggestions["sigmoid_family"] = True

                # Try to detect which type
                y_min, y_max = np.min(y), np.max(y)
                y_range = y_max - y_min

                # If range is bounded (like tanh from -1 to 1)
                if y_range < 3 and is_monotonic:
                    suggestions["tanh"] = True

                # If starts near 0 and grows (like softplus, ReLU-like)
                if y_min >= -0.5 and is_monotonic_increasing:
                    suggestions["softplus"] = True

            # Saturation on both sides (like sigmoid)
            if (
                left_var / total_var < 0.1
                and right_var / total_var < 0.1
                and is_monotonic
            ):
                suggestions["sigmoid"] = True

        # --- CURVATURE-BASED SOFTPLUS DETECTION ---
        # Softplus has: monotonic increasing, starts near 0, accelerating then decelerating growth
        # d²y/dx² > 0 (convex) but approaches 0 as x → ∞
        if is_monotonic_increasing and np.min(y) >= -0.5:
            # Check if growth rate decreases (concave-like in derivative)
            dy = np.diff(y)
            dx = np.diff(x)
            valid = dx > 1e-9
            if np.sum(valid) > 3:
                growth_rate = dy[valid] / dx[valid]
                # If growth starts low, increases, then levels off → Softplus candidate
                early_growth = np.mean(growth_rate[: len(growth_rate) // 3])
                late_growth = np.mean(growth_rate[2 * len(growth_rate) // 3 :])
                if late_growth > early_growth * 0.5 and late_growth < 2.0:
                    suggestions["softplus"] = True

        return suggestions

    except Exception:
        return {}


def generate_candidate_features(
    X_data: Any,
    variable_names: list[str],
    include_transcendentals: bool = True,
    y_data: Any = None,
) -> tuple[Any, list[str]]:
    """Generates a dictionary of candidate functions (features) for symbolic regression.

    Args:
        X_data: numpy array of shape (n_samples, n_variables)
        variable_names: list of strings ['x', 'y', ...]
        include_transcendentals: If False, generates only polynomials and rationals (Stage 1).

    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    import numpy as np

    # Ensure input is numpy array
    X_data = np.array(X_data, dtype=float)
    if len(X_data.shape) == 1:
        X_data = X_data.reshape(-1, 1)

    n_samples, n_vars = X_data.shape
    features = []
    feature_names = []

    # 1. Bias term (Constant)
    features.append(np.ones(n_samples))
    feature_names.append("1")

    # 2. Simple Polynomials (Degree 1 to 3)
    # x, y, x^2, y^2, x^3...
    for i in range(n_vars):
        col = X_data[:, i]
        name = variable_names[i]

        # Power 1
        features.append(col)
        feature_names.append(name)

        # Power 2
        features.append(col**2)
        feature_names.append(f"{name}^2")

        # Power 3
        features.append(col**3)
        feature_names.append(f"{name}^3")

        # Power 4, 5, 10 (High degree scan)
        # 4 is useful for Inverse Quartic laws; 5, 10 for specific poly fits
        for p in [4, 5, 10]:
            features.append(col**p)
            feature_names.append(f"{name}^{p}")

    # 3. Interactions (x*y, x^2*y, x*y^2)
    if n_vars > 1:
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # x * y
                col = X_data[:, i] * X_data[:, j]
                name = f"{variable_names[i]}*{variable_names[j]}"
                features.append(col)
                feature_names.append(name)

                # x^2 * y and x * y^2 (Cubic Interactions)
                # x^2 * y and x * y^2 (Cubic Interactions)
                # Allow universally for Phase 1 to enable "Blindfold Physics" (variable agnostic)
                # OMP Structural Boosting will handle overfitting.

                # x^2 * y (e.g., r^2 * h)
                col_sq_i = (X_data[:, i] ** 2) * X_data[:, j]
                name_sq_i = f"{variable_names[i]}^2*{variable_names[j]}"
                features.append(col_sq_i)
                feature_names.append(name_sq_i)

                # x * y^2 (e.g., m * v^2)
                col_sq_j = X_data[:, i] * (X_data[:, j] ** 2)
                name_sq_j = f"{variable_names[i]}*{variable_names[j]}^2"
                features.append(col_sq_j)
                feature_names.append(name_sq_j)

                # --- NEW: Sqrt Interactions (sqrt(x*y)) ---
                # Only if both columns are non-negative
                if np.all(X_data[:, i] >= 0) and np.all(X_data[:, j] >= 0):
                    col_sqrt_int = np.sqrt(X_data[:, i] * X_data[:, j])
                    features.append(col_sqrt_int)
                    feature_names.append(
                        f"sqrt({variable_names[i]}*{variable_names[j]})"
                    )

    # 3b. Triple Interactions (x*y*z) - CRITICAL for physics like m*g*h, E=mc^2*t
    if n_vars >= 3:
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                for k in range(j + 1, n_vars):
                    # x * y * z
                    col = X_data[:, i] * X_data[:, j] * X_data[:, k]
                    name = (
                        f"{variable_names[i]}*{variable_names[j]}*{variable_names[k]}"
                    )
                    features.append(col)
                    feature_names.append(name)

    # --- NEW: SHIFTED RATIONALS (Doppler Shift, Singularity detection) ---
    # Moved outside of 'transcendentals' because 1/(C-x) is algebraic/rational
    for i in range(n_vars):
        col = X_data[:, i]
        name = variable_names[i]

        # 1 / (C - x) or 1 / (x - C)
        col_max = np.max(col)
        col_min = np.min(col)

        # Doppler Shift often involves Speed of Sound (340) or Light (3e8)
        possible_poles = [340.0, 30.0, 100.0, 3e8]
        # Also try "max + small_delta" or "min - small_delta"
        if np.isfinite(col_max):
            possible_poles.append(col_max + 1.0)
        if np.isfinite(col_min):
            possible_poles.append(col_min - 1.0)

        for pole in possible_poles:
            # 1 / (C - x)
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = pole - col
                # Relaxed singularity check: just ensure not ALL are zero
                if np.count_nonzero(np.abs(denom) < 1e-9) == 0:
                    inv_shifted = 1.0 / denom
                    if (
                        np.all(np.isfinite(inv_shifted))
                        and np.max(np.abs(inv_shifted)) < 1e100
                    ):
                        features.append(inv_shifted)
                        feature_names.append(f"1/({pole}-{name})")

    # --- NEW: TRANSCENDENTAL FUNCTIONS ---
    if include_transcendentals:
        # Global Transcendentals (Power Bases)
        # Check for 2^x, 10^x
        for i in range(n_vars):
            col = X_data[:, i]
            name = variable_names[i]

            # Protected Power Bases
            with np.errstate(over="ignore"):
                pow2 = 2.0**col
                if np.all(np.isfinite(pow2)) and np.max(np.abs(pow2)) < 1e100:
                    features.append(pow2)
                    feature_names.append(f"2^{name}")

                pow10 = 10.0**col
                if np.all(np.isfinite(pow10)) and np.max(np.abs(pow10)) < 1e100:
                    features.append(pow10)
                    feature_names.append(f"10^{name}")

        for i in range(n_vars):
            col = X_data[:, i]
            name = variable_names[i]

            # Sine (sin(x))
            features.append(np.sin(col))
            feature_names.append(f"sin({name})")

            # sin(x)/x (Sinc function - critical for signal processing)
            if not np.any(np.isclose(col, 0, atol=1e-10)):
                sinc_col = np.sin(col) / col
                features.append(sinc_col)
                feature_names.append(f"sin({name})/{name}")

        # Argument Scaling: Frequency (sin(2x), sin(pi*x))
        features.append(np.sin(2 * col))
        feature_names.append(f"sin(2*{name})")
        features.append(np.sin(np.pi * col))
        feature_names.append(f"sin(pi*{name})")

        # Cosine (cos(x))
        features.append(np.cos(col))
        feature_names.append(f"cos({name})")

        # Argument Scaling: Frequency Scan (sin(kx), cos(kx))
        # PURE DISCOVERY - No training wheels. Learn the hard way.
        freq_candidates = set()  # Start with NOTHING. Discovery must find it.

        if y_data is not None:
            try:
                detected_freqs = detect_frequency(col, y_data)
                for k in detected_freqs:
                    freq_candidates.add(k)
                    # Trust the detector's harmonic expansion
            except:
                pass

        for k in sorted(freq_candidates):
            k_int = int(k) if isinstance(k, float) and k.is_integer() else k
            features.append(np.sin(k_int * col))
            feature_names.append(f"sin({k_int}*{name})")
            features.append(np.cos(k_int * col))
            feature_names.append(f"cos({k_int}*{name})")

        features.append(np.sin(np.pi * col))
        feature_names.append(f"sin(pi*{name})")

        # Exponential (exp(x) or e^x)
        # Be careful with overflow! Maybe clip values or only use for small inputs
        with np.errstate(over="ignore"):
            exp_col = np.exp(col)
            # Only add if values aren't infinite and not too huge
            if np.all(np.isfinite(exp_col)) and np.max(np.abs(exp_col)) < 1e100:
                features.append(exp_col)
                feature_names.append(f"exp({name})")

        # Exponential Decay (exp(-x))
        with np.errstate(over="ignore"):
            exp_neg_col = np.exp(-col)
            if np.all(np.isfinite(exp_neg_col)) and np.max(np.abs(exp_neg_col)) < 1e100:
                features.append(exp_neg_col)
                feature_names.append(f"exp(-{name})")

        # Arrhenius / Inverse Exponential (exp(A/x), exp(-A/x))
        # Scan common activation energy scalings
        numerators = [1.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

        with np.errstate(all="ignore"):
            if not np.any(np.isclose(col, 0, atol=1e-9)):
                inv_col_base = 1.0 / col

                for num in numerators:
                    inv_col = num * inv_col_base

                    # exp(A/x)
                    # Only do positive scaling if necessary (often Arrhenius is exp(-A/T))
                    # But sometimes we need exp(A/T)? Maybe rarely.
                    # Let's stick to base 1.0 for positive, and scan negative.
                    if num == 1.0:
                        exp_inv = np.exp(inv_col)
                        if (
                            np.all(np.isfinite(exp_inv))
                            and np.max(np.abs(exp_inv)) < 1e100
                        ):
                            features.append(exp_inv)
                            feature_names.append(f"exp(1/{name})")

                    # exp(-A/T) - The Standard Arrhenius
                    exp_neg_inv = np.exp(-inv_col)
                    if (
                        np.all(np.isfinite(exp_neg_inv))
                        and np.max(np.abs(exp_neg_inv)) < 1e100
                    ):
                        features.append(exp_neg_inv)
                        # Use clean naming
                        if num == 1.0:
                            feature_names.append(f"exp(-1/{name})")
                        else:
                            feature_names.append(f"exp(-{int(num)}/{name})")

        # --- NEW: GAUSSIAN (exp(-x^2)) - Bell Curve, Normal Distribution ---
        # This is FUNDAMENTALLY different from exp(-x):
        # - exp(-x) decays linearly in log scale
        # - exp(-x^2) has the bell curve shape
        with np.errstate(over="ignore"):
            gaussian_col = np.exp(-(col**2))
            if (
                np.all(np.isfinite(gaussian_col))
                and np.max(np.abs(gaussian_col)) < 1e100
            ):
                features.append(gaussian_col)
                feature_names.append(f"exp(-{name}^2)")

                # Damped Harmonic Motion Interactions (exp(-x) * sin(x), etc.)
                # Only add if decay term is valid

                # exp(-x) * sin(x)
                features.append(exp_neg_col * np.sin(col))
                feature_names.append(f"exp(-{name})*sin({name})")

                # exp(-x) * cos(x)
                features.append(exp_neg_col * np.cos(col))
                feature_names.append(f"exp(-{name})*cos({name})")

                # exp(-x) * sin(2x)
                features.append(exp_neg_col * np.sin(2 * col))
                feature_names.append(f"exp(-{name})*sin(2*{name})")

                # exp(-x) * cos(2x)
                features.append(exp_neg_col * np.cos(2 * col))
                feature_names.append(f"exp(-{name})*cos(2*{name})")

        # Logarithm (log(x))
        # Logarithm (log(x))
        # Only valid for positive inputs
        if np.all(col > 0):
            features.append(np.log(col))
            feature_names.append(f"log({name})")

            # log2(x) - Entropy (bits)
            features.append(np.log2(col))
            feature_names.append(f"log2({name})")

            # log10(x) - Decibels, pH, etc.
            features.append(np.log10(col))
            feature_names.append(f"log10({name})")

            # --- NEW: LOG-NORMAL (exp(-log(x)^2)) ---
            # Common in distribution of sizes
            with np.errstate(over="ignore"):
                log_sq = np.log(col) ** 2
                # exp(-log(x)^2)
                log_norm = np.exp(-log_sq)
                features.append(log_norm)
                feature_names.append(f"exp(-log({name})^2)")

                # exp(-log(x)^2 / 2) -> Standard LogNormal kernel with sigma=1
                log_norm_2 = np.exp(-log_sq / 2.0)
                features.append(log_norm_2)
                feature_names.append(f"exp(-log({name})^2/2)")

        # x * log(x) - Entropy, Information Theory
        # Valid for x >= 0 (limit x->0 is 0)
        if np.all(col >= 0):
            # Compute x * log(x) safely
            with np.errstate(invalid="ignore", divide="ignore"):
                # Use a mask where x > 0
                x_log_x = np.zeros_like(col)
                mask_pos = col > 1e-12
                if np.any(mask_pos):
                    x_log_x[mask_pos] = col[mask_pos] * np.log(col[mask_pos])
                
                features.append(x_log_x)
                feature_names.append(f"{name}*log({name})")
                
                # Also add (x*log(x))^2
                features.append(x_log_x**2)
                feature_names.append(f"({name}*log({name}))^2")

        # --- NEW: VARIABLE-TRANSCENDENTAL PRODUCTS (Growing Wave, Modulated Signals) ---
        # These are CRITICAL for physics: x*sin(x), x*cos(x), x*exp(x)

        # x * sin(x) - Growing sine wave, modulated signals
        features.append(col * np.sin(col))
        feature_names.append(f"{name}*sin({name})")

        # x * cos(x) - Growing cosine wave
        features.append(col * np.cos(col))
        feature_names.append(f"{name}*cos({name})")

        # x^2 * sin(x) - Polynomial-modulated wave
        features.append((col**2) * np.sin(col))
        feature_names.append(f"{name}^2*sin({name})")

        # x^2 * cos(x) - Polynomial-modulated wave
        features.append((col**2) * np.cos(col))
        feature_names.append(f"{name}^2*cos({name})")

        # --- NEW: HYPERBOLIC FUNCTIONS (Catenary, Special Relativity, etc.) ---
        # sinh(x) = (exp(x) - exp(-x)) / 2
        # cosh(x) = (exp(x) + exp(-x)) / 2
        with np.errstate(over="ignore"):
            sinh_col = np.sinh(col)
            cosh_col = np.cosh(col)
            if np.all(np.isfinite(sinh_col)) and np.max(np.abs(sinh_col)) < 1e100:
                features.append(sinh_col)
                feature_names.append(f"sinh({name})")
            if np.all(np.isfinite(cosh_col)) and np.max(np.abs(cosh_col)) < 1e100:
                features.append(cosh_col)
                feature_names.append(f"cosh({name})")

        # --- NEW: ACTIVATION FUNCTIONS (Sigmoid, Softplus, Tanh) ---
        # Critical for Neural Network behaviors and biological growth
        with np.errstate(over="ignore", invalid="ignore"):
            # Sigmoid: 1 / (1 + exp(-x))
            exp_neg = np.exp(-col)
            if np.all(np.isfinite(exp_neg)):
                sigmoid = 1.0 / (1.0 + exp_neg)
                features.append(sigmoid)
                feature_names.append(f"1/(1+exp(-{name}))")

            # Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
            features.append(np.tanh(col))
            feature_names.append(f"tanh({name})")

            # Softplus: ln(1 + exp(x))
            # Use log1p for numerical stability
            exp_pos = np.exp(col)
            if np.all(np.isfinite(exp_pos)):
                softplus = np.log1p(exp_pos)
                features.append(softplus)
                feature_names.append(f"log(1+exp({name}))")

                # Alternative Softplus: ln(1 + exp(-x))
                # Common in physics
                if np.all(np.isfinite(exp_neg)):
                    softplus_neg = np.log1p(exp_neg)
                    features.append(softplus_neg)
                    feature_names.append(f"log(1+exp(-{name}))")

        # --- PHASE 4: SATURATION-DETECTED SIGMOID VARIANTS ---
        # DYNAMIC DISCOVERY: If data shows saturation, add scaled variants
        if y_data is not None:
            try:
                saturation_hints = detect_saturation(col, y_data)
                if saturation_hints.get("softplus") or saturation_hints.get(
                    "sigmoid_family"
                ):
                    # Add scaled Softplus: k * log(1 + exp(x/k))
                    for k in [0.5, 2.0, 5.0, 10.0]:
                        with np.errstate(all="ignore"):
                            scaled_softplus = k * np.log1p(np.exp(col / k))
                            if np.all(np.isfinite(scaled_softplus)):
                                features.append(scaled_softplus)
                                feature_names.append(f"{k}*log(1+exp({name}/{k}))")

                if saturation_hints.get("tanh") or saturation_hints.get("sigmoid"):
                    # Add steeper Tanh: tanh(k*x)
                    for k in [2.0, 5.0, 10.0, 50.0, 100.0]:
                        features.append(np.tanh(k * col))
                        feature_names.append(f"tanh({k}*{name})")
            except:
                pass

        # --- PHASE 2 & 3: GENIUS MODE FEATURES ---

        # 1. Self Power (x^x)
        if np.all(col > 0):
            with np.errstate(all="ignore"):
                x_x = np.power(col, col)
                if np.all(np.isfinite(x_x)) and np.max(np.abs(x_x)) < 1e100:
                    features.append(x_x)
                    feature_names.append(f"{name}^{name}")

        # 2. Oscillator Singularity (sin(1/x))
        with np.errstate(all="ignore"):
            if np.count_nonzero(np.abs(col) < 1e-9) == 0:
                sin_inv = np.sin(1.0 / col)
                features.append(sin_inv)
                feature_names.append(f"sin(1/{name})")

        # 3. Lorentzian / Cauchy (1 / (1 + k*x^2))
        for k in [1, 4, 10, 25, 100]:
            lor_denom = 1.0 + k * (col**2)
            lor_feat = 1.0 / lor_denom
            features.append(lor_feat)
            feature_names.append(f"1/(1+{k}*{name}^2)")

        # 4. Tanh with Shelf (tanh(k*(x-c)))
        tanh_shifts = [0.5, 0.0, 1.0]
        tanh_scales = [100.0, 10.0, 50.0]
        for c in tanh_shifts:
            for k in tanh_scales:
                t_arg = k * (col - c)
                t_feat = np.tanh(t_arg)
                features.append(t_feat)
                feature_names.append(f"tanh({k}*({name}-{c}))")

        # 5. Complex Composites (Ackley components)

        # exp(sin(x))
        with np.errstate(all="ignore"):
            exp_sin = np.exp(np.sin(col))
            if np.all(np.isfinite(exp_sin)):
                features.append(exp_sin)
                feature_names.append(f"exp(sin({name}))")

        # exp(cos(kx)) - Scan k=1, 2pi, 2
        k_vals = [1.0, 2.0, np.pi, 2 * np.pi]
        for k in k_vals:
            with np.errstate(all="ignore"):
                arg = np.cos(k * col)
                feat = np.exp(arg)
                if np.all(np.isfinite(feat)):
                    k_str = f"{k:.2f}" if (k != 1.0 and k != 2.0) else f"{int(k)}"
                    if abs(k - np.pi) < 1e-5:
                        k_str = "pi"
                    if abs(k - 2 * np.pi) < 1e-5:
                        k_str = "2*pi"
                    fname = f"cos({k_str}*{name})" if k != 1.0 else f"cos({name})"
                    features.append(feat)
                    feature_names.append(f"exp({fname})")

        # exp(-k * sqrt(x^2))
        ks_ackley = [0.2, 0.5, 1.0]
        abs_col = np.abs(col)
        for k in ks_ackley:
            with np.errstate(all="ignore"):
                arg = -k * abs_col
                feat = np.exp(arg)
                feat_name_inner = f"-{k}*sqrt({name}^2)"
                if np.all(np.isfinite(feat)):
                    features.append(feat)
                    feature_names.append(f"exp({feat_name_inner})")

    # --- NEW: RATIONAL FUNCTIONS (1/x) ---
    # This helps find physics laws like Inverse Square Law

    # --- NEW: LORENTZ FACTOR / RELATIVISTIC (1/sqrt(1-x^2/c^2)) ---
    if include_transcendentals:
        for i in range(n_vars):
            col = X_data[:, i]
            name = variable_names[i]

            # Standard Relativistic: 1/sqrt(1 - v^2) (assuming c=1)
            # Check domain: 1 - v^2 > 0  => |v| < 1
            with np.errstate(invalid="ignore"):
                one_minus_v2 = 1.0 - col**2
                if np.all(one_minus_v2 > 0):
                    lorentz = 1.0 / np.sqrt(one_minus_v2)
                    features.append(lorentz)
                    feature_names.append(f"1/sqrt(1-{name}^2)")

    for i in range(n_vars):
        col = X_data[:, i]
        name = variable_names[i]

        # Avoid division by zero
        if not np.any(np.isclose(col, 0, atol=1e-10)):
            features.append(1 / col)
            feature_names.append(f"1/{name}")

            features.append(1 / (col**2))
            feature_names.append(f"1/{name}^2")

            # Lennard-Jones (1/r^6, 1/r^12) and others
            # PURE DISCOVERY - No training wheels. Learn the hard way.
            pow_candidates = set()  # Start with NOTHING.

            if y_data is not None:
                try:
                    detected = detect_power_laws(col, y_data)
                    for e in detected:
                        if e < -1.0:  # Negative powers (Inverse)
                            pow_candidates.add(abs(e))
                except:
                    pass

            for p in sorted(list(pow_candidates)):
                # Handle fractional powers? 2.5?
                with np.errstate(all="ignore"):
                    if isinstance(p, float) and p.is_integer():
                        p = int(p)

                    inv_p = 1.0 / (col**p)
                    if np.all(np.isfinite(inv_p)):
                        features.append(inv_p)
                        feature_names.append(f"1/{name}^{p}")

    # --- NEW: RATIONAL INTERACTIONS (x/y, x*y/z) ---
    # Critical for Ideal Gas Law (P = nT/V) and others
    if n_vars > 1:
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue

                # Feature: x / y
                col_i = X_data[:, i]
                col_j = X_data[:, j]
                name_i = variable_names[i]
                name_j = variable_names[j]

                if not np.any(np.isclose(col_j, 0, atol=1e-10)):
                    features.append(col_i / col_j)
                    feature_names.append(f"{name_i}/{name_j}")

    if n_vars > 2:
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                for k in range(n_vars):
                    if k == i or k == j:
                        continue

                    # Feature: x * y / z
                    col_i = X_data[:, i]
                    col_j = X_data[:, j]
                    col_k = X_data[:, k]
                    name_i = variable_names[i]
                    name_j = variable_names[j]
                    name_k = variable_names[k]

                    if not np.any(np.isclose(col_k, 0, atol=1e-10)):
                        # x * y / z
                        features.append((col_i * col_j) / col_k)
                        name_new = f"{name_i}*{name_j}/{name_k}"
                        feature_names.append(name_new)
                        # print(f"DEBUG GEN: {name_new}", flush=True)

                        # x * y / z^2 (Inverse Square Product)
                        features.append((col_i * col_j) / (col_k**2))
                        feature_names.append(f"{name_i}*{name_j}/{name_k}^2")
                        # print(f"DEBUG GEN: {name_i}*{name_j}/{name_k}^2", flush=True)

    # Feature: x * y * z / w (Triple Product Ratio for Reynolds Number)
    if n_vars > 3:
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                for k in range(j + 1, n_vars):
                    for idx_l in range(n_vars):
                        if idx_l in [i, j, k]:
                            continue

                        col_i = X_data[:, i]
                        col_j = X_data[:, j]
                        col_k = X_data[:, k]
                        col_l = X_data[:, idx_l]

                        if not np.any(np.isclose(col_l, 0, atol=1e-10)):
                            features.append((col_i * col_j * col_k) / col_l)
                            feature_names.append(
                                f"{variable_names[i]}*{variable_names[j]}*{variable_names[k]}/{variable_names[idx_l]}"
                            )

                            # Triple Product Inverse Quartic (for Hagen-Poiseuille: mu*L*Q/r^4)
                            features.append((col_i * col_j * col_k) / (col_l**4))
                            feature_names.append(
                                f"{variable_names[i]}*{variable_names[j]}*{variable_names[k]}/{variable_names[idx_l]}^4"
                            )

    # --- NEW: GEOMETRIC INTERACTIONS (Cone/Pyramid) ---
    # x * sqrt(x^2 + y^2) - Algebraic, so allowed without transcendentals
    if n_vars >= 2:
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                # r * sqrt(r^2 + h^2)
                col_i = X_data[:, i]
                col_j = X_data[:, j]
                sum_sq = col_i**2 + col_j**2
                sqrt_sum = np.sqrt(sum_sq)

                features.append(col_i * sqrt_sum)
                feature_names.append(
                    f"{variable_names[i]}*sqrt({variable_names[i]}^2+{variable_names[j]}^2)"
                )

    # --- NEW: QUANTUM PHYSICS INTERACTIONS (Planck's Law) ---
    if include_transcendentals:
        for i in range(n_vars):
            col = X_data[:, i]
            name = variable_names[i]

            with np.errstate(over="ignore", invalid="ignore"):
                # exp(x) - 1
                exp_minus_one = np.exp(col) - 1.0

                # Check if valid and not zero (to avoid division by zero)
                # We'll use a mask or safe division
                valid_denom = np.all(np.isfinite(exp_minus_one)) and not np.any(
                    np.isclose(exp_minus_one, 0, atol=1e-10)
                )

                if valid_denom:
                    # x^3 / (exp(x) - 1)
                    term1 = (col**3) / exp_minus_one
                    if np.all(np.isfinite(term1)):
                        features.append(term1)
                        feature_names.append(f"{name}^3/(exp({name})-1)")

                    # x^5 / (exp(x) - 1)
                    term2 = (col**5) / exp_minus_one
                    if np.all(np.isfinite(term2)):
                        features.append(term2)
                        feature_names.append(f"{name}^5/(exp({name})-1)")

    # --- NEW: TRANSCENDENTAL FUNCTIONS (x^x, x*log(x), interactions) ---
    if include_transcendentals:
        # 1. Transcendental Interactions (e.g. exp(-t) * cos(2t))
        # We need to explicitly generate product of Exp and Trig columns
        # because the generic interaction loop (lines 543+) only handles Initial columns (x, y).
        # But we haven't generated exp/sin columns yet!
        # Wait, the transcendental generation loop is BELOW here (Lines 735+ in original).
        # I should insert my interactions AFTER generation?
        pass

    # --- NEW: SELF-POWER FUNCTIONS (x^x) ---
    if include_transcendentals:
        for i in range(n_vars):
            col = X_data[:, i]
            name = variable_names[i]

            # x^x is only valid for x > 0 (to stay real)
            if np.all(col > 0):
                with np.errstate(over="ignore", invalid="ignore"):
                    # Use power(col, col)
                    self_pow = np.power(col, col)
                    if (
                        np.all(np.isfinite(self_pow))
                        and np.max(np.abs(self_pow)) < 1e100
                    ):
                        features.append(self_pow)
                        feature_names.append(f"{name}^{name}")

        # --- NEW: TRANSCENDENTAL-POLYNOMIAL INTERACTIONS (x*exp(x)) ---
        # Critical for Taylor series vs Exact form disambiguation (e.g. x*exp(x))
        for i in range(n_vars):
            col = X_data[:, i]
            name = variable_names[i]

            with np.errstate(over="ignore", invalid="ignore"):
                exp_col = np.exp(col)
                exp_neg_col = np.exp(-col)
                exp_gauss = np.exp(-(col**2))

                # Check validity before adding
                has_exp = (
                    np.all(np.isfinite(exp_col)) and np.max(np.abs(exp_col)) < 1e100
                )
                has_exp_neg = (
                    np.all(np.isfinite(exp_neg_col))
                    and np.max(np.abs(exp_neg_col)) < 1e100
                )
                has_gauss = np.all(np.isfinite(exp_gauss))

                # x * exp(x), x^2 * exp(x)
                if has_exp:
                    features.append(col * exp_col)
                    feature_names.append(f"{name}*exp({name})")
                    features.append(col**2 * exp_col)
                    feature_names.append(f"{name}^2*exp({name})")

                # x * exp(-x), x^2 * exp(-x) - Gamma distribution shapes
                if has_exp_neg:
                    features.append(col * exp_neg_col)
                    feature_names.append(f"{name}*exp(-{name})")
                    features.append(col**2 * exp_neg_col)
                    feature_names.append(f"{name}^2*exp(-{name})")

                # x * exp(-x^2) - Gaussian Derivative (Hermite polynomials)
                if has_gauss:
                    features.append(col * exp_gauss)
                    feature_names.append(f"{name}*exp(-{name}^2)")

    # --- NEW: KNOWLEDGE EXPANSION (INVERSE TRIG, PIECEWISE, SPECIAL) ---
    if include_transcendentals:
        from scipy.special import erf, gamma

        for i in range(n_vars):
            col = X_data[:, i]
            name = variable_names[i]

            # 1. Inverse Trigonometric
            # Arcsin/Arccos valid for [-1, 1]
            if np.all(np.abs(col) <= 1.0):
                features.append(np.arcsin(col))
                feature_names.append(f"asin({name})")
                features.append(np.arccos(col))
                feature_names.append(f"acos({name})")

            # Arctan valid everywhere
            features.append(np.arctan(col))
            feature_names.append(f"atan({name})")

            # 2. Reciprocal Trigonometric
            # Tan (sin/cos). Valid if cos != 0.
            # Avoid asymptotes
            if not np.any(np.isclose(np.cos(col), 0, atol=1e-5)):
                features.append(np.tan(col))
                feature_names.append(f"tan({name})")

            # 3. Piecewise / Discontinuous (Fundamental for Engineering/AI)
            # Abs |x|
            features.append(np.abs(col))
            feature_names.append(f"abs({name})")

            # Sign sign(x)
            features.append(np.sign(col))
            feature_names.append(f"sign({name})")

            # ReLU max(0, x) (AI)
            features.append(np.maximum(0, col))
            feature_names.append(f"relu({name})")

            # Step / Floor / Ceil
            features.append(np.floor(col))
            feature_names.append(f"floor({name})")
            features.append(np.ceil(col))
            feature_names.append(f"ceil({name})")

            # 4. Special Functions (Physics/Prob)
            # Error Function erf(x)
            features.append(erf(col))
            feature_names.append(f"erf({name})")

            # Gamma Function (Factorial). Valid for x > 0 (roughly) or non-integer negative
            # We restrict to positive for safety
            if np.all(col > 0):
                with np.errstate(all="ignore"):
                    g_val = gamma(col)
                    if np.all(np.isfinite(g_val)) and np.max(np.abs(g_val)) < 1e100:
                        features.append(g_val)
                        feature_names.append(f"gamma({name})")

    return np.column_stack(features), feature_names


def check_log_linear_transformations(
    X_data: Any, y_data: Any, variable_names: list[str]
) -> tuple[bool, str | None]:
    """Check for simple log-linear relationships (exponential and power laws).

    Args:
        X_data: Input data (n_samples, n_vars)
        y_data: Output data (n_samples,)
        variable_names: List of variable names

    Returns:
        Tuple (success, function_string)
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression

    X_data = np.array(X_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

    if len(X_data.shape) == 1:
        X_data = X_data.reshape(-1, 1)

    n_samples, n_vars = X_data.shape

    # Only support single variable for simple checks for now
    if n_vars != 1:
        return False, None

    x = X_data[:, 0]
    y = y_data
    var_name = variable_names[0]

    # 1. Check Exponential: y = A * e^(Bx)  => ln(y) = ln(A) + Bx
    # Valid only if all y > 0
    if np.all(y > 0):
        try:
            log_y = np.log(y)
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), log_y)
            r2 = model.score(x.reshape(-1, 1), log_y)

            if r2 > 0.999:  # Strong fit
                B = model.coef_[0]
                ln_A = model.intercept_
                A = np.exp(ln_A)

                # Format nicely
                def _fmt_val(val):
                    import numpy as np

                    if abs(val - np.pi) < 1e-4:
                        return "pi"
                    if abs(val - 2 * np.pi) < 1e-4:
                        return "2*pi"
                    if abs(val - 0.5 * np.pi) < 1e-4:
                        return "0.5*pi"
                    return f"{val:.10g}"

                A_str = _fmt_val(A)
                B_str = _fmt_val(B)

                if abs(A - 1.0) < 0.01:
                    A_str = ""
                else:
                    A_str = f"{A_str}*"

                if abs(B - 1.0) < 0.01:
                    return True, f"{A_str}exp({var_name})"
                else:
                    return True, f"{A_str}exp({B_str}*{var_name})"
        except Exception:
            pass

    # 2. Check Power Law: y = A * x^B => ln(y) = ln(A) + B*ln(x)
    # Valid only if all x > 0 and y > 0
    if np.all(x > 0) and np.all(y > 0):
        try:
            log_x = np.log(x)
            log_y = np.log(y)
            model = LinearRegression()
            model.fit(log_x.reshape(-1, 1), log_y)
            r2 = model.score(log_x.reshape(-1, 1), log_y)

            if r2 > 0.999:  # Strong fit
                B = model.coef_[0]
                ln_A = model.intercept_
                A = np.exp(ln_A)

                # Format nicely
                # Format nicely using robust logic (same as regression_solver._symbolify_coefficient)
                def _fmt_val(val):
                    if abs(val) < 1e-6:
                        return "0"

                    # 1. Round to integer
                    rounded = round(val)
                    if abs(val - rounded) < 0.001 and abs(rounded) > 0.5:
                        return str(int(rounded))

                    # 2. Pi and Pi fractions
                    import sympy as sp

                    pi_val = float(sp.pi.evalf())

                    # Check specific range including Sphere Volume 4/3 etc.
                    for denom in [1, 2, 3, 4, 6]:
                        for num in range(-15, 16):
                            if num == 0:
                                continue
                            expected = (num / denom) * pi_val
                            if abs(val - expected) < 0.001:
                                if denom == 1:
                                    if num == 1:
                                        return "pi"
                                    if num == -1:
                                        return "-pi"
                                    return f"{num}*pi"
                                else:
                                    return (
                                        f"{num}/{denom}*pi"
                                        if num > 0
                                        else f"({num}/{denom})*pi"
                                    )

                    # 3. Simple fractions
                    for denom in [2, 3, 4, 5, 8, 10]:
                        for num in range(-20, 21):
                            if num == 0:
                                continue
                            expected = num / denom
                            if abs(val - expected) < 0.001:
                                return (
                                    f"{num}/{denom}" if num > 0 else f"({num}/{denom})"
                                )

                    return f"{val:.10g}"

                A_str = _fmt_val(A)
                B_str = _fmt_val(B)

                if abs(A - 1.0) < 0.01:
                    A_str = ""
                else:
                    A_str = f"{A_str}*"

                return True, f"{A_str}{var_name}^{B_str}"
        except Exception:
            pass

    return False, None
