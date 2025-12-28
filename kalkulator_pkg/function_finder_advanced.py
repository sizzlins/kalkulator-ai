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
from decimal import Decimal
from decimal import getcontext
from fractions import Fraction
from typing import Any

import numpy as np
import sympy as sp

try:
    import mpmath  # noqa: F401

    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

from .config import ABSOLUTE_TOLERANCE
from .config import CONSTANT_DETECTION_TOLERANCE
from .config import LASSO_LAMBDA
from .config import OMP_MAX_ITERATIONS
from .config import RELATIVE_TOLERANCE

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

    # Fast check for integers
    if abs(float_val - round(float_val)) < tolerance:
        return sp.Integer(round(float_val))

    # Explicit check for common log constants (often missed by nsimplify)
    # This specifically addresses the user's issue with discovering exp(log(2)*x)
    log_candidates = [
        (sp.log(2), 0.69314718056),
        (sp.log(3), 1.09861228867),
        (sp.log(10), 2.30258509299),
        (sp.pi, 3.14159265359),
        (sp.E, 2.71828182846),
    ]

    for sym_cand, val_cand in log_candidates:
        if abs(float_val - val_cand) < tolerance:
            return sym_cand

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
            float_val, tolerance=tolerance, constants=constants_list, rational=True
        )

        # Check if simplified result actually contains our constants
        # (nsimplify sometimes just returns a fraction)
        if simplified.has(sp.pi) or simplified.has(sp.E) or simplified.has(sp.sqrt):
            # Double check numerical accuracy
            if abs(float_val - float(simplified.evalf())) < tolerance:
                # Extra check: Is it simple?
                den = sp.denom(simplified)
                if abs(den) <= 24:  # Reject weird denominators
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
        except Exception:
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
        except Exception:
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


def detect_poles_from_data(
    X_data: np.ndarray, y_data: np.ndarray
) -> list[tuple[float, int]]:
    """Detect pole locations and orders from nan/inf values in y.

    When y approaches infinity (or is nan), the corresponding x value is a pole.
    We estimate the pole order n using log-log slope analysis.

    Args:
        X_data: Input data array (n_samples, n_vars)
        y_data: Output data array (n_samples,)

    Returns:
        List of (pole_location, estimated_order) tuples
    """
    import numpy as np

    if y_data is None or len(y_data) == 0:
        return []

    # Ensure X_data is 2D
    if X_data.ndim == 1:
        X_data = X_data.reshape(-1, 1)

    poles = []

    # Find nan/inf indices
    invalid_mask = ~np.isfinite(y_data)
    invalid_indices = np.where(invalid_mask)[0]

    for idx in invalid_indices:
        if X_data.shape[1] >= 1:
            pole_x = X_data[idx, 0]

            # Find points NEAR the pole (within some distance)
            distances = np.abs(X_data[:, 0] - pole_x)
            valid_mask = np.isfinite(y_data) & (distances > 1e-9)

            # Get nearby valid points
            nearby_mask = valid_mask & (distances < 2.0)  # Within distance 2
            n_nearby = np.sum(nearby_mask)

            if n_nearby >= 3:
                # Estimate pole order using log-log slope
                x_nearby = X_data[nearby_mask, 0]
                y_nearby = y_data[nearby_mask]

                # log|y| ≈ -n * log|x - pole| + const
                log_dist = np.log(np.abs(x_nearby - pole_x) + 1e-15)
                log_y = np.log(np.abs(y_nearby) + 1e-15)

                # Filter out invalid log values
                valid_log = np.isfinite(log_dist) & np.isfinite(log_y)
                if np.sum(valid_log) >= 3:
                    try:
                        # Linear regression: slope = -n
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", np.RankWarning)
                            coeffs = np.polyfit(log_dist[valid_log], log_y[valid_log], 1)
                        estimated_n = -coeffs[0]

                        # Round to nearest integer, clamp to 1-4
                        n = int(round(estimated_n))
                        n = max(1, min(4, n))  # Clamp to [1, 4]

                        poles.append((float(pole_x), n))
                    except Exception:
                        # Fallback: assume order 1
                        poles.append((float(pole_x), 1))
            else:
                # Not enough data, assume order 1
                poles.append((float(pole_x), 1))

    # Remove duplicates (round to avoid floating point issues)
    seen = set()
    unique_poles = []
    for pole_x, n in poles:
        key = round(pole_x, 6)
        if key not in seen:
            seen.add(key)
            unique_poles.append((pole_x, n))

    return unique_poles


def generate_candidate_features(
    X_data: Any,
    variable_names: list[str],
    include_transcendentals: bool = True,
    y_data: Any = None,
    X_original: Any = None,
    y_original: Any = None,
) -> tuple[Any, list[str]]:
    """Generates a dictionary of candidate functions (features) for symbolic regression.

    Args:
        X_data: numpy array of shape (n_samples, n_variables) - FILTERED data for features
        variable_names: list of strings ['x', 'y', ...]
        include_transcendentals: If False, generates only polynomials and rationals (Stage 1).
        y_data: Filtered y data (for frequency detection, etc.)
        X_original: Original unfiltered X data (for pole detection)
        y_original: Original unfiltered y data (for pole detection)

    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    import numpy as np

    # Ensure input is numpy array
    X_data = np.array(X_data, dtype=float)
    if len(X_data.shape) == 1:
        X_data = X_data.reshape(-1, 1)

    n_samples, n_vars = X_data.shape

    # Ensure we have enough variable names
    # If the user provided fewer names than columns (e.g. data has 2 cols but user said "find f(x)"),
    # pad with default names (x0, x1, etc.) or generic names to prevent IndexError.
    if variable_names:
        standard_defaults = ["x", "y", "z", "t", "u", "v"]
        used_names = set(variable_names)

        while len(variable_names) < n_vars:
            # Find first default not used
            next_name = None
            for name in standard_defaults:
                if name not in used_names:
                    next_name = name
                    break

            if not next_name:
                next_name = f"x_{len(variable_names)}"

            variable_names.append(next_name)
            used_names.add(next_name)
    else:
        # Should be handled by caller, but safe fallback
        variable_names = [f"x_{i}" for i in range(n_vars)]

    features = []
    feature_names = []

    # 1. Bias term (Constant)
    features.append(np.ones(n_samples))
    feature_names.append("1")

    # 2. Simple Polynomials (Degree 1 to 3)
    # x, y, x^2, y^2, x^3...
    for i in range(n_vars):
        col = X_data[:, i]
        # Robustness Check (Rule 5): Ensure enough names
        if i < len(variable_names):
            name = variable_names[i]
        else:
            name = f"var_{i}"  # Fallback name if missing

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

    # --- DYNAMIC POLE DETECTION (from nan/inf in y_data) ---
    # Detects poles and their orders using log-log slope analysis
    # Use ORIGINAL (unfiltered) data if provided, otherwise fall back to filtered data
    X_for_poles = X_original if X_original is not None else X_data
    y_for_poles = y_original if y_original is not None else y_data

    if y_for_poles is not None:
        try:
            detected_poles = detect_poles_from_data(X_for_poles, y_for_poles)

            for pole_x, pole_order in detected_poles:
                col = X_data[:, 0]  # Use FILTERED X for feature generation
                name = variable_names[0] if variable_names else "x"

                with np.errstate(divide="ignore", invalid="ignore"):
                    denom = col - pole_x

                    # Mask: non-pole points (where denom is not near zero)
                    non_pole_mask = np.abs(denom) > 1e-9

                    # Generate features for detected pole order and lower orders
                    for n in range(1, pole_order + 1):
                        # 1/(x-pole)^n
                        inv_n = np.where(non_pole_mask, 1.0 / (denom**n), np.nan)

                        # Check that non-pole values are finite and bounded
                        valid_vals = inv_n[non_pole_mask]
                        if (
                            len(valid_vals) > 0
                            and np.all(np.isfinite(valid_vals))
                            and np.max(np.abs(valid_vals)) < 1e100
                        ):
                            features.append(inv_n)
                            if n == 1:
                                feature_names.append(f"1/({name}-{pole_x})")
                            else:
                                feature_names.append(f"1/({name}-{pole_x})^{n}")

                            # Also add x/(x-pole)^n for numerator terms
                            x_over_pole_n = np.where(
                                non_pole_mask, col * (1.0 / (denom**n)), np.nan
                            )
                            valid_x_vals = x_over_pole_n[non_pole_mask]
                            if (
                                len(valid_x_vals) > 0
                                and np.all(np.isfinite(valid_x_vals))
                                and np.max(np.abs(valid_x_vals)) < 1e100
                            ):
                                features.append(x_over_pole_n)
                                if n == 1:
                                    feature_names.append(f"{name}/({name}-{pole_x})")
                                else:
                                    feature_names.append(
                                        f"{name}/({name}-{pole_x})^{n}"
                                    )
        except Exception:
            pass  # Fail gracefully if pole detection fails

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

            # Refactored Transcendental Features logic
            _add_transcendental_features(col, name, features, feature_names, y_data)
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
                except Exception:
                    pass

            for p in sorted(pow_candidates):
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
                        # x * y / z^2 (Inverse Square Product)
                        features.append((col_i * col_j) / (col_k**2))
                        feature_names.append(f"{name_i}*{name_j}/{name_k}^2")
                        # print(f"DEBUG GEN: {name_i}*{name_j}/{name_k}^2", flush=True)

    # --- NEW: LAMBERT W FUNCTION (x*e^x = y => x = W(y)) ---
    # Critical for inverting x^x, x*log(x), etc.
    if include_transcendentals:
        try:
            from scipy.special import lambertw

            for i in range(n_vars):
                col = X_data[:, i]
                name = variable_names[i]

                # Standard W(x) - principal branch
                # Real branch only
                with np.errstate(all="ignore"):
                    # lambertw returns complex, we want real part if imag is negligible
                    w_val = lambertw(col)
                    if np.all(np.abs(np.imag(w_val)) < 1e-9):
                        w_real = np.real(w_val)
                        if np.all(np.isfinite(w_real)):
                            features.append(w_real)
                            feature_names.append(f"LambertW({name})")

                # GENIUS FEATURE: Inverse of x^x
                # x^x = y  =>  x = exp(W(log(y)))
                # This pattern is specifically requested and mathematically significant.
                if np.all(col > 0):
                    with np.errstate(all="ignore"):
                        log_col = np.log(col)
                        w_log = lambertw(log_col)
                        if np.all(np.abs(np.imag(w_log)) < 1e-9):
                            # exp(W(log(x)))
                            feat = np.exp(np.real(w_log))
                            if np.all(np.isfinite(feat)):
                                features.append(feat)
                                feature_names.append(f"exp(LambertW(log({name})))")
                                # print(f"DEBUG: Generated exp(LambertW(log({name})))", flush=True)
        except ImportError:
            pass
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

                if has_gauss:
                    features.append(col * exp_gauss)
                    feature_names.append(f"{name}*exp(-{name}^2)")

        # --- NEW: TRANSCENDENTAL INTERACTIONS (exp*exp, trig*exp, trig*trig) ---
        # Explicit interaction pass after all unary transcendentals are generated
        _add_transcendental_interactions(features, feature_names, n_vars, variable_names)

    # --- NEW: KNOWLEDGE EXPANSION (INVERSE TRIG, PIECEWISE, SPECIAL) ---
    if include_transcendentals:
        from scipy.special import erf
        from scipy.special import gamma

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

    # Defensive Check (The Wall Rule): Ensure atomic consistency
    if len(features) != len(feature_names):
        # Critical Logic Error - Crash immediately with clear info
        raise RuntimeError(
            f"Feature Gen Mismatch: {len(features)} features vs {len(feature_names)} names. This is a compiler bug."
        )

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


def _add_transcendental_features(col, name, features, feature_names, y_data):
    """Helper to add transcendental features for a single variable with robust NaN/Inf handling."""
    local_feats = []
    local_names = []

    # 1. Sine and Cosine (base frequency)
    local_feats.append(np.sin(col))
    local_names.append(f"sin({name})")
    local_feats.append(np.cos(col))  # ADDED: cos(x) base for sin(x+y) decomposition
    local_names.append(f"cos({name})")
    
    # Sinc
    with np.errstate(divide='ignore', invalid='ignore'):
        if not np.any(np.isclose(col, 0, atol=1e-9)):
            local_feats.append(np.sin(col)/col)
            local_names.append(f"sin({name})/{name}")
    
    # Frequency scan
    freq_candidates = {1.0, 2.0, np.pi}
    if y_data is not None:
        try:
             detected = detect_frequency(col, y_data)
             freq_candidates.update(detected)
        except Exception: pass
        
    for k in sorted(freq_candidates):
        if k == 1.0: continue
        k_val = k
        k_str = f"{k:.2g}"
        if abs(k - np.pi) < 1e-5: k_str = "pi"
        if abs(k - 2*np.pi) < 1e-5: k_str = "2*pi"
        
        local_feats.append(np.sin(k_val * col))
        local_names.append(f"sin({k_str}*{name})")
        local_feats.append(np.cos(k_val * col))
        local_names.append(f"cos({k_str}*{name})")

    # 2. Exponentials
    with np.errstate(over="ignore"):
         # exp(x)
         local_feats.append(np.exp(col))
         local_names.append(f"exp({name})")
         # exp(-x)
         local_feats.append(np.exp(-col))
         local_names.append(f"exp(-{name})")
         # Gaussian exp(-x^2)
         local_feats.append(np.exp(-(col**2)))
         local_names.append(f"exp(-{name}^2)")
         
         # Arrhenius exp(-A/x)
         if not np.any(np.isclose(col, 0, atol=1e-9)):
             inv = 1.0/col
             for A in [1, 10, 100]:
                 local_feats.append(np.exp(-A*inv))
                 local_names.append(f"exp(-{A}/{name})")

    # 3. Logarithms (positive only)
    if np.all(col > 1e-9):
         local_feats.append(np.log(col))
         local_names.append(f"log({name})")
         local_feats.append(np.log10(col))
         local_names.append(f"log10({name})")
         # x * log(x)
         local_feats.append(col * np.log(col))
         local_names.append(f"{name}*log({name})")
         # Log-Normal exp(-log(x)^2)
         log_x = np.log(col)
         local_feats.append(np.exp(-(log_x**2)))
         local_names.append(f"exp(-log({name})^2)")

    # 4. Hyperbolic
    with np.errstate(over="ignore"):
         local_feats.append(np.sinh(col))
         local_names.append(f"sinh({name})")
         local_feats.append(np.cosh(col))
         local_names.append(f"cosh({name})")
         local_feats.append(np.tanh(col))
         local_names.append(f"tanh({name})")
         
    # 5. Activation Functions
    with np.errstate(over="ignore"):
         # Sigmoid 1/(1+exp(-x))
         ex = np.exp(-col)
         local_feats.append(1.0/(1.0+ex))
         local_names.append(f"sigmoid({name})")
         # Softplus log(1+exp(x))
         local_feats.append(np.log1p(np.exp(col)))
         local_names.append(f"softplus({name})")

    # 6. Hybrid/Composite
    # x*sin(x)
    local_feats.append(col * np.sin(col))
    local_names.append(f"{name}*sin({name})")
    
    # Filter valid features
    for f, n in zip(local_feats, local_names):
         if np.all(np.isfinite(f)) and np.max(np.abs(f)) < 1e100:
              features.append(f)
              feature_names.append(n)


def _add_transcendental_interactions(
    features: list[Any],
    feature_names: list[str],
    n_vars: int,
    variable_names: list[str],
    max_interaction_count: int = 10000
) -> None:
    """Add pairwise interactions between transcendental features (e.g. exp(x)*exp(y)).

    This explicitly targets coupled physical laws like:
    - exp(x+y) = exp(x)*exp(y) (Thermal/Diffusion)
    - exp(x-y) = exp(x)*exp(-y)
    - sin(x)*exp(-y) (Damped Oscillation)
    - sin(x)*cos(y) (Wave Interference)

    Args:
        features: List of feature arrays (will be appended to)
        feature_names: List of feature names strings (will be appended to)
        n_vars: Number of base variables
        variable_names: Names of base variables
        max_interaction_count: Safety limit to prevent combinatorial explosion
    """
    import numpy as np

    # 1. Identify "Primary Transcendental" features already generated
    # We look for indices of features starting with "exp", "sin", "cos"
    # We DO NOT interact things that are already interactions (avoid exp(x)*exp(y)*sin(z))
    # We strictly look for "Unary" transcendentals of base variables.
    
    trans_indices = []
    trans_types = []  # "exp", "trig"

    for idx, name in enumerate(feature_names):
        # Check if it's a simple unary transcendental
        # Valid: "exp(x)", "sin(y)", "cos(x)", "exp(-y)"
        # Invalid: "x*exp(x)" (already mixed), "exp(x+y)" (if it existed)
        
        is_unary_trans = False
        t_type = None

        if name.startswith("exp(") and name.count("(") == 1:
            is_unary_trans = True
            t_type = "exp"
        elif (name.startswith("sin(") or name.startswith("cos(")) and name.count("(") == 1:
            is_unary_trans = True
            t_type = "trig"
        
        # Verify it only contains ONE variable
        if is_unary_trans:
            # Check variable containment
            # "exp(x)" -> contains 'x'
            # CRITICAL FIX: "exp" contains "x", so simple substring check fails for variable "x"
            # We must check INSIDE the parentheses.
            
            try:
                start_idx = name.find("(")
                end_idx = name.rfind(")")
                if start_idx != -1 and end_idx != -1:
                    inner_content = name[start_idx+1:end_idx]
                    
                    # Check variables in inner content
                    vars_in_name = 0
                    found_var = None
                    for v in variable_names:
                        if v in inner_content:
                             vars_in_name += 1
                             found_var = v
                    
                    # Special check: If multiple vars found, ensure they aren't substrings of each other?
                    # For now, strict check.
                    if vars_in_name == 1:
                        trans_indices.append(idx)
                        trans_types.append(t_type)
            except Exception:
                pass
    
    # 2. Generate Pairwise Interactions
    # We only interact features coming from DIFFERENT variables to avoid redundancy
    # (e.g. exp(x)*exp(x) = exp(2x) which is usually covered)
    # Actually, for sin(x)*cos(x) = 0.5*sin(2x), it might be useful, but let's prioritize inter-variable first.
    
    count = 0
    n_trans = len(trans_indices)
    
    # Quadratic loop over identified transcendentals
    for i in range(n_trans):
        idx_i = trans_indices[i]
        name_i = feature_names[idx_i]
        col_i = features[idx_i]
        type_i = trans_types[i]
        
        # Extract which variable is in name_i using ROBUST parsing
        var_i = None
        try:
            start_idx = name_i.find("(")
            end_idx = name_i.rfind(")")
            if start_idx != -1 and end_idx != -1:
                inner = name_i[start_idx+1:end_idx]
                for v in variable_names:
                    if v in inner:
                        var_i = v
                        break
        except Exception:
            pass
            
        # Fallback if parsing failed (shouldn't happen for these features)
        if var_i is None:
             for v in variable_names:
                if v in name_i:
                    var_i = v
                    break

        for j in range(i + 1, n_trans):
            idx_j = trans_indices[j]
            name_j = feature_names[idx_j]
            col_j = features[idx_j]
            type_j = trans_types[j]

            # Extract variable j using ROBUST parsing
            var_j = None
            try:
                start_idx = name_j.find("(")
                end_idx = name_j.rfind(")")
                if start_idx != -1 and end_idx != -1:
                    inner = name_j[start_idx+1:end_idx]
                    for v in variable_names:
                        if v in inner:
                            var_j = v
                            break
            except Exception:
                pass
                
            if var_j is None:
                 for v in variable_names:
                    if v in name_j:
                        var_j = v
                        break
            
            # Constraint: Must be different variables
            # We want exp(x)*exp(y), not exp(x)*sin(x) (which is locally handled in _add_transcendental_features)
            # or exp(x)*exp(-x) (=1)
            if var_i == var_j:
                continue
                
            # Constraint: Limit types of interactions
            # exp*exp (Thermal) -> High Priority
            # trig*exp (Damping) -> High Priority
            # trig*trig (Interference) -> High Priority
            # We allow all mixed types between diff variables.

            if count >= max_interaction_count:
                return

            new_name = f"{name_i}*{name_j}"
            
            # Compute product
            with np.errstate(over="ignore", invalid="ignore"):
                new_col = col_i * col_j
                
                # Validation (Constitution Rule 5)
                if np.all(np.isfinite(new_col)) and np.max(np.abs(new_col)) < 1e100:
                     # Check if it's not all zero
                    if not np.all(np.abs(new_col) < 1e-10):
                        features.append(new_col)
                        feature_names.append(new_name)
                        count += 1

