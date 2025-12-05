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

import sympy as sp

try:
    import mpmath

    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

from .config import (
    ABSOLUTE_TOLERANCE,
    CONSTANT_DETECTION_TOLERANCE,
    LASSO_LAMBDA,
    OMP_MAX_ITERATIONS,
    RELATIVE_TOLERANCE,
    RESIDUAL_THRESHOLD,
    USE_AIC_BIC,
)

# Set high precision for Decimal
getcontext().prec = 50

# Library of known constants for detection
KNOWN_CONSTANTS = {
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "sqrt(2)": sp.sqrt(2),
    "sqrt(3)": sp.sqrt(3),
    "sqrt(5)": sp.sqrt(5),
    "log(2)": sp.log(2),
    "log(10)": sp.log(10),
    "ln(2)": sp.log(2),
    "ln(10)": sp.log(10),
    "gamma": sp.EulerGamma,
    "EulerGamma": sp.EulerGamma,
    "catalan": sp.Catalan,
    "Catalan": sp.Catalan,
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

    # First, try SymPy's nsimplify
    try:
        # Try to simplify to a known constant
        simplified = sp.nsimplify(
            float_val,
            tolerance=tolerance,
            full=True,
        )

        # Check if it matches any known constant
        for const_name, const_symbol in KNOWN_CONSTANTS.items():
            const_val = float(sp.N(const_symbol))
            if abs(float_val - const_val) / (abs(const_val) + 1e-10) < tolerance:
                return const_symbol

        # Check if simplified is close to the original
        if isinstance(simplified, (sp.Number, sp.Rational, sp.Integer)):
            simplified_val = float(sp.N(simplified))
            if (
                abs(float_val - simplified_val) / (abs(simplified_val) + 1e-10)
                < tolerance
            ):
                # Check if it's a known constant expression
                if abs(simplified_val - math.pi) < tolerance * abs(math.pi):
                    return sp.pi
                elif abs(simplified_val - math.e) < tolerance * abs(math.e):
                    return sp.E
                elif abs(simplified_val - math.sqrt(2)) < tolerance * abs(math.sqrt(2)):
                    return sp.sqrt(2)
                elif abs(simplified_val - math.sqrt(3)) < tolerance * abs(math.sqrt(3)):
                    return sp.sqrt(3)
    except (ValueError, TypeError, AttributeError):
        pass

    # Direct comparison with known constants
    for const_name, const_symbol in KNOWN_CONSTANTS.items():
        try:
            const_val = float(sp.N(const_symbol))
            if abs(float_val - const_val) / (abs(const_val) + 1e-10) < tolerance:
                return const_symbol
        except (ValueError, TypeError):
            continue

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
    selected = []
    coefficients = np.zeros(n_features)

    for iteration in range(min(max_nonzero, max_iterations)):
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
        from sklearn.linear_model import Lasso

        # Convert to numpy arrays
        import numpy as np

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

        for iteration in range(max_iterations):
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
        from sklearn.linear_model import lasso_path
        import numpy as np

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


def generate_candidate_features(
    X_data: Any, variable_names: list[str]
) -> tuple[Any, list[str]]:
    """Generates a dictionary of candidate functions (features) for symbolic regression.

    Args:
        X_data: numpy array of shape (n_samples, n_variables)
        variable_names: list of strings ['x', 'y', ...]

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

        # Power 3 (Optional, good for polynomials)
        features.append(col**3)
        feature_names.append(f"{name}^3")

    # 3. Interactions (x*y, x^2*y, x*y^2)
    if n_vars > 1:
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # x * y
                col = X_data[:, i] * X_data[:, j]
                name = f"{variable_names[i]}*{variable_names[j]}"
                features.append(col)
                feature_names.append(name)

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

    # --- NEW: TRANSCENDENTAL FUNCTIONS ---

    for i in range(n_vars):
        col = X_data[:, i]
        name = variable_names[i]

        # Sine (sin(x))
        features.append(np.sin(col))
        feature_names.append(f"sin({name})")

        # Argument Scaling: Frequency (sin(2x), sin(pi*x))
        features.append(np.sin(2 * col))
        feature_names.append(f"sin(2*{name})")
        features.append(np.sin(np.pi * col))
        feature_names.append(f"sin(pi*{name})")

        # Cosine (cos(x))
        features.append(np.cos(col))
        feature_names.append(f"cos({name})")

        # Argument Scaling: Frequency (cos(2x), cos(pi*x))
        features.append(np.cos(2 * col))
        feature_names.append(f"cos(2*{name})")
        features.append(np.cos(np.pi * col))
        feature_names.append(f"cos(pi*{name})")

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

        # Logarithm (log(x))
        # Only valid for positive inputs
        if np.all(col > 0):
            features.append(np.log(col))
            feature_names.append(f"log({name})")

            # x * log(x) - Entropy, information theory
            features.append(col * np.log(col))
            feature_names.append(f"{name}*log({name})")

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

    # --- NEW: RATIONAL FUNCTIONS (1/x) ---
    # This helps find physics laws like Inverse Square Law
    for i in range(n_vars):
        col = X_data[:, i]
        name = variable_names[i]

        # Avoid division by zero
        if not np.any(np.isclose(col, 0, atol=1e-10)):
            features.append(1 / col)
            feature_names.append(f"1/{name}")

            features.append(1 / (col**2))
            feature_names.append(f"1/{name}^2")

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
                A_str = f"{A:.4g}"
                B_str = f"{B:.4g}"

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
                A_str = f"{A:.4g}"
                B_str = f"{B:.4g}"

                if abs(A - 1.0) < 0.01:
                    A_str = ""
                else:
                    A_str = f"{A_str}*"

                return True, f"{A_str}{var_name}^{B_str}"
        except Exception:
            pass

    return False, None
