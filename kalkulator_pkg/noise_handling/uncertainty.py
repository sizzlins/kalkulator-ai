"""Uncertainty quantification for regression models.

Provides:
- Bootstrap confidence intervals for coefficients
- Prediction intervals for new data
- Statistical significance testing
- Error propagation
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import stats


def bootstrap_confidence_interval(
    X: np.ndarray,
    y: np.ndarray,
    fit_function: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, float]],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> dict:
    """Bootstrap confidence intervals for regression coefficients.

    Uses resampling with replacement to estimate the distribution of
    coefficients and construct confidence intervals.

    Args:
        X: Input features of shape (n_samples, n_features)
        y: Target values
        fit_function: Function(X, y) -> (coefficients, intercept)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Dict with:
        - 'coef_mean': Mean of coefficient estimates
        - 'coef_std': Standard deviation of estimates
        - 'coef_ci_lower': Lower confidence bounds
        - 'coef_ci_upper': Upper confidence bounds
        - 'intercept_mean', 'intercept_std', 'intercept_ci': Same for intercept
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples = len(y)

    if random_state is not None:
        np.random.seed(random_state)

    # Collect bootstrap estimates
    coef_samples = []
    intercept_samples = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        try:
            coef, intercept = fit_function(X_boot, y_boot)
            coef_samples.append(coef)
            intercept_samples.append(intercept)
        except Exception:
            continue

    if not coef_samples:
        raise ValueError("All bootstrap samples failed")

    coef_samples = np.array(coef_samples)
    intercept_samples = np.array(intercept_samples)

    # Calculate statistics
    alpha = 1 - confidence

    result = {
        "coef_mean": np.mean(coef_samples, axis=0),
        "coef_std": np.std(coef_samples, axis=0),
        "coef_ci_lower": np.percentile(coef_samples, 100 * alpha / 2, axis=0),
        "coef_ci_upper": np.percentile(coef_samples, 100 * (1 - alpha / 2), axis=0),
        "intercept_mean": np.mean(intercept_samples),
        "intercept_std": np.std(intercept_samples),
        "intercept_ci_lower": np.percentile(intercept_samples, 100 * alpha / 2),
        "intercept_ci_upper": np.percentile(intercept_samples, 100 * (1 - alpha / 2)),
        "n_successful": len(coef_samples),
    }

    return result


def prediction_interval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_new: np.ndarray,
    model: Any,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate prediction intervals for new data points.

    Uses the t-distribution to account for uncertainty in both the
    model parameters and the residual variance.

    Args:
        X_train: Training features
        y_train: Training targets
        X_new: New points to predict
        model: Fitted model with predict() method
        confidence: Confidence level

    Returns:
        Tuple of (predictions, lower_bounds, upper_bounds)
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    X_new = np.asarray(X_new)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_new.ndim == 1:
        X_new = X_new.reshape(-1, 1)

    n_samples, n_features = X_train.shape

    # Get predictions
    y_pred_train = model.predict(X_train)
    y_pred_new = model.predict(X_new)

    # Calculate residual standard error
    residuals = y_train - y_pred_train
    dof = n_samples - n_features - 1  # Degrees of freedom
    if dof <= 0:
        dof = 1

    mse = np.sum(residuals**2) / dof
    se = np.sqrt(mse)

    # For prediction intervals, we need leverage values
    # Using a simplified approach: constant variance
    try:
        # Add intercept column
        X_aug = np.column_stack([np.ones(n_samples), X_train])
        X_new_aug = np.column_stack([np.ones(len(X_new)), X_new])

        # Hat matrix diagonal (leverage)
        XtX_inv = np.linalg.inv(X_aug.T @ X_aug + 1e-6 * np.eye(X_aug.shape[1]))
        leverage = np.sum((X_new_aug @ XtX_inv) * X_new_aug, axis=1)

        # Prediction standard error
        pred_se = se * np.sqrt(1 + leverage)

    except Exception:
        # Fallback to simple approach
        pred_se = se * np.ones(len(X_new))

    # t-value for confidence level
    t_value = stats.t.ppf((1 + confidence) / 2, dof)

    lower = y_pred_new - t_value * pred_se
    upper = y_pred_new + t_value * pred_se

    return y_pred_new, lower, upper


def coefficient_significance(
    X: np.ndarray, y: np.ndarray, coefficients: np.ndarray, intercept: float
) -> dict:
    """Test statistical significance of regression coefficients.

    Performs t-tests to determine if each coefficient is significantly
    different from zero.

    Args:
        X: Input features
        y: Target values
        coefficients: Estimated coefficients
        intercept: Estimated intercept

    Returns:
        Dict with t-statistics, p-values, and significance flags
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    coefficients = np.asarray(coefficients).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    # Calculate predictions and residuals
    y_pred = X @ coefficients + intercept
    residuals = y - y_pred

    # Residual standard error
    dof = n_samples - n_features - 1
    if dof <= 0:
        dof = 1
    mse = np.sum(residuals**2) / dof

    # Standard errors of coefficients
    try:
        X_aug = np.column_stack([np.ones(n_samples), X])
        XtX_inv = np.linalg.inv(X_aug.T @ X_aug + 1e-6 * np.eye(X_aug.shape[1]))
        var_coef = mse * np.diag(XtX_inv)
        se_intercept = np.sqrt(var_coef[0])
        se_coef = np.sqrt(var_coef[1:])
    except Exception:
        se_coef = np.ones(n_features)
        se_intercept = 1.0

    # t-statistics
    t_coef = coefficients / (se_coef + 1e-10)
    t_intercept = intercept / (se_intercept + 1e-10)

    # p-values (two-tailed)
    p_coef = 2 * (1 - stats.t.cdf(np.abs(t_coef), dof))
    p_intercept = 2 * (1 - stats.t.cdf(np.abs(t_intercept), dof))

    return {
        "coef_t_stats": t_coef,
        "coef_p_values": p_coef,
        "coef_significant_05": p_coef < 0.05,
        "coef_significant_01": p_coef < 0.01,
        "coef_se": se_coef,
        "intercept_t_stat": t_intercept,
        "intercept_p_value": p_intercept,
        "intercept_significant": p_intercept < 0.05,
        "intercept_se": se_intercept,
        "r_squared": 1 - np.sum(residuals**2) / np.sum((y - np.mean(y)) ** 2),
        "adj_r_squared": 1
        - (1 - (1 - np.sum(residuals**2) / np.sum((y - np.mean(y)) ** 2)))
        * (n_samples - 1)
        / dof,
    }


def monte_carlo_uncertainty(
    X: np.ndarray,
    y: np.ndarray,
    y_errors: np.ndarray | float,
    fit_function: Callable,
    n_simulations: int = 1000,
    random_state: int | None = None,
) -> dict:
    """Monte Carlo propagation of measurement uncertainties.

    Simulates measurement errors to propagate uncertainty to the
    fitted coefficients.

    Args:
        X: Input features
        y: Target values (central values)
        y_errors: Standard deviations of y measurements (scalar or array)
        fit_function: Function(X, y) -> (coefficients, intercept)
        n_simulations: Number of Monte Carlo samples
        random_state: Random seed

    Returns:
        Dict with coefficient distributions and uncertainties
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if isinstance(y_errors, (int, float)):
        y_errors = np.full_like(y, y_errors)
    else:
        y_errors = np.asarray(y_errors).ravel()

    if random_state is not None:
        np.random.seed(random_state)

    coef_samples = []
    intercept_samples = []

    for _ in range(n_simulations):
        # Add noise according to measurement errors
        y_noisy = y + np.random.normal(0, y_errors)

        try:
            coef, intercept = fit_function(X, y_noisy)
            coef_samples.append(coef)
            intercept_samples.append(intercept)
        except Exception:
            continue

    if not coef_samples:
        raise ValueError("All Monte Carlo samples failed")

    coef_samples = np.array(coef_samples)
    intercept_samples = np.array(intercept_samples)

    return {
        "coef_mean": np.mean(coef_samples, axis=0),
        "coef_std": np.std(coef_samples, axis=0),
        "coef_samples": coef_samples,
        "intercept_mean": np.mean(intercept_samples),
        "intercept_std": np.std(intercept_samples),
        "intercept_samples": intercept_samples,
    }


def format_with_uncertainty(value: float, uncertainty: float, sig_figs: int = 2) -> str:
    """Format a value with its uncertainty.

    Args:
        value: Central value
        uncertainty: Standard deviation or error
        sig_figs: Significant figures in uncertainty

    Returns:
        Formatted string like "3.14 ± 0.02"
    """
    if uncertainty <= 0:
        return f"{value:.6g}"

    # Determine precision from uncertainty
    import math

    exp = math.floor(math.log10(abs(uncertainty))) - (sig_figs - 1)

    # Round both to this precision
    factor = 10 ** (-exp)
    val_rounded = round(value * factor) / factor
    unc_rounded = round(uncertainty * factor) / factor

    # Format with appropriate decimal places
    if exp >= 0:
        return f"{int(val_rounded)} ± {int(unc_rounded)}"
    else:
        decimals = -exp
        return f"{val_rounded:.{decimals}f} ± {unc_rounded:.{decimals}f}"
