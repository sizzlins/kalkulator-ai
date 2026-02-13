"""Monomial Heuristic for Multivariate Functions.

Detects f(x,y) = C * x^a * y^b using log-linear regression.

v1.0: Initial implementation with sign restoration (Gemini's refinement).
"""

import numpy as np
from typing import Optional


def detect_monomial_structure(
    X: np.ndarray,
    y: np.ndarray,
    var_names: list[str],
    verbose: bool = False
) -> list[str]:
    """
    Detect f(x,y) = C * x^a * y^b using log-linear regression.
    
    Algorithm:
    1. Transform: ln|z| = ln|C| + a*ln|x| + b*ln|y|
    2. Solve for exponents using least squares
    3. Round to integers if close
    4. Restore sign of C by testing both ±C
    5. Verify with R² > 0.99
    
    Args:
        X: Feature matrix (N x D)
        y: Target values (N,)
        var_names: List of variable names ['x', 'y', ...]
        verbose: Print debug information
    
    Returns:
        List of seed expressions if monomial detected, else []
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_vars = X.shape[1]
    if len(var_names) != n_vars:
        if verbose:
            print(f"[Monomial] Variable name mismatch: {len(var_names)} vs {n_vars}")
        return []
    
    # 1. Filter zeros and very small numbers to avoid log errors
    valid_mask = np.abs(y) > 1e-6
    for i in range(n_vars):
        valid_mask &= np.abs(X[:, i]) > 1e-6
    
    n_valid = np.sum(valid_mask)
    if n_valid < max(5, n_vars + 2):  # Need enough points for regression
        if verbose:
            print(f"[Monomial] Not enough valid points: {n_valid}")
        return []
    
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    # Handle complex values - use real part only
    if np.iscomplexobj(y_clean):
        y_clean = np.real(y_clean)
    if np.iscomplexobj(X_clean):
        X_clean = np.real(X_clean)
        
    # Re-filter zeros after real cast (e.g. real(3j) -> 0.0)
    # Use Safe Log Pattern equivalent by filtering rows
    safe_mask = np.abs(y_clean) > 1e-9
    for i in range(n_vars):
        safe_mask &= np.abs(X_clean[:, i]) > 1e-9
        
    if np.sum(safe_mask) < max(5, n_vars + 2):
        if verbose:
            print("[Monomial] Not enough points after safe-log filtering")
        return []

    X_clean = X_clean[safe_mask]
    y_clean = y_clean[safe_mask]
    
    # 2. Log-Linear Transform: ln|y| = ln|C| + a*ln|x0| + b*ln|x1|...
    try:
        log_y = np.log(np.abs(y_clean))
        log_X = np.log(np.abs(X_clean))
    except (ValueError, RuntimeWarning):
        if verbose:
            print("[Monomial] Log transform failed")
        return []
    
    # Prepend column of ones for the intercept (ln|C|)
    A = np.column_stack([np.ones(len(log_y)), log_X])
    
    # 3. Solve Least Squares
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, log_y, rcond=None)
    except np.linalg.LinAlgError:
        if verbose:
            print("[Monomial] Least squares failed")
        return []
    
    ln_C_mag = coeffs[0]
    exponents = coeffs[1:]
    
    if verbose:
        print(f"[Monomial] Raw exponents: {exponents}")
        print(f"[Monomial] Raw ln|C|: {ln_C_mag}")
    
    # 4. Rounding & Integer Check
    C_mag = np.exp(ln_C_mag)
    exponents_rounded = np.round(exponents).astype(int)
    
    # Check if exponents are close to integers (tolerance 0.15)
    if not np.allclose(exponents, exponents_rounded, atol=0.15):
        if verbose:
            print(f"[Monomial] Exponents not close to integers: {exponents}")
        return []
    
    # 5. Sign Restoration (Gemini's fix)
    # Reconstruct with positive C
    y_pred_pos = C_mag * np.prod(
        np.sign(X_clean) ** exponents_rounded * np.abs(X_clean) ** exponents_rounded,
        axis=1
    )
    
    # Calculate error for C positive vs negative
    error_pos = np.median(np.abs(y_clean - y_pred_pos))
    error_neg = np.median(np.abs(y_clean - (-y_pred_pos)))
    
    # Round C to integer if close
    C_rounded = round(C_mag)
    if abs(C_mag - C_rounded) < 0.5:
        final_C = C_rounded
    else:
        final_C = round(C_mag, 4)  # Keep 4 decimal places
    
    if error_neg < error_pos:
        final_C = -final_C
        if verbose:
            print(f"[Monomial] Detected negative coefficient")
    
    if verbose:
        print(f"[Monomial] Final C: {final_C}, exponents: {exponents_rounded}")
    
    # 6. Final Verification (R² check)
    y_final_pred = final_C * np.prod(
        np.sign(X_clean) ** exponents_rounded * np.abs(X_clean) ** exponents_rounded,
        axis=1
    )
    
    ss_res = np.sum((y_clean - y_final_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    if verbose:
        print(f"[Monomial] R² = {r2:.6f}")
    
    if r2 < 0.99:
        if verbose:
            print(f"[Monomial] R² too low ({r2:.4f}), rejecting")
        return []
    
    # 7. Construct the expression string
    terms = []
    
    # Add coefficient if not 1 or -1
    if final_C == -1:
        terms.append("-")
    elif final_C != 1:
        terms.append(str(final_C))
    
    for i, exp in enumerate(exponents_rounded):
        if exp == 0:
            continue
        var = var_names[i]
        if exp == 1:
            terms.append(var)
        else:
            terms.append(f"{var}**{exp}")
    
    if not terms:
        return []  # All exponents were 0
    
    # Join with multiplication
    expr = "*".join(terms)
    if expr.startswith("-*"):
        expr = "-" + expr[2:]  # Fix "-*x" to "-x"
    
    if verbose:
        print(f"[Monomial] Detected: {expr}")
    
    return [expr]


def check_dynamic_range(y: np.ndarray, verbose: bool = False) -> bool:
    """
    Check if data has high dynamic range suggesting polynomial growth.
    
    Returns True if IQR filtering should be skipped.
    """
    y_abs = np.abs(y[np.isfinite(y)])
    if len(y_abs) < 5:
        return False
    
    # Use percentiles to avoid single outliers dominating
    y_99 = np.percentile(y_abs, 99)
    y_1 = np.percentile(y_abs, 1) + 1e-9  # Avoid division by zero
    
    dynamic_range = y_99 / y_1
    
    if dynamic_range > 1e5:  # 100,000x difference
        if verbose:
            print(f"[Safety] High Dynamic Range ({dynamic_range:.1e}) detected.")
            print("[Safety] Skipping IQR outlier filtering to preserve polynomial tail.")
        return True
    
    return False
