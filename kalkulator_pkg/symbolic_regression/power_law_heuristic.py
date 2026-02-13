"""Power Law Heuristic for Multivariate Functions.

Detects f(x,y) = x^g(y) patterns where g(y) is an unknown function.

Algorithm (Gemini's method):
1. Zero-Power Baseline: If y=0 gives output=1 for all x → pattern is x^g(y) where g(0)=0
2. Reference Point: If y=1 gives output=sqrt(x) → g(1)=0.5
3. Exponent Fitting: With 2+ reference points, fit g(y) from candidates
4. Verification: Check against full dataset

v1.0: Initial implementation.
"""

import numpy as np
from typing import Optional, Tuple


# Candidate exponent functions g(y)
EXPONENT_CANDIDATES = [
    ("sqrt(y)/2", lambda y: np.sqrt(np.maximum(y, 0)) / 2),
    ("sqrt(y)", lambda y: np.sqrt(np.maximum(y, 0))),
    ("y/2", lambda y: y / 2),
    ("y", lambda y: y),
    ("log(y+1)", lambda y: np.log(np.abs(y) + 1)),
    ("y**0.5", lambda y: np.power(np.maximum(y, 0), 0.5)),
]


def detect_power_law_structure(
    X: np.ndarray,
    y: np.ndarray,
    var_names: list[str],
    verbose: bool = False
) -> list[str]:
    """
    Detect f(x,y) = x^g(y) patterns.
    
    Args:
        X: Feature matrix (N x 2) - expects 2 variables
        y: Target values (N,)
        var_names: List of variable names ['x', 'y']
        verbose: Print debug information
    
    Returns:
        List of seed expressions if power-law detected, else []
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_vars = X.shape[1]
    if n_vars != 2:
        if verbose:
            print(f"[PowerLaw] Requires exactly 2 variables, got {n_vars}")
        return []
    
    if len(var_names) != 2:
        var_names = ['x', 'y']
    
    # Handle complex values
    if np.iscomplexobj(y):
        y = np.real(y)
    if np.iscomplexobj(X):
        X = np.real(X)
    
    # Try both orderings: x^g(y) and y^g(x)
    for base_idx, exp_idx in [(0, 1), (1, 0)]:
        base_var = var_names[base_idx]
        exp_var = var_names[exp_idx]
        
        result = _detect_power_law_single(
            X[:, base_idx], X[:, exp_idx], y,
            base_var, exp_var, verbose
        )
        
        if result:
            return result
    
    return []


def _detect_power_law_single(
    x_base: np.ndarray,
    x_exp: np.ndarray,
    y: np.ndarray,
    base_var: str,
    exp_var: str,
    verbose: bool
) -> list[str]:
    """
    Check if f(x,y) = base_var^g(exp_var).
    """
    n = len(y)
    
    # Step 1: Zero-Power Baseline Check
    # Find points where exp_var ≈ 0
    zero_mask = np.abs(x_exp) < 0.05
    n_zero = np.sum(zero_mask)
    
    if n_zero < 3:
        if verbose:
            print(f"[PowerLaw] Not enough zero-exponent points: {n_zero}")
        return []
    
    outputs_at_zero = y[zero_mask]
    
    # Check if all outputs ≈ 1 (x^0 = 1)
    if not np.allclose(np.real(outputs_at_zero), 1.0, rtol=0.05, atol=0.05):
        if verbose:
            print(f"[PowerLaw] Outputs at {exp_var}=0 are not ~1: mean={np.mean(outputs_at_zero):.4f}")
        return []
    
    if verbose:
        print(f"[PowerLaw] Zero-power check PASSED: {n_zero} points at {exp_var}≈0 give output≈1")
    
    # Step 2: Reference Point Extraction
    # Find g(y) values at reference points
    reference_points = {}
    
    for ref_val in [1, 4, 9, 16]:  # sqrt(1)=1, sqrt(4)=2, sqrt(9)=3, sqrt(16)=4
        mask = np.abs(x_exp - ref_val) < 0.1
        if np.sum(mask) < 3:
            continue
        
        x_vals = x_base[mask]
        y_vals = y[mask]
        
        # Filter positive values for log
        pos_mask = (x_vals > 0.1) & np.isfinite(y_vals) & (y_vals > 0)
        if np.sum(pos_mask) < 3:
            continue
        
        # Fit log(y) = k * log(x) using least squares
        # This means y = x^k, so k is the exponent
        log_x = np.log(x_vals[pos_mask])
        log_y = np.log(y_vals[pos_mask])
        
        try:
            # Weighted least squares
            coeffs = np.polyfit(log_x, log_y, 1)
            k = coeffs[0]  # slope = exponent
            
            # Calculate R² for this fit
            y_pred = coeffs[0] * log_x + coeffs[1]
            ss_res = np.sum((log_y - y_pred) ** 2)
            ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            
            if r2 > 0.95:  # Good fit
                reference_points[ref_val] = k
                if verbose:
                    print(f"[PowerLaw] At {exp_var}={ref_val}: g({exp_var})={k:.4f} (R²={r2:.4f})")
        except (ValueError, np.linalg.LinAlgError):
            continue
    
    if len(reference_points) < 2:
        if verbose:
            print(f"[PowerLaw] Not enough reference points: {len(reference_points)}")
        return []
    
    # Add g(0) = 0 as an implicit reference
    reference_points[0] = 0.0
    
    # Step 3: Exponent Function Fitting
    g_func_name, g_func, best_mse = _fit_exponent_function(reference_points, verbose)
    
    if g_func_name is None:
        if verbose:
            print("[PowerLaw] Could not fit exponent function")
        return []
    
    # Step 4: Verification
    # Test the candidate on the full dataset
    valid_mask = (x_base > 0.1) & np.isfinite(y)
    if np.sum(valid_mask) < 10:
        return []
    
    x_valid = x_base[valid_mask]
    exp_valid = x_exp[valid_mask]
    y_valid = y[valid_mask]
    
    try:
        exponents = g_func(exp_valid)
        y_pred = np.power(x_valid, exponents)
        
        # Handle complex results
        if np.iscomplexobj(y_pred):
            y_pred = np.real(y_pred)
        
        # Calculate R²
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        if verbose:
            print(f"[PowerLaw] Verification R² = {r2:.6f}")
        
        if r2 > 0.95:
            # Build expression: base_var^(g_func_name with exp_var)
            g_expr = g_func_name.replace('y', exp_var)
            expr = f"{base_var}**({g_expr})"
            
            if verbose:
                print(f"[PowerLaw] Detected: {expr}")
            
            return [expr]
    except Exception as e:
        if verbose:
            print(f"[PowerLaw] Verification failed: {e}")
    
    return []


def _fit_exponent_function(
    reference_points: dict,
    verbose: bool
) -> Tuple[Optional[str], Optional[callable], float]:
    """
    Given g(0)=0, g(1)=k1, g(4)=k2, etc., find g(y).
    
    Returns (name, function, mse) or (None, None, inf) if no fit.
    """
    y_refs = np.array(list(reference_points.keys()))
    g_vals = np.array(list(reference_points.values()))
    
    best_name = None
    best_func = None
    best_mse = float('inf')
    
    for name, func in EXPONENT_CANDIDATES:
        try:
            predicted = func(y_refs)
            mse = np.mean((g_vals - predicted) ** 2)
            
            if verbose:
                print(f"[PowerLaw] Testing g(y)={name}: MSE={mse:.6f}")
            
            if mse < best_mse:
                best_mse = mse
                best_name = name
                best_func = func
        except Exception:
            continue
    
    if best_mse < 0.01:  # Good fit threshold
        if verbose:
            print(f"[PowerLaw] Best fit: g(y)={best_name} (MSE={best_mse:.6f})")
        return best_name, best_func, best_mse
    
    return None, None, float('inf')
