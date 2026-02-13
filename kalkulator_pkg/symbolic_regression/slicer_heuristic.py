"""Variable Separability Heuristic for Multivariate Functions.

Detects separable structure like f(x,y) = g(x) + h(y) by analyzing
zero-points and constant slices.

v1.0: Initial implementation with Tier 1 (Zero-Point) and Tier 2 (Constant-Slice).
"""

import numpy as np
from typing import Callable, Optional


def detect_separable_structure(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: list[str],
    find_func: Callable,
    verbose: bool = False
) -> list[str]:
    """
    Detect if f(x,y) = g(x) + h(y) using zero-point and slicing techniques.
    
    Args:
        X: Feature matrix (N x D) where D >= 2
        y: Target values (N,)
        variable_names: List of variable names ['x', 'y', ...]
        find_func: Callable to run 1D regression, signature:
                   find_func(X_1d, y, var_names) -> dict with 'expression', 'mse'
        verbose: Print debug information
    
    Returns:
        List of seed expressions if separable structure detected, else []
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_vars = X.shape[1]
    if n_vars < 2:
        return []  # Need at least 2 variables for separability
    
    if len(variable_names) != n_vars:
        if verbose:
            print(f"[Slicer] Variable name mismatch: {len(variable_names)} vs {n_vars} columns")
        return []
    
    all_seeds = []

    # Tier 1: Zero-Point Detection (fast path)
    zero_seeds = _detect_via_zero_points(X, y, variable_names, find_func, verbose)
    if zero_seeds:
        all_seeds.extend(zero_seeds)
    
    # Tier 2: Constant-Slice Detection (fallback)
    # Only run if Tier 1 failed? Or always?
    # Original logic was "fallback". Let's keep it fallback-ish but allow Tier 3.
    if not zero_seeds:
        slice_seeds = _detect_via_slicing(X, y, variable_names, find_func, verbose)
        all_seeds.extend(slice_seeds)
    
    # Tier 3: Log-Space Slicer (Detects Multiplicative/Power Laws)
    # f(x,y) = g(x)*h(y)  => log(f) = log(g) + log(h)
    # f(x,y) = x^g(y)     => log(f) = g(y)*log(x) (Multiplicative in log-space? No, separable in log-space)
    # Condition: All y must be positive
    if np.all(y > 1e-9):
        try:
            log_y = np.log(y)
            # Recursively call detection on log data? 
            # Or just call zero-point detection? Slicing works too.
            # Let's call zero-point for speed.
            log_seeds = _detect_via_zero_points(X, log_y, variable_names, find_func, verbose=False)
            
            for seed in log_seeds:
                # Invert transform: seed -> exp(seed)
                # But Slicer returns full expressions like "sin(x) + y^2"
                # So we wrap in exp(...)
                # Check complexity?
                wrapped = f"exp({seed})"
                all_seeds.append(wrapped)
                if verbose:
                    print(f"[Slicer] Found Log-Space seed: {wrapped}")
        except Exception as e:
            if verbose:
                print(f"[Slicer] Log-Space check failed: {e}")
                
    return all_seeds


def _detect_via_zero_points(
    X: np.ndarray,
    y: np.ndarray,
    var_names: list[str],
    find_func: Callable,
    verbose: bool
) -> list[str]:
    """
    Find points where x≈0 or y≈0 and run 1D regression.
    
    When x=0: f(0, y) = g(0) + h(y) ≈ h(y) + constant
    This reveals h(y) (possibly with an offset).
    """
    n_vars = len(var_names)
    discovered = {}  # var_name -> expression string
    
    for i in range(n_vars):
        # Use absolute tolerance for exact zero detection
        # Cross-hair sampling generates exact 0s, so use tight absolute tolerance
        # Also check relative tolerance for scaled data
        col_std = np.std(np.real(X[:, i]))  # Use real part for complex data
        rel_tolerance = 0.01 * col_std if col_std > 1e-10 else 1e-5
        abs_tolerance = 0.01  # For exact zeros from cross-hair sampling
        tolerance = max(abs_tolerance, rel_tolerance)
        
        # Find points where this variable is ~0
        X_col_real = np.real(X[:, i])  # Handle complex data
        
        # Priority: Check for EXACT zeros (from CrossHair sampling)
        exact_mask = np.abs(X_col_real) < 1e-9
        n_exact = np.sum(exact_mask)
        
        if n_exact >= 5:
            # We have good cross-hair data! Use ONLY these points.
            if verbose:
                print(f"[ZeroPoint] Using {n_exact} exact zeros for {var_names[i]}")
            mask = exact_mask
            n_points = n_exact
        else:
            # Fallback to approximate zeros
            mask = np.abs(X_col_real) < tolerance
            n_points = np.sum(mask)
        
        if n_points >= 5:  # Need at least 5 points for regression
            # The "other" variables become the 1D input
            other_indices = [j for j in range(n_vars) if j != i]
            
            if len(other_indices) == 1:
                # Simple 2D case: f(x, y) with one variable at zero
                other_idx = other_indices[0]
                other_var = var_names[other_idx]
                
                X_slice = X[mask, other_idx]
                y_slice = y[mask]
                
                # Filter real-only points (exclude complex outputs)
                real_mask = ~np.iscomplex(y_slice) | (np.abs(np.imag(y_slice)) < 1e-9)
                X_slice = np.real(X_slice[real_mask]).reshape(-1, 1)
                y_slice = np.real(y_slice[real_mask])
                
                if len(y_slice) < 5:
                    continue  # Not enough real points
                
                if verbose:
                    print(f"[ZeroPoint] Found {len(y_slice)} real points where {var_names[i]} ≈ 0")
                
                expr = None
                mse = float('inf')
                
                # First, try the user-provided find_func
                try:
                    result = find_func(X_slice, y_slice, [other_var])
                    
                    if result and isinstance(result, dict):
                        expr = result.get('expression', '')
                        mse = result.get('mse', float('inf'))
                        if verbose:
                            print(f"[ZeroPoint] find_func returned: {expr} (MSE={mse:.4g})")
                except Exception as e:
                    if verbose:
                        print(f"[ZeroPoint] find_func failed: {e}")
                
                # FALLBACK: Try direct sqrt pattern detection
                # If find_func failed or returned a poor fit, check if y = sqrt(x)
                if mse > 0.01:
                    x_vals = X_slice.flatten()
                    # Only check positive x values for sqrt
                    pos_mask = x_vals > 0.01
                    if np.sum(pos_mask) >= 3:
                        x_pos = x_vals[pos_mask]
                        y_pos = y_slice[pos_mask]
                        
                        # Test: y = sqrt(x)
                        y_pred_sqrt = np.sqrt(x_pos)
                        mse_sqrt = np.mean((y_pos - y_pred_sqrt) ** 2)
                        
                        if verbose:
                            print(f"[ZeroPoint] Testing sqrt({other_var}): MSE={mse_sqrt:.4g}")
                        
                        if mse_sqrt < 0.01:
                            expr = f"sqrt({other_var})"
                            mse = mse_sqrt
                            if verbose:
                                print(f"[ZeroPoint] Direct sqrt match! sqrt({other_var}) MSE={mse:.4g}")
                
                if expr and mse < 0.1:  # Reasonable fit
                    discovered[other_var] = expr
                    if verbose:
                        print(f"[ZeroPoint] At {var_names[i]}=0: f({other_var}) = {expr} (MSE={mse:.4g})")
    
    # Combine discovered terms with bias correction
    if len(discovered) >= 2:
        return _combine_with_bias_correction(discovered, X, y, var_names, verbose)
    
    return []


def _detect_via_slicing(
    X: np.ndarray,
    y: np.ndarray,
    var_names: list[str],
    find_func: Callable,
    verbose: bool
) -> list[str]:
    """
    Fallback: Find constant slices where one variable doesn't vary much.
    
    Strategy:
    1. For each variable, find clusters of similar values
    2. Use largest cluster as a "constant slice"
    3. Run 1D regression on remaining variables
    """
    n_vars = len(var_names)
    discovered = {}
    
    for i in range(n_vars):
        col = X[:, i]
        
        # Simple binning: divide into 10 bins, find densest bin
        n_bins = 10
        hist, bin_edges = np.histogram(col, bins=n_bins)
        densest_bin = np.argmax(hist)
        
        # Points in densest bin
        bin_low = bin_edges[densest_bin]
        bin_high = bin_edges[densest_bin + 1]
        mask = (col >= bin_low) & (col <= bin_high)
        n_points = np.sum(mask)
        
        if n_points >= 10:  # Need more points for slice regression
            other_indices = [j for j in range(n_vars) if j != i]
            
            if len(other_indices) == 1:
                other_idx = other_indices[0]
                other_var = var_names[other_idx]
                
                X_slice = X[mask, other_idx].reshape(-1, 1)
                y_slice = y[mask]
                
                slice_center = (bin_low + bin_high) / 2
                
                if verbose:
                    print(f"[Slicer] Found {n_points} points where {var_names[i]} ≈ {slice_center:.3f}")
                
                try:
                    result = find_func(X_slice, y_slice, [other_var])
                    
                    if result and isinstance(result, dict):
                        expr = result.get('expression', '')
                        mse = result.get('mse', float('inf'))
                        
                        if expr and mse < 0.5:  # More lenient for slices
                            discovered[other_var] = expr
                            if verbose:
                                print(f"[Slicer] At {var_names[i]}≈{slice_center:.3f}: f({other_var}) = {expr}")
                except Exception:
                    pass
    
    if len(discovered) >= 2:
        return _combine_with_bias_correction(discovered, X, y, var_names, verbose)
    
    return []


def _combine_with_bias_correction(
    discovered: dict[str, str],
    X: np.ndarray,
    y: np.ndarray,
    var_names: list[str],
    verbose: bool
) -> list[str]:
    """
    Combine discovered partial expressions and correct for constant offset.
    
    Fix for Gemini's "Constant Offset Double-Count" problem:
    If sin(x) + 1 discovered for x-slice and cos(y) for y-slice,
    naive combination gives sin(x) + 1 + cos(y) = sin(x) + cos(y) + 1 (wrong!).
    
    Solution: Evaluate combined expression, compute mean residual as bias.
    """
    # Build base expression
    base_terms = list(discovered.values())
    base_expr_str = " + ".join(base_terms)
    
    if verbose:
        print(f"[Slicer] Combining terms: {base_expr_str}")
    
    # Try to evaluate and compute bias
    try:
        # Build evaluation context
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr
        
        # Create symbols
        symbols = {name: sp.Symbol(name) for name in var_names}
        
        # Parse the combined expression
        expr = parse_expr(base_expr_str, local_dict=symbols)
        
        # Evaluate on full dataset
        y_pred = np.zeros(len(y))
        for idx in range(len(y)):
            point = {var_names[j]: float(X[idx, j]) for j in range(len(var_names))}
            try:
                val = float(expr.evalf(subs=point))
                y_pred[idx] = val if np.isfinite(val) else np.nan
            except:
                y_pred[idx] = np.nan
        
        # Filter valid predictions
        valid_mask = np.isfinite(y_pred) & np.isfinite(y)
        if np.sum(valid_mask) < 10:
            if verbose:
                print("[Slicer] Not enough valid predictions for bias correction")
            return [base_expr_str]
        
        # Compute bias (mean residual)
        # Fix: Only use real part for complex numbers
        residuals = y[valid_mask] - y_pred[valid_mask]
        if np.iscomplexobj(residuals):
            residuals = np.real(residuals)
        
        bias = np.mean(residuals)
        
        # Construct final expression
        if abs(bias) > 0.01:
            # Round bias to reasonable precision
            bias_rounded = round(bias, 4)
            final_expr = f"{base_expr_str} + {bias_rounded}"
            if verbose:
                print(f"[Slicer] Bias correction: {bias_rounded}")
        else:
            final_expr = base_expr_str
        
        # Verify final MSE
        # (not strictly necessary but good for confidence)
        
        if verbose:
            print(f"[Slicer] Final seed: {final_expr}")
        
        return [final_expr]
        
    except Exception as e:
        if verbose:
            print(f"[Slicer] Bias correction failed: {e}")
        # Return raw combination as fallback
        return [base_expr_str]


def _simple_find_wrapper(X_1d, y, var_names):
    """
    Simple wrapper for testing - fits basic polynomials.
    In production, use the real find_function_from_data.
    """
    from numpy.polynomial import polynomial as P
    
    x = X_1d.flatten()
    
    # Try polynomial fits of increasing degree
    best_expr = None
    best_mse = float('inf')
    
    for degree in [1, 2, 3]:
        try:
            coeffs = np.polyfit(x, y, degree)
            y_pred = np.polyval(coeffs, x)
            mse = np.mean((y - y_pred) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                # Build expression string
                terms = []
                var = var_names[0]
                for i, c in enumerate(coeffs):
                    power = degree - i
                    if abs(c) > 1e-6:
                        if power == 0:
                            terms.append(f"{c:.4g}")
                        elif power == 1:
                            terms.append(f"{c:.4g}*{var}")
                        else:
                            terms.append(f"{c:.4g}*{var}**{power}")
                best_expr = " + ".join(terms)
        except:
            continue
    
    return {'expression': best_expr, 'mse': best_mse}
