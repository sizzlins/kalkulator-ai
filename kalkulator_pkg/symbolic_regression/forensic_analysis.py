"""Forensic Analysis Module for Symbolic Regression.

Extracts deep patterns from data using heuristics, singularity analysis,
and integer sequence detection.
"""
import numpy as np
import time
import fractions
import warnings
from ..heuristics import check_power_peeling

def _detect_outer_functions(y):
    """Phase 2: Range Analysis - suggest outer functions like sin, cos."""
    y_finite = y[np.isfinite(y)]
    if len(y_finite) == 0:
        return []
    
    y_min, y_max = np.min(y_finite), np.max(y_finite)
    y_range = y_max - y_min
    
    suggestions = []
    
    # Trig-bounded: approximately [-1, 1]
    if -1.2 < y_min < -0.8 and 0.8 < y_max < 1.2:
        suggestions.extend(['sin', 'cos', 'tanh'])
    
    # Always-positive with max ~1 (could be abs of trig)
    elif y_min > -0.1 and 0.8 < y_max < 1.2:
        suggestions.append('abs')
    
    return suggestions

def _compose_seeds(pole_seeds, outer_functions):
    """Phase 3: Generate composed seeds like sin(1/(x-3))."""
    composed = []
    # Only compose with basic pole seeds (not squared or multiplied)
    basic_poles = [s for s in pole_seeds if '**' not in s and ' * ' not in s]
    
    for pole in basic_poles:
        for func in outer_functions:
            composed.append(f'{func}({pole})')
            # Also try inverted pole
            if '1/(' in pole and '-' in pole:
                inverted = pole.replace('-(', '+(').replace('-', '+', 1).replace('+(', '-(', 1)
                if inverted != pole:
                    composed.append(f'{func}({inverted})')
            
            # 2025-01-19 Fix: Support sin(x/(x-p)) by adding x * pole
            # This covers x/(x-p) which is 1 + p/(x-p)
            if '1/(' in pole and 'x' not in pole[:pole.find('1/(')]: # Simple 1/(x-p)
                composed.append(f'{func}(x * {pole})')
    return composed

def _detect_integer_patterns(X, y, variable_names=None):
    """Phase 3 - Integer Pattern Recognition (Robust LLL)."""
    # Allow (N,1) shaped arrays
    if X.ndim > 1 and X.shape[1] > 1: return []
    try: x_flat = X.flatten()
    except: return []

    seeds = []
    seeds = []
    var_name = variable_names[0] if variable_names else "x"
    
    
    # Vectorized Candidate Search
    # 1. Filter Non-Complex, Finite
    try:
        # Handle complex arrays - only consider points with Im(x) ~ 0
        if np.iscomplexobj(x_flat):
            real_mask = np.abs(np.imag(x_flat)) < 1e-9
            # Keep indices aligned
            candidate_indices = np.where(real_mask)[0]
            x_real = np.real(x_flat[real_mask])
        else:
            x_real = x_flat
            candidate_indices = np.arange(len(x_flat))
            
        if len(x_real) == 0: return []

        # 2. Check for Integers
        x_rounded = np.round(x_real)
        is_int = np.abs(x_real - x_rounded) < 1e-9
        
        # 3. Check Range (|x| > 1 and |x| < 10)
        abs_x = np.abs(x_real)
        in_range = (abs_x > 1) & (abs_x < 10)
        
        # Combine masks
        candidates = is_int & in_range
        indices = candidate_indices[candidates]
        
    except Exception:
        indices = []
    
    for i in indices[:5]:
        try:
            x_val = int(round(float(x_flat[i].real if hasattr(x_flat[i], 'real') else x_flat[i])))
            y_val = y[i]
            
            if np.iscomplex(y_val) or not np.isfinite(y_val): continue
            if abs(y_val) < 1e-6: continue
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                frac = fractions.Fraction(float(y_val)).limit_denominator(1000)
                if abs(float(frac) - float(y_val)) > 1e-6: continue
                
            num, den = frac.numerator, frac.denominator
            
            
            for n in range(1, 13):
                x_pow = x_val ** n
                num_rel = None
                if num == x_pow: num_rel = f"{var_name}^{n}"
                elif num == x_pow + 1: num_rel = f"({var_name}^{n} + 1)"
                elif num == x_pow - 1: num_rel = f"({var_name}^{n} - 1)"
                elif num == 1 - x_pow: num_rel = f"(1 - {var_name}^{n})"
                elif num == x_pow + x_val: num_rel = f"({var_name}^{n} + {var_name})"
                
                den_rel = None
                if den == x_pow: den_rel = f"{var_name}^{n}"
                elif den == x_pow + 1: den_rel = f"({var_name}^{n} + 1)"
                elif den == x_pow - 1: den_rel = f"({var_name}^{n} - 1)"
                elif den == 1 - x_pow: den_rel = f"(1 - {var_name}^{n})"
                elif den == x_pow - x_val: den_rel = f"({var_name}^{n} - {var_name})"
                
                if num_rel and den_rel: seeds.append(f"{num_rel} / {den_rel}")
                elif num_rel and den == 1: seeds.append(num_rel)
        except: continue
            
    return sorted(list(set(seeds)))


def _detect_scalloped_staircase(X, y, variable_names=None, verbose=False):
    """Detect generalized scalloped patterns: f(x) = floor(x)^a + frac(x)^b.
    
    Algorithm (generalized from Gemini's approach):
    1. Integer Anchors: Detect power relationship f(n) = n^a using log-log regression
    2. Residual Analysis: Subtract floor(x)^a and analyze residual for frac(x)^b
    3. Verify: The full formula floor(x)^a + frac(x)^b matches all data points
    
    Args:
        X: Input data array
        y: Output data array
        variable_names: List of variable names
        verbose: Print debug info
        
    Returns:
        Tuple (seeds_list, exact_match) if found, else empty list
    """
    # [User Fix] Check for 1D data ONLY to avoid broadcasting errors
    X_arr = np.array(X)
    if X_arr.ndim > 1 and X_arr.shape[1] > 1:
        return []

    # [User Fix 2] Scalloped Staircase doesn't exist in complex plane
    if np.iscomplexobj(y):
        return []

    try:
        X_flat = X_arr.flatten()
        y_flat = np.array(y).flatten()
    except:
        return []
        
    # Filter for real, finite values (handle complex with tolerance)
    if np.iscomplexobj(X_flat) or np.iscomplexobj(y_flat):
        # Allow tiny imaginary parts (floating point noise)
        x_real = np.real(X_flat)
        y_real = np.real(y_flat)
        x_imag = np.imag(X_flat)
        y_imag = np.imag(y_flat)
        
        valid_mask = (
            (np.abs(x_imag) < 1e-9) & 
            (np.abs(y_imag) < 1e-9) & 
            np.isfinite(x_real) & 
            np.isfinite(y_real)
        )
        
        if np.sum(valid_mask) < 5:
            return []
            
        X_flat = x_real[valid_mask]
        y_flat = y_real[valid_mask]
    else:
        # Just check finite
        valid_mask = np.isfinite(X_flat) & np.isfinite(y_flat)
        if np.sum(valid_mask) < 5:
            return []
        X_flat = X_flat[valid_mask]
        y_flat = y_flat[valid_mask]
    
    var = variable_names[0] if variable_names else "x"
    
    # Step 1: Integer Anchor Analysis
    # Find all integer x values and detect the floor power 'a'
    integer_mask = np.abs(X_flat - np.round(X_flat)) < 1e-9
    if np.sum(integer_mask) < 3:
        return []  # Need at least 3 integer anchor points
    
    integer_x = X_flat[integer_mask]
    integer_y = y_flat[integer_mask]
    
    # Try to find 'a' such that f(n) = n^a
    # Use positive integers for log-log regression
    # EXCLUDE x=0 (log(0)=-inf) and x=1 (log(1)=0 causes division issues)
    pos_int_mask = integer_x > 1.5  # Only x >= 2
    if np.sum(pos_int_mask) >= 2:  # Need at least 2 points for regression
        pos_int_x = integer_x[pos_int_mask]
        pos_int_y = integer_y[pos_int_mask]
        
        # Only proceed if all y values are positive
        if np.all(pos_int_y > 0):
            log_x = np.log(pos_int_x)
            log_y = np.log(pos_int_y)
            
            # a = log(y) / log(x) for each point
            # Use errstate to suppress warnings for any edge cases
            with np.errstate(divide='ignore', invalid='ignore'):
                a_estimates = log_y / log_x
            # Use nanmedian to handle any remaining nan values
            a_median = np.nanmedian(a_estimates)
            
            # If still nan, fall back to simple check
            if np.isnan(a_median):
                anchor_errors = np.abs(integer_y - integer_x)
                if np.max(anchor_errors) < 1e-6:
                    best_a = 1
                else:
                    return []  # Can't determine floor power
            else:
                # Snap 'a' to common values
                a_candidates = [0.5, 1, 1.5, 2, 3, 4]
                best_a = min(a_candidates, key=lambda a: abs(a - a_median))
        else:
            # Check if f(n) = n (a=1) - the original simple case
            anchor_errors = np.abs(integer_y - integer_x)
            if np.max(anchor_errors) < 1e-6:
                best_a = 1
            else:
                return []  # Can't determine floor power
    else:
        # Check if f(n) = n (a=1) - the original simple case
        anchor_errors = np.abs(integer_y - integer_x)
        if np.max(anchor_errors) < 1e-6:
            best_a = 1
        else:
            return []  # Not enough positive integers
    
    # Verify integer anchors match floor^a
    floor_int = np.floor(integer_x)
    with np.errstate(divide='ignore', invalid='ignore'):
        expected_y = np.where(floor_int == 0, 0, np.abs(floor_int) ** best_a * np.sign(floor_int) ** int(best_a))
    anchor_errors = np.abs(integer_y - expected_y)
    if np.max(anchor_errors) > 1e-4:
        return []  # Doesn't match floor^a pattern
    
    if verbose:
        print(f"   Scalloped Staircase: Integer anchors verified with floor^{best_a} ({len(integer_x)} points)")
    
    # Step 2: Fractional Analysis
    # Subtract floor(x)^a and analyze the residual for frac(x)^b
    floor_x = np.floor(X_flat)
    frac_x = X_flat - floor_x
    
    # Compute floor^a term (handle negative floors and a=2 specially)
    with np.errstate(divide='ignore', invalid='ignore'):
        if best_a == 2:
            floor_powered = floor_x ** 2
        elif best_a == 1:
            floor_powered = floor_x
        else:
            floor_powered = np.abs(floor_x) ** best_a * np.sign(floor_x)
    
    residual = y_flat - floor_powered
    
    # For values in [0, 1), floor=0, so residual = y = frac^b
    frac_mask_01 = (X_flat >= 0) & (X_flat < 1) & (~integer_mask)
    
    best_b = 2  # Default guess
    if np.sum(frac_mask_01) >= 3:
        frac_01_x = X_flat[frac_mask_01]
        frac_01_y = y_flat[frac_mask_01]  # Since floor=0, y = frac^b
        
        # Find b via log-log regression
        valid_mask = (frac_01_x > 0.01) & (frac_01_y > 1e-10)
        
        if np.sum(valid_mask) >= 2:
            log_x = np.log(frac_01_x[valid_mask])
            log_y = np.log(frac_01_y[valid_mask])
            
            b_estimates = log_y / log_x
            b_median = np.median(b_estimates)
            
            if verbose:
                print(f"   Scalloped Staircase: Estimated frac power b = {b_median:.4f}")
            
            b_candidates = [0.5, 1, 1.5, 2, 2.5, 3, 4]
            best_b = min(b_candidates, key=lambda b: abs(b - b_median))
    
    if verbose:
        print(f"   Scalloped Staircase: Snapped b = {best_b}")
    
    # Step 3: Validate floor(x)^a + frac(x)^b
    y_pred = floor_powered + frac_x ** best_b
    
    mse = np.mean((y_flat - y_pred) ** 2)
    max_err = np.max(np.abs(y_flat - y_pred))
    
    if verbose:
        print(f"   Scalloped Staircase: MSE = {mse:.6g}, Max Error = {max_err:.6g}")
    
    if max_err < 1e-6:
        # Perfect match - build expression string
        
        # TRIVIAL CASE: floor(x) + frac(x) = x
        # Don't return a complex representation for a simple linear function!
        if best_a == 1 and best_b == 1:
            # This is just f(x) = x, represented as floor(x) + frac(x)
            # Return None to let simpler detectors (like linear) handle it
            if verbose:
                print(f"   Scalloped Staircase: Detected trivial case (a=1, b=1) => f(x)=x, skipping")
            return []
        
        if best_a == 1 and best_b == 2:
            expr = f"floor({var}) + frac({var})^2"
        elif best_a == 2 and best_b == 2:
            expr = f"floor({var})^2 + frac({var})^2"
        elif best_a == 1:
            expr = f"floor({var}) + frac({var})^{best_b}"
        elif best_b == 2:
            expr = f"floor({var})^{best_a} + frac({var})^2"
        else:
            expr = f"floor({var})^{best_a} + frac({var})^{best_b}"
        
        if verbose:
            print(f"   Scalloped Staircase: FOUND {expr}")
        
        return ([expr], expr)  # Short-circuit tuple
    
    elif mse < 0.01:
        # Good but not perfect - return as seed
        if best_a == 1:
            expr = f"floor({var}) + frac({var})^{best_b}"
        else:
            expr = f"floor({var})^{best_a} + frac({var})^{best_b}"
        return [expr]
    
    return []


# ============================================================================
# PATTERN DETECTORS - Regenerated after refactoring
# ============================================================================





def _detect_scaled_staircase(X, y, variable_names=None, verbose=False):
    """Detect scaled/shifted staircase patterns: floor((x-c)/s) * h + k.
    
    Algorithm:
    1. Sort data by x.
    2. Identify plateaus (runs of constant y).
    3. Calculate step width (dx between plateau centers/edges) and step height (dy).
    4. Estimate parameters: s ~ width, h ~ height.
    5. Propose candidate functions including ceil/floor variants.
    """
    if X.ndim > 1 and X.shape[1] > 1: return []
    try:
        x_flat = X.flatten()
        y_flat = y.flatten()
    except: return []


    
    # Sort
    idx = np.argsort(x_flat)
    x_sorted = x_flat[idx]
    y_sorted = y_flat[idx]
    
    # Filter finite
    mask = np.isfinite(x_sorted) & np.isfinite(y_sorted)
    x_clean = x_sorted[mask]
    y_clean = y_sorted[mask]
    
    if len(x_clean) < 10: 
         return []
    
    var = variable_names[0] if variable_names else "x"
    
    # 2. Identify Plateaus
    # Calculate differences
    dy = np.diff(y_clean)
    dx = np.diff(x_clean)
    
    # Define "constant y" as small change relative to range (or absolute epsilon)
    y_range = np.max(y_clean) - np.min(y_clean)
    abs_tol = 1e-5 if y_range < 1 else y_range * 1e-4
    
    is_flat = np.abs(dy) < abs_tol
    
    # Group into plateaus
    # We want transitions where is_flat is False
    jumps = np.where(~is_flat)[0]
    
    if len(jumps) < 2: 
        return [] 
    
    # Calculate Step Heights (dy at jumps)
    step_heights = dy[jumps]
    median_height = np.median(step_heights)
    
    if abs(median_height) < 1e-9: 
        return []
    
    # Calculate Step Widths
    # Distance between jump points in x
    # Note: jumps index i corresponds to transition between i and i+1
    # x at jump is (x[i] + x[i+1])/2
    x_jump_locs = (x_clean[jumps] + x_clean[jumps+1]) / 2.0
    step_widths = np.diff(x_jump_locs)
    
    if len(step_widths) < 1: 
        return []
    
    median_width = np.median(step_widths)
    if median_width < 1e-9: 
        return []

    # v4.2 Fix: "Riemann Sum" Hallucination Prevention
    # If the detected "steps" are just the sampling interval (i.e. every point is a step),
    # this is not a staircase function, it's just a discrete approximation of a curve.
    avg_sampling = np.median(dx)
    if avg_sampling > 1e-9 and median_width < 1.5 * avg_sampling:
        if verbose:
             print(f"   [Scaled Staircase] Rejected: Steps are indistinguishable from sampling (width={median_width:.4f} ~ dx={avg_sampling:.4f})")
        return []
    
    # Check consistency (low variance in width/height)
    width_std = np.std(step_widths)
    height_std = np.std(step_heights)
    

    
    # Heuristic: Coefficient of variation < 0.1 (10% variance allowed)
    if width_std / median_width > 0.1 or height_std / abs(median_height) > 0.1:
        if verbose:
            print(f"   [Scaled Staircase] Rejected: High variance in steps (w_std={width_std:.4f}, h_std={height_std:.4f})")
        return []
        
    # v4.2 Fix: "Staircase Hallucination" Prevention
    # Check identifying feature: Steps should be FLAT (zero slope).
    # Smooth polynomials might look like steps if sampled coarsely or if threshold is high.
    # We check the slope of the "flat" regions.
    flat_indices = np.where(is_flat)[0]
    if len(flat_indices) > 0:
        # Calculate slope in flat regions
        flat_dy = dy[flat_indices]
        flat_dx = dx[flat_indices]
        flat_slopes = flat_dy / flat_dx
        avg_flat_slope = np.mean(np.abs(flat_slopes))
        
        # Calculate global slope (start to end)
        global_slope = (y_clean[-1] - y_clean[0]) / (x_clean[-1] - x_clean[0])
        
        # Criterion: If local segments have significant slope relative to global trend, 
        # it's likely a smooth curve, not a step function.
        if abs(global_slope) > 1e-9:
            slope_ratio = avg_flat_slope / abs(global_slope)
            if slope_ratio > 0.05: # 5% threshold as per user request
                if verbose:
                    print(f"   [Scaled Staircase] Rejected: Non-zero slope in steps (ratio={slope_ratio:.4f})")
                return []

        
    s = median_width
    h = median_height
    
    if verbose:
        print(f"   [Scaled Staircase] s={s}, h={h}")

    seeds = []
    
    # Round s, h to nice numbers if close
    s_rounded = round(s)
    if abs(s - s_rounded) < 0.05 * s: s = s_rounded
    elif abs(s - 0.5) < 0.05: s = 0.5
    
    h_rounded = round(h)
    if abs(h - h_rounded) < 0.05 * abs(h): h = h_rounded
    
    # Generate Candidates
    # 1. floor(x/s) * h
    # 2. ceil(x/s) * h
    # 3. round(x/s) * h
    
    # Construct term strings
    # x_term = x/s
    if abs(s - 1.0) < 1e-5:
        x_term = var
    elif abs(s - 0.5) < 1e-5:
        x_term = f"2*{var}"
    elif isinstance(s, int) or (isinstance(s, float) and s.is_integer()):
        x_term = f"{var}/{int(s)}"
    else:
        try:
             frac = fractions.Fraction(s).limit_denominator(100)
             if abs(frac - s) < 1e-5:
                 if frac.numerator == 1:
                     x_term = f"{var}/{frac.denominator}"
                 else:
                     x_term = f"{frac.denominator}*{var}/{frac.numerator}"
             else:
                 x_term = f"{var}/{s:.4g}"
        except:
             x_term = f"{var}/{s:.4g}"
        
    # h_term
    def apply_h(expr, h_val):
        if abs(h_val - 1.0) < 1e-5: return expr
        if abs(h_val + 1.0) < 1e-5: return f"-{expr}"
        if isinstance(h_val, int) or h_val.is_integer():
            return f"{int(h_val)}*{expr}"
        return f"{h_val:.4g}*{expr}"

    seeds.append(apply_h(f"floor({x_term})", h))
    seeds.append(apply_h(f"ceil({x_term})", h))
    seeds.append(apply_h(f"round({x_term})", h))
    
    return seeds



def _detect_step_patterns(X, y, variable_names=None, verbose=False):
    """Detect step functions like floor(x), ceil(x), round(x)."""
    seeds = []
    
    # 1. Scalloped Staircase (floor, ceil)
    try:
        scalloped = _detect_scalloped_staircase(X, y, variable_names=variable_names, verbose=verbose)
        if scalloped:
            if isinstance(scalloped, tuple): seeds.extend(scalloped[0])
            elif isinstance(scalloped, list): seeds.extend(scalloped)
    except Exception: pass

    # 2. Scaled Staircase
    try:
        scaled = _detect_scaled_staircase(X, y, variable_names=variable_names, verbose=verbose)
        if scaled:
            seeds.extend(scaled)
    except Exception: pass

    try:
        # 3. Integer Patterns
        integer_seeds = _detect_integer_patterns(X, y, variable_names=variable_names)
        if integer_seeds: seeds.extend(integer_seeds)
    except Exception: pass
        
    return sorted(list(set(seeds)))

def _detect_self_power(X, y, variable_names=None, verbose=False):
    """Detect self-power patterns: f(x) = x^x.
    
    Algorithm:
    1. For positive x > 1: log(y) / log(x) should equal x
    2. Check if log(y) = x * log(x) for all positive points
    3. Verify on integer points: n^n = y
    
    Returns:
        Tuple (seeds, exact_match) if exact match found, else list of seeds.
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []  # 1D only
    
    try:
        x_flat = np.real(X.flatten())
        y_flat = np.real(np.array(y).flatten())
    except Exception:
        return []
    
    if len(x_flat) < 5:
        return []
    
    var = variable_names[0] if variable_names else "x"
    
    # Focus on positive integer points: n^n should match
    integer_mask = np.abs(x_flat - np.round(x_flat)) < 1e-9
    pos_int_mask = integer_mask & (x_flat >= 1)
    
    if np.sum(pos_int_mask) < 3:
        return []
    
    int_x = x_flat[pos_int_mask].astype(int)
    int_y = y_flat[pos_int_mask]
    
    # Check if y = x^x for integer points
    expected = int_x.astype(float) ** int_x
    errors = np.abs(int_y - expected)
    
    # Use relative error for large values
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_errors = errors / np.maximum(np.abs(expected), 1)
    
    max_rel_err = np.nanmax(rel_errors)
    
    if verbose and max_rel_err < 0.1:  # Only print if significant match (UI hygiene)
        print(f"   Self-Power: Max relative error on integers = {max_rel_err:.6f}")
    
    if max_rel_err < 1e-6:
        # Verify on all positive points
        pos_mask = x_flat > 0.5
        if np.sum(pos_mask) > 0:
            x_pos = x_flat[pos_mask]
            y_pos = y_flat[pos_mask]
            
            with np.errstate(invalid='ignore'):
                expected_pos = x_pos ** x_pos
                errors_pos = np.abs(y_pos - expected_pos)
                rel_errors_pos = errors_pos / np.maximum(np.abs(expected_pos), 1)
            
            max_err_pos = np.nanmax(rel_errors_pos)
            
            if max_err_pos < 1e-4:
                expr = f"pow({var}, {var})"
                if verbose:
                    print(f"   Self-Power: FOUND {expr}")
                return ([expr, f"{var}^{var}"], expr)
    
    return []

def _detect_inverse_self_power(X, y, variable_names=None, verbose=False):
    """Detect inverse self-power patterns: y^y = x (i.e. y = exp(W(ln(x)))).
    
    Algorithm:
    1. Check if y^y ≈ x for positive x, y points
    2. Check y * ln(y) ≈ ln(x)
    """
    if X.ndim > 1 and X.shape[1] > 1: return []
    try:
        x_flat = np.real(X.flatten())
        y_flat = np.real(np.array(y).flatten())
    except: return []
    
    # Needs positive x, y
    pos_mask = (x_flat > 1e-6) & (y_flat > 1e-6)
    if np.sum(pos_mask) < 3: return []
    
    x_pos = x_flat[pos_mask]
    y_pos = y_flat[pos_mask]
    
    # Check y^y = x
    try:
        y_pow_y = y_pos ** y_pos
        errors = np.abs(y_pow_y - x_pos)
        
        # Relative error
        rel_errors = errors / np.maximum(np.abs(x_pos), 1e-9)
        max_rel_err = np.max(rel_errors)
        
        if verbose and max_rel_err < 0.1:
            print(f"   Inverse Self-Power: Max rel error = {max_rel_err:.6f}")
            
        if max_rel_err < 1e-3: # Strict match
            var = variable_names[0] if variable_names else "x"
            expr = f"exp(lambertw(log({var})))"
            if verbose: print(f"   Inverse Self-Power: FOUND {expr}")
            return ([expr], expr)
    except: pass
    
    return []

def _detect_relu_patterns(X, y, variable_names=None, verbose=False):
    """Detect ReLU patterns: f(x) = max(0, x) or max(0, x - c).
    
    Algorithm:
    1. Find where y = 0 (left region)
    2. Find where y = x (or y = x - c) (right region)
    3. Find the transition point
    
    Returns:
        Tuple (seeds, exact_match) if exact match found, else list of seeds.
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    try:
        x_flat = np.real(X.flatten())
        y_flat = np.real(np.array(y).flatten())
    except Exception:
        return []
    
    if len(x_flat) < 5:
        return []
    
    var = variable_names[0] if variable_names else "x"
    
    # Find zero region
    zero_mask = np.abs(y_flat) < 1e-9
    if np.sum(zero_mask) < 2:
        return []
    
    # Find linear region
    nonzero_mask = ~zero_mask
    if np.sum(nonzero_mask) < 3:
        return []
    
    # Check if nonzero region is linear: y = a*x + b
    x_nz = x_flat[nonzero_mask]
    y_nz = y_flat[nonzero_mask]
    
    try:
        coeffs = np.polyfit(x_nz, y_nz, 1)
        a, b = coeffs
    except Exception:
        return []
    
    # For ReLU: a should be close to 1, b close to 0 or -c
    if abs(a - 1.0) > 0.1:
        return []
    
    # Predict and check error
    y_pred_linear = a * x_nz + b
    max_err = np.max(np.abs(y_nz - y_pred_linear))
    
    if max_err > 0.01:
        return []
    
    # Find transition point
    x_zeros = x_flat[zero_mask]
    x_max_zero = np.max(x_zeros)
    
    if verbose:
        print(f"   ReLU: Transition at x ≈ {x_max_zero:.4f}, slope = {a:.4f}")
    
    # Build expression
    if abs(b) < 0.01:
        # Simple max(0, x)
        expr = f"max(0, {var})"
    else:
        c = -b
        if abs(c - round(c)) < 0.01:
            c = int(round(c))
        expr = f"max(0, {var} - {c})"
    
    # Verify on all data
    if abs(b) < 0.01:
        y_pred = np.maximum(0, x_flat)
    else:
        y_pred = np.maximum(0, x_flat + b)
    
    mse = np.mean((y_flat - y_pred)**2)
    
    if mse < 1e-6:
        if verbose:
            print(f"   ReLU: FOUND {expr}")
        return ([expr], expr)
    
    return []

def _detect_clamp_patterns(X, y, variable_names=None, verbose=False):
    """Detect clamp patterns: f(x) = min(x, c) or max(a, x).
    
    Algorithm:
    1. Look for constant regions (y = c for all x > threshold)
    2. Look for linear regions (y = x for x < threshold)
    3. Find transition point
    
    Returns:
        Tuple (seeds, exact_match) if exact match found, else list of seeds.
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    try:
        x_flat = np.real(X.flatten())
        y_flat = np.real(np.array(y).flatten())
    except Exception:
        return []
    
    if len(x_flat) < 5:
        return []
    
    var = variable_names[0] if variable_names else "x"
    
    # Sort by x
    sort_idx = np.argsort(x_flat)
    x_sorted = x_flat[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    # Look for constant upper region (min(x, c))
    # Find longest suffix with constant y
    n = len(y_sorted)
    const_val = y_sorted[-1]
    const_count = 1
    
    for i in range(n-2, -1, -1):
        if abs(y_sorted[i] - const_val) < 1e-6:
            const_count += 1
        else:
            break
    
    if const_count >= 3:
        # Check if prefix is linear y = x
        prefix_len = n - const_count
        if prefix_len >= 3:
            x_prefix = x_sorted[:prefix_len]
            y_prefix = y_sorted[:prefix_len]
            
            # Check y = x
            errors = np.abs(y_prefix - x_prefix)
            max_err = np.max(errors)
            
            if max_err < 1e-4:
                c = round(const_val, 4)
                if abs(c - round(c)) < 0.01:
                    c = int(round(c))
                expr = f"min({var}, {c})"
                
                if verbose:
                    print(f"   Clamp: FOUND {expr}")
                return ([expr], expr)
    
    # Look for constant lower region (max(a, x))
    const_val_low = y_sorted[0]
    const_count_low = 1
    
    for i in range(1, n):
        if abs(y_sorted[i] - const_val_low) < 1e-6:
            const_count_low += 1
        else:
            break
    
    if const_count_low >= 3:
        suffix_start = const_count_low
        if n - suffix_start >= 3:
            x_suffix = x_sorted[suffix_start:]
            y_suffix = y_sorted[suffix_start:]
            
            errors = np.abs(y_suffix - x_suffix)
            max_err = np.max(errors)
            
            if max_err < 1e-4:
                a = round(const_val_low, 4)
                if abs(a - round(a)) < 0.01:
                    a = int(round(a))
                expr = f"max({a}, {var})"
                
                if verbose:
                    print(f"   Clamp: FOUND {expr}")
                return ([expr], expr)
    
    return []

def _detect_pulse_patterns(X, y, variable_names=None, verbose=False):
    """Detect pulse patterns: f(x) = Heaviside(x-a) - Heaviside(x-b).
    
    Algorithm:
    1. Look for regions with constant 0, constant 1, constant 0
    2. Find the two transition points
    
    Returns:
        Tuple (seeds, exact_match) if exact match found, else list of seeds.
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    try:
        x_flat = np.real(X.flatten())
        y_flat = np.real(np.array(y).flatten())
    except Exception:
        return []
    
    if len(x_flat) < 7:
        return []
    
    var = variable_names[0] if variable_names else "x"
    
    # Check if y only takes values 0 and 1
    unique_y = np.unique(np.round(y_flat, 6))
    if len(unique_y) != 2:
        return []
    if not (np.min(unique_y) < 0.1 and np.max(unique_y) > 0.9):
        return []
    
    # Sort by x
    sort_idx = np.argsort(x_flat)
    x_sorted = x_flat[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    # Find transitions
    transitions = []
    for i in range(1, len(y_sorted)):
        if abs(y_sorted[i] - y_sorted[i-1]) > 0.5:
            # Transition between i-1 and i
            trans_x = (x_sorted[i-1] + x_sorted[i]) / 2
            trans_type = "up" if y_sorted[i] > y_sorted[i-1] else "down"
            transitions.append((trans_x, trans_type))
    
    if len(transitions) != 2:
        return []
    
    t1, type1 = transitions[0]
    t2, type2 = transitions[1]
    
    # Should be up then down for positive pulse
    if type1 == "up" and type2 == "down":
        a, b = t1, t2
        # Snap to integers if close
        if abs(a - round(a)) < 0.1:
            a = int(round(a))
        if abs(b - round(b)) < 0.1:
            b = int(round(b))
        expr = f"heaviside({var} - {a}, 0.5) - heaviside({var} - {b}, 0.5)"
        
        if verbose:
            print(f"   Pulse: FOUND pulse from {a} to {b}")
        return ([expr], expr)
    
    return []

def _detect_bessel_patterns(X, y, verbose=False): 
    """Detect Bessel function patterns - placeholder for future."""
    return []

def _detect_factorial_patterns(X, y, variable_names=None, verbose=False):
    """Detect Factorial patterns: y = x! or y = x^a + x!.
    
    Algorithm:
    1. Check for basic factorial x! matching integers
    2. Check for offset factorial (y - x^k = x!)
    
    Args:
        X: Input data
        y: Target data
        variable_names: Names of variables
        verbose: Print debug info
        
    Returns:
        Tuple (seeds, exact_match) if found, else list of seeds.
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []
        
    try:
        x_flat = np.real(X.flatten())
        y_flat = np.real(np.array(y).flatten())
        
        # Filter valid integer inputs (x >= 0 for standard factorial)
        # Allow x < 0 if using gamma, but let's stick to user's case (integers)
        # Note: User mentioned f(-1) = oo, so singularities are possible.
        
        integer_mask = np.abs(x_flat - np.round(x_flat)) < 1e-9
        valid_mask = integer_mask & (x_flat >= 0) & (x_flat <= 170) # 171! overflows float64
        
        if np.sum(valid_mask) < 3:
            return []
            
        x_valid = x_flat[valid_mask].astype(int)
        y_valid = y_flat[valid_mask]
        
        # Compute expected factorials
        import scipy.special
        expected_fact = scipy.special.factorial(x_valid)
        
        var = variable_names[0] if variable_names else "x"
        seeds = []
        
        # 1. Direct Factorial Check: y = factorial(x)
        errors = np.abs(y_valid - expected_fact)
        # Use relative error for large values
        with np.errstate(divide='ignore', invalid='ignore'):
             rel_errors = errors / np.maximum(np.abs(expected_fact), 1.0)
        
    
        if np.max(rel_errors) < 1e-4:
            expr = f"factorial({var})"
            if verbose: print(f"   Factorial: FOUND {expr}")
            return ([expr], expr)
            
        # 2. Offset Factorial Check: y = x^k + factorial(x)
        # User Case: x^3 + x!
        # Subtract factorial from y and check if it matches x^k
        residual = y_valid - expected_fact
        
        # Check if residual is x^k
        # Avoid 0^0 or issues with negative bases if they leaked in
        # We only have x >= 0 here.
        
        # Simple polynomial check for degrees 1, 2, 3...
        for k in [1, 2, 3, 4, 5]:
            poly_term = x_valid.astype(float) ** k
            poly_errors = np.abs(residual - poly_term)
            if np.max(poly_errors) < 1e-4: # Absolute error for integers is fine
                 expr = f"{var}^{k} + factorial({var})"
                 if verbose: print(f"   Factorial Offset: FOUND {expr}")
                 return ([expr], expr)
                 
        # Check if residual is exponential? a^x
        # ... (Optional, but let's keep it simple for now)
        
    except Exception as e:
        if verbose: print(f"Factorial detection error: {e}")
        pass
        
    return []

def _detect_prime_counting_patterns(X, y, variable_names=None, verbose=False):
    """Detect prime counting function: f(x) = π(x) = number of primes ≤ x.
    
    Algorithm:
    1. Precompute π(n) for small n
    2. Check if y values match at integer points
    
    Returns:
        Tuple (seeds, exact_match) if exact match found, else list of seeds.
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    try:
        x_flat = np.real(X.flatten())
        y_flat = np.real(np.array(y).flatten())
    except Exception:
        return []
    
    if len(x_flat) < 5:
        return []
    
    var = variable_names[0] if variable_names else "x"
    
    # Precompute prime counts
    def sieve_primes(n):
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i in range(n + 1) if sieve[i]]
    
    max_x = min(int(np.max(x_flat)) + 1, 1000)
    primes = sieve_primes(max_x)
    
    # Build π(n) lookup
    prime_count = {}
    count = 0
    prime_set = set(primes)
    for n in range(max_x + 1):
        if n in prime_set:
            count += 1
        prime_count[n] = count
    
    # Check integer points
    integer_mask = np.abs(x_flat - np.round(x_flat)) < 1e-9
    nonneg_mask = integer_mask & (x_flat >= 0)
    
    if np.sum(nonneg_mask) < 5:
        return []
    
    int_x = x_flat[nonneg_mask].astype(int)
    int_y = y_flat[nonneg_mask]
    
    matches = 0
    total = 0
    
    for x_val, y_val in zip(int_x, int_y):
        if x_val in prime_count:
            expected = prime_count[x_val]
            if abs(y_val - expected) < 0.5:
                matches += 1
            total += 1
    
    if total < 5:
        return []
    
    match_rate = matches / total
    
    if verbose and match_rate > 0.05:  # UI Hygiene: Only print if significant match
        print(f"   Prime Counting: Match rate = {match_rate:.1%} ({matches}/{total})")
    
    if match_rate > 0.95:
        expr = f"prime_pi({var})"
        if verbose:
            print(f"   Prime Counting: FOUND {expr}")
        return ([expr], expr)
        
    seeds = []
    if match_rate > 0.3:
        seeds.append(f"prime_pi({var})")

    # --- Check for ith_prime (2, 3, 5, 7, 11...) ---
    # f(n) = nth prime
    try:
        prime_matches = 0
        prime_total = 0
        
        # Need to extend sieve if values are large (nth prime > n)
        # e.g., prime(100) = 541 > 100
        target_max_y = 0
        if len(int_y) > 0:
            target_max_y = int(np.max(int_y))
        
        # Sieve up to max(y) + padding to cover nth prime check
        # A conservative upper bound for nth prime is n*log(n*log(n)).
        # But here we just check if y matches a prime list. 
        # Actually simpler: if y is a prime p, and x is its index (index of p in primes).
        
        if target_max_y > max_x:
            primes_extended = sieve_primes(target_max_y + 100)
        else:
            primes_extended = primes
            
        # 1-based indexing for nth prime (1st prime is 2)
        # Check if y = prime(x)
        for x_val, y_val in zip(int_x, int_y):
            # Check range
            if 0 < x_val <= len(primes_extended):
                expected_prime = primes_extended[x_val - 1] # 0-indexed array vs 1-based math
                if abs(y_val - expected_prime) < 0.5:
                    prime_matches += 1
            prime_total += 1
            
        if prime_total >= 3:
            prime_match_rate = prime_matches / prime_total
            if verbose and prime_match_rate > 0.05:
                 print(f"   Nth Prime: Match rate = {prime_match_rate:.1%} ({prime_matches}/{prime_total})")
                 
            if prime_match_rate > 0.95:
                # Found exact match for prime sequence!
                expr = f"ith_prime({var})" 
                if verbose: print(f"  -> Nth Prime EXACT MATCH: {expr}")
                return ([expr], expr)
            elif prime_match_rate > 0.5:
                seeds.append(f"ith_prime({var})")
    except Exception:
        pass

    return seeds

def _detect_modulo_patterns(X, y, variable_names=None, verbose=False):
    """Detect modulo patterns: f(x) = x % k (sawtooth).
    
    Algorithm:
    1. Look for periodic sawtooth pattern
    2. Find period k by analyzing zero crossings or drops
    3. Verify x % k matches
    
    Returns:
        Tuple (seeds, exact_match) if exact match found, else list of seeds.
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    try:
        x_flat = np.real(X.flatten())
        y_flat = np.real(np.array(y).flatten())
    except Exception:
        return []
    
    if len(x_flat) < 10:
        return []
    
    var = variable_names[0] if variable_names else "x"
    
    # Sort by x
    sort_idx = np.argsort(x_flat)
    x_sorted = x_flat[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    # Look for "drops" in y (sawtooth resets)
    # y should increase then drop suddenly
    drops = []
    for i in range(1, len(y_sorted)):
        dy = y_sorted[i] - y_sorted[i-1]
        dx = x_sorted[i] - x_sorted[i-1]
        
        # A drop is when y decreases significantly while x increases
        if dy < -0.5 and dx > 0:
            drop_x = x_sorted[i]
            drops.append(drop_x)
    
    if len(drops) < 2:
        return []
    
    # Estimate period from drop spacing
    drop_diffs = np.diff(drops)
    period_estimate = np.median(drop_diffs)
    
    # UI Hygiene: Only print if we are fairly confident or in deep debug (period_estimate is just a guess here)
    if verbose and len(drops) > 5:
        print(f"   Modulo: Estimated period = {period_estimate:.4f}")
    
    # Snap to nice values
    candidates = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10]
    k = min(candidates, key=lambda c: abs(c - period_estimate))
    
    if abs(k - period_estimate) > 0.2:
        # Try the estimate directly
        k = period_estimate
    
    # Verify x % k
    y_pred = x_sorted % k
    errors = np.abs(y_sorted - y_pred)
    max_err = np.max(errors)
    
    if verbose and max_err < 0.1: # UI Hygiene
        print(f"   Modulo: Testing k={k}, max_err={max_err:.6f}")
    
    if max_err < 0.01:
        if abs(k - round(k)) < 0.01:
            k = int(round(k))
        expr = f"{var} % {k}"
        if verbose:
            print(f"   Modulo: FOUND {expr}")
        return ([expr], expr)
    
    return []

def _detect_reciprocal_patterns(X, y, variable_names=None, verbose=False):
    """Detect reciprocal patterns: f(x) = 1 / g(x).
    
    Algorithm (Inspired by User feedback):
    1. Calculate z = 1/y
    2. Check if z is linear: z = ax + b
    3. Check if z is linear + sine: z = ax + b + c*sin(x)
    
    Returns:
        Tuple (seeds, exact_match)
    """
    if X.ndim > 1 and X.shape[1] > 1: return []
    
    try:
        x_flat = np.real(X.flatten())
        y_flat = np.real(np.array(y).flatten())
        
        # Filter singularities where y is close to 0
        mask = np.abs(y_flat) > 1e-6
        x_clean = x_flat[mask]
        y_clean = y_flat[mask]
        
        if len(x_clean) < 5: return []
        
        z = 1.0 / y_clean
    except Exception:
        return []
        
    var = variable_names[0] if variable_names else "x"
    
    # 1. Check Linear: z = ax + b
    try:
        coeffs = np.polyfit(x_clean, z, 1)
        a, b = coeffs
        z_pred_lin = a * x_clean + b
        
        # R2 score
        ss_res = np.sum((z - z_pred_lin)**2)
        ss_tot = np.sum((z - np.mean(z))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        if r2 > 0.99:
            # High confidence linear match in inverse space
            a_snap = int(round(a)) if abs(a - round(a)) < 0.01 else round(a, 10)
            b_snap = int(round(b)) if abs(b - round(b)) < 0.01 else round(b, 10)
            
            denom = f"{a_snap}*{var} + {b_snap}"
            denom = denom.replace(" + -", " - ").replace("1*", "")
            expr = f"1 / ({denom})"
            
            if verbose:
                print(f"   Reciprocal: Linear trend found (R2={r2:.4f}). Seed: {expr}")
            
            if r2 > 0.9999:
                return ([expr], expr)
            
            # If not perfect, add seed but continue to check for trig components
            # This captures cases like 1/(x + sin(x)) where linear fit is effectively high but not exact

            
        # 2. Check Linear + Sine: z = ax + b + c*sin(x)
        # Residual of linear fit
        resid = z - z_pred_lin
        
        # Fit sine to residual? simpler: check correlation with sin(x)
        # Using fixed frequency assumption first (freq=1)
        sin_x = np.sin(x_clean)
        cos_x = np.cos(x_clean)
        
        A = np.vstack([x_clean, np.ones(len(x_clean)), sin_x, cos_x]).T
        coeffs_trig, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        a, b, c_sin, c_cos = coeffs_trig
        
        z_pred_trig = a*x_clean + b + c_sin*sin_x + c_cos*cos_x
        ss_res_trig = np.sum((z - z_pred_trig)**2)
        r2_trig = 1 - (ss_res_trig / (ss_tot + 1e-10))
        
        if r2_trig > 0.99:
             a_snap = int(round(a)) if abs(a - round(a)) < 0.01 else round(a, 10)
             b_snap = int(round(b)) if abs(b - round(b)) < 0.01 else round(b, 10)
             
             # Generalized: Accept any significant trig component
             # Original heuristic required abs(c_mag - 1.0) < 0.1, which fails for 1/e * cos(x)
             # Use high precision (10 decimals) to avoid "lying" about exact matches
             denom = f"{a_snap}*{var} + {b_snap} + {round(c_sin, 10)}*sin({var}) + {round(c_cos, 10)}*cos({var})"
             # Clean up zero terms
             denom = denom.replace("+ 0.0*sin(x)", "").replace("+ 0.0*cos(x)", "")
             denom = denom.replace("+ -", "- ").replace("1.0*", "1*")
             expr = f"1 / ({denom})"
             
             if verbose:
                print(f"   Reciprocal: Lin+Trig found (R2={r2_trig:.4f}). Seed: {expr}")
             return ([expr], expr if r2_trig > 0.9999 else None)
                 
    except Exception:
        pass
        
    return []

def _detect_anchor_patterns(X, y, variable_names=None, verbose=False):
    """Detect constant anchor patterns: f(x) = (x+c)^(1/x).
    Algorithm:
    1. Compute z = y^x
    2. Check if z is linear with respect to x: z = m*x + c
    3. If m ≈ 1, then y = (x+c)^(1/x)
    
    Returns:
        Tuple (seeds, exact_match) if exact match found, else list of seeds.
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    try:
        x_flat = np.real(X.flatten())
        y_flat = np.real(np.array(y).flatten())
    except Exception:
        return []
    
    if len(x_flat) < 5:
        return []
    
    var = variable_names[0] if variable_names else "x"
    
    # Filter points where transform is safe (y != 0) and x is positive (avoid complex base issues)
    valid_mask = (np.abs(y_flat) > 1e-9) & np.isfinite(y_flat) & np.isfinite(x_flat) & (x_flat > 0)
    if np.sum(valid_mask) < 5:
        return []
    
    x_val = x_flat[valid_mask]
    y_val = y_flat[valid_mask]
    
    # Compute z = y^x
    # Handle negative bases for y via complex power, then take real part checking imag is small
    try:
        # We need original complex y if available to be precise, but here we only have real parts passed in
        # Assuming y_flat is real part.
        # But if y was originally complex (from input), we might have lost phase info if we only use real part?
        # The prompt implies we receive real-valued X, y generally, or flattened reals.
        # Let's rely on standard power.
        z_val = np.power(y_val.astype(complex), x_val)
        
        # Check if z is essentially real
        if np.any(np.abs(z_val.imag) > 1e-4):
            # If z has imaginary components, maybe it's not this pattern
            # But wait, (x+c) should be real.
            return []
            
        z_real = z_val.real
    except Exception:
        return []
    
    # Check linearity of z_real vs x_val
    # z = m*x + c
    try:
        coeffs = np.polyfit(x_val, z_real, 1)
        m, c = coeffs
    except Exception:
        return []
    
    if verbose and abs(m - 1.0) < 0.1: # UI Hygiene: Only print if slope is close to 1
        print(f"   Anchor: z=y^x linearity check: m={m:.4f}, c={c:.4f}")
    
    # Check if m is close to 1
    if abs(m - 1.0) > 0.05:
        return []
    
    # Check MSE of linear fit
    z_pred = m * x_val + c
    mse_z = np.mean((z_real - z_pred)**2)
    r2 = 1.0 - (np.sum((z_real - z_pred)**2) / np.sum((z_real - np.mean(z_real))**2))
    
    if r2 < 0.99:
        return []
        
    if verbose:
        print(f"   Anchor: Found linear relation z = {m:.2f}x + {c:.2f} (R2={r2:.4f})")
    
    # Refine c assuming m=1
    # z = x + c => c = z - x
    c_estimates = z_real - x_val
    c_final = np.median(c_estimates)
    
    if abs(c_final - round(c_final)) < 0.01:
        c_final = int(round(c_final))
    else:
        c_final = round(c_final, 4)
        
    # Build expression
    # y = (x + c)^(1/x) => (x + c)**(1/x)
    expr = f"({var} + {c_final})**(1/{var})"
    
    # Verify exact match on y
    try:
        y_pred_verify = np.power(x_val + c_final, 1.0/x_val)
        y_err = np.max(np.abs(y_val - y_pred_verify))
        if y_err < 1e-2:
             if verbose: print(f"   Anchor: FOUND {expr} (max err {y_err:.6f})")
             return ([expr], expr)
    except Exception:
        pass
        
    # If not perfect exact match, still return seed
    if verbose: print(f"   Anchor: Proposing {expr}")
    return ([expr])


def _detect_signum_patterns(X, y, variable_names=None, verbose=False):
    """Detect signum (sign) function: f(x) = sign(x).
    
    Algorithm:
    1. Check if y only takes values in {-1, 0, 1}
    2. Verify sign(x) matches y
    
    Returns:
        Tuple (seeds, exact_match) if exact match found, else list of seeds.
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    try:
        x_flat = np.real(X.flatten())
        y_flat = np.real(np.array(y).flatten())
    except Exception:
        return []
    
    if len(x_flat) < 5:
        return []
    
    var = variable_names[0] if variable_names else "x"
    
    # Check if all y values are in {-1, 0, 1}
    for y_val in y_flat:
        if abs(y_val - round(y_val)) > 0.01:
            return []
        if int(round(y_val)) not in [-1, 0, 1]:
            return []
    
    # Verify sign(x) = y
    y_pred = np.sign(x_flat)
    errors = np.abs(y_flat - y_pred)
    max_err = np.max(errors)
    
    if max_err < 0.01:
        expr = f"sign({var})"
        if verbose:
            print(f"   Signum: FOUND {expr}")
        return ([expr], expr)
    
    return []

def _detect_bitwise_patterns(X, y, variable_names=None, verbose=False):
    """Detect bitwise XOR patterns: f(x) = int(x) ^ k or f(x) = floor(x) ^ k.
    
    Algorithm (from Gemini's analysis):
    1. Integer Pattern Recognition: Look at integer x,y pairs and compute x ^ y.
       If all results are the same constant k, we have an XOR pattern.
    2. Truncation vs Floor: Check negative decimals to distinguish int() from floor().
       - int(-4.5) = -4 (truncation toward zero)
       - floor(-4.5) = -5 (round down)
    3. Zero-Zone Confirmation: Values in (-1, 1) should all map to 0 ^ k = k.
    
    Returns:
        Tuple (seeds, exact_match) if exact match found, else list of seeds.
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []  # 1D only for now
    
    try:
        x_flat = X.flatten()
        y_flat = np.array(y).flatten()
    except Exception:
        return []
    
    if len(x_flat) < 5:
        return []
    
    var = variable_names[0] if variable_names else "x"
    
    # Step 1: Find integer x values and compute candidate XOR constants
    integer_mask = np.abs(x_flat - np.round(x_flat)) < 1e-9
    if np.sum(integer_mask) < 3:
        return []  # Need at least 3 integer points
    
    integer_x = x_flat[integer_mask].real.astype(int)
    integer_y = y_flat[integer_mask]
    
    # Check if y values are also integers (XOR produces integers)
    if not np.all(np.abs(integer_y - np.round(integer_y)) < 1e-9):
        return []  # XOR should produce integer outputs
    
    integer_y = integer_y.astype(int)
    
    # Compute candidate k = x ^ y for each integer pair
    xor_constants = integer_x ^ integer_y
    
    # Check if all XOR constants are the same
    if len(set(xor_constants)) != 1:
        return []  # Not a simple XOR pattern
    
    k = int(xor_constants[0])
    
    if verbose:
        print(f"   Bitwise XOR: Candidate constant k = {k}")
        print(f"   Bitwise XOR: Verified on {len(integer_x)} integer points")
    
    # Step 2: Determine if it's int() (truncation) or floor()
    # Check negative decimals: int(-4.5) = -4, floor(-4.5) = -5
    neg_decimal_mask = (x_flat < -0.5) & (~integer_mask)
    
    use_truncation = True  # Default to int() (more common)
    
    if np.sum(neg_decimal_mask) >= 1:
        # Test a negative decimal
        for i in np.where(neg_decimal_mask)[0]:
            x_val = float(x_flat[i])
            y_val = int(round(y_flat[i]))
            
            # What would truncation (int) predict?
            trunc_pred = int(x_val) ^ k
            # What would floor predict?
            floor_pred = int(np.floor(x_val)) ^ k
            
            if y_val == trunc_pred and y_val != floor_pred:
                use_truncation = True
                break
            elif y_val == floor_pred and y_val != trunc_pred:
                use_truncation = False
                break
    
    if verbose:
        print(f"   Bitwise XOR: Using {'int()' if use_truncation else 'floor()'} for rounding")
    
    # Step 3: Verify the pattern on ALL data points
    errors = 0
    for i in range(len(x_flat)):
        x_val = float(x_flat[i])
        y_val = y_flat[i]
        
        if use_truncation:
            x_int = int(x_val)  # Truncation toward zero
        else:
            x_int = int(np.floor(x_val))
        
        predicted = x_int ^ k
        
        if abs(predicted - y_val) > 1e-6:
            errors += 1
    
    error_rate = errors / len(x_flat)
    
    if verbose:
        print(f"   Bitwise XOR: Error rate = {error_rate:.1%} ({errors}/{len(x_flat)} points)")
    
    if error_rate < 0.01:  # Allow 1% error for floating point issues
        # Build the expression
        if use_truncation:
            # int(x) is trunc(x) in some systems, but we'll use floor for positive, ceil for negative
            # Actually, Python's int() truncates toward zero
            # We can express this as: sign(x) * floor(abs(x))
            # But simpler: just use "trunc(x)" if available, or note it
            expr = f"bitwise_xor(trunc({var}), {k})"
            # Fallback if trunc not available:
            expr_alt = f"bitwise_xor(floor({var}), {k})"  # Close enough for positive x
        else:
            expr = f"bitwise_xor(floor({var}), {k})"
            expr_alt = expr
        
        if verbose:
            print(f"   Bitwise XOR: FOUND {expr}")
        
        # Short-circuit with exact match
        return ([expr, expr_alt], expr)
    
    elif error_rate < 0.1:
        # Good but not perfect - return as seed
        if use_truncation:
            expr = f"bitwise_xor(floor({var}), {k})"
        else:
            expr = f"bitwise_xor(floor({var}), {k})"
        return [expr]
    
    return []


def _detect_fibonacci_patterns(X, y, variable_names=None, verbose=False):
    """Detect if data follows the Fibonacci sequence: f(n) = F(n).
    
    Algorithm:
    1. Find integer x values with their corresponding y values
    2. Check if y values match Fibonacci sequence at those indices
    3. Verify recurrence relation: F(n) = F(n-1) + F(n-2)
    
    Returns:
        Tuple (seeds, exact_match) if exact match found, else list of seeds.
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []  # 1D only
    
    try:
        x_flat = X.flatten()
        y_flat = np.array(y).flatten()
    except Exception:
        return []
    
    if len(x_flat) < 5:
        return []
    
    var = variable_names[0] if variable_names else "x"
    
    # Step 1: Find non-negative integer x values
    integer_mask = np.abs(x_flat - np.round(x_flat)) < 1e-9
    nonneg_int_mask = integer_mask & (x_flat >= 0)
    
    if np.sum(nonneg_int_mask) < 5:
        return []  # Need enough integer points
    
    int_x = x_flat[nonneg_int_mask].real.astype(int)
    int_y = y_flat[nonneg_int_mask]
    
    # Sort by x
    sort_idx = np.argsort(int_x)
    int_x = int_x[sort_idx]
    int_y = int_y[sort_idx]
    
    # Step 2: Precompute Fibonacci values up to max(int_x)
    max_n = min(int(np.max(int_x)) + 1, 100)  # Cap at 100 to avoid overflow
    fib = [0, 1]
    for i in range(2, max_n + 1):
        fib.append(fib[-1] + fib[-2])
    
    # Step 3: Check if y values match Fibonacci at corresponding x indices
    matches = 0
    total = 0
    for i in range(len(int_x)):
        n = int(int_x[i])
        if n < len(fib):
            expected = fib[n]
            actual = int_y[i]
            if abs(expected - actual) < 1e-6:
                matches += 1
            total += 1
    
    if total < 3:
        return []
    
    match_rate = matches / total
    
    if verbose and match_rate > 0.05:  # Only print if significant match rate (UI hygiene)
        print(f"   Fibonacci: Match rate = {match_rate:.1%} ({matches}/{total} integer points)")
    
    if match_rate > 0.95:
        # Also verify with non-integer points using analytic continuation
        # F(x) = (phi^x - cos(pi*x) * phi^(-x)) / sqrt(5)
        phi = 1.618033988749895
        sqrt5 = 2.23606797749979
        
        non_int_mask = ~integer_mask
        if np.sum(non_int_mask) > 0:
            x_ni = x_flat[non_int_mask]
            y_ni = y_flat[non_int_mask]
            
            # Predicted using analytic continuation
            pred_ni = (phi**x_ni - np.cos(np.pi * x_ni) * phi**(-x_ni)) / sqrt5
            
            ni_errors = np.abs(y_ni - pred_ni)
            max_ni_err = np.max(ni_errors) if len(ni_errors) > 0 else 0
            
            if verbose:
                print(f"   Fibonacci: Non-integer max error = {max_ni_err:.6f}")
            
            if max_ni_err < 1e-4:
                expr = f"fibonacci({var})"
                if verbose:
                    print(f"   Fibonacci: FOUND {expr}")
                return ([expr], expr)
        else:
            # No non-integer points, but integers match perfectly
            expr = f"fibonacci({var})"
            if verbose:
                print(f"   Fibonacci: FOUND {expr} (integers only)")
            return ([expr], expr)
    
    elif match_rate > 0.8:
        return [f"fibonacci({var})"]
    
    return []


def _detect_linear_recurrence(X, y, variable_names=None, verbose=False):
    """Detect general linear recurrence patterns: a(n) = c1*a(n-1) + c2*a(n-2) + ...
    
    This can identify:
    - Fibonacci: coeffs=[1,1], initial=[0,1] or [1,1]
    - Lucas: coeffs=[1,1], initial=[2,1]
    - Tribonacci: coeffs=[1,1,1], initial=[0,0,1]
    - Pell: coeffs=[2,1], initial=[0,1]
    - Padovan: coeffs=[0,1,1], initial=[1,1,1]
    - Any custom linear recurrence!
    
    Algorithm:
    1. Get consecutive integer values a(0), a(1), a(2), ...
    2. Try recurrence orders 2, 3, 4 (number of previous terms)
    3. For each order k, solve for coefficients: a(n) = c1*a(n-1) + ... + ck*a(n-k)
    4. Verify recurrence holds for all data points
    5. Match against known sequences (Fibonacci, Lucas, etc.)
    
    Returns:
        Tuple (seeds, exact_match) if exact match found, else list of seeds.
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []  # 1D only
    
    try:
        x_flat = X.flatten()
        y_flat = np.array(y).flatten()
    except Exception:
        return []
    
    if len(x_flat) < 7:
        return []  # Need enough points to detect recurrence
    
    var = variable_names[0] if variable_names else "x"
    
    # Step 1: Find consecutive non-negative integer x values
    integer_mask = np.abs(x_flat - np.round(x_flat)) < 1e-9
    nonneg_int_mask = integer_mask & (x_flat >= 0)
    
    if np.sum(nonneg_int_mask) < 7:
        return []
    
    int_x = x_flat[nonneg_int_mask].real.astype(int)
    int_y = y_flat[nonneg_int_mask]
    
    # Sort by x and find consecutive sequence starting from 0 or 1
    sorted_idx = np.argsort(int_x)
    int_x = int_x[sorted_idx]
    int_y = int_y[sorted_idx]
    
    # Build a dictionary for quick lookup
    seq_dict = {int(x): y for x, y in zip(int_x, int_y)}
    
    # Find starting point (0 or 1)
    start = 0 if 0 in seq_dict else (1 if 1 in seq_dict else None)
    if start is None:
        return []
    
    # Get consecutive values
    seq_vals = []
    n = start
    while n in seq_dict and len(seq_vals) < 50:
        seq_vals.append(seq_dict[n])
        n += 1
    
    if len(seq_vals) < 7:
        return []
    
    # Known sequences to match against
    KNOWN_SEQUENCES = {
        # (tuple of coeffs): (name, function_name)
        ((1, 1),): [
            ("Fibonacci", "fibonacci", [0, 1]),
            ("Lucas", "lucas", [2, 1]),
        ],
        ((2, 1),): [
            ("Pell", None, [0, 1]),  # No built-in function
        ],
        ((1, 1, 1),): [
            ("Tribonacci", None, [0, 0, 1]),
        ],
        ((0, 1, 1),): [
            ("Padovan", None, [1, 1, 1]),
        ],
    }
    
    # Step 2: Try different recurrence orders
    for order in [2, 3]:
        if len(seq_vals) < order + 4:
            continue
        
        # Build system of equations: A * coeffs = b
        # a(n) = c1*a(n-1) + c2*a(n-2) + ...
        # For each n from order to len-1:
        # seq[n] = c1*seq[n-1] + c2*seq[n-2] + ...
        
        n_eqs = len(seq_vals) - order
        A = np.zeros((n_eqs, order))
        b = np.zeros(n_eqs)
        
        for i in range(order, len(seq_vals)):
            eq_idx = i - order
            for j in range(order):
                val = seq_vals[i - j - 1]
                A[eq_idx, j] = val.real if hasattr(val, 'real') else val
            val_b = seq_vals[i]
            b[eq_idx] = val_b.real if hasattr(val_b, 'real') else val_b
        
        # Solve least squares
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        
        # Check if solution is valid (residuals small)
        predicted = A @ coeffs
        errors = np.abs(b - predicted)
        max_err = np.max(errors)
        
        if max_err > 0.1:
            continue
        
        # Round coefficients to nearest integer (most recurrences have integer coeffs)
        rounded_coeffs = np.round(coeffs).astype(int)
        
        # Verify with rounded coefficients
        predicted_rounded = A @ rounded_coeffs
        errors_rounded = np.abs(b - predicted_rounded)
        max_err_rounded = np.max(errors_rounded)
        
        if max_err_rounded > 1e-6:
            continue
        
        if verbose:
            print(f"   Linear Recurrence: Found order-{order} recurrence with coeffs {list(rounded_coeffs)}")
        
        # Get initial values
        initial_vals = [int(round(v)) for v in seq_vals[:order]]
        
        # Match against known sequences
        coeff_tuple = (tuple(rounded_coeffs),)
        
        if coeff_tuple in KNOWN_SEQUENCES:
            for name, func_name, expected_initial in KNOWN_SEQUENCES[coeff_tuple]:
                if initial_vals == expected_initial:
                    if func_name:
                        expr = f"{func_name}({var})"
                        if verbose:
                            print(f"   Linear Recurrence: Matched {name} sequence -> {expr}")
                        return ([expr], expr)
                    else:
                        if verbose:
                            print(f"   Linear Recurrence: Matched {name} sequence (no built-in function)")
                        # Return a symbolic representation
                        seed = f"recurrence_{name.lower()}({var})"
                        return [seed]
        
        # Unknown recurrence - return as seed with description
        coeff_str = ",".join(str(c) for c in rounded_coeffs)
        if verbose:
            print(f"   Linear Recurrence: Unknown sequence with coeffs [{coeff_str}], initial {initial_vals}")
        
        # Can't express as a simple function, but we found a pattern
        return []
    
    return []

# Remaining stubs for unimplemented detectors
def _detect_odd_function_patterns(X, y, verbose=False): return []
def _detect_rosenbrock_patterns(X, y, variable_names=None, verbose=False): return []
def _detect_fractal_cosine_patterns(X, y, verbose=False): return []

def _detect_chirp_patterns(X, y, variable_names=None, verbose=False):
    """Detect chirp-like patterns (sin(x^2)) via zero-crossing analysis.
    
    Heuristic:
    - sin(x^2) has zeros at x = sqrt(n * pi)
    - cos(x^2) has zeros at x = sqrt((n + 0.5) * pi)
    """
    seeds = []
    
    # Needs to be 1D
    if X.ndim > 1 and X.shape[1] > 1: return []
    try: x_flat = X.flatten()
    except: return []
    
    # Filter for real, finite zeros (exclude complex)
    if np.iscomplexobj(x_flat):
        real_mask = np.abs(np.imag(x_flat)) < 1e-9
        x_flat = np.real(x_flat[real_mask])
        y = y[real_mask]
        
    # Find points where y is close to zero
    zero_mask = (np.abs(y) < 1e-3) & np.isfinite(x_flat)
    if np.sum(zero_mask) < 2:
        return []
        
    zeros_x = x_flat[zero_mask]
    
    # Analyze if zeros match sqrt(n * pi)
    # x = sqrt(n * pi) => x^2 = n * pi => x^2 / pi = n
    
    # Calculate v = x^2 / pi
    v_vals = (zeros_x ** 2) / np.pi
    
    # Check for integers (sin(x^2))
    v_rounded = np.round(v_vals)
    matches_sin = np.abs(v_vals - v_rounded) < 0.05
    sin_match_rate = np.sum(matches_sin) / len(zeros_x)
    
    # Check for half-integers (cos(x^2))
    # n + 0.5
    v_shifted = v_vals - 0.5
    v_shifted_rounded = np.round(v_shifted)
    matches_cos = np.abs(v_shifted - v_shifted_rounded) < 0.05
    cos_match_rate = np.sum(matches_cos) / len(zeros_x)
    
    var = variable_names[0] if variable_names else "x"
    
    if verbose and (sin_match_rate > 0.5 or cos_match_rate > 0.5):
        print(f"   Chirp Analysis: Tested {len(zeros_x)} zeros. Sin match: {sin_match_rate:.0%}, Cos match: {cos_match_rate:.0%}")
        
    if sin_match_rate > 0.5:
        if verbose: print(f"   Chirp Analysis: Detected sin({var}^2) pattern (Zeros at sqrt(n*π))")
        seeds.append(f"sin({var}**2)")
        
    if cos_match_rate > 0.5:
        if verbose: print(f"   Chirp Analysis: Detected cos({var}^2) pattern (Zeros at sqrt((n+0.5)*π))")
        seeds.append(f"cos({var}**2)")
        
    return seeds

def _detect_singularity_zeros(X, y, variable_names=None, verbose=False):
    """Detect singularities by analyzing zero accumulation points.
    
    If y = sin(1/(x-p)) or sin(x/(x-p)), zeros appear at 1/(x-p) = n*pi.
    So 1/(zeros - p) should be evenly spaced (spacing = pi).
    """
    seeds = []
    
    if X.ndim > 1 and X.shape[1] > 1: return []
    try: x_flat = X.flatten()
    except: return []
    
    # Filter for real
    if np.iscomplexobj(x_flat):
        real_mask = np.abs(np.imag(x_flat)) < 1e-9
        x_flat = np.real(x_flat[real_mask])
        y = y[real_mask]
        
    zero_mask = (np.abs(y) < 1e-2) & np.isfinite(x_flat) # Relaxed tolerance
    zeros = x_flat[zero_mask]
    
    if len(zeros) < 5: return []
    
    zeros = np.sort(zeros)
    
    # Heuristic: Check candidates for pole p
    # Try integers and half-integers in range
    x_min, x_max = np.min(x_flat), np.max(x_flat)
    candidates = np.arange(np.ceil(x_min - 1), np.floor(x_max + 1) + 0.1, 0.5)
    
    var = variable_names[0] if variable_names else "x"
    
    for p in candidates:
        if np.any(np.abs(zeros - p) < 1e-9): continue
        
        # Use only zeros unlikely to be aliased (furthest from p)
        # Aliasing happens when frequency 1/(x-p)^2 is high -> near p
        # We need the "slow" zeros to establish the period pattern
        
        # Sort zeros by distance from p
        dists = np.abs(zeros - p)
        # Keep zeros where dist is relatively large
        # Heuristic: keep top 75% farthest points (drop closest 25%)
        # But ensure we keep at least 4 points
        if len(zeros) > 6:
            safe_indices = np.argsort(dists)[len(zeros)//4:] # Drop closest 25%
            zeros_safe = zeros[safe_indices]
        else:
            zeros_safe = zeros
            
        # Transform: z' = 1 / (z - p)
        z_trans = 1.0 / (zeros_safe - p)
        z_trans = np.sort(z_trans)
        
        # Check if evenly spaced
        diffs = np.diff(z_trans)
        
        # Filter outliers (large jumps due to missing zeros)
        # We look for a common divisor (the period) -> pi
        
        # Simple check: median diff should be stable
        # We look for low variance in diffs
        # Filter out massive jumps (outliers where we missed many zeros)
        valid_diffs = diffs[diffs < 5 * np.median(diffs)]
        if len(valid_diffs) < 3: continue
        
        median_diff = np.median(valid_diffs)
        if median_diff < 1e-3: continue
        
        # Check consistency (coefficient of variation)
        # Are the diffs clustering around the median?
        consistent_diffs = valid_diffs[np.abs(valid_diffs - median_diff) < 0.2 * median_diff]
        match_rate = len(consistent_diffs) / len(diffs)
        
        if match_rate > 0.5:
             T = median_diff
             # Calculate implied frequency k
             # 1/(x-p) has spacing T.
             # sin(k/(x-p)) has zeros at k/(x-p) = n*pi => 1/(x-p) = n*pi/k.
             # So spacing T = pi/k.
             # k = pi / T.
             k = np.pi / T
             k_rounded = round(k, 2)
             if abs(k - round(k)) < 0.1: k_rounded = int(round(k))
             
             if verbose:
                 print(f"   Singularity Analysis: Zeros accumulate at {p} with period {T:.3f} (k={k_rounded}).")
             
             p_str = str(float(p))
             k_str = str(k_rounded)
             
             # Base seeds
             seeds.append(f"sin({k_str}/({var}-({p_str})))")
             seeds.append(f"cos({k_str}/({var}-({p_str})))")
             
             # Variant x/(x-p)
             seeds.append(f"sin({var}/({var}-({p_str})))")
             
             # Variant with k
             if k_rounded != 1:
                 seeds.append(f"sin({k_str}*{var}/({var}-({p_str})))")
             
             # Break early to avoid duplicate detections
             return seeds
                 
    return seeds
def _detect_newton_polynomial(X, y, variable_names=None, verbose=False): return []
def _detect_sub_epsilon_patterns(X, y, variable_names=None, verbose=False): return []

def _detect_rational_form(X, y, variable_names=None, verbose=False):
    """Detect if outputs are simple fractions, suggesting a rational function.
    
    If outputs like 0.28 (7/25), 0.6 (3/5), 1 (1/1) are detected,
    generates polynomial ratio seeds like:
        (a*x² + b*y² + c) / (d*x² + e*y² + f)
        (A - B²) / (A + B²) where A, B are polynomials
    
    This enables discovery of rational forms equivalent to trig expressions.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_vars = X.shape[1]
    if n_vars < 2:
        return []
    
    seeds = []
    
    # Get variable names
    if variable_names and len(variable_names) >= 2:
        v0, v1 = variable_names[0], variable_names[1]
    else:
        v0, v1 = "x", "y"
    
    # Check if outputs look like simple fractions
    y_finite = y[np.isfinite(y)]
    if len(y_finite) < 5:
        return []
    
    # Count how many outputs are close to simple fractions (p/q with q <= 100)
    fraction_count = 0
    for val in y_finite[:50]:  # Sample first 50 points
        if abs(val) > 10:  # Skip large values
            continue
        # Try to find a simple fraction approximation
        for q in range(1, 101):
            prod = val * q
            # Robust extraction of real part
            try:
                if hasattr(prod, 'imag') and abs(prod.imag) > 1e-9:
                    continue
                if hasattr(prod, 'real'):
                    prod = prod.real
                p = round(prod)
            except: continue

            if abs(val - p/q) < 1e-9:
                fraction_count += 1
                break
    
    fraction_ratio = fraction_count / min(len(y_finite), 50)
    
    # If >50% of outputs are simple fractions, generate rational seeds
    if fraction_ratio > 0.5:
        if verbose:
            print(f"  -> Rational form suspected ({fraction_ratio*100:.0f}% simple fractions)")
        
        # Generate polynomial ratio seeds with INDEPENDENT coefficients
        # Pattern: (a*y² - (x²+y²-c)²) / (a*y² + (x²+y²-c)²)
        # The y² coefficient (a) and constant offset (c) are independent
        for a_coef in [4, 16]:  # y² coefficient
            for c_offset in [4, 16]:  # Constant offset in (x²+y²-c)
                A = f"{a_coef}*{v1}**2"
                B = f"({v0}**2+{v1}**2-{c_offset})"
                seeds.append(f"({A}-{B}**2)/({A}+{B}**2)")
                # Also try negated version
                seeds.append(f"({B}**2-{A})/({B}**2+{A})")
        
        # Simpler quadratic ratios
        seeds.append(f"({v0}**2-{v1}**2)/({v0}**2+{v1}**2)")  # Hyperbolic-like
        seeds.append(f"(4*{v1}**2-{v0}**2)/(4*{v1}**2+{v0}**2)")  # Scaled
        seeds.append(f"({v0}*{v1})/({v0}**2+{v1}**2)")  # Product ratio
        
        # Distance-based ratios (inspired by bipolar coordinates)
        for a in [2, 4]:
            seeds.append(f"(({v0}-{a})**2+{v1}**2-({v0}+{a})**2-{v1}**2)/(({v0}-{a})**2+{v1}**2+({v0}+{a})**2+{v1}**2)")
    
    return seeds

# v4.4 Audit Remediation: Removed 'Bipolar Poles' heuristic. 
# It was flagging as overfitting to Feynman benchmarks.
def _detect_bipolar_poles(X, y, variable_names=None, verbose=False):
    return []

def _detect_power_law_patterns(X, y, variable_names=None, verbose=False):
    """Detect simple power laws y = c * x^k via log-log analysis.
    
    Checks if log(y) / log(x) is constant -> y = x^k.
    Also handles offsets if needed (future), but primarily for x^k.
    Target case: x * sqrt(x) -> x^1.5
    """
    seeds = []
    
    # 1D only
    if X.ndim > 1 and X.shape[1] > 1: return []
    try: x_flat = X.flatten()
    except: return []
    
    # Filter for positive x, y (for log)
    mask = (x_flat > 1e-6) & (y > 1e-6)
    if np.sum(mask) < 5: return []
    
    x_valid = x_flat[mask]
    y_valid = y[mask]
    
    x_log = np.log(x_valid)
    y_log = np.log(y_valid)
    
    # Avoid division by zero where log(x) ~ 0 (i.e., x ~ 1)
    # y = x^k -> if x=1, y=1 regardless of k. So x=1 provides no info about k.
    valid_ratios_mask = np.abs(x_log) > 1e-6
    if np.sum(valid_ratios_mask) < 3: return []
    
    # 1. Simple ratio check (y = x^k)
    ratios = y_log[valid_ratios_mask] / x_log[valid_ratios_mask]
    k_mean = np.mean(ratios)
    k_std = np.std(ratios)
    
    if k_std < 1e-2: # Very consistent exponent
        if verbose:
            print(f"   Power Law Analysis: Found y = x^{k_mean:.3f} (std={k_std:.4f})")
        
        # Round to nice numbers
        k_rounded = round(k_mean, 3)
        if abs(k_mean - round(k_mean)) < 1e-3:
            k_rounded = int(round(k_mean))
        elif abs(k_mean - round(k_mean * 2) / 2) < 1e-3: # Half-integers (1.5, 2.5)
            k_rounded = round(k_mean * 2) / 2
        
        var = variable_names[0] if variable_names else "x"
        
        # Explicit seeds
        seeds.append(f"pow({var}, {k_rounded})")
        
        # Special case: 1.5 -> x * sqrt(x)
        if abs(k_rounded - 1.5) < 1e-9:
             seeds.append(f"{var} * sqrt({var})")
        
        # Special case: 0.5 -> sqrt(x)
        if abs(k_rounded - 0.5) < 1e-9:
             seeds.append(f"sqrt({var})")
             
        # SHORT-CIRCUIT: If match is near-perfect, return it immediately to bypass evolution risks
        if k_std < 1e-3:
             # Return the most "natural" form first
             if abs(k_rounded - 1.5) < 1e-9:
                 return (seeds, f"{var} * sqrt({var})")
             elif abs(k_rounded - 0.5) < 1e-9:
                 return (seeds, f"sqrt({var})")
             else:
                 return (seeds, f"pow({var}, {k_rounded})")
             
        return seeds
        
    # 2. Try Linear Regression (Robust for A*x^k with intercept issues)
    try:
        from .. import heuristics
        X_2d = X.reshape(-1, 1)
        # Check log-linear fit (y = A*x^B)
        success_ln, func_str_ln = heuristics.check_log_linear_transformations(
            X_2d, y, variable_names or ["x"]
        )
        if success_ln:
            if verbose: print(f"   Power Law Analysis (Linear Reg): Found {func_str_ln}")
            seeds.append(func_str_ln)
            # Short-circuit if strong match
            return (seeds, func_str_ln)
    except Exception:
        pass

    return seeds

def _detect_power_peeling(ctx, X, y, variable_names=None):
    """Detect if y = Base(x)^x via Rational Analysis on y^(1/x)."""
    seeds = []
    
    # Skip for multivariate data - this is a 1D-only heuristic
    if X.ndim > 1 and X.shape[1] > 1:
        return []

    # 1. Compute z = y^(1/x)
    # Filter to valid points where calculation is possible
    valid_points = []
    
    # Handle both flat arrays and (N,1) shapes
    X_flat = X.flatten()
    y_flat = y.flatten()
    
    for i in range(len(y_flat)):
        xi, yi = X_flat[i], y_flat[i]
        if abs(xi) < 1e-6: continue # Avoid 1/0
        
        try:
            # We want base = y^(1/x).
            # If y is complex, this is tricky. Complex power have multiple branches.
            # But Python's ** (pow) usually picks the principal branch.
            # base = yi ** (1.0/xi)
            # However, ((x-1)/(x+1)) might be negative, so base is negative.
            # If base is negative, base^x is complex. yi is complex.
            # yi^(1/xi) should recover base.
            
            # Suppress overflow/invalid warnings - we handle these cases gracefully
            with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
                # Check for zero to avoid divide by zero warning, although errstate might catch it
                if abs(xi) < 1e-9: continue
                zi = yi ** (1.0/xi)
            
            # Rational Analysis currently only supports Real numbers?
            # Yes, standard SVD/Rational solver expects real coefficients.
            # If base is real (like (x-1)/(x+1)), then zi should be real (or close to it).
            if isinstance(zi, complex):
                if abs(zi.imag) < 1e-4: zi = zi.real
                else: continue # Skip complex bases for now (Rational Finder might not handle them)
                
            # Stability check: reject massive values that cause overflow in variance calculation
            if np.isfinite(zi) and abs(zi) < 1e10:
                valid_points.append(((xi,), zi))
        except: continue
        
    if len(valid_points) < 5: 
        # print(f"Power Peeling: Only {len(valid_points)} valid points (need 5).") # Noise
        return []
    
    # 2. Run Rational Analysis on z
    # Local import to avoid circular dependency
    try:
        from kalkulator_pkg.function_manager import find_function_from_data
        # print(f"Power Peeling: Running Rational Analysis on {len(valid_points)} points...")
        param_var = variable_names[0] if variable_names else "x"
        success, func_str, _, note = find_function_from_data(ctx, valid_points, param_names=[param_var])
        print(f"Power Peeling Result: {success}, {func_str}, Note: {note}")
        
        if success and func_str:
            import re
            mse_match = re.search(r"MSE=([\d.eE+-]+)", str(note))
            mse = float(mse_match.group(1)) if mse_match else 1.0
            
            # Use ^ for compatibility with parser/symbolify
            base_seed = f"({func_str})^x"
            seeds.append(base_seed)
            
            # Short-Circuit if fit is very good
            if mse < 1e-9:
                return (seeds, base_seed)
        
        if success:
             # Found a rational base!
             # Return (base)**x
             return [f"({func_str})**x"]
    except ImportError:
        pass
    return seeds

def _detect_tower_patterns(X, y, variable_names=None, verbose=False, banned_operators=None):
    """Detect tower functions y = x^g(x) using check_power_peeling."""
    seeds = []
    # 1D only
    if X.ndim > 1 and X.shape[1] > 1:
        return []
        
    # Prepare data
    X_list = X.tolist() if hasattr(X, "tolist") else list(X)
    y_list = y.tolist() if hasattr(y, "tolist") else list(y)
    
    names = variable_names if variable_names else ["x"]
    
    success, expr, mse = check_power_peeling(X_list, y_list, names, verbose=verbose, banned_operators=banned_operators)
    
    if success and expr:
         seeds.append(expr)
         if mse < 1e-9:
             return ([expr], expr)
             
    return seeds

def _detect_trig_composites(y_data):
    """Detect potential deep nested trigonometric functions."""
    # Heuristic: If bounded between [-1.5, 1.5] but noisy/high-frequency
    # Relaxed bounds to catch sin(complex) which can exceed 1
    y_finite = y_data[np.isfinite(y_data)]
    if len(y_finite) == 0: return []
    
    y_min, y_max = np.min(y_finite), np.max(y_finite)
    if -1.5 < y_min < -0.5 and 0.5 < y_max < 1.5:
        # It's definitely sinusodial-ish.
        # Suggest deep nests which are hard to find randomly.
        return [
            "sin(tan(x))", "cos(tan(x))",
            "sin(cos(x))", "cos(sin(x))",
            "sin(cos(tan(x)))", "cos(sin(tan(x)))",
            "tan(sin(x))", "tan(cos(x))"
        ]
    return []

def _detect_triangle_wave(X, y, verbose=False):
    """Detect triangle wave functions: abs(x - round(x)) = distance to nearest integer.
    
    Triangle waves are piecewise linear functions that repeat with period 1.
    Common forms:
    - abs(x - round(x)) = distance to nearest integer
    - 0.5 - abs(frac(x) - 0.5) = same function, different form
    - abs(frac(x + 0.5) - 0.5) = shifted variant
    """
    seeds = []
    
    # Need X to be 1D
    if X.ndim > 1 and X.shape[1] > 1: 
        return []
    try: 
        x_flat = X.flatten()
    except: 
        return []
    
    # Filter for real, finite values (exclude complex)
    is_real_x = ~np.iscomplex(x_flat) if np.iscomplexobj(x_flat) else np.ones(len(x_flat), dtype=bool)
    is_real_y = ~np.iscomplex(y) if np.iscomplexobj(y) else np.ones(len(y), dtype=bool)
    valid_mask = np.isfinite(x_flat) & np.isfinite(y) & is_real_x & is_real_y
    if np.sum(valid_mask) < 5:
        return []
    
    x_valid = np.real(x_flat[valid_mask]).astype(float)
    y_valid = np.real(y[valid_mask]).astype(float)
    
    # Triangle wave signature checks:
    # 1. Y values should be in [0, 0.5] for standard triangle wave
    y_min, y_max = np.min(y_valid), np.max(y_valid)
    
    if verbose:
        print(f"\n[SV] TRIANGLE WAVE DETECTOR:")
        print(f"     Y range: [{y_min:.4g}, {y_max:.4g}]")
    
    # Check if bounded like triangle wave [0, 0.5] or scaled variant
    if y_min < -0.1 or y_max > 1.0:
        if verbose:
            print(f"     -> Y range outside [0, 1], not a standard triangle wave")
        return []
    
    # Test: abs(x - round(x)) - distance to nearest integer
    try:
        y_pred_triangle = np.abs(x_valid - np.round(x_valid))
        mse_triangle = np.mean((y_valid - y_pred_triangle) ** 2)
        max_err_triangle = np.max(np.abs(y_valid - y_pred_triangle))
        
        if verbose:
            print(f"     Testing abs(x - round(x)):")
            print(f"       MSE={mse_triangle:.6g}, max_err={max_err_triangle:.6g}")
        
        if max_err_triangle < 1e-6:
            if verbose:
                print(f"     -> PERFECT MATCH: abs(x - round(x))")
            seeds.append("abs(x - floor(x + 0.5))")  # Lowercase for GP engine compatibility
            seeds.append("abs(x - ceil(x - 0.5))")   # Alternative form
            return seeds
        elif mse_triangle < 0.01:
            if verbose:
                print(f"     -> Good match for triangle wave seed")
            seeds.append("abs(x - floor(x + 0.5))")
    except Exception as e:
        if verbose:
            print(f"     -> Error testing triangle: {e}")
    
    # Test: 0.5 - abs(frac(x) - 0.5) = alternative triangle wave form
    try:
        frac_x = x_valid - np.floor(x_valid)
        y_pred_frac = 0.5 - np.abs(frac_x - 0.5)
        mse_frac = np.mean((y_valid - y_pred_frac) ** 2)
        max_err_frac = np.max(np.abs(y_valid - y_pred_frac))
        
        if verbose:
            print(f"     Testing 0.5 - abs(frac(x) - 0.5):")
            print(f"       MSE={mse_frac:.6g}, max_err={max_err_frac:.6g}")
        
        if max_err_frac < 1e-6:
            if verbose:
                print(f"     -> PERFECT MATCH: 0.5 - abs(frac(x) - 0.5)")
            seeds.append("0.5 - abs(frac(x) - 0.5)")
            return seeds
    except Exception as e:
        if verbose:
            print(f"     -> Error testing frac form: {e}")
    
    # Test: sawtooth wave frac(x) or x - floor(x)
    try:
        y_pred_sawtooth = frac_x
        mse_sawtooth = np.mean((y_valid - y_pred_sawtooth) ** 2)
        max_err_sawtooth = np.max(np.abs(y_valid - y_pred_sawtooth))
        
        if verbose:
            print(f"     Testing frac(x) = x - floor(x):")
            print(f"       MSE={mse_sawtooth:.6g}, max_err={max_err_sawtooth:.6g}")
        
        if max_err_sawtooth < 1e-6:
            if verbose:
                print(f"     -> PERFECT MATCH: frac(x)")
            seeds.append("frac(x)")
            seeds.append("x - floor(x)")
            return seeds
    except:
        pass
    
    return seeds

def _detect_floor_wave_patterns(X, y, variable_names=None, verbose=False):
    """Detect quantized sine waves: floor(A * sin(w*x)) or ceil(A * sin(w*x)).
    
    Heuristic based on User Feedback:
    1.  Discrete Integer Values: The function output jumps between integers.
    2.  Asymmetry: floor(A*sin(x)) has range [-A, A-1]. ceil has [-A+1, A].
    3.  Frequency: Underlying periodic signal.
    """
    seeds = []
    
    # 1. Basic Input Checks
    if X.ndim > 1 and X.shape[1] > 1: return []
    try: x_flat = X.flatten()
    except: return []
    
    # Filter for valid finite data
    mask = np.isfinite(x_flat) & np.isfinite(y)
    if np.sum(mask) < 10: return []
    
    x_valid = x_flat[mask]
    y_valid = y[mask]
    
    # 2. Integer Check
    # Check if mostly integers (allow small noise)
    y_round = np.round(y_valid)
    mse_int = np.mean((y_valid - y_round)**2)
    if mse_int > 0.01:
        # Not integers, maybe scaled? 
        return []

    # 3. Range & Asymmetry Analysis
    y_max = np.max(y_round)
    y_min = np.min(y_round)
    y_range = y_max - y_min
    
    # Estimate Amplitude A
    est_A_floor = -y_min
    est_A_ceil = y_max

    var = variable_names[0] if variable_names else "x"

    candidates = []
    
    if abs(y_max - (est_A_floor - 1)) < 0.5:
        candidates.append(("floor", est_A_floor))
    
    # Check Ceil Hypothesis (Min = -A+1, Max = A)
    if abs(y_min - (-est_A_ceil + 1)) < 0.5:
        candidates.append(("ceil", est_A_ceil))
        
    # Check Round Hypothesis (Symmetric: [-A, A])
    if abs(est_A_floor - y_max) < 0.5 and abs(est_A_floor + y_min) < 0.5:
        candidates.append(("round", est_A_floor))

    if not candidates:
        return []

    # 4. Frequency Detection
    # Using zero crossings on the discrete data might be noisy, but let's try.
    # Helper: try to find 'k' such that sign(sin(k*x)) matches sign(y)
    
    from ..heuristics import detect_frequency
    freqs = detect_frequency(x_valid, y_valid)
    if not freqs:
        freqs = [1.0] # Default
        
    # 5. Validation
    for func_type, A in candidates:
        for w in freqs + [1.0]: # Always try w=1
            
            # Construct candidate y
            try:
                # Use simple python math for vector ops
                arg = A * np.sin(w * x_valid)
                
                if func_type == "floor":
                    y_pred = np.floor(arg)
                elif func_type == "ceil":
                    y_pred = np.ceil(arg)
                elif func_type == "round":
                    y_pred = np.round(arg)
                else:
                    continue
                    
                # Check match
                # Allow slight mismatches due to sampling at transition points
                diff = np.abs(y_pred - y_valid)
                match_ratio = np.sum(diff < 0.1) / len(diff)
                
                if match_ratio > 0.9: # 90% match
                    w_str = f"*{w}" if abs(w - 1.0) > 1e-6 else ""
                    seed = f"{func_type}({A}*sin({var}{w_str}))"
                    if seed not in seeds:
                        seeds.append(seed)
                        if verbose: print(f"   [Forensic] Floor Wave Exact Match: {seed} (acc={match_ratio:.2%})")
            except:
                pass
                
    return seeds
    
def _detect_general_staircase(X, y, variable_names=None, verbose=False):
    """Detect general staircase functions: floor((x+c)/k), ceil(mx), etc.
    
    Algorithm:
    1. Step Analysis: Group constant Y values.
    2. Slope Calculation: Rate of change between steps (Steps/X).
    3. Rounding Check: Validates floor/ceil/round against linear trend.
    """
    seeds = []
    
    # 1D only
    if X.ndim > 1 and X.shape[1] > 1: return []
    try: x_flat = X.flatten()
    except: return []

    # Valid data only
    mask = np.isfinite(x_flat) & np.isfinite(y)
    x_valid = x_flat[mask]
    y_valid = y[mask]
    
    if len(x_valid) < 6: return []

    # Sort
    idx = np.argsort(x_valid)
    x_sorted = x_valid[idx]
    y_sorted = y_valid[idx]

    # 1. Step Analysis
    # Get unique Y values to identify "Plateaus"
    # Round Y to handle minor noise
    y_rounded = np.round(y_sorted, 2) 
    unique_y, indices = np.unique(y_rounded, return_index=True)
    
    # If Y changes too frequently (like x), it's not a staircase
    # Staircase has few unique Y values relative to N samples? Not necessarily.
    # Staircase steps can be short (1 sample).
    # Better metric: Discrete Jumps.
    
    # Check if Y is mostly integers (primary target for floor/ceil)
    if np.mean(np.abs(y_sorted - np.round(y_sorted)) < 1e-3) < 0.9:
        return [] # Not integers, probably not floor/ceil

    # Calculate Slope 'm' from overall trend (Robust LinearFit)
    try:
        coeffs = np.polyfit(x_sorted, y_sorted, 1)
        m, c = coeffs
        
        # Refine slope to likely ratios (1, 0.5, 0.33, 2, 3...)
        # Invert slope to find 'k' in x/k
        if abs(m) > 1e-6:
            k = 1.0 / m
            # If k is close to integer?
            k_int = round(k)
            if abs(k - k_int) < 0.1:
                m_refined = 1.0 / k_int
                type_str = f"x/{k_int}"
            else:
                m_int = round(m, 2)
                m_refined = m_int
                type_str = f"{m_int}*x"
        else:
            return []

        if verbose:
            print(f"   [Staircase] Trend m={m:.3f}, suggests {type_str}")

        # Test Hypotheses
        # y = floor(m*x + c)
        # y = ceil(m*x + c)
        # y = round(m*x + c)
        
        # Optimize 'c' for each hypothesis
        # For floor(m*x + c) = y, we need m*x + c >= y AND m*x + c < y + 1
        # c >= y - m*x  AND  c < y + 1 - m*x
        # We can scan a few c values around the linear intercept
        
        var = variable_names[0] if variable_names else "x"
        
        # Simplified hypothesis testing:
        # Construct line with refined slope
        trend = m_refined * x_sorted
        
        best_match = None
        best_acc = 0
        
        # Test offsets for 'c'
        # Often c is 0, 0.5, 1, -0.5
        for c_test in [0, 0.5, -0.5, 1, -1, 0.1, c]:
            linear = trend + c_test
            
            # Floor
            acc_fl = np.mean(np.abs(np.floor(linear) - y_sorted) < 0.1)
            if acc_fl > best_acc:
                best_acc = acc_fl
                base_expr = f"{m_refined}*{var} + {c_test}" if c_test != 0 else f"{m_refined}*{var}"
                func_name = "floor"
                
            # Ceil
            acc_cl = np.mean(np.abs(np.ceil(linear) - y_sorted) < 0.1)
            if acc_cl > best_acc:
                best_acc = acc_cl
                base_expr = f"{m_refined}*{var} + {c_test}" if c_test != 0 else f"{m_refined}*{var}"
                func_name = "ceil"
                
            # Round
            acc_rd = np.mean(np.abs(np.round(linear) - y_sorted) < 0.1)
            if acc_rd > best_acc:
                best_acc = acc_rd
                base_expr = f"{m_refined}*{var} + {c_test}" if c_test != 0 else f"{m_refined}*{var}"
                func_name = "round"

        if best_acc > 0.9: # 90% Match
            # Cleanup expression: 0.5*x -> x/2
            if abs(m_refined - 0.5) < 1e-9:
                base_expr = base_expr.replace("0.5*x", "x/2")
                base_expr = base_expr.replace("0.5*", "/2") # General check
            elif abs(m_refined - 1.0/3) < 1e-9:
                # 0.3333*x -> x/3
                pass

            seed = f"{func_name}({base_expr})"
            
            # Special Case: Floor Div // 
            # If floor(x/k), suggest floor(x/k) AND (x//k) if parser supported it
            # For now, floor(x/k) is the target.
            
            seeds.append(seed)
            if verbose:
                print(f"   [Staircase] MATCH: {seed} (acc={best_acc:.2%})")

    except Exception as e:
        if verbose: print(f"   [Staircase] Error: {e}")

    return seeds

def _detect_rapid_growth_poly(X, y, verbose=False):
    """Detect high-degree polynomials like 1 - x^8 via log-slope analysis."""
    seeds = []
    
    # 2025-01-16 Fix: Handle multivariate data (prevent broadcast error)
    # This heuristic is designed for univariate analysis.
    if X.ndim > 1 and X.shape[1] > 1:
        return []

    # Filter for large values where x^n dominates
    mask = (np.abs(y) > 100) & (np.abs(X.flatten()) > 1.5)
    if np.sum(mask) < 3: return []
    
    x_large = X.flatten()[mask]
    y_large = y[mask]
    
    # Check y ~ x^n
    # log|y| ~ n * log|x|
    try:
        log_x = np.log(np.abs(x_large))
        log_y = np.log(np.abs(y_large))
        n_est = np.median(log_y / log_x)
        
        if verbose: print(f"   Rapid Growth: estimated degree {n_est:.2f}")
        
        # Check if close to integer
        if abs(n_est - round(n_est)) < 0.1 and round(n_est) > 3:
            n = int(round(n_est))
            # Determine sign/offset by checking sign of y vs x^n
            # x^n vs y
            y_pred = x_large ** n
            
            # Check 1 - x^n
            if np.median(np.abs((1 - y_pred) - y_large)) < 1e-3 * np.median(np.abs(y_large)):
                 seeds.append(f"1 - x^{n}")
                 
            # Check x^n - 1
            if np.median(np.abs((y_pred - 1) - y_large)) < 1e-3 * np.median(np.abs(y_large)):
                 seeds.append(f"x^{n} - 1")
                 
            # Check -x^n
            if np.median(np.abs((-y_pred) - y_large)) < 1e-3 * np.median(np.abs(y_large)):
                 seeds.append(f"-x^{n}")
    except Exception:
        # Power peeling heuristic failed, ignore
        pass
    
    return seeds

def _detect_symmetry_pole(X, y, variable_names=None, verbose=False):
    """Detect points of symmetry which often indicate poles or centers."""
    if X.ndim > 1 and X.shape[1] > 1: return []
    
    seeds = []
    
    # Sort data
    xy = sorted(zip(X.flatten(), y), key=lambda p: p[0])
    xs = np.array([p[0] for p in xy])
    ys = np.array([p[1] for p in xy])
    
    # Filter out complex values
    xs_real = []
    ys_real = []
    for x_val, y_val in zip(xs, ys):
        if np.iscomplex(x_val) or np.iscomplex(y_val):
            if abs(np.imag(x_val)) < 1e-9 and abs(np.imag(y_val)) < 1e-9:
                xs_real.append(float(np.real(x_val)))
                ys_real.append(float(np.real(y_val)))
        else:
            # Use np.real to avoid ComplexWarning even if value is incidentally complex type
            xs_real.append(float(np.real(x_val)))
            ys_real.append(float(np.real(y_val)))
            
    if len(xs_real) < 4: return []
    
    xs = np.array(xs_real)
    ys = np.array(ys_real)
    
    n_points = len(xs)
    
    # Check each integer and half-integer in range as candidate center
    x_min, x_max = np.min(xs), np.max(xs)
    candidates = []
    for c in np.arange(np.ceil(x_min), np.floor(x_max) + 0.1, 0.5):
        candidates.append(c)
        
    var = variable_names[0] if variable_names else "x"
    
    for c in candidates:
        # Check odd symmetry around c: f(c+h) = -f(c-h)
        # Find pairs (x1, x2) such that (x1+x2)/2 ≈ c
        
        errors = []
        count = 0
        
        for i in range(n_points):
            x1 = xs[i]
            if x1 >= c: break # Only check left side
            
            # Find mirror point x2 ≈ 2c - x1
            target_x2 = 2*c - x1
            
            # Find closest actual point
            idx2 = np.argmin(np.abs(xs - target_x2))
            x2 = xs[idx2]
            
            if abs(x2 - target_x2) < 1e-4:
                y1, y2 = ys[i], ys[idx2]
                
                # Odd symmetry: y1 + y2 ≈ 0
                errors.append(abs(y1 + y2))
                count += 1
        
        if count > 2:
            mean_sym_err = np.mean(errors)
            # If symmetry error is small relative to magnitude
            y_mag = np.mean(np.abs(ys))
            if mean_sym_err < 0.05 * y_mag:
                if verbose: print(f"   Symmetry Analysis: Found ODD symmetry check at {c} (err={mean_sym_err:.4f})")
                
                c_str = str(float(c))
                seeds.append(f"1/({var}-locked({c_str}))")
                seeds.append(f"1/({var}-locked({c_str}))**2")
    
    return seeds

def _detect_zero_patterns(X, y, variable_names=None, verbose=False):
    """Detect periodicity via zero analysis (zeros at k*pi)."""
    if X.ndim > 1 and X.shape[1] > 1: return []
    try: x_flat = X.flatten()
    except: return []

    # Filter for zeros
    # Use localized tolerance based on y range or simple epsilon
    # For user case: f(pi, 0.006)=0 => 0.006 might be y value? 
    # No, f(pi, y) = sin(pi)*y = 0.
    # So we look for points where output is exactly 0 or very close.
    zero_mask = (np.abs(y) < 1e-3) & np.isfinite(x_flat)
    if np.sum(zero_mask) < 2: return []
    
    zeros_x = x_flat[zero_mask]
    var = variable_names[0] if variable_names else "x"
    seeds = []
    
    if verbose:
        print(f"   [Forensic] Analyzing {len(zeros_x)} zeros for periodicity...")
    
    # Check for PI multiples (sin(x))
    # x / pi should be integer
    x_div_pi = zeros_x / np.pi
    # Allow some noise
    matches_sin = np.abs(x_div_pi - np.round(x_div_pi)) < 0.05
    match_rate_sin = np.mean(matches_sin)
    
    if match_rate_sin > 0.6: # Relaxed threshold as some zeros might be from other factors (like y=0)
       if verbose: print(f"   [Forensic] Zeros match k*pi ({match_rate_sin:.0%}) -> sin({var})")
       seeds.append(f"sin({var})")
       
    # Check for PI/2 + k*PI (cos(x))
    # (x / pi) - 0.5 should be integer
    matches_cos = np.abs((x_div_pi - 0.5) - np.round(x_div_pi - 0.5)) < 0.05
    match_rate_cos = np.mean(matches_cos)
    
    if match_rate_cos > 0.6:
       if verbose: print(f"   [Forensic] Zeros match k*pi + pi/2 ({match_rate_cos:.0%}) -> cos({var})")
       seeds.append(f"cos({var})")
    
    return seeds


def _detect_complex_offset(X, y, variable_names=None, verbose=False):
    """Detect functions like 7 + sqrt(x) via complex residue analysis.
    
    If inputs are real/complex mixture, and complex outputs appearing for negative inputs
    share a constant REAL part (e.g., 7 + 3j), we can peel off the 7.
    """
    seeds = []
    
    y = np.array(y)
    
    # Check for complex values
    if not np.iscomplexobj(y): 
        return []
        
    complex_mask = np.abs(np.imag(y)) > 1e-9
    
    if np.sum(complex_mask) < 2:
        return []
        
    y_complex = y[complex_mask]
    
    # 1. Check if Real parts are constant
    real_parts = np.real(y_complex)
    mean_real = np.mean(real_parts)
    std_real = np.std(real_parts)
    
    # Tolerance relative to magnitude
    if std_real < 1e-5: # Tight tolerance for constant offset
        offset = mean_real
        if verbose:
             print(f"   [Forensic] Complex Offset Detected: Const Real Part = {offset:.4g}")
             
        # Suggest simple offset seeds
        # We don't solve the residue yet (leave that to evolution/SVD), just hint the structure
        seeds.append(f"sqrt(x) + {offset}")
        seeds.append(f"{offset} + sqrt(abs(x))")
        
    return seeds

def _detect_bitwise_patterns(X, y, variable_names=None, verbose=False):
    """Detect simple bitwise patterns like x^k for integers."""
    # Placeholder for Bitwise XOR detection (referenced in generate_pattern_seeds)
    return []

def _detect_fibonacci_patterns(X, y, variable_names=None, verbose=False):
    """Detect Fibonacci sequence patterns."""
    # Placeholder for Fibonacci detection (referenced in generate_pattern_seeds)
    return []

def _detect_symmetry(X, y, variable_names=None):
    """
    Detects if f(args) is invariant under permutation and suggests max/min/median.
    Optimized O(N log N) using Sort & Group.
    """
    if X.ndim < 2 or X.shape[1] < 2:
        return []

    # 1. Fast Symmetry Check via Sorting
    # Round inputs to handle float noise during sorting
    X_sig = np.round(np.sort(X, axis=1), 6)
    
    # Find duplicate signatures (permutations)
    unique_sigs, inverse_indices = np.unique(X_sig, axis=0, return_inverse=True)
    
    symmetry_violation = 0.0
    valid_groups = 0
    
    for i in range(len(unique_sigs)):
        # Get indices of all rows belonging to this permutation group
        group_indices = np.where(inverse_indices == i)[0]
        
        if len(group_indices) > 1:
            # Check if outputs are identical
            y_group = y[group_indices]
            variance = np.var(y_group)
            symmetry_violation += variance
            valid_groups += 1
            
    # If no permutations found, we can't prove symmetry (or lack thereof) from data
    if valid_groups == 0:
        return []

    seeds = []
    # 2. If Symmetric (Low Violation), Test Candidates
    if symmetry_violation < 1e-5:
        var_str = ", ".join(variable_names) if variable_names else ",".join([f"x{i}" for i in range(X.shape[1])])
        
        # Helper R2
        def _quick_r2(y_true, y_pred):
            res_ss = np.sum((y_true - y_pred)**2)
            tot_ss = np.sum((y_true - np.mean(y_true))**2)
            return 1 - res_ss / (tot_ss + 1e-10)

        # Test MAX
        y_max = np.max(X, axis=1)
        if _quick_r2(y, y_max) > 0.99:
            seeds.append(f"max({var_str})")
            
        # Test MIN
        y_min = np.min(X, axis=1)
        if _quick_r2(y, y_min) > 0.99:
            seeds.append(f"min({var_str})")
        
        # Test MEDIAN
        if X.shape[1] >= 3:
             y_med = np.median(X, axis=1)
             if _quick_r2(y, y_med) > 0.99:
                 # Seed as median(vars) - hope user has it or we define it later?
                 # Or use composition: median3(a,b,c) = sum - min - max
                 # But sticking to atomic seed is safer if we add the op.
                 # If op missing, it might parse error.
                 # But let's add it. `genetic_engine` parser needs it defined.
                 # `expression_tree.py` handles ops.
                 # For now, let's trust the user's snippet.
                 # But if 'median' isn't in genetic_config, it's useless complexity.
                 # I'll add 'median' to the seed.
                 seeds.append(f"median({var_str})")
                 
    return seeds


def generate_pattern_seeds(ctx, X, y, variable_names=None, verbose=False, banned_operators=None):
    """Detect patterns in data and return seed expression strings."""
    t0 = time.perf_counter()
    FORENSIC_TIMEOUT = 10.0  # seconds — hard cap to prevent hanging
    seeds = []
    pole_seeds = []  # Agent Handoff: Initialize early to prevent NameError at line 2440
    
    def _check_timeout():
        """Return True if forensic analysis has exceeded its time budget."""
        return time.perf_counter() - t0 > FORENSIC_TIMEOUT
    
    # Ensure X is 2D
    X = np.array(X)
    if X.ndim == 1: X = X.reshape(-1, 1)

    # Helper to check if array is object-like string/mixed
    def _is_object_like(arr):
        return arr.dtype == object or np.issubdtype(arr.dtype, np.str_) or np.issubdtype(arr.dtype, np.unicode_)

    # Safe conversion to complex/float to avoid object arrays (which trip isfinite)
    try:
        # Initial array conversion if list
        if isinstance(y, list) or isinstance(y, tuple):
            y = np.array(y)
            
        # Try converting to complex first (handles both real and complex)
        if _is_object_like(y):
             y_clean = []
             for val in y.flatten():
                 try:
                     y_clean.append(complex(val))
                 except:
                     y_clean.append(np.nan)
             y = np.array(y_clean)
        else:
             y = np.asanyarray(y, dtype=complex)
             
        # Same for X (already np.array but might be object)
        if _is_object_like(X):
             X_clean = []
             for row in X:
                 row_clean = []
                 for val in row:
                     try:
                         row_clean.append(complex(val))
                     except:
                         row_clean.append(np.nan)
                 X_clean.append(row_clean)
             X = np.array(X_clean)
        else:
             X = np.asanyarray(X, dtype=complex)

        # Downgrade to float if purely real (cleaner for some heuristics)
        if np.all(np.abs(np.imag(y)) < 1e-15):
             y = np.real(y)
        if np.all(np.abs(np.imag(X)) < 1e-15):
             X = np.real(X)
             
    
    except Exception as e:
        if verbose: print(f"[Forensic] Input conversion warning: {e}")
        # Fallback to original, might crash later but we tried
        pass
    
    if verbose: print(f"[Forensic] Input X.dtype={X.dtype}, y.dtype={y.dtype}")

    n_vars = X.shape[1]
    derived_vars = variable_names if variable_names and len(variable_names) == n_vars else [f"x{k}" for k in range(n_vars)]
    var = derived_vars[0]

    # 0. TRIVIAL IDENTITY CHECK: f(x) = x
    # This must come first to prevent overly complex detectors from matching simple linear functions
    try:
        y_arr = np.array(y).flatten()
        x_arr = X[:, 0].flatten() if X.ndim > 1 else X.flatten()
        identity_err = np.max(np.abs(y_arr - x_arr))
        if identity_err < 1e-9:
            if verbose:
                print(f"  -> Trivial Identity Detected: f({var}) = {var}")
            return ([var], var)
    except Exception:
        pass  # Continue to other detectors

    # 0.05 SYMMETRY CHECK (max, min, median)
    # Check early to avoid linear cheats (sum/2) for symmetric functions
    try:
        sym_seeds = _detect_symmetry(X, y, variable_names=derived_vars)
        if sym_seeds:
            if verbose: print(f"  -> Symmetry Detected: {sym_seeds}")
            # If we found it with R2 > 0.99 (implied by _detect_symmetry), 
            # return it as a strong candidate (Exact Match behavior)
            return (sym_seeds, sym_seeds[0])
    except Exception as e:
        if verbose: print(f"  -> Symmetry detection error: {e}")

    # 0.1 NEW: CHECK POWER PEELING (Tower Functions)
    # Detects x^x, x^sqrt(x), etc.
    try:
         t_seeds = _detect_tower_patterns(X, y, variable_names=derived_vars, verbose=verbose, banned_operators=banned_operators)
         # Returns tuple (seeds, exact_match) or list
         if isinstance(t_seeds, tuple) and t_seeds[1]:
              if verbose: print(f"  -> Tower Pattern Exact Match: {t_seeds[1]}")
              return ([t_seeds[1]], t_seeds[1])
         elif isinstance(t_seeds, list):
              seeds.extend(t_seeds)
         elif isinstance(t_seeds, tuple):
              seeds.extend(t_seeds[0])
    except Exception:
         pass

    # 0.2 NEW: CHECK COMPLEX OFFSET (7 + sqrt(x))
    try:
        offset_seeds = _detect_complex_offset(X, y, variable_names=derived_vars, verbose=verbose)
        if offset_seeds:
            seeds.extend(offset_seeds)
    except Exception:
        pass

    # 0.5. PRODUCT INTERACTION CHECK (NEW: sin(x)*y etc.)
    # Explicitly seed separable combinations which are common but hard to evolve
    if len(derived_vars) > 1:
        import itertools
        if verbose: print(f"  -> Testing {len(derived_vars)} variables for interactions...")
        
        # 1. Simple Products: x*y, x*y*z
        # 2. Trig Products: sin(x)*y, cos(x)*y
        for v1, v2 in itertools.combinations(derived_vars, 2):
            seeds.append(f"{v1} * {v2}")
            seeds.append(f"{v1} / {v2}")
            seeds.append(f"{v1} + {v2}") # Linear combination
            
            # Trig interactions
            seeds.append(f"sin({v1}) * {v2}")
            seeds.append(f"cos({v1}) * {v2}")
            seeds.append(f"{v1} * sin({v2})")
            seeds.append(f"{v1} * cos({v2})")
            
            # Exponential interactions
            seeds.append(f"exp({v1}) * {v2}")
            seeds.append(f"{v1} * exp({v2})")

            # Trig Rotations (Gemini Theory: sin(x+y) etc.)
            seeds.append(f"sin({v1} + {v2})")
            seeds.append(f"cos({v1} + {v2})")
            seeds.append(f"sin({v1} - {v2})")
            seeds.append(f"cos({v1} - {v2})")
            
        if verbose: print(f"  -> Added {len(seeds)} interaction seeds")

    # 0.5. MONOMIAL PATTERN CHECK (C * x^a * y^b)
    # Detect f(x,y) = C * x^a * y^b using log-linear regression
    if len(derived_vars) >= 1:
        try:
            from .monomial_heuristic import detect_monomial_structure
            
            monomial_seeds = detect_monomial_structure(
                X, y, derived_vars, verbose=verbose
            )
            if monomial_seeds:
                if verbose: print(f"[Monomial] Detected: {monomial_seeds}")
                seeds.extend(monomial_seeds)
        except Exception as e:
            if verbose: print(f"[Monomial] Error: {e}")

    # 0.6. VARIABLE SEPARABILITY CHECK (NEW: sqrt(x) + sqrt(y))
    # Detect f(x,y) = g(x) + h(y) using zero-point and slicing techniques
    if len(derived_vars) >= 2:
        try:
            from .slicer_heuristic import detect_separable_structure
            from ..function_manager import find_function_from_data
            
            # Wrapper to adapt find_function_from_data to expected signature
            def _find_wrapper(X_1d, y_1d, var_names):
                try:
                    result = find_function_from_data(X_1d, y_1d, var_names)
                    if result and 'expression' in result:
                        return {'expression': result['expression'], 'mse': result.get('mse', 0.1)}
                except Exception:
                    pass
                return None
            
            slicer_seeds = detect_separable_structure(
                X, y, derived_vars, _find_wrapper, verbose=verbose
            )
            if slicer_seeds:
                if verbose: print(f"[Slicer] Detected separable structure: {slicer_seeds}")
                seeds.extend(slicer_seeds)
        except Exception as e:
            if verbose: print(f"[Slicer] Error: {e}")

    # 0.7. POWER LAW PATTERN CHECK (x^g(y))
    # Detect f(x,y) = x^g(y) where g(y) is like sqrt(y)/2
    if len(derived_vars) == 2:
        try:
            from .power_law_heuristic import detect_power_law_structure
            
            power_seeds = detect_power_law_structure(
                X, y, derived_vars, verbose=verbose
            )
            if power_seeds:
                if verbose: print(f"[PowerLaw] Detected: {power_seeds}")
                seeds.extend(power_seeds)
        except Exception as e:
            if verbose: print(f"[PowerLaw] Error: {e}")

    # 1. Step Function
    if _check_timeout():
        if verbose: print("[Forensic] Timeout reached after early detectors, returning collected seeds.")
        return seeds if seeds else []

    step_patterns = _detect_step_patterns(X, y, variable_names=derived_vars, verbose=verbose)
    if verbose: print("[DEBUG] Finished _detect_step_patterns")
    if verbose: print("[DEBUG] Finished _detect_step_patterns")
    if step_patterns: return (step_patterns, step_patterns[0]) # Match return signature
    
    # 1.1. Zero Pattern Analysis (NEW: sin(x) via zeros at k*pi)
    try:
        zero_patterns = _detect_zero_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if zero_patterns:
            if verbose: print(f"  -> Zero Patterns found: {zero_patterns}")
            seeds.extend(zero_patterns)
    except Exception as e:
        if verbose: print(f"  -> Zero Pattern detection error: {e}")
    
    # 1.2. Factorial Detection (Moved to Top for Priority)
    # Check early as it is specific and unlikely to false positive
    try:
        fact_result = _detect_factorial_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if fact_result:
            if isinstance(fact_result, tuple):
                if verbose: print(f"  -> Factorial EXACT MATCH: {fact_result[1]}")
                return fact_result
            if verbose: print(f"  -> Factorial found: {fact_result}")
            seeds.extend(fact_result)
    except Exception as e:
        if verbose: print(f"  -> Factorial detection error: {e}")
    
    
    # 1.5. Scalloped Staircase Detection (NEW: floor(x) + frac(x)^k)
    # Check this early since it's a very specific pattern
    try:
        scalloped_result = _detect_scalloped_staircase(X, y, variable_names=derived_vars, verbose=verbose)
        if scalloped_result:
            # Check for short-circuit tuple (seeds, best_match)
            if isinstance(scalloped_result, tuple):
                if verbose: print(f"  -> Scalloped Staircase EXACT MATCH: {scalloped_result[1]}")
                return scalloped_result
            if verbose: print(f"  -> Scalloped Staircase found: {scalloped_result}")
            seeds.extend(scalloped_result)
            
        # 1.5.1 General Staircase (Linear -> Rounding)
        general_staircase = _detect_general_staircase(X, y, variable_names=derived_vars, verbose=verbose)
        if general_staircase:
            if verbose: print(f"  -> General Staircase found: {general_staircase}")
            seeds.extend(general_staircase)
            
    except Exception as e:
        if verbose: print(f"  -> Scalloped Staircase detection error: {e}")
    
    # 1.6. Bitwise XOR Detection (NEW: int(x) ^ k)
    # Check early since it's a very specific integer pattern
    try:
        bitwise_result = _detect_bitwise_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if bitwise_result:
            # Check for short-circuit tuple (seeds, best_match)
            if isinstance(bitwise_result, tuple):
                if verbose: print(f"  -> Bitwise XOR EXACT MATCH: {bitwise_result[1]}")
                return bitwise_result
            if verbose: print(f"  -> Bitwise XOR found: {bitwise_result}")
            seeds.extend(bitwise_result)
    except Exception as e:
        if verbose: print(f"  -> Bitwise XOR detection error: {e}")
    
    # 1.7. Fibonacci Sequence Detection (NEW: fibonacci(x))
    # Check early since it's a very specific integer recurrence pattern
    try:
        fib_result = _detect_fibonacci_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if fib_result:
            # Check for short-circuit tuple (seeds, best_match)
            if isinstance(fib_result, tuple):
                if verbose: print(f"  -> Fibonacci EXACT MATCH: {fib_result[1]}")
                return fib_result
            if verbose: print(f"  -> Fibonacci found: {fib_result}")
            seeds.extend(fib_result)
    except Exception as e:
        if verbose: print(f"  -> Fibonacci detection error: {e}")
    
    # 1.8. General Linear Recurrence Detection (Lucas, Tribonacci, Pell, etc.)
    # Detects any sequence of the form a(n) = c1*a(n-1) + c2*a(n-2) + ...
    if _check_timeout():
        if verbose: print("[Forensic] Timeout reached after integer detectors, returning collected seeds.")
        return seeds if seeds else []
    try:
        recurrence_result = _detect_linear_recurrence(X, y, variable_names=derived_vars, verbose=verbose)
        if recurrence_result:
            # Check for short-circuit tuple (seeds, best_match)
            if isinstance(recurrence_result, tuple):
                if verbose: print(f"  -> Linear Recurrence EXACT MATCH: {recurrence_result[1]}")
                return recurrence_result
            if verbose: print(f"  -> Linear Recurrence found: {recurrence_result}")
            seeds.extend(recurrence_result)
    except Exception as e:
        if verbose: print(f"  -> Linear Recurrence detection error: {e}")
    
    # 1.9. Self-Power Detection (NEW: x^x)
    try:
        self_power_result = _detect_self_power(X, y, variable_names=derived_vars, verbose=verbose)
        if self_power_result:
            if isinstance(self_power_result, tuple):
                if verbose: print(f"  -> Self-Power EXACT MATCH: {self_power_result[1]}")
                return self_power_result
            if verbose: print(f"  -> Self-Power found: {self_power_result}")
            seeds.extend(self_power_result)
    except Exception as e:
        if verbose: print(f"  -> Self-Power detection error: {e}")
    
    # 1.9.5. Inverse Self-Power Detection (NEW: y^y = x -> exp(W(log(x))))
    try:
        inv_self_power_result = _detect_inverse_self_power(X, y, variable_names=derived_vars, verbose=verbose)
        if inv_self_power_result:
             if isinstance(inv_self_power_result, tuple):
                 if verbose: print(f"  -> Inverse Self-Power EXACT MATCH: {inv_self_power_result[1]}")
                 return inv_self_power_result
             if verbose: print(f"  -> Inverse Self-Power found: {inv_self_power_result}")
             seeds.extend(inv_self_power_result)
    except Exception as e:
        if verbose: print(f"  -> Inverse Self-Power detection error: {e}")
    
    # 1.10. Modulo Detection (NEW: x % k)
    try:
        modulo_result = _detect_modulo_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if modulo_result:
            if isinstance(modulo_result, tuple):
                
                return modulo_result
            
            seeds.extend(modulo_result)
    except Exception as e:
        if verbose: print(f"  -> Modulo detection error: {e}")
    
    # 1.11. Signum Detection (NEW: sign(x))
    try:
        signum_result = _detect_signum_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if signum_result:
            if isinstance(signum_result, tuple):
                
                return signum_result
            
            seeds.extend(signum_result)
    except Exception as e:
        if verbose: print(f"  -> Signum detection error: {e}")
    
    # 1.12. ReLU Detection (NEW: max(0, x))
    try:
        relu_result = _detect_relu_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if relu_result:
            if isinstance(relu_result, tuple):
                
                return relu_result
            
            seeds.extend(relu_result)
    except Exception as e:
        if verbose: print(f"  -> ReLU detection error: {e}")
    
    # 1.13. Clamp Detection (NEW: min(x, c) or max(a, x))
    try:
        clamp_result = _detect_clamp_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if clamp_result:
            if isinstance(clamp_result, tuple):
                
                return clamp_result
            
            seeds.extend(clamp_result)
    except Exception as e:
        if verbose: print(f"  -> Clamp detection error: {e}")
    
    # 1.14. Pulse Detection (NEW: Heaviside(x-a) - Heaviside(x-b))
    try:
        pulse_result = _detect_pulse_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if pulse_result:
            if isinstance(pulse_result, tuple):
                
                return pulse_result
            if verbose: print(f"  -> Pulse found: {pulse_result}")
            seeds.extend(pulse_result)
    except Exception as e:
        if verbose: print(f"  -> Pulse detection error: {e}")

    # 1.15. Floor/Ceil Wave Detection (NEW: floor(A * sin(x)))
    # Based on user feedback about asymmetric integer ranges
    try:
        floor_wave_result = _detect_floor_wave_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if floor_wave_result:
            if isinstance(floor_wave_result, tuple):
                if verbose: print(f"  -> Floor Wave EXACT MATCH: {floor_wave_result[1]}")
                return floor_wave_result
            if verbose: print(f"  -> Floor Wave found: {floor_wave_result}")
            seeds.extend(floor_wave_result)
    except Exception as e:
        if verbose: print(f"  -> Floor Wave detection error: {e}")
    


    
    # 1.15. Prime Counting Detection (NEW: π(x))
    try:
        prime_result = _detect_prime_counting_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if prime_result:
            if isinstance(prime_result, tuple):
                if verbose: print(f"  -> Prime Counting EXACT MATCH: {prime_result[1]}")
                return prime_result
            if verbose: print(f"  -> Prime Counting found: {prime_result}")
            seeds.extend(prime_result)
    except Exception as e:
        if verbose: print(f"  -> Prime Counting detection error: {e}")
    
    # 1.16. Anchor Pattern Detection (NEW: (x+c)^(1/x))
    try:
        anchor_result = _detect_anchor_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if anchor_result:
            if isinstance(anchor_result, tuple):
                if verbose: print(f"  -> Anchor EXACT MATCH: {anchor_result[1]}")
                return anchor_result
            if verbose: print(f"  -> Anchor found: {anchor_result}")
            seeds.extend(anchor_result)
    except Exception as e:
        if verbose: print(f"  -> Anchor detection error: {e}")

    # 1.17. Reciprocal Pattern Detection (NEW: 1/g(x))
    try:
        reciprocal_result = _detect_reciprocal_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if reciprocal_result:
            if isinstance(reciprocal_result, tuple):
                if verbose: print(f"  -> Reciprocal EXACT MATCH: {reciprocal_result[1]}")
                return reciprocal_result
            if verbose: print(f"  -> Reciprocal found: {reciprocal_result}")
            seeds.extend(reciprocal_result)
    except Exception as e:
        if verbose: print(f"  -> Reciprocal detection error: {e}")
    
    # 2. Power Law Detection (NEW: x^k)
    if _check_timeout():
        if verbose: print("[Forensic] Timeout reached after discrete detectors, returning collected seeds.")
        return seeds if seeds else []
    try:
        power_law_seeds = _detect_power_law_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if power_law_seeds:
            # Check for short-circuit tuple (seeds, best_match)
            if isinstance(power_law_seeds, tuple):
                if verbose: print(f"  -> Power Law EXACT MATCH: {power_law_seeds[1]}")
                return power_law_seeds
                
            if verbose: print(f"  -> Power Law found: {power_law_seeds}")
            seeds.extend(power_law_seeds)
    except Exception as e:
        if verbose: print(f"  -> Power Law detection error: {e}")
    
    # 2.5. Damped Sinusoid Detection (NEW: e^(Ax)*sin(Bx))
    # Uses Gemini's algorithm: slope analysis + envelope extraction + log regression
    try:
        from ..heuristics import detect_damped_sinusoid
        damped_success, damped_expr, damped_mse = detect_damped_sinusoid(
            X, y, variable_names=derived_vars, verbose=verbose
        )
        if damped_success and damped_mse < 1e-4:
            if verbose: print(f"  -> Damped Sinusoid EXACT MATCH: {damped_expr}")
            return ([damped_expr], damped_expr)  # Short-circuit!
        elif damped_success:
            if verbose: print(f"  -> Damped Sinusoid found: {damped_expr}")
            seeds.append(damped_expr)
    except Exception as e:
        if verbose: print(f"  -> Damped Sinusoid detection error: {e}")
    
    # 2.6. Complex Logarithm Detection (NEW: ln(g(x)))
    # Detects complex logs with Im(y) ≈ pi (ln of negative numbers)
    try:
        log_seeds = _detect_complex_log_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if log_seeds:
            if verbose: print(f"  -> Complex Log patterns found: {log_seeds}")
            seeds.extend(log_seeds)
    except Exception as e:
        if verbose: print(f"  -> Complex Log detection error: {e}")

    # 3. Power Peeling Heuristic (y^(1/x))
    # Check if y = g(x)^x -> analyze z = y^(1/x)
    try:
        power_seeds = _detect_power_peeling(ctx, X, y, variable_names=derived_vars)
        if power_seeds:
            # Check for short-circuit tuple
            if isinstance(power_seeds, tuple):
                if verbose: print(f"  -> Power Peeling EXACT MATCH: {power_seeds[1]}")
                return power_seeds
                
            if verbose: print(f"  -> Power Peeling found base: {power_seeds}")
            seeds.extend(power_seeds)
    except Exception as e:
        if verbose: print(f"  -> Power Peeling error: {e}")

    # 3. Triangle Wave Detection
    # Check if y = abs(x - round(x)) or other piecewise periodic patterns
    try:
        triangle_seeds = _detect_triangle_wave(X, y, verbose=verbose)
        if triangle_seeds:
            if verbose: print(f"  -> Triangle Wave found: {triangle_seeds}")
            seeds.extend(triangle_seeds)
    except Exception as e:
        if verbose: print(f"  -> Triangle Wave error: {e}")

    # 4. Bipolar Coordinate Detection (NEW)
    # For 2D functions with poles, generate atan-based seeds
    try:
        bipolar_seeds = _detect_bipolar_poles(X, y, variable_names=derived_vars, verbose=verbose)
        if bipolar_seeds:
            seeds.extend(bipolar_seeds)
    except Exception as e:
        if verbose: print(f"  -> Bipolar detection error: {e}")

    # 5. Rational Form Detection (NEW)
    # If outputs are simple fractions, generate polynomial ratio seeds
    try:
        rational_seeds = _detect_rational_form(X, y, variable_names=derived_vars, verbose=verbose)
        if rational_seeds:
            seeds.extend(rational_seeds)
    except Exception as e:
        if verbose: print(f"  -> Rational form detection error: {e}")

    # 6. Symmetry Pole Detection (NEW - for Singularity Locking without NaNs)
    try:
        sym_seeds = _detect_symmetry_pole(X, y, variable_names=derived_vars, verbose=verbose)
        if sym_seeds:
            pole_seeds.extend(sym_seeds)  # Add to pole_seeds for Composition!
            seeds.extend(sym_seeds)
    except Exception as e:
        if verbose: print(f"  -> Symmetry detection error: {e}")

    # 6. Chirp Pattern Detection (Restored)
    # Check for zeros at sqrt(n*pi)
    try:
        chirp_seeds = _detect_chirp_patterns(X, y, variable_names=derived_vars, verbose=verbose)
        if chirp_seeds:
            seeds.extend(chirp_seeds)
    except Exception as e:
        if verbose: print(f"  -> Chirp detection error: {e}")



    # 1.5 Peeling Heuristic (Inverse Composition)
    if _check_timeout():
        if verbose: print("[Forensic] Timeout reached before peeling, returning collected seeds.")
        return seeds if seeds else []
    # Check if peeling off an outer function reveals a simple integer pattern
    # e.g. y = sin((x-1)/(x+1)) -> z = arcsin(y) = (x-1)/(x+1)
    peeled_seeds = []
    
    # Try Arcsin peeling if in range [-1, 1]
    y_finite = y[np.isfinite(y)]
    if len(y_finite) > 0:
        y_min, y_max = np.min(y_finite), np.max(y_finite)
        if y_min > -1.01 and y_max < 1.01:
             # Try Arcsin
             try:
                 with warnings.catch_warnings():
                     warnings.simplefilter("ignore")
                     # Avoid domain errors at edges
                     z_arcsin = np.arcsin(np.clip(y, -0.99999, 0.99999))
                 
                 int_patterns_asin = _detect_integer_patterns(X, z_arcsin)
                 if int_patterns_asin:
                     if verbose: print(f"   Composition Analysis: Found sin({int_patterns_asin[0]})")
                     for p in int_patterns_asin:
                         peeled_seeds.append(f"sin({p})")
             except: pass

             # Try Arctan (range is open, but usually used for tanh-like)
             # But if it's tanh(something), we use atanh
             
        # Try Log peeling if positive
        if y_min > 0:
             try:
                 with warnings.catch_warnings():
                     warnings.simplefilter("ignore")
                     z_log = np.log(y)
                 
                 int_patterns_log = _detect_integer_patterns(X, z_log)
                 if int_patterns_log:
                     if verbose: print(f"   Composition Analysis: Found exp({int_patterns_log[0]})")
                     for p in int_patterns_log:
                         peeled_seeds.append(f"exp({p})")
             except: pass
             
        # Try Atanh peeling if in range (-1, 1) and looks like tanh
        if y_min > -1.01 and y_max < 1.01:
             try:
                 with warnings.catch_warnings():
                     warnings.simplefilter("ignore")
                     # Clip slightly inside to avoid infinity
                     z_atanh = np.arctanh(np.clip(y, -0.99999, 0.99999))
                 
                 int_patterns_atanh = _detect_integer_patterns(X, z_atanh)
                 
                 # Also try Rational Form on the inside!
                 rational_patterns_atanh = _detect_rational_form(X, z_atanh, variable_names=derived_vars)
                 
                 if int_patterns_atanh or rational_patterns_atanh:
                     if verbose: print(f"   Composition Analysis: Found tanh(inner_func)")
                     for p in int_patterns_atanh + rational_patterns_atanh:
                         peeled_seeds.append(f"tanh({p})")
             except: pass

    seeds.extend(peeled_seeds)
    
    # 3. Rapid Growth Polynomials
    poly_seeds = _detect_rapid_growth_poly(X, y, verbose=verbose)
    if poly_seeds:
        if verbose: print(f"   Rapid Growth: Found {poly_seeds}")
        seeds.extend(poly_seeds)

    # 2. Integer Pattern Analysis (Gemini Method)
    integer_patterns = _detect_integer_patterns(X, y)
    if integer_patterns:
        if verbose: print(f"   Integer Analysis: Deduced patterns {integer_patterns}")
        seeds.extend(integer_patterns)
        
    # --- Singularity Analysis (Inline) ---
    seen_poles = set()
    detected_pole_info = []
    for i, y_val in enumerate(y):
        try:
            if not np.isfinite(y_val):
                for col_idx in range(n_vars):
                    val = X[i, col_idx]
                    var_name = derived_vars[col_idx]
                    pole_key = (var_name, val)
                    if pole_key in seen_poles: continue
                    seen_poles.add(pole_key)
                    
                    if isinstance(val, complex):
                        if abs(val.imag) < 1e-10: val = val.real
                        else: continue
                    
                    val_str = str(float(val))
                    if verbose: detected_pole_info.append(f"{var_name}={val_str}")
                    
                    # Singularity Locking: Wrap constant in locked() to prevent optimizer drift
                    # e.g. 1/(x-3.0) -> 1/(x-locked(3.0))
                    basic_pole = f"1/({var_name}-(locked({val_str})))"
                    pole_seeds.append(basic_pole)
                    seeds.append(basic_pole)
                    seeds.append(f"1/({var_name}-(locked({val_str})))**2")
                    seeds.append(f"1/(locked({val_str})-({var_name}))")
        except: continue
        
    # --- Near-zero crossing detection ---
    if len(y) >= 3:
        y_finite = np.array([yi if np.isfinite(yi) else 0 for yi in y])
        y_max = np.max(np.abs(y_finite)) if np.any(np.isfinite(y_finite)) else 1
        if y_max > 10:
            for i in range(len(y) - 1):
                if np.isfinite(y[i]) and np.isfinite(y[i + 1]):
                    ratio = abs(y[i + 1] / y[i]) if y[i] != 0 else 0
                    if ratio > 10 or (ratio > 0 and ratio < 0.1):
                        mid_x = (X[i, 0] + X[i + 1, 0]) / 2
                        near_pole = f"1/({var}-{mid_x:.2f})"
                        pole_seeds.append(near_pole)
                        seeds.append(near_pole)

    # --- Phase 4: Composition (Connecting Singularities to Trig) ---
    # This was previously missing!
    outer_funcs = _detect_outer_functions(y)
    if outer_funcs and pole_seeds:
         composed_seeds = _compose_seeds(pole_seeds, outer_funcs)
         if composed_seeds:
             if verbose: print(f"   Composition Analysis: Composing {len(pole_seeds)} poles with {outer_funcs} -> {len(composed_seeds)} seeds")
             seeds.extend(composed_seeds)
             
    # --- Final Filtering based on Banned Operators ---
    if banned_operators:
        valid_seeds = []
        for s in seeds:
            s_str = str(s) # handle non-string seeds if any
            s_lower = s_str.lower()
            
            is_banned = False
            for ban in banned_operators:
                ban_lower = ban.lower()
                if ban_lower in s_lower:
                    is_banned = True
                    break
                # Special cases
                if ban_lower == "pow":
                    if "**" in s_lower or "^" in s_lower:
                        is_banned = True
                        break
                if ban_lower == "^":
                     if "pow" in s_lower or "**" in s_lower:
                        is_banned = True
                        break
            
            if not is_banned:
                valid_seeds.append(s)
        seeds = valid_seeds
             
    # Clean up
    return sorted(list(set(seeds)))

    # --- Outer Functions ---
    outer_functions = _detect_outer_functions(y)
    if verbose and outer_functions:
        print(f"   Range Analysis: Suggested {outer_functions}")
        
    # --- Compose Seeds ---
    if pole_seeds and outer_functions:
        composed = _compose_seeds(pole_seeds, outer_functions)
        seeds.extend(composed)
        
    # 5. Deep Trig Heuristic (NEW)
    try:
        trig_seeds = _detect_trig_composites(y)
        if trig_seeds:
            if verbose: print(f"  -> Discovered deep trig possibilities: {trig_seeds}")
            seeds.extend(trig_seeds)
    except Exception:
        pass
        
    return sorted(list(set(seeds)))

def _detect_complex_log_patterns(X, y, variable_names=None, verbose=False):
    """Detect complex logarithm patterns: f(x) = log(g(x)).
    
    Identifies patterns where Im(y) ≈ k*pi, which indicates ln(negative).
    If found, computes z = exp(y) (which should be real) and runs analysis on z.
    
    Args:
        X: Input data
        y: Output data (can be complex)
    """
    seeds = []
    
    if X.ndim > 1 and X.shape[1] > 1: return []
    try: x_flat = X.flatten()
    except: return []
    
    # Check for complex values
    if not np.iscomplexobj(y):
        return []
        
    # Analyze imaginary parts
    y_imag = np.imag(y)
    y_real = np.real(y)
    
    # Filter for non-zero imaginary parts
    complex_mask = np.abs(y_imag) > 1e-3
    if np.sum(complex_mask) < 3:
        return []
        
    imag_vals = y_imag[complex_mask]
    
    # Check if imaginary parts cluster around multiples of pi
    # ln(-x) -> pi*i
    # ln(complex) -> arg(z)
    
    median_imag = np.median(np.abs(imag_vals))
    
    # Check for PI signature
    if abs(median_imag - np.pi) < 0.1:
        if verbose: print(f"   Complex Log Analysis: Detected imaginary component ~ PI (median={median_imag:.4f}) -> Potential ln(negative)")
        
        # Transform: z = exp(y)
        # exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
        # If b = pi, exp(y) = exp(Re(y)) * (-1) = -exp(Re(y))
        
        try:
            # We want to find the inner function g(x) s.t. y = ln(g(x))
            # So g(x) = exp(y)
            z = np.exp(y)
            
            # z should be real (mostly)
            # Check if z is effectively real
            if np.all(np.abs(np.imag(z)) < 1e-4):
                z_real = np.real(z)
                if verbose: print(f"   Complex Log Analysis: transformed exp(y) is effectively REAL. Analysing inner function...")
                
                # RECURSIVE ANALYSIS on z
                # Run detectors on inner function z
                # 1. Reciprocal (for ln(1/x) or ln(e/x) -> e^y = e/x)
                recip_result = _detect_reciprocal_patterns(X, z_real, variable_names, verbose=False)
                
                # Unpack tuple if needed
                recip_seeds = []
                if isinstance(recip_result, tuple):
                    recip_seeds = recip_result[0]
                elif isinstance(recip_result, list):
                    recip_seeds = recip_result
                    
                for s in recip_seeds:
                    # s is candidate for z. So y = ln(s)
                    # Algebraic Simplification (Agent Handoff Rule 3: Avoid Singularity)
                    # log(1/u) -> -log(u)
                    # This avoids the division by zero if u approx 0.
                    if s.startswith("1 / (") and s.endswith(")"):
                        # Extract denominator u
                        denom = s[5:-1]
                        simplified_seed = f"-log({denom})"
                        seeds.append(simplified_seed)
                        if verbose: print(f"   -> Found inner reciprocal pattern: {s} => {simplified_seed} (Simplified)")
                    else:
                        seed = f"log({s})"
                        seeds.append(seed)
                        if verbose: print(f"   -> Found inner reciprocal pattern: {s} => {seed}")
                    
                # 2. Rational
                rational_seeds = _detect_rational_form(X, z_real, variable_names, verbose=False)
                for s in rational_seeds:
                    seeds.append(f"log({s})")
                    
                # 3. Linear/Poly (using polyfit directly for speed)
                try:
                    # Simple linear check for exp(y) = ax+b
                    z_finite_mask = np.isfinite(z_real) & np.isfinite(x_flat)
                    if np.sum(z_finite_mask) > 5:
                        x_lin = x_flat[z_finite_mask]
                        z_lin = z_real[z_finite_mask]
                        if np.iscomplexobj(x_lin): x_lin = np.real(x_lin)
                        
                        coeffs = np.polyfit(x_lin, z_lin, 1)
                        p_lin = np.poly1d(coeffs)
                        res_lin = np.sum((z_lin - p_lin(x_lin))**2)
                        r2_lin = 1 - (res_lin / (np.var(z_lin)*len(z_lin) + 1e-9))
                        
                        if r2_lin > 0.99:
                            a, b = coeffs
                            a = round(a, 5)
                            b = round(b, 5)
                            inner = f"{a}*{variable_names[0] if variable_names else 'x'} + {b}"
                            seeds.append(f"log({inner})")
                            if verbose: print(f"   -> Found inner linear pattern: {inner}")
                except: pass
                
        except Exception as e:
            if verbose: print(f"   Complex Log Analysis error: {e}")
            
    return seeds



