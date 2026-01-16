"""Forensic Analysis Module for Symbolic Regression.

Extracts deep patterns from data using heuristics, singularity analysis,
and integer sequence detection.
"""
import numpy as np
import time
import math
import warnings
import fractions

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
    return composed

def _detect_integer_patterns(X, y):
    """The 'Gemini Method': Phase 3 - Integer Pattern Recognition."""
    # Allow (N,1) shaped arrays
    if X.ndim > 1 and X.shape[1] > 1: return []
    try: x_flat = X.flatten()
    except: return []

    seeds = []
    var_name = "x"
    
    indices = []
    for i, x_val in enumerate(x_flat):
        if np.iscomplex(x_val) or (hasattr(x_val, 'imag') and abs(x_val.imag) > 1e-9): continue
        try:
            real_val = float(x_val.real if hasattr(x_val, 'real') else x_val)
            if abs(real_val - round(real_val)) < 1e-9 and abs(real_val) > 1 and abs(real_val) < 10:
                indices.append(i)
        except: continue
    
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
            
    return list(set(seeds))

# Stubs for other detectors (to be ported fully later)
def _detect_step_patterns(X, y): return []
def _detect_self_power(X, y, verbose=False): return []
def _detect_relu_patterns(X, y): return []
def _detect_clamp_patterns(X, y, verbose=False): return []
def _detect_pulse_patterns(X, y, verbose=False): return []
def _detect_bessel_patterns(X, y, verbose=False): return []
def _detect_gamma_patterns(X, y, verbose=False): return []
def _detect_prime_counting_patterns(X, y): return []
def _detect_bitwise_patterns(X, y): return []
def _detect_modulo_patterns(X, y, verbose=False): return []
def _detect_fibonacci_patterns(X, y, verbose=False): return []
def _detect_anchor_patterns(X, y, verbose=False): return []
def _detect_odd_function_patterns(X, y, verbose=False): return []
def _detect_signum_patterns(X, y, variable_names=None, verbose=False): return []
def _detect_rosenbrock_patterns(X, y, variable_names=None, verbose=False): return []
def _detect_fractal_cosine_patterns(X, y, verbose=False): return []
def _detect_chirp_patterns(X, y, variable_names=None, verbose=False): return []
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
            p = round(val * q)
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

def _detect_bipolar_poles(X, y, variable_names=None, verbose=False):
    """Detect bipolar coordinate patterns from singularities.
    
    Looks for poles where y becomes non-finite (AccumBounds, nan, inf).
    For 2D data: generates atan(y/(x-pole)), atan((x-pole)/y), and
    cos(k*(...)) wrapped versions for various multipliers k.
    
    This enables discovery of interference patterns like:
        cos(16*(atan((x-2)/y) + atan(y/(x+2))))
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Only for 2D functions
    n_vars = X.shape[1]
    if n_vars < 2:
        return []
    
    seeds = []
    
    # Get variable names
    if variable_names and len(variable_names) >= 2:
        v0, v1 = variable_names[0], variable_names[1]
    else:
        v0, v1 = "x", "y"
    
    # Find rows with non-finite y (poles)
    bad_mask = ~np.isfinite(y)
    if not np.any(bad_mask):
        return []
    
    # Analyze poles in first dimension (x)
    bad_x_vals = X[bad_mask, 0]
    if len(bad_x_vals) > 0:
        # Find unique pole locations
        unique_poles = np.unique(np.round(bad_x_vals, 6))
        for pole in unique_poles:
            if np.isfinite(pole) and abs(pole) < 100:
                pole_int = int(pole) if pole == int(pole) else pole
                # Generate atan seeds with this pole
                if pole_int >= 0:
                    seeds.append(f"atan({v1}/({v0}-{pole_int}))")  # atan(y/(x-pole))
                    seeds.append(f"atan(({v0}-{pole_int})/{v1})")  # atan((x-pole)/y)
                    seeds.append(f"atan({v1}/({v0}+{abs(pole_int)}))")  # atan(y/(x+pole))
                else:
                    seeds.append(f"atan({v1}/({v0}+{abs(pole_int)}))")
                    seeds.append(f"atan(({v0}+{abs(pole_int)})/{v1})")
                    
                # Complementary pole: if we found -2, also try +2 (hidden pole)
                comp_pole = -pole_int
                if abs(comp_pole) < 100:
                    if comp_pole >= 0:
                        seeds.append(f"atan({v1}/({v0}-{comp_pole}))")
                        seeds.append(f"atan(({v0}-{comp_pole})/{v1})")
                    else:
                        seeds.append(f"atan({v1}/({v0}+{abs(comp_pole)}))")
                        seeds.append(f"atan(({v0}+{abs(comp_pole)})/{v1})")
    
    # Analyze poles in second dimension (y)
    bad_y_vals = X[bad_mask, 1]
    if len(bad_y_vals) > 0:
        unique_y_poles = np.unique(np.round(bad_y_vals, 6))
        for pole in unique_y_poles:
            if np.isfinite(pole) and abs(pole) < 100:
                pole_int = int(pole) if pole == int(pole) else pole
                if pole_int >= 0:
                    seeds.append(f"atan({v0}/({v1}-{pole_int}))")
                    seeds.append(f"atan(({v1}-{pole_int})/{v0})")
                else:
                    seeds.append(f"atan({v0}/({v1}+{abs(pole_int)}))")
                    seeds.append(f"atan(({v1}+{abs(pole_int)})/{v0})")
    
    # Check if output is bounded (suggests cos/sin wrapper)
    y_finite = y[np.isfinite(y)]
    if len(y_finite) > 0:
        y_min, y_max = np.min(y_finite), np.max(y_finite)
        is_bounded = -1.5 < y_min < -0.5 and 0.5 < y_max < 1.5
        
        if is_bounded and len(seeds) > 0:
            # Generate cos(k*...) wrapped versions for various multipliers
            wrapped = []
            base_seeds = seeds.copy()
            for multiplier in [2, 4, 8, 16]:  # Common angular multipliers
                for seed in base_seeds[:6]:  # Wrap first 6 base seeds
                    wrapped.append(f"cos({multiplier}*{seed})")
                    wrapped.append(f"sin({multiplier}*{seed})")
            
            # EXPLICIT bipolar combinations using atan2 to avoid division by zero
            # atan2(a, b) = atan(a/b) but handles b=0 gracefully
            # Pattern: cos(k*(atan2(y, x+a) + atan2(x-a, y)))
            for a in [2, 3, 4]:  # Common pole separations
                for k in range(1, 21):  # All multipliers 1-20 (not hardcoded)
                    # Using atan2 for division-free evaluation
                    wrapped.append(f"cos({k}*(atan2({v1},{v0}+{a})+atan2({v0}-{a},{v1})))")
                    wrapped.append(f"cos({k}*(atan2({v1},{v0}-{a})+atan2({v0}+{a},{v1})))")
                    wrapped.append(f"cos({k}*(atan2({v1},{v0}+{a})-atan2({v0}-{a},{v1})))")
                    # Also keep atan versions for cases where y≠0 (better simplification)
                    wrapped.append(f"cos({k}*(atan({v1}/({v0}+{a}))+atan(({v0}-{a})/{v1})))")
            
            # Also try sums of detected atans for bipolar patterns
            if len(base_seeds) >= 2:
                for k in [8, 16]:
                    wrapped.append(f"cos({k}*({base_seeds[0]}+{base_seeds[1]}))")
                    wrapped.append(f"cos({k}*({base_seeds[0]}-{base_seeds[1]}))")
            
            seeds.extend(wrapped)
    
    if verbose and seeds:
        print(f"  -> Bipolar poles detected, generated {len(seeds)} atan-based seeds")
    
    return list(set(seeds))  # Remove duplicates

def _detect_power_peeling(X, y):
    """Detect if y = Base(x)^x via Rational Analysis on y^(1/x)."""
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
            with np.errstate(invalid='ignore', over='ignore'):
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
        print(f"Power Peeling: Only {len(valid_points)} valid points (need 5).")
        return []
    
    # 2. Run Rational Analysis on z
    # Local import to avoid circular dependency
    try:
        from kalkulator_pkg.function_manager import find_function_from_data
        print(f"Power Peeling: Running Rational Analysis on {len(valid_points)} points...")
        success, func_str, _, note = find_function_from_data(valid_points, param_names=["x"])
        print(f"Power Peeling Result: {success}, {func_str}, Note: {note}")
        
        if success:
             # Found a rational base!
             # Return (base)**x
             return [f"({func_str})**x"]
    except ImportError:
        pass
        
    return []

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
            print(f"     → Y range outside [0, 1], not a standard triangle wave")
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
                print(f"     → PERFECT MATCH: abs(x - round(x))")
            seeds.append("abs(x - floor(x + 0.5))")  # Lowercase for GP engine compatibility
            seeds.append("abs(x - ceil(x - 0.5))")   # Alternative form
            return seeds
        elif mse_triangle < 0.01:
            if verbose:
                print(f"     → Good match for triangle wave seed")
            seeds.append("abs(x - floor(x + 0.5))")
    except Exception as e:
        if verbose:
            print(f"     → Error testing triangle: {e}")
    
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
                print(f"     → PERFECT MATCH: 0.5 - abs(frac(x) - 0.5)")
            seeds.append("0.5 - abs(frac(x) - 0.5)")
            return seeds
    except Exception as e:
        if verbose:
            print(f"     → Error testing frac form: {e}")
    
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
                print(f"     → PERFECT MATCH: frac(x)")
            seeds.append("frac(x)")
            seeds.append("x - floor(x)")
            return seeds
    except:
        pass
    
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
    except: pass
    
    return seeds

def generate_pattern_seeds(X, y, variable_names=None, verbose=False):
    """Detect patterns in data and return seed expression strings."""
    t0 = time.perf_counter()
    seeds = []
    pole_seeds = []
    
    # Ensure X is 2D
    X = np.array(X)
    if X.ndim == 1: X = X.reshape(-1, 1)
    
    n_vars = X.shape[1]
    derived_vars = variable_names if variable_names and len(variable_names) == n_vars else [f"x{k}" for k in range(n_vars)]
    var = derived_vars[0]

    # 1. Step Function
    step_patterns = _detect_step_patterns(X, y)
    if step_patterns: return (step_patterns, step_patterns[0]) # Match return signature
    
    # 2. Power Peeling Heuristic (NEW)
    # Check if y = g(x)^x -> analyze z = y^(1/x)
    try:
        power_seeds = _detect_power_peeling(X, y)
        if power_seeds:
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

    # 1.5 Peeling Heuristic (Inverse Composition)
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
                 if int_patterns_atanh:
                     if verbose: print(f"   Composition Analysis: Found tanh({int_patterns_atanh[0]})")
                     for p in int_patterns_atanh:
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
                    
                    basic_pole = f"1/({var_name}-({val_str}))"
                    pole_seeds.append(basic_pole)
                    seeds.append(basic_pole)
                    seeds.append(f"1/({var_name}-({val_str}))**2")
                    seeds.append(f"1/({val_str}-({var_name}))")
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
    except: pass
        
    return list(set(seeds))
