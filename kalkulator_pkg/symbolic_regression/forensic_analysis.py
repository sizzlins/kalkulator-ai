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
            
            for n in [1, 2, 3]:
                x_pow = x_val ** n
                num_rel = None
                if num == x_pow: num_rel = f"{var_name}^{n}"
                elif num == x_pow + 1: num_rel = f"({var_name}^{n} + 1)"
                elif num == x_pow - 1: num_rel = f"({var_name}^{n} - 1)"
                elif num == x_pow + x_val: num_rel = f"({var_name}^{n} + {var_name})"
                
                den_rel = None
                if den == x_pow: den_rel = f"{var_name}^{n}"
                elif den == x_pow + 1: den_rel = f"({var_name}^{n} + 1)"
                elif den == x_pow - 1: den_rel = f"({var_name}^{n} - 1)"
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
                 # Clip slightly inside to avoid infinity
                 z_atanh = np.arctanh(np.clip(y, -0.99999, 0.99999))
                 int_patterns_atanh = _detect_integer_patterns(X, z_atanh)
                 if int_patterns_atanh:
                     if verbose: print(f"   Composition Analysis: Found tanh({int_patterns_atanh[0]})")
                     for p in int_patterns_atanh:
                         peeled_seeds.append(f"tanh({p})")
             except: pass

    seeds.extend(peeled_seeds)
    
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
        
    return list(set(seeds))
