
from typing import Any, List, Tuple, Optional, Dict
import numpy as np
import math
from .utils.numeric import eval_to_float
from .heuristics import (
    solve_rational_function_svd,
    check_log_linear_transformations,
    detect_symbolic_constant
)

def check_triangle_wave(
    data_points: List[Tuple[Any, Any]], 
    param_names: List[str], 
    verbose: bool = False
) -> Optional[str]:
    """Check for triangle wave pattern: abs(x - round(x))."""
    if len(data_points) < 5 or len(param_names) != 1:
        return None
        
    try:
        x_vals = []
        y_vals = []
        for p in data_points:
            x_in = p[0] if not isinstance(p[0], (list, tuple)) else p[0][0]
            x_vals.append(eval_to_float(x_in))
            y_vals.append(eval_to_float(p[1]))
            
        x_arr = np.array(x_vals)
        y_arr = np.array(y_vals)
        
        # Filter finite
        valid = np.isfinite(x_arr) & np.isfinite(y_arr)
        if np.sum(valid) < 5: return None
        
        x_valid = x_arr[valid]
        y_valid = y_arr[valid]
        
        y_pred = np.abs(x_valid - np.round(x_valid))
        max_err = np.max(np.abs(y_valid - y_pred))
        
        if max_err < 1e-6:
             var = param_names[0]
             return f"Abs({var} - floor({var} + 0.5))"
             
    except Exception:
        pass
    return None

def check_harmonic(
    data_points: List[Tuple[Any, Any]], 
    param_names: List[str], 
    verbose: bool = False
) -> Optional[str]:
    """Check for simple sin/cos patterns."""
    if len(data_points) < 3 or len(param_names) != 1:
        return None
        
    try:
        x_vals = []
        y_vals = []
        for p in data_points:
            x_in = p[0] if not isinstance(p[0], (list, tuple)) else p[0][0]
            x_vals.append(eval_to_float(x_in))
            y_vals.append(eval_to_float(p[1]))
            
        var = param_names[0]
        
        freqs = [0.5, 1, 2, 3, 4, 5, np.pi]
        labels = ["0.5", "1", "2", "3", "4", "5", "pi"]
        
        for freq, label in zip(freqs, labels):
            # Cosine
            cos_vals = np.array([np.cos(freq * x) for x in x_vals])
            errs = np.abs(cos_vals - np.array(y_vals))
            if np.max(errs) < 1e-3:
                return f"cos({var})" if label == "1" else f"cos({label}*{var})"
                
            # Sine
            sin_vals = np.array([np.sin(freq * x) for x in x_vals])
            errs = np.abs(sin_vals - np.array(y_vals))
            if np.max(errs) < 1e-3:
                return f"sin({var})" if label == "1" else f"sin({label}*{var})"
                
    except Exception:
        pass
    return None

def check_sqrt_poly(
    data_points: List[Tuple[Any, Any]],
    param_names: List[str],
    verbose: bool = False
) -> Optional[str]:
    """Check for y = sqrt(P(x))."""
    if len(param_names) != 1: return None
    
    try:
         x_vals = []
         y_vals = []
         for p in data_points:
            x_in = p[0] if not isinstance(p[0], (list, tuple)) else p[0][0]
            x_vals.append(eval_to_float(x_in))
            y_vals.append(eval_to_float(p[1]))
            
         y_arr = np.array(y_vals)
         # Require positive y (simplify)
         if np.any(y_arr < 0): return None
         
         y_sq = y_arr ** 2
         coeffs = np.polyfit(x_vals, y_sq, 2)
         
         p = np.poly1d(coeffs)
         y_sq_pred = p(x_vals)
         mse = np.mean((y_sq - y_sq_pred)**2)
         
         if mse < 1e-6:
             # Construct poly string
             a, b, c = coeffs
             # Snap to 0/integers
             def snap(v):
                 if abs(v) < 1e-9: return 0
                 if abs(v - round(v)) < 1e-9: return int(round(v))
                 return v
             a, b, c = snap(a), snap(b), snap(c)
             
             var = param_names[0]
             terms = []
             if a != 0: terms.append(f"{a}*{var}^2" if a != 1 else f"{var}^2")
             if b != 0: terms.append(f"{b}*{var}" if b != 1 else f"{var}")
             if c != 0: terms.append(f"{c}")
             
             if not terms: return "0"
             poly_str = "+".join(terms).replace("+-", "-")
             return f"sqrt({poly_str})"
             
    except Exception:
        pass
    return None

def check_rational_svd(
    data_points: List[Tuple[Any, Any]],
    param_names: List[str],
    verbose: bool = False
) -> Optional[str]:
    """Check for rational function using SVD."""
    if len(param_names) != 1: return None
    
    try:
        # Adapt format for solve_rational_function_svd
        X_data = [[eval_to_float(p[0] if not isinstance(p[0], (list, tuple)) else p[0][0])] for p in data_points]
        y_data = [eval_to_float(p[1]) for p in data_points]
        
        deg = 4 if len(data_points) >= 12 else 2
        success, func, mse = solve_rational_function_svd(
            X_data, y_data, param_names, 
            max_numerator_degree=deg, max_denominator_degree=deg,
            verbose=verbose
        )
        
        if success and mse < 1e-6:
            return func
            
    except Exception:
        pass
    return None

def check_advanced_heuristics(
    data_points: List[Tuple[Any, Any]],
    param_names: List[str],
    verbose: bool = False
) -> Optional[str]:
    """Run advanced heuristics (Rational, Harmonic, etc.)."""
    
    # 1. Rational SVD (Strong solver for fractions)
    # We do this AFTER standard regression (Linear/Poly) in strategy
    res = check_rational_svd(data_points, param_names, verbose)
    if res: return res
    
    # 2. Harmonic
    res = check_harmonic(data_points, param_names, verbose)
    if res: return res
    
    # 3. Sqrt(Poly)
    res = check_sqrt_poly(data_points, param_names, verbose)
    if res: return res
    
    # 4. Log-Linear (Power Law, Exp)
    try:
        X_data = [
            eval_to_float(p[0] if not isinstance(p[0], (list, tuple)) else p[0][0])
            for p in data_points
        ]
        y_data = [eval_to_float(p[1]) for p in data_points]
        
        success, func_str = check_log_linear_transformations(X_data, y_data, param_names)
        if success: return func_str
    except Exception:
        pass
        
    return None
