import math
import sympy as sp
from typing import Optional, Tuple, List

# Fundamental constants to check against
CONSTANTS = [
    (sp.E, "e", math.e),
    (sp.pi, "pi", math.pi),
    (sp.E * sp.pi, "e*pi", math.e * math.pi),
    (sp.pi**2, "pi^2", math.pi**2),
    (sp.E**2, "e^2", math.e**2),
]

def reconstruct_constant(value: float, tolerance: float = 1e-4, max_denom: int = 10, verbose: bool = False) -> Optional[str]:
    """
    Attempt to reconstruct a float value as a symbolic expression involving e and pi.
    Args:
        value: The float value to reconstruct.
        tolerance: Maximum allowed absolute error.
        max_denom: Maximum denominator for rational approximations.
        verbose: Whether to print candidates checked.
    Returns:
        String representation...
    """
    import sys
    
    if abs(value) < 1e-10: return "0"
    
    if verbose:
        print(f"[SV] CONSTANT RECONSTRUCTION:\n     Analyzing coeff: {value:.11f}", file=sys.stderr)
        print(f"     Checking candidates: ['int', 'rational', 'pi', 'e', 'linear_combos']", file=sys.stderr)

    # 1. Rational Check
    import fractions
    try:
        f = fractions.Fraction(value).limit_denominator(max_denom)
        err = abs(value - float(f))
        if verbose and err > 1e-10:
             print(f"     ❌ Match: {f} (Error: {err:.2e})", file=sys.stderr)
             
        if err < tolerance:
            if verbose: print(f"     ✅ MATCH: {f} (Error: {err:.2e}) -> Snapped!", file=sys.stderr)
            return str(f)
    except: pass

    # 2. Single Scaling
    for sym, name, val in CONSTANTS:
        if abs(val) < 1e-9: continue
        
        # Check ratio
        ratio = value / val
        try:
            frac = fractions.Fraction(ratio).limit_denominator(max_denom)
            cand_val = float(frac) * val
            err = abs(value - cand_val)
            
            cand_name = f"{frac}*{name}"
            if frac == 1: cand_name = name
            elif frac == -1: cand_name = f"-{name}"
            
            if verbose and err > tolerance: # Only show interesting misses? Or some?
                # To avoid spam, maybe only show if error is "kinda close" (e.g. < 0.1)
                if err < 0.5:
                     print(f"     ❌ Match: {cand_name} (Error: {err:.2e})", file=sys.stderr)

            if err < tolerance / abs(val): # Scaled or absolute? Logic above used abs check.
                # Re-check stricter tolerance for consistency
                if err < tolerance:
                    if verbose: print(f"     ✅ MATCH: {cand_name} (Error: {err:.2e}) -> Snapped!", file=sys.stderr)
                    return cand_name
        except: pass

    # 3. Linear Combination
    search_range = range(-3, 4)
    best_expr = None
    best_err = float('inf')
    
    # Track top rejection for verbose
    best_reject = None
    best_reject_err = float('inf')

    for a in search_range:
        for b in search_range:
            for c in search_range:
                if a == 0 and b == 0 and c == 0: continue
                
                candidate_val = a * math.e + b * math.pi + c
                err = abs(value - candidate_val)
                
                # Format for display
                parts = []
                if a != 0: parts.append(f"{a}*e" if a != 1 else "e")
                if b != 0: parts.append(f"{b}*pi" if b != 1 else "pi")
                if c != 0: parts.append(str(c))
                
                expr_str = ""
                for p in parts:
                    if not expr_str: expr_str = p
                    else:
                        if p.startswith("-"): expr_str += f" - {p[1:]}"
                        else: expr_str += f" + {p}"
                
                # Cleanup 1*e
                expr_str = expr_str.replace("1*e", "e").replace("1*pi", "pi")

                if err < tolerance and err < best_err:
                    best_err = err
                    best_expr = expr_str
                
                # Track best non-match for logging context
                if err < best_reject_err and err > tolerance:
                    best_reject_err = err
                    best_reject = expr_str

    if best_expr:
        if verbose: print(f"     ✅ MATCH: {best_expr} (Error: {best_err:.2e}) -> Snapped!", file=sys.stderr)
        return best_expr
    
    if verbose and best_reject:
         print(f"     ❌ Best Approx: {best_reject} (Error: {best_reject_err:.2e})", file=sys.stderr)

    return None

def reconstruct_coefficients(coeffs: List[float], tolerance: float = 1e-4) -> Tuple[List[str], bool]:
    """
    Reconstruct a list of coefficients. 
    Returns list of strings and a boolean indicating if ANY substitution happened.
    """
    res = []
    any_sub = False
    for c in coeffs:
        rec = reconstruct_constant(c, tolerance)
        if rec:
            res.append(f"({rec})") # Wrap in parens for safety
            any_sub = True
        else:
            res.append(str(c))
    return res, any_sub
