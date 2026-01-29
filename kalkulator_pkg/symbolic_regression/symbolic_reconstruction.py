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

    # v4.0 Audit Remediation: Removed Rational Reconstruction (The "Rational Trap")
    # We do not attempt to force floats into fractions like 22/7.
    # The data must speak for itself.
    # import fractions
    # try:
    #     f = fractions.Fraction(value).limit_denominator(max_denom)
    #     ...
    # except: pass

    # 2. Single Scaling
    for sym, name, val in CONSTANTS:
        if abs(val) < 1e-9: continue
        
        # Check ratio (Integer only)
        ratio = value / val
        try:
            # v4.0 Audit: Only allow nice integer multiples (e.g. 2pi, 3pi), NOT 13/7pi.
            # We also allow simple fractions like pi/2 (0.5)
            
            near_int = round(ratio)
            if abs(ratio - near_int) < 0.01: # Strict 1% for integer multiples
                cand_val = near_int * val
                err = abs(value - cand_val)
                
                cand_name = f"{near_int}*{name}"
                if near_int == 1: cand_name = name
                elif near_int == -1: cand_name = f"-{name}"
                elif near_int == 0: cand_name = "0"
                
                if err < tolerance:
                     if verbose: print(f"     ✅ MATCH: {cand_name} (Error: {err:.2e}) -> Snapped!", file=sys.stderr)
                     return cand_name
                     
            # Optional: Check for halves (0.5, 1.5, etc) - Common in physics (1/2 mv^2)
            # Remove if audit demands absolute purity, but 0.5 is very common.
            # Let's keep 0.5 but strict.
            if abs(2*ratio - round(2*ratio)) < 0.01:
                 halves = round(2*ratio)
                 if halves % 2 != 0: # Only odd halves (1/2, 3/2)
                     cand_val = (halves / 2.0) * val
                     err = abs(value - cand_val)
                     if err < tolerance:
                         return f"{halves}/2 * {name}"

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
