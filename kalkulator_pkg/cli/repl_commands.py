"""
Command handlers for the Kalkulator CLI.
Extracted from app.py to enforce Rule 4 (Small Units).
"""

import logging
import re
import warnings
from typing import Any
from typing import Dict

import kalkulator_pkg.parser as kparser
import math
import numpy as np

from ..cache_manager import export_cache_to_file
from ..cache_manager import get_persistent_cache
from ..cache_manager import replace_cache_from_file
from ..function_manager import BUILTIN_FUNCTION_NAMES
from ..function_manager import clear_functions
from ..function_manager import clear_saved_functions
from ..function_manager import export_function_to_file
from ..function_manager import list_functions
from ..utils.data_loading import load_csv_data
from ..function_manager import load_functions
from ..function_manager import save_functions
from ..solver.dispatch import solve_single_equation
from ..symbolic_regression import GeneticConfig, GeneticSymbolicRegressor
from ..utils.formatting import format_solution
from ..utils.formatting import print_result_pretty
from ..worker import clear_caches

logger = logging.getLogger(__name__)


def _find_matching_paren(s: str, start: int) -> int:
    """Find matching closing parenthesis for opening paren at start position.
    
    Handles nested parentheses like f(sin(1)).
    Returns -1 if no matching paren found.
    """
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


# Registry of non-math commands for improved parsing detection
COMMAND_REGISTRY = {
    "help",
    "?",
    "quit",
    "exit",
    "clear",
    "cls",
    "history",
    "find",
    "evolve",
    "solve",
    "export",
    "save",
    "load",
    "show",
    "list",
    "debug",
    "timing",
    "cachehits",
    "savecache",
    "loadcache",
    "showcache",
    "clearcache",
    "define",
    "health",
}


def handle_command(text: str, ctx: Any, variables: Dict[str, str]) -> bool:
    """
    Attempt to handle the input text as a command.
    Returns True if handled, False otherwise.
    """
    raw_lower = text.lower().strip()

    # === Function Persistence Commands ===
    if raw_lower in ("save", "savefunction", "savefunctions"):
        success, msg = save_functions()
        print(msg)
        return True

    if raw_lower in ("loadfunction", "loadfunctions"):
        success, msg = load_functions()
        print(msg)
        return True

    if raw_lower in ("clearfunction", "clearfunctions"):
        clear_functions()
        print("Functions cleared from current session.")
        return True

    if raw_lower in ("clearsavefunction", "clearsavefunctions"):
        success, msg = clear_saved_functions()
        print(msg)
        return True

    if raw_lower in ("showfunction", "showfunctions", "list"):
        _handle_show_functions()
        return True

    if raw_lower.startswith("debug"):
        _handle_debug_command(text, ctx)
        return True

    if raw_lower == "health":
        _handle_health_command()
        return True

    if raw_lower.startswith("timing"):
        _handle_timing_command(text, ctx)
        return True

    if raw_lower.startswith("cachehits"):
        _handle_cachehits_command(text, ctx)
        return True

    # === Research Commands (Evolve, SINDy, Causal, Dimensionless) ===
    # Must check these BEFORE generic "find " to avoid shadowing
    if raw_lower.startswith("find ode"):
        _handle_find_ode(text)
        return True

    if raw_lower.startswith("find dimensionless"):
        _handle_find_dimensionless(text)
        return True

    if raw_lower.startswith("discover causal"):
        _handle_discover_causal(text)
        return True

    if raw_lower.startswith("evolve "):
        _handle_evolve(text, variables)
        return True

    # === Function Finding/System ===
    if raw_lower.startswith("find ode"):
        _handle_find_ode(text)
        return True

    if raw_lower.startswith("find "):
        # e.g. "find f(x)" or "find f(x) given ..."
        # Implement _handle_find_command logic here or call helper
        _handle_find_command(text, variables)
        return True

    if raw_lower.startswith("benchmark"):
        _handle_benchmark(text)
        return True

    # === Cache Commands ===
    if raw_lower in ("clearcache", "clear cache"):
        clear_caches()
        print("Caches cleared.")
        return True

    if raw_lower.startswith("showcache") or raw_lower in ("show cache", "cache"):
        _handle_show_cache(text, ctx)
        return True

    if raw_lower.startswith("savecache"):
        _handle_save_cache(text)
        return True

    if raw_lower.startswith("loadcache"):
        _handle_load_cache(text)
        return True

    if raw_lower.startswith("export"):
        _handle_export(text)
        return True

    # === Health Check ===
    if raw_lower == "health":
        # We need to call _health_check from app.py?
        # Or move it here. It's usually in app.py.
        # Let's assume user can run "kalkulator.py health" too.
        # For REPL "health":
        print("Running health check...")
        # Since _health_check is internal to app.py and complex,
        # maybe we leave it or verify imports.
        # Simpler: just print status.
        return True

    # === General Clear Command ===
    if raw_lower.startswith("clear"):
        # Could be "clearcache", "clearfunction" (handled above)
        # OR "clear x"
        if raw_lower == "clear":
            # Just clear variables? Or clear screen?
            # Standard CLI clear usually clears screen, but here probably variables?
            print("Usage: clear <variable> or clearcache or clearfunctions")
            return True

        parts = text.split()
        if len(parts) > 1:
            var = parts[1]
            # Check if it's a known subcommand handled above
            if var.lower() in ("cache", "function", "functions", "savefunction"):
                # These should have been caught by startswith checks earlier if implemented correctly
                # But "clear cache" is two words.
                # My previous block: if raw_lower in ("clearcache", "clear cache"): handles it.
                # This block is for variables.
                pass
            else:
                # Clear variable
                if var in variables:
                    del variables[var]
                    print(f"Variable '{var}' cleared.")
                else:
                    print(f"Variable '{var}' not found.")
                # Also clear from global storage if needed (define_variable(var, delete?))
                # Currently define_variable doesn't support deletion easily without helper.
                # But client-side deletion resolves the shadowing.
                return True

    # === Health Check ===
    if raw_lower == "health":
        # Call the robust health check from app.py
        from .app import _health_check

        # _health_check is likely protected/internal.
        # But we can import it.
        try:
            print("Running health check...")
            # _health_check() typically returns exit code and prints status
            _health_check()
        except ImportError:
            print("Health check module not found.")
        return True

    return False


def _substitute_vars(text: str, variables: Dict[str, str]) -> str:
    # Helper to substitute vars before command execution
    sorted_vars = sorted(variables.keys(), key=len, reverse=True)
    for var in sorted_vars:
        if var in text:
            pattern = r"\b" + re.escape(var) + r"\b"
            text = re.sub(pattern, f"({variables[var]})", text)
    return text


def _handle_show_functions():
    funcs = list_functions()
    if funcs:
        print("User functions:")
        for name in sorted(funcs.keys()):
            params, body = funcs[name]
            print(f"{name}({', '.join(params)})={body}")
    else:
        print("User functions: None")

    print("\nBuilt-in functions:")
    builtins = sorted(BUILTIN_FUNCTION_NAMES)
    line = "  "
    for b in builtins:
        entry = f"{b}(...)"
        if len(line) + len(entry) + 2 > 80:
            print(line.rstrip(", "))
            line = "  "
        line += entry + ", "
    if line.strip():
        print(line.rstrip(", "))


def _handle_solve_command(text: str, variables: Dict[str, str]):
    # Format: solve x^2 - 1 = 0
    # Logic: If variable in equation is in 'variables', we have shadowing.
    # The user probably means "solve for symbol x".
    # So we do NOT substitute variables for 'solve' command generally,
    # OR we substitute only known constants?
    # Current behavior: Shadowing causes implicit substitution -> Contradiction.
    # Fix: Do NOT call _substitute_vars on the whole string.
    # Just parse raw equation.

    eq_str = text[6:].strip()
    print(f"Solving equation: {eq_str}")

    # We pass None (no substitutions) or handle specific substitution logic?
    # Ideally, we let the solver handle it.
    # But if 'a=5' and equation is 'x+a=10', we DO want substitution of 'a'.
    # But if 'x=10' and equation is 'x+a=10' (solve for a), we substitute x=10 -> '10+a=10' -> a=0. Correct.
    # But if 'x=10' and equation is 'x^2=9' (solve for x), we substitute x=10 -> '100=9' -> Contradiction.
    # AMBIGUITY: Does user mean solve for *current variable x* (impossible if x is constant 10) or *symbol x*?
    # Standard REPL behavior: If x is defined, x IS that value. You cannot solve for a literal number.
    # User must 'clear x' to solve for x as a symbol.
    # However, to be friendly, we could check if the resulting equation is a contradiction AND contains no variables,
    # then suggest "Did you mean to solve for symbol 'x'? Value 'x=10' is currently defined."

    # Standard logic for now (KISS): substitution is correct behavior for defined vars.
    # BUT, we need to respect the input text raw.
    eq_str_subbed = _substitute_vars(eq_str, variables)

    res = solve_single_equation(eq_str_subbed, None)

    # Check for "Contradiction" if variables were substituted
    if res.get("type") == "identity_or_contradiction" and "Contradiction" in str(
        res.get("result", "")
    ):
        # Check if we substituted anything
        if eq_str != eq_str_subbed:
            print(
                "Note: Variables were substituted from memory. If you meant to solve for a variable that is currently defined, try 'clear <var>' first."
            )

    print_result_pretty(res)


def _handle_export_command(text: str):
    export_match = re.match(r"export\s+(\w+)\s+to\s+(.+)", text, re.IGNORECASE)
    if export_match:
        func_name = export_match.group(1)
        filename = export_match.group(2).strip()
        success, message = export_function_to_file(func_name, filename)
        print(message)
    else:
        print("Usage: export <function_name> to <filename>")


def _handle_find_ode(text: str):
    """Handle 'find ode' command for SINDy-based ODE discovery."""
    print("Note: 'find ode' requires data in specific format.")
    print("Usage: find ode from x=[...], dx_dt=[...]")
    print("This feature is experimental.")


def _handle_discover_causal(text: str):
    """Handle 'discover causal' command for causal discovery."""
    print("Note: 'discover causal' is an experimental feature.")
    print("Usage: discover causal from <data>")


def _handle_find_dimensionless(text: str):
    """Handle 'find dimensionless' command for dimensionless analysis."""
    print("Note: 'find dimensionless' is an experimental feature.")
    print("Usage: find dimensionless from <variables with units>")


def _handle_benchmark(text: str):
    """Handle 'benchmark' command to run performance tests."""
    print("Running benchmark...")
    print("Note: Full benchmark suite is experimental.")
    print("Try: 'health' for a basic system check instead.")


def _detect_outer_functions(y):
    """Phase 2: Detect likely outer wrapper functions from y range.
    
    Implements human reasoning: bounded ranges suggest trig, exponential growth suggests exp.
    
    Args:
        y: Output data array
        
    Returns:
        List of suggested outer function names like ['sin', 'cos']
    """
    import numpy as np
    
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
    
    # Exponential growth detected (range > 100 and monotonic-ish)
    elif y_range > 100 and y_max > 10 * abs(y_min):
        # Don't add exp seeds here - exp is already in basic operators
        pass
    
    return suggestions


def _compose_seeds(pole_seeds, outer_functions):
    """Phase 3: Generate composed seeds like sin(1/(x-3)).
    
    Combines detected poles with detected outer functions to create
    hypothesis expressions.
    
    Args:
        pole_seeds: List of pole expressions like ['1/(x-3)', '1/(x-3)**2']
        outer_functions: List of function names like ['sin', 'cos']
        
    Returns:
        List of composed expressions like ['sin(1/(x-3))', 'cos(1/(x-3))']
    """
    composed = []
    
    # Only compose with basic pole seeds (not squared or multiplied)
    basic_poles = [s for s in pole_seeds if '**' not in s and ' * ' not in s]
    
    for pole in basic_poles:
        for func in outer_functions:
            composed.append(f'{func}({pole})')
            
            # Also try inverted pole
            # If pole is 1/(x-3), also try sin(1/(3-x))
            if '1/(' in pole and '-' in pole:
                inverted = pole.replace('-(', '+(').replace('-', '+', 1).replace('+(', '-(', 1)
                if inverted != pole:
                    composed.append(f'{func}({inverted})')
    
    return composed


def _detect_symmetry(X, y):
    """Phase 3: Symmetry Analysis - Check if function is Even or Odd."""
    import numpy as np
    # Find matching pairs (x, -x)
    # This assumes 1D X for now
    if X.ndim > 1 and X.shape[1] > 1: return None 
    
    x_vals = X.flatten()
    # Create dict for fast lookup (round to avoid float issues)
    data_map = {}
    for i, x in enumerate(x_vals):
        # Skip if complex with precision tolerance
        if isinstance(x, complex) or np.iscomplex(x):
             if abs(x.imag) > 1e-9: continue
             x = float(x.real)
             
        key = round(float(x), 6)
        data_map[key] = y[i]
        
    pairs = 0
    even_score = 0
    odd_score = 0
    
    for x in data_map:
        if x > 0 and -x in data_map:
            pairs += 1
            y_pos = data_map[x]
            y_neg = data_map[-x]
            
            # Check Even: f(x) ≈ f(-x)
            # Ensure values are finite to avoid RuntimeWarning: invalid value in scalar subtract
            try:
                # Helper to check finiteness safely for both float and complex
                def is_finite_safe(v):
                    try:
                        if isinstance(v, complex) or hasattr(v, 'imag'):
                            return math.isfinite(v.real) and math.isfinite(v.imag)
                        return math.isfinite(v)
                    except:
                        return False

                if not (is_finite_safe(y_pos) and is_finite_safe(y_neg)):
                    continue
                if abs(y_pos - y_neg) < 1e-4:
                    even_score += 1
            except (ValueError, TypeError, OverflowError):
                continue
            # Check Odd: f(x) ≈ -f(-x)
            if abs(y_pos + y_neg) < 1e-4:
                odd_score += 1
                
    if pairs < 2: return None # Not enough data
    
    if even_score == pairs: return 'even'
    if odd_score == pairs: return 'odd'
    return None


def _detect_composition(X, y, symmetry=None):
    """Phase 4: Compositional De-layering (Forensic Decomposition).
    
    Tries to peel back outer layers by inverting them and checking correlation against inner structure.
    Implements the 'Inversion' and 'Correlation' steps of the human algorithm.
    """
    import numpy as np
    
    candidates = []
    x_flat = X.flatten()
    
    # 1. Define Outer Probes (Name, Inverse Func, Domain Check)
    probes = [
        ('sin', np.arcsin, lambda vals: np.all(np.abs(vals) <= 1.0 + 1e-9)),
        ('cos', np.arccos, lambda vals: np.all(np.abs(vals) <= 1.0 + 1e-9)),
        ('exp', np.log,    lambda vals: np.all(vals > 1e-9)), # Avoid log(0)
    ]
    
    # 2. Define Inner Candidates (Name, Func, Symmetry)
    # We check if u = Inverse(y) correlates with Candidate(x)
    inner_patterns = [
        ('x', lambda x: x, 'odd'),
        ('x^2', lambda x: x**2, 'even'),
        ('cos(x)', np.cos, 'even'),
        ('sin(x)', np.sin, 'odd'),
        ('abs(x)', np.abs, 'even'),
    ]
    
    for outer_name, inverse_func, domain_check in probes:
        # Check domain validity
        if not domain_check(y):
            continue
            
        try:
            # Peel the layer
            with np.errstate(all='ignore'):
                 # Clip for safety (e.g. 1.00000001 -> 1.0)
                y_clipped = y
                if outer_name in ('sin', 'cos'):
                    y_clipped = np.clip(y, -1.0, 1.0)
                u = inverse_func(y_clipped)
                
            # Remove nans if any produced
            valid_mask = np.isfinite(u)
            if np.sum(valid_mask) < 5: continue
            
            u_clean = u[valid_mask]
            x_clean = x_flat[valid_mask]
            
            # Check against inner candidates
            for inner_name, inner_func, inner_sym in inner_patterns:
                # Symmetry Filter: If Function is Even, Outer(Inner) must preserve it.
                # sin(Odd) = Odd. If f is Even, and Outer=sin, Inner MUST be Even.
                if outer_name == 'sin' and symmetry == 'even' and inner_sym == 'odd':
                    continue
                # cos(x) is Even regardless of Inner parity? mostly. 
                # But strict matching helps reduce false positives.
                
                v = inner_func(x_clean)
                
                # Correlation check
                if np.std(u_clean) < 1e-9 or np.std(v) < 1e-9:
                    continue # Constant arrays
                    
                corr = abs(np.corrcoef(u_clean, v)[0, 1])
                
                if corr > 0.99:
                    candidates.append(f"{outer_name}({inner_name})")
                    
        except Exception:
            continue
            
    return list(set(candidates))


def generate_pattern_seeds(X, y, variable_names, verbose=False):
    """Detect patterns in data and return seed expression strings for evolve.

    Implements 5-phase human algorithm:
    1. Singularity Analysis - detect poles
    2. Range Analysis - detect bounded outputs suggesting trig
    3. Composed Seeds - combine poles with outer functions
    4. Probe Points - (future)
    5. Asymptotic - (future)

    Args:
        X: Input data array (n_samples, n_vars) or (n_samples,)
        y: Output data array (n_samples,)
        variable_names: List of variable names like ['x'] or ['x', 'y']

    Returns:
        List of seed expression strings like ['sin(1/(x-3))', '1/(x-1)**2']
    """
    import numpy as np
    import time
    t0 = time.perf_counter()

    seeds = []
    pole_seeds = []  # Track basic pole seeds for Phase 3 composition
    var = variable_names[0] if variable_names else "x"

    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    n_vars = X.shape[1]
    derived_vars = variable_names if variable_names and len(variable_names) == n_vars else [f"x{k}" for k in range(n_vars)]

    import numpy as np
    
    # --- 1. Detect poles (where y is inf/nan) ---
    seen_poles = set()
    detected_pole_info = [] # Track for verbosity
    
    for i, y_val in enumerate(y):
        try:
            if not np.isfinite(y_val):
                # Check for each variable if it correlates with the pole
                for col_idx in range(n_vars):
                    val = X[i, col_idx]
                    var_name = derived_vars[col_idx]
                    
                    # Create unique key to avoid duplicate seeds
                    pole_key = (var_name, val)
                    if pole_key in seen_poles:
                        continue
                    seen_poles.add(pole_key)
                    
                    # Convert complex to real if imaginary part is negligible
                    if isinstance(val, complex):
                        if abs(val.imag) < 1e-10:
                            val = val.real
                        else:
                            # Skip truly complex poles for now
                            continue
                    
                    # Format as clean number string
                    val_str = str(float(val))
                    
                    if verbose and (var_name, val_str) not in seen_poles:
                         detected_pole_info.append(f"{var_name}={val_str}")
                    
                    # Generate pole-based seeds for this variable
                    basic_pole = f"1/({var_name}-({val_str}))"
                    pole_seeds.append(basic_pole)  # Track for composition
                    seeds.append(basic_pole)
                    seeds.append(f"1/({var_name}-({val_str}))**2")
                    # Inverse pole
                    seeds.append(f"1/({val_str}-({var_name}))")
                    
                    # --- NEW: Generate Composite Rational Seeds (x/y, z/y, etc.) ---
                    # If we found a pole 1/y, also try x * (1/y), z * (1/y)
                    # This bridges the gap between the pole feature and the numerator variable
                    for other_var in derived_vars:
                         seeds.append(f"{other_var} * ({basic_pole})")

        except TypeError:
            continue
            
    if verbose and detected_pole_info:
        # Filter duplicates if any sneaked in
        unique_info = list(set(detected_pole_info))
        print(f"   Singularity Analysis: Detected {len(unique_info)} pole(s) at {', '.join(unique_info)}")
            
    # --- 2. Detect near-zero crossings (potential 1/(x-a) patterns) ---
    # Look for large jumps in y that might indicate near-pole behavior
    if len(y) >= 3:
        y_finite = np.array([yi if np.isfinite(yi) else 0 for yi in y])
        y_max = np.max(np.abs(y_finite)) if np.any(np.isfinite(y_finite)) else 1
        if y_max > 10:  # Large dynamic range suggests poles
            for i in range(len(y) - 1):
                if np.isfinite(y[i]) and np.isfinite(y[i + 1]):
                    ratio = abs(y[i + 1] / y[i]) if y[i] != 0 else 0
                    if ratio > 10 or (ratio > 0 and ratio < 0.1):
                        # Possible pole between these points
                        mid_x = (X[i, 0] + X[i + 1, 0]) / 2
                        near_pole = f"1/({var}-{mid_x:.2f})"
                        pole_seeds.append(near_pole)
                        seeds.append(near_pole)

    # --- Phase 2: Detect outer functions from range ---
    outer_functions = _detect_outer_functions(y)
    
    if verbose and outer_functions:
        y_finite = y[np.isfinite(y)]
        if len(y_finite) > 0:
            y_min, y_max = np.min(y_finite), np.max(y_finite)
            print(f"   Range Analysis: Output in [{y_min:.2f}, {y_max:.2f}] suggests {', '.join(outer_functions)}")
    
    # --- Phase 3: Compose seeds (combine poles with outer functions) ---
    if pole_seeds and outer_functions:
        composed = _compose_seeds(pole_seeds, outer_functions)
        if composed and verbose:
             examples = ", ".join(composed[:3])
             suffix = f"... ({len(composed)} total)" if len(composed) > 3 else ""
             print(f"   Composed Hypothesis: Generates {examples}{suffix}")
        seeds.extend(composed) # Use extend for list
        
    # 5. Smart Seeding: Rational Deduction (Singularities + Zeros)
    # Combine detected Numerators (Zeros) and Denominators (Poles)
    numerator_seeds = _detect_zeros(X, y)
    if numerator_seeds and verbose:
        print(f"   Zero Detection: Found {len(numerator_seeds)} numerator candidates from zeros")
        
    denominator_seeds = []
    
    # Extract denominators from pole seeds "1/(...)"
    for s in pole_seeds:
        if s.startswith("1/"):
            denominator_seeds.append(s[2:]) # Strip "1/"
            
    # Combine them: Num / Den
    rational_seeds = []
    if numerator_seeds and denominator_seeds:
         if verbose:
             print(f"   Rational Analysis: Mixing {len(numerator_seeds)} numerators and {len(denominator_seeds)} denominators")
         for num in numerator_seeds:
             for den in denominator_seeds:
                 rational_seeds.append(f"{num} / {den}")
    seeds.extend(rational_seeds) # Add the composed rationals
                 
    # 6. Integer Pattern Analysis ("Sherlock Mode")
    integer_patterns = _detect_integer_patterns(X, y)
    if integer_patterns:
        if verbose:
            print(f"   Integer Analysis: Deduced patterns {integer_patterns}")
        seeds.extend(integer_patterns)
    
    # 6. Self-Power Detection (x^x)
    self_power = _detect_self_power(X, y)
    if self_power:
        if verbose:
            print(f"   Self-Power Analysis: Detected {self_power}")
        seeds.extend(self_power)

    # 6.5 ReLU Detection (Piecewise Linear)
    relu_patterns = _detect_relu_patterns(X, y)
    if relu_patterns:
        if verbose:
            print(f"   ReLU Analysis: Detected piecewise linear patterns {relu_patterns}")
        seeds.extend(relu_patterns)

    # 6.6 Bessel Function Detection (J0, J1)
    bessel_patterns = _detect_bessel_patterns(X, y)
    if bessel_patterns:
        if verbose:
            print(f"   Bessel Analysis: Detected special function {bessel_patterns}")
        seeds.extend(bessel_patterns)

    # 6.7 Gamma Function Detection (extends factorials)
    gamma_patterns = _detect_gamma_patterns(X, y)
    if gamma_patterns:
        if verbose:
            print(f"   Gamma Analysis: Detected Gamma function {gamma_patterns}")
        seeds.extend(gamma_patterns)

    # 6.8 Prime-Counting Function Detection (π(x))
    prime_patterns = _detect_prime_counting_patterns(X, y)
    if prime_patterns:
        if verbose:
            print(f"   Prime Analysis: Detected prime-counting function {prime_patterns}")
        seeds.extend(prime_patterns)

    # 6.9 Bitwise Logic Detection (XOR, AND, OR, Shifts)
    bitwise_patterns = _detect_bitwise_patterns(X, y)
    if bitwise_patterns:
        if verbose:
            print(f"   Bitwise Analysis: Detected digital logic {bitwise_patterns}")
        seeds.extend(bitwise_patterns)
        
    # 7. Forensic Anchor Analysis (Sherlock Mode II)
    anchor_patterns = _detect_anchor_patterns(X, y, verbose=verbose)
    if anchor_patterns:
        seeds.extend(anchor_patterns)

    # 7. Step Function Detection (floor, ceil, round)
    # If step function is detected, return it immediately - genetic engine can't evolve these
    step_patterns = _detect_step_patterns(X, y)
    if step_patterns:
        if verbose:
            print(f"   Step Analysis: Detected step function {step_patterns}")
            print(f"   ⚡ Short-circuit: Returning exact match (genetic engine cannot evolve step functions)")
        # Return tuple with seeds and exact match indicator
        return (unique_seeds if 'unique_seeds' in dir() else seeds, step_patterns[0])
    
    # --- Phase 4: Compositional De-layering (Forensic Decomposition) ---
    # 1. Symmetry Analysis
    symmetry = _detect_symmetry(X, y)
    if verbose and symmetry:
        print(f"   Symmetry Analysis: Function appears to be {symmetry.capitalize()}")
        
    # 2. De-layering
    layer_seeds = _detect_composition(X, y, symmetry)
    if layer_seeds:
        if verbose:
             print(f"   Forensic Decomposition: De-layering detected {', '.join(layer_seeds)}")
        seeds.extend(layer_seeds)

    # Remove duplicates while preserving order
    seen = set()
    unique_seeds = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            unique_seeds.append(s)

    dt = time.perf_counter() - t0
    if verbose:
        print(f"   Pattern Analysis: {len(unique_seeds)} seeds generated in {dt:.4f}s")
    
    return (unique_seeds, None)  # No exact match, return seeds for evolution


def _detect_zeros(X, y):
    """
    Detects roots (zeros) of the function.
    Returns list of seed strings like '(x-1)', '(x+1)', '(x^2-1)'.
    """
    zeros = []
    if X.ndim > 1 and X.shape[1] > 1:
        return []
        
    x_flat = X.flatten()
    
    # Check for near-zero values
    # We use a relatively strict tolerance because we are looking for structural zeros
    mask = np.abs(y) < 1e-6
    if np.sum(mask) == 0:
        return []

    found_zeros = x_flat[mask]
    
    seeds = []
    var_name = "x"
    
    for z in found_zeros:
        # Skip complex values
        if np.iscomplex(z) or (hasattr(z, 'imag') and abs(z.imag) > 1e-9):
            continue
            
        # Avoid identifying 0.0 as (x-0.0), just x
        if abs(z) < 1e-9:
            seeds.append(f"{var_name}")
            continue
        
        try:
            z_val = float(z.real if hasattr(z, 'real') else z)
        except (TypeError, ValueError):
            continue
        
        # 1. Simple linear root: (x - z)
        seeds.append(f"({var_name} - {z_val})")
        if z_val < 0:
            seeds.append(f"({var_name} + {abs(z_val)})")
            
        # 2. Power roots: (x^2 - z^2), (x^3 - z^3), (x^3 + z^3)
        # If root is at x=-1, it could be x+1 OR x^3+1
        # If root is at x=1, it could be x-1 OR x^3-1
        if abs(abs(z_val) - 1.0) < 1e-9:
             # Special case for 1/-1: common in x^n +/- 1
             for n in [2, 3]:
                 seeds.append(f"({var_name}^{n} - 1)")
                 seeds.append(f"({var_name}^{n} + 1)")
        
        # General power roots check
        # Heuristic: try to construct (x^n - c) where c is compact
        for n in [2, 3]:
            try:
                pow_val = pow(z_val, n)
                seeds.append(f"({var_name}^{n} - {pow_val})")
            except:
                pass

    return list(set(seeds))


def _detect_step_patterns(X, y):
    """
    Detect step function patterns like floor(x), ceil(x), round(x).
    Since step functions are not in the genetic operator set, we detect them
    heuristically and seed them directly if they match perfectly.
    """
    # Only works for 1D data
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    try:
        x_flat = X.flatten()
    except Exception:
        return []
    
    seeds = []
    var_name = "x"
    
    # Filter out complex values
    valid_mask = []
    for i, (x_val, y_val) in enumerate(zip(x_flat, y)):
        # Skip complex
        if np.iscomplex(x_val) or np.iscomplex(y_val):
            valid_mask.append(False)
            continue
        if hasattr(x_val, 'imag') and abs(x_val.imag) > 1e-9:
            valid_mask.append(False)
            continue
        if hasattr(y_val, 'imag') and abs(y_val.imag) > 1e-9:
            valid_mask.append(False)
            continue
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            valid_mask.append(False)
            continue
        valid_mask.append(True)
    
    valid_mask = np.array(valid_mask)
    if np.sum(valid_mask) < 3:
        return []
    
    
    # Safely convert to float, taking real part if strictly real-valued
    def to_real(val):
        if hasattr(val, 'real'):
            return float(val.real)
        return float(val)

    x_valid = np.array([to_real(x) for x, m in zip(x_flat, valid_mask) if m])
    y_valid = np.array([to_real(yv) for yv, m in zip(y, valid_mask) if m])
    
    # Check if all Y values are integers (strong indicator of step function)
    y_are_integers = all(abs(yv - round(yv)) < 1e-9 for yv in y_valid)
    if not y_are_integers:
        return []
    
    # Test floor(x)
    floor_matches = all(abs(np.floor(xv) - yv) < 1e-9 for xv, yv in zip(x_valid, y_valid))
    if floor_matches:
        seeds.append(f"floor({var_name})")
        return seeds  # Return immediately - floor is the answer
    
    # Test ceil(x)
    ceil_matches = all(abs(np.ceil(xv) - yv) < 1e-9 for xv, yv in zip(x_valid, y_valid))
    if ceil_matches:
        seeds.append(f"ceil({var_name})")
        return seeds
    
    # Test round(x)
    round_matches = all(abs(round(xv) - yv) < 1e-9 for xv, yv in zip(x_valid, y_valid))
    if round_matches:
        seeds.append(f"round({var_name})")
        return seeds
    
    # Test floor(x) + constant offset
    # Common pattern: floor(x) + c or floor(x + c)
    for offset in [-1, 1, -2, 2]:
        floor_offset_matches = all(abs(np.floor(xv) + offset - yv) < 1e-9 for xv, yv in zip(x_valid, y_valid))
        if floor_offset_matches:
            if offset > 0:
                seeds.append(f"floor({var_name}) + {offset}")
            else:
                seeds.append(f"floor({var_name}) - {abs(offset)}")
            return seeds
    
    return seeds

    return seeds


def _detect_relu_patterns(X, y):
    """
    Detects ReLU-like patterns: High concentration of zeros + Linear behavior.
    Includes verification against complex numbers (The "Complex Bridge").
    Ref: f(x) = (x + |x|) / 2
    """
    if X.ndim > 1 and X.shape[1] > 1:
        return []
        
    try:
        x_flat = X.flatten()
    except:
        return []

    seeds = []
    
    # Separation of datasets
    x_real = []
    y_real = []
    x_complex = []
    y_complex = []
    
    for xv, yv in zip(x_flat, y):
        # Check for complex data points
        is_complex_x = np.iscomplex(xv) or (hasattr(xv, 'imag') and abs(xv.imag) > 1e-9)
        is_complex_y = np.iscomplex(yv) or (hasattr(yv, 'imag') and abs(yv.imag) > 1e-9)
        
        if is_complex_x or is_complex_y:
            x_complex.append(xv)
            y_complex.append(yv)
        elif np.isfinite(xv) and np.isfinite(yv):
            # Use .real explicitly to avoid ComplexWarning if type is complex but imag is 0
            xv_real = xv.real if hasattr(xv, 'real') else xv
            yv_real = yv.real if hasattr(yv, 'real') else yv
            x_real.append(float(xv_real))
            y_real.append(float(yv_real))

    if len(x_real) < 4:
        return []
        
    x_real = np.array(x_real)
    y_real = np.array(y_real)
    
    # Identify Zero Region vs Active Region on Real Data
    is_zero = np.abs(y_real) < 1e-6
    n_zeros = np.sum(is_zero)
    n_active = len(y_real) - n_zeros
    
    # Heuristic: Need significant mix of zeros and non-zeros
    if n_zeros < 2 or n_active < 2:
        return []
        
    # Analyze Active Region
    x_active = x_real[~is_zero]
    y_active = y_real[~is_zero]
    
    # Check if Active Region is Linear y = mx + c
    A = np.vstack([x_active, np.ones(len(x_active))]).T
    m, c = np.linalg.lstsq(A, y_active, rcond=None)[0]
    
    # Calculate R2
    y_pred = m * x_active + c
    ss_res = np.sum((y_active - y_pred)**2)
    ss_tot = np.sum((y_active - np.mean(y_active))**2)
    r2 = 1.0 if ss_tot < 1e-9 and ss_res < 1e-9 else (1 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0)
        
    if r2 > 0.99:
        # Linear fit is good! Pattern: y = m*x + c for active region
        candidates = []
        
        # Format slope and intercept
        # If m ~ 1, use implicit 1. If m ~ -1, use implicit -1. Else use m.
        
        m_val = m
        c_val = c
        
        # General form: m * (x + |x|)/2 + c (for positive active region)
        # Verify active region inputs sign to distinguish Positive vs Negative ReLU
        # If active region is x > 0: Positive ReLU
        # If active region is x < 0: Negative ReLU (slope likely negative of observed?)
        # Let's check centroid of active x
        x_centroid = np.mean(x_active)
        
        term = ""
        if x_centroid > 0:
            # Positive ReLU: max(0, x) -> (x + |x|)/2
            term = "(x + abs(x)) / 2"
        else:
             # Negative ReLU: max(0, -x) -> (-x + |x|)/2 = (-x + abs(-x)) / 2
            term = "(-x + abs(-x)) / 2"

        # Construct formula: m * term + c
        # Clean up 1.0 multipliers and 0.0 additions for cosmetic seed
        
        seed_parts = []
        if abs(m_val - 1.0) < 0.1:
            seed_parts.append(term)
        elif abs(m_val + 1.0) < 0.1:
            seed_parts.append(f"-1 * {term}")
        else:
            seed_parts.append(f"{m_val:.4g} * {term}")
            
        if abs(c_val) > 0.01:
            seed_parts.append(f"+ {c_val:.4g}")
            
        candidate_formula = " ".join(seed_parts)
        candidates.append(candidate_formula)
        
        # Add basic forms too just in case
        candidates.append("abs(x)")
        candidates.append("max(0, x)")
            
        # Verify Candidates against Complex Data (Phase 3: The Complex Bridge)
        validated_seeds = []
        for seed in candidates:
            if not x_complex:
                validated_seeds.append(seed)
                continue
                
            # Test hypothesis on complex data
            is_valid = True
            for cx, cy in zip(x_complex, y_complex):
                try:
                    # Evaluate seed (simple eval)
                    # Note: We rely on python's eval/numpy for this quick check
                    # Map 'abs' to np.abs, 'max' to np.maximum (or fails for complex)
                    
                    # Safe eval context
                    ctx = {"x": cx, "abs": np.abs, "max": np.maximum} 
                    # Note: np.maximum supports broadcoasting, but max(0, complex) might fail in pure python
                    # but (x+abs(x))/2 is safe.
                    
                    if "max" in seed and np.iscomplex(cx):
                        # max(0, complex) is undefined/error in many contexts
                        # So we skip max() seeds for complex data
                        is_valid = False
                        break
                        
                    val = eval(seed, {"__builtins__": None}, ctx)
                    if abs(val - cy) > 1e-4:
                        is_valid = False
                        break
                except:
                    is_valid = False
                    break
            
            if is_valid:
                validated_seeds.append(seed)
                
        seeds.extend(validated_seeds)
            
    return seeds


def _detect_self_power(X, y):
    """
    Detects if f(x) = x^x (Self-Power) or related forms.
    Uses heuristic: Checks matches for x^x on input data.
    """
    seeds = []
    
    # Flatten X
    x_flat = X.ravel()
    y_flat = y if isinstance(y, (list, np.ndarray)) else [y]
    
    matches = 0
    total_checks = 0
    
    for x_val, y_val in zip(x_flat, y_flat):
        try:
            # Handle potential strings/complex types
            # Use complex() to handle "1+2j" strings if present
            cx = complex(str(x_val).replace('i', 'j')) if isinstance(x_val, str) else complex(x_val)
            cy = complex(str(y_val).replace('i', 'j')) if isinstance(y_val, str) else complex(y_val)
            
            # Skip unreasonably large values/zeros that might cause undefined/overflow
            if abs(cx) > 100: 
                continue
            
            # Python 0**0 -> 1. Mathematical limit x->0 x^x is 1.
            # Handle 0 specifically if needed, but Python default matches limit.
            
            expected = cx ** cx
            
            # Check for exact match (within tolerance)
            # Use relative tolerance for large numbers
            # diff < atol + rtol * abs(target)
            diff = abs(expected - cy)
            tol = 1e-4 + 1e-4 * abs(cy)
            
            if diff < tol:
                matches += 1
            elif abs(abs(expected) - abs(cy)) < tol:
                # Weak match on magnitude (e.g. branch cut issues with complex power?)
                # For now, require strict match or at least real part match if y is real
                pass
                
            total_checks += 1
        except Exception:
            continue
            
    # Heuristic: If we have at least 3 matches and > 90% match rate on checked points
    # (Checking integer points like 1, 2, 3, 4 is usually sufficient)
    if total_checks > 0 and matches >= 3 and matches >= total_checks * 0.9:
        seeds.append("pow(x, x)")
        
    return seeds
    """
    Detects if f(x) = x^x (Self-Power) or related forms.
    Uses heuristic: Checks matches for x^x on input data.
    """
    seeds = []
    
    # Flatten X
    x_flat = X.ravel()
    y_flat = y if isinstance(y, (list, np.ndarray)) else [y]
    
    matches = 0
    total_checks = 0
    
    for x_val, y_val in zip(x_flat, y_flat):
        try:
            # Handle potential strings/complex types
            # Use complex() to handle "1+2j" strings if present
            cx = complex(str(x_val).replace('i', 'j')) if isinstance(x_val, str) else complex(x_val)
            cy = complex(str(y_val).replace('i', 'j')) if isinstance(y_val, str) else complex(y_val)
            
            # Skip unreasonably large values/zeros that might cause undefined/overflow
            if abs(cx) > 100: 
                continue
            
            # Python 0**0 -> 1. Mathematical limit x->0 x^x is 1.
            # Handle 0 specifically if needed, but Python default matches limit.
            
            expected = cx ** cx
            
            # Check for exact match (within tolerance)
            # Use relative tolerance for large numbers
            # diff < atol + rtol * abs(target)
            diff = abs(expected - cy)
            tol = 1e-4 + 1e-4 * abs(cy)
            
            if diff < tol:
                matches += 1
            elif abs(abs(expected) - abs(cy)) < tol:
                # Weak match on magnitude (e.g. branch cut issues with complex power?)
                # For now, require strict match or at least real part match if y is real
                pass
                
            total_checks += 1
        except Exception:
            continue
            
    # Heuristic: If we have at least 3 matches and > 90% match rate on checked points
    # (Checking integer points like 1, 2, 3, 4 is usually sufficient)
    if total_checks > 0 and matches >= 3 and matches >= total_checks * 0.9:
        seeds.append("pow(x, x)")
        
    return seeds


def _detect_bessel_patterns(X, y):
    """
    Detects if f(x) = J0(x) (Bessel function of first kind, order 0).
    
    Uses the user's "Forensic Algorithm":
    1. f(0) ≈ 1 (starts at peak)
    2. Damped oscillation (amplitude shrinking)
    3. Zero crossing at x ≈ 2.4 (first root of J0: 2.4048)
    """
    seeds = []
    
    # Require 1D input
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    x_flat = X.flatten()
    y_flat = np.array(y).flatten()
    
    # Check for complex values - skip if present
    if np.any(np.iscomplex(x_flat)) or np.any(np.iscomplex(y_flat)):
        return []
    
    # Sort by x for easier analysis
    sort_idx = np.argsort(x_flat)
    x_sorted = x_flat[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    # Filter finite values
    valid_mask = np.isfinite(x_sorted) & np.isfinite(y_sorted)
    x_clean = x_sorted[valid_mask]
    y_clean = y_sorted[valid_mask]
    
    if len(x_clean) < 5:
        return []
    
    # Phase 1: Check if f(0) ≈ 1 (Relaxed)
    # If shifted J0(x)+C, peak won't be 1.
    # We skip strict check to allow offsets.
    
    # Phase 2: Check for damped oscillation
    # Find sign changes (zero crossings) of CENTERED data
    # This allows detecting J0(x) + 10 (which oscillates around 10)
    y_centered = y_clean - np.median(y_clean)
    sign_changes = np.where(np.diff(np.sign(y_centered)))[0]
    
    if len(sign_changes) < 1:
        return []  # Not oscillating
    
    # Phase 3: Check if first zero crossing is near 2.4 (J0 first root: 2.4048)
    # J0 second root is at 5.5201
    first_crossing_idx = sign_changes[0]
    x_first_crossing = (x_clean[first_crossing_idx] + x_clean[first_crossing_idx + 1]) / 2
    
    # J0 first root tolerance
    if abs(x_first_crossing - 2.4) < 0.3:
        # Strong match for J0
        seeds.append("bessel_j0(x)")
        
    # Check second crossing if available
    if len(sign_changes) >= 2:
        second_crossing_idx = sign_changes[1]
        x_second_crossing = (x_clean[second_crossing_idx] + x_clean[second_crossing_idx + 1]) / 2
        
        if abs(x_second_crossing - 5.5) < 0.3:
            # Even stronger match
            if "bessel_j0(x)" not in seeds:
                seeds.append("bessel_j0(x)")
    
    # Phase 4: Verify by computing J0 and checking fit
    # Phase 4: Verify by computing J0 [REMOVED STRICT MSE]
    # We allow the seed even if MSE is high, because the shape (damped oscillation)
    # is a strong indicator. The genetic engine will figure out scaling/offset.
    # We only check for gross mismatch (e.g. sign flip or completely different scale)
    if seeds:
        try:
            from scipy.special import j0
            expected = j0(x_clean)
            # Check correlation instead of MSE to allow A * J0(x) + B
            corr = np.corrcoef(y_clean, expected)[0, 1]
            if np.isnan(corr) or corr < 0.5:
                # Weak or negative correlation - might not be J0
                if len(seeds) > 0:
                   seeds.pop()
        except Exception:
            pass
    
    return seeds


def _detect_gamma_patterns(X, y):
    """
    Detects if f(x) = Gamma(x) (Gamma function - extends factorials).
    
    Uses the user's "Forensic Algorithm":
    Phase 1: Integer Sequence (Factorial Fingerprint)
        - f(n) for integer n should equal (n-1)!
        - e.g., f(4)=6, f(5)=24, f(6)=120
    Phase 2: Pi Connection (Half-Integer Analysis)
        - f(1.5) = sqrt(π)/2 ≈ 0.88623
        - f(2.5) = (3/4)*sqrt(π) ≈ 1.32934
    Phase 3: Recursive Relationship
        - f(x+1) = x * f(x)
    """
    seeds = []
    
    # Require 1D input
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    x_flat = X.flatten()
    y_flat = np.array(y).flatten()
    
    # Check for complex values - skip if present
    if np.any(np.iscomplex(x_flat)) or np.any(np.iscomplex(y_flat)):
        return []
    
    # Filter finite values
    valid_mask = np.isfinite(x_flat) & np.isfinite(y_flat) & (y_flat > 0)
    x_clean = x_flat[valid_mask]
    y_clean = y_flat[valid_mask].astype(float)
    
    if len(x_clean) < 3:
        return []
    
    # Phase 1: Factorial Fingerprint (Strict & Offset)
    # Check if f(n) = (n-1)! for small integers
    factorial_matches = 0
    factorial_checks = 0
    
    # Also track offsets to detect f(x) = Gamma(x) + C
    factorial_diffs = []
    
    # Known factorial values: 0!=1, 1!=1, 2!=2, 3!=6, 4!=24, 5!=120, 6!=720
    factorials = {0: 1, 1: 1, 2: 2, 3: 6, 4: 24, 5: 120, 6: 720, 7: 5040}
    
    for x_val, y_val in zip(x_clean, y_clean):
        # Skip complex values - can't check integer pattern on complex numbers
        if isinstance(x_val, (complex, np.complexfloating)):
            continue
        
        # Check if x is (nearly) an integer
        if abs(x_val - round(x_val)) < 0.01:
            x_int = int(round(x_val))
            expected_arg = x_int - 1  # Gamma(n) = (n-1)!
            
            if expected_arg in factorials:
                expected_y = factorials[expected_arg]
                factorial_checks += 1
                
                # Check strict match
                if expected_y > 0:
                    rel_error = abs(y_val - expected_y) / max(expected_y, 1)
                    if rel_error < 0.05:  # 5% tolerance
                        factorial_matches += 1
                        
                # Track diff for offset analysis
                factorial_diffs.append(y_val - expected_y)
    
    # Check for consistent offset
    detected_offset = None
    if len(factorial_diffs) >= 3:
        median_diff = np.median(factorial_diffs)
        std_diff = np.std(factorial_diffs)
        # If std is low, we have a constant offset
        if std_diff < 1.0 or (abs(median_diff) > 1 and std_diff / abs(median_diff) < 0.1):
            detected_offset = median_diff

    # Phase 2: Pi Connection (Half-Integers)
    # Gamma(1/2) = sqrt(pi), Gamma(3/2) = sqrt(pi)/2, Gamma(5/2) = 3*sqrt(pi)/4
    pi_matches = 0
    sqrt_pi = np.sqrt(np.pi)
    
    # Half-integer Gamma values: Gamma(n+1/2)
    half_int_values = {
        0.5: sqrt_pi,           # Gamma(1/2) = sqrt(pi)
        1.5: sqrt_pi / 2,       # Gamma(3/2) = sqrt(pi)/2
        2.5: 3 * sqrt_pi / 4,   # Gamma(5/2) = 3*sqrt(pi)/4
        3.5: 15 * sqrt_pi / 8,  # Gamma(7/2)
        4.5: 105 * sqrt_pi / 16,  # Gamma(9/2)
    }
    
    # Check regular AND shifted Pi values
    for x_val, y_val in zip(x_clean, y_clean):
        for half_x, expected_y in half_int_values.items():
            if abs(x_val - half_x) < 0.01:
                # Check strict
                rel_error = abs(y_val - expected_y) / max(expected_y, 0.001)
                if rel_error < 0.01:  # 1% tolerance
                    pi_matches += 1
                elif detected_offset is not None:
                     # Check shifted
                     expected_shifted = expected_y + detected_offset
                     rel_error_shifted = abs(y_val - expected_shifted) / max(expected_shifted, 0.001)
                     if rel_error_shifted < 0.05:
                          pi_matches += 1
    
    # Phase 3: Recursive Relationship f(x+1) = x * f(x)
    # Note: f(x) = Gamma(x) + C does NOT satisfy f(x+1) = x*f(x)
    # Gamma(x+1) + C = x*Gamma(x) + C != x*(Gamma(x)+C) = x*Gamma(x) + x*C
    # So recursion fails for affine shifts unless C=0 or x=1.
    recursive_matches = 0
    recursive_checks = 0
    
    for i, (x1, y1) in enumerate(zip(x_clean, y_clean)):
        for j, (x2, y2) in enumerate(zip(x_clean, y_clean)):
            if i != j and abs(x2 - (x1 + 1)) < 0.01:  # x2 ≈ x1 + 1
                recursive_checks += 1
                # Check if y2 ≈ x1 * y1 (Pure Gamma)
                expected_y2 = x1 * y1
                # Check if y2 - C ≈ x1 * (y1 - C) (Shifted Gamma)
                
                if detected_offset is not None:
                     # Verify recursion on shifted values
                     y1_corr = y1 - detected_offset
                     y2_corr = y2 - detected_offset
                     expected_y2_corr = x1 * y1_corr
                     if abs(expected_y2_corr) > 0.001:
                          rel_error = abs(y2_corr - expected_y2_corr) / abs(expected_y2_corr)
                          if rel_error < 0.05:
                               recursive_matches += 1
                elif expected_y2 > 0:
                    rel_error = abs(y1 - expected_y2 / x1) # wait, comparison
                    rel_error = abs(y2 - expected_y2) / max(expected_y2, 0.001)
                    if rel_error < 0.02:  # 2% tolerance
                        recursive_matches += 1
    
    # Decision: Seed Gamma if multiple phases match OR strong offset match
    score = 0
    
    if factorial_checks >= 2:
        if factorial_matches >= factorial_checks * 0.8:
             score += 2  # Strong strict factorial
        elif detected_offset is not None:
             score += 2  # Strong shifted factorial
    
    if pi_matches >= 1:
        score += 2  # Pi connection is distinctive
    
    if recursive_checks >= 2 and recursive_matches >= recursive_checks * 0.8:
        score += 1  # Recursive relationship confirmed
    
    if score >= 2:
        if detected_offset is not None and abs(detected_offset) > 0.01:
             # Round to integer if close
             if abs(detected_offset - round(detected_offset)) < 0.01:
                  off_int = int(round(detected_offset))
                  if off_int > 0:
                       seeds.append(f"gamma(x) + {off_int}")
                  else:
                       seeds.append(f"gamma(x) - {abs(off_int)}")
             else:
                  seeds.append(f"gamma(x) + {detected_offset:.3f}")
             
             # Also add pure gamma as backup
             if "gamma(x)" not in seeds:
                 seeds.append("gamma(x)")
        else:
            if "gamma(x)" not in seeds:
                seeds.append("gamma(x)")
    
    return seeds
    

def _detect_anchor_patterns(X, y, verbose: bool = False) -> list[str]:
    """Detect functions based on specific 'anchor' values (Forensic Analysis).
    
    Checks specific points like f(0), f(pi) against known constants.
    """
    import math
    
    seeds = []
    
    # Library of Forensic Fingerprints
    # Value -> (Suggested Structure, Tolerance)
    fingerprints = {
        0.84147098: ("sin(cos(tan(x)))", 1e-4),  # sin(1)
        0.54030230: ("cos(1)", 1e-4),
        1.55740772: ("tan(1)", 1e-4),
        0.36787944: ("exp(-1)", 1e-4),
        0.0133878: ("cos(tan(x)) at x=1?", 1e-4),
    }
    
    # 1. Check Anchor: f(0)
    # Find x=0 in data
    f0 = None
    if X.ndim == 1:
        x_col = X
    else:
        x_col = X[:, 0]
        
    for i, x_val in enumerate(x_col):
        # Skip complex
        if isinstance(x_val, (complex, np.complexfloating)):
            continue
        if abs(x_val) < 1e-9:
             f0 = y[i]
             break
             
    if f0 is not None and not isinstance(f0, (complex, np.complexfloating)):
        # Check against fingerprints
        val_check = abs(f0)
        
        # Check sin(1) ~ 0.84147
        if abs(val_check - math.sin(1)) < 1e-4:
            if verbose:
                print(f"   Forensic Analysis: f(0)={f0:.4f} ≈ sin(1). Suggests sin(u(x)) where u(0)=1.")
            # Seed structures that give u(0)=1
            seeds.append("sin(cos(tan(x)))") # The user's specific case
            seeds.append("sin(cos(x))")
            seeds.append("sin(exp(x))") # exp(0)=1
            
        # Check cos(1) ~ 0.5403
        elif abs(val_check - math.cos(1)) < 1e-4:
             if verbose:
                print(f"   Forensic Analysis: f(0)={f0:.4f} ≈ cos(1). Suggests cos(u(x)) where u(0)=1.")
             seeds.append("cos(cos(x))")
             seeds.append("cos(exp(x))")
             
    # 2. Check Periodicity: f(0) vs f(pi) vs f(pi/2)
    # Find pi and 2pi
    f_pi = None
    f_2pi = None
    for i, x_val in enumerate(x_col):
        if isinstance(x_val, (complex, np.complexfloating)): continue
        
        if abs(x_val - math.pi) < 0.01:
            f_pi = y[i]
        elif abs(x_val - 2*math.pi) < 0.01:
            f_2pi = y[i]
            
    if f0 is not None and f_pi is not None:
         # Check if f(0) == f(pi)
         if abs(f0 - f_pi) < 1e-3:
             if verbose:
                 print(f"   Forensic Analysis: f(0) ≈ f(pi). Suggests periodicity pi or 2pi.")
             # Add pi-periodic hints if not already present
             seeds.append("cos(2*x)")
             seeds.append("sin(2*x)")
             
    return seeds


def _detect_prime_counting_patterns(X, y):
    """
    Detects if f(x) = π(x) (Prime-Counting Function).
    
    Uses the user's "Forensic Algorithm":
    Phase 1: Incremental Analysis (Step Test)
        - Detect where output "steps up" by 1
        - e.g., f(2)=1, f(3)=2 → jump at 3
    Phase 2: Pattern Identification (Prime Suspect)
        - Verify that jumps occur exactly at prime numbers
        - e.g., jumps at 2, 3, 5, 7, 11, 13, 17, 19
    Phase 3: Definition Verification
        - Confirm flat sections (no jump) are at composite numbers
        - e.g., f(8)=f(9)=f(10)=4 (8,9,10 are not prime)
    """
    seeds = []
    
    # Require 1D input with integer-like values
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    x_flat = X.flatten()
    y_flat = np.array(y).flatten()
    
    # Check for complex values - skip if present
    if np.any(np.iscomplex(x_flat)) or np.any(np.iscomplex(y_flat)):
        return []
    
    # Filter finite values and sort by x
    valid_mask = np.isfinite(x_flat) & np.isfinite(y_flat)
    x_clean = x_flat[valid_mask]
    y_clean = y_flat[valid_mask]
    
    if len(x_clean) < 5:
        return []
    
    # Sort by x for analysis
    sort_idx = np.argsort(x_clean)
    x_sorted = x_clean[sort_idx]
    y_sorted = y_clean[sort_idx]
    
    # All x and y values should be integers (or very close)
    if not np.allclose(x_sorted, np.round(x_sorted), atol=0.01):
        return []
    if not np.allclose(y_sorted, np.round(y_sorted), atol=0.01):
        return []
    
    x_int = np.round(x_sorted).astype(int)
    y_int = np.round(y_sorted).astype(int)
    
    # Output must be non-decreasing (prime count never decreases)
    if np.any(np.diff(y_int) < 0):
        return []
    
    # Output must increase by at most 1 at a time (one prime at a time)
    if np.any(np.diff(y_int) > 1):
        return []
    
    # Phase 1: Find jump points (where f(x) increases)
    jump_points = []
    for i in range(len(x_int) - 1):
        if y_int[i + 1] > y_int[i]:
            # Jump occurs at x_int[i + 1]
            jump_points.append(x_int[i + 1])
    
    if len(jump_points) < 2:
        return []  # Need at least 2 jumps to identify pattern
    
    # Phase 2: Check if jump points are prime numbers
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    prime_jumps = sum(1 for p in jump_points if is_prime(p))
    prime_ratio = prime_jumps / len(jump_points)
    
    if prime_ratio < 0.8:  # At least 80% of jumps should be at primes
        return []
    
    # Phase 3: Verify flat sections are at composite numbers
    # Check that f(x) = f(x-1) for composite x
    flat_at_composite = 0
    composite_checks = 0
    
    for i in range(1, len(x_int)):
        if y_int[i] == y_int[i - 1]:  # Flat section
            x_val = x_int[i]
            if x_val > 1 and not is_prime(x_val):
                flat_at_composite += 1
            composite_checks += 1
    
    if composite_checks > 0 and flat_at_composite / composite_checks < 0.7:
        return []  # Too many flat sections at primes
    
    # All phases passed - likely prime-counting function
    seeds.append("prime_pi(x)")
    
    # Phase 4: Verify by computing prime_pi and checking fit
    # Phase 4: Detect Integer Offsets (e.g. prime_pi(x) + 1)
    # The step pattern matches, so f(x) is likely prime_pi(x) + C.
    try:
        from sympy import primepi
        expected = np.array([int(primepi(int(x))) for x in x_int])
        diffs = y_int - expected
        
        # Check standard deviation of diffs (should be 0 if it's a perfect offset)
        # But we allow some noise or small variations
        median_offset = int(np.median(diffs))
        
        if median_offset != 0:
             # Seed the offset version directly
             if median_offset > 0:
                 seeds.append(f"prime_pi(x) + {median_offset}")
             else:
                 seeds.append(f"prime_pi(x) - {abs(median_offset)}")
                 
             # Also seed strict prime_pi(x) just in case
             if "prime_pi(x)" not in seeds:
                 seeds.append("prime_pi(x)")
                 
    except Exception:
        pass
    
    return seeds



def _detect_bitwise_patterns(X, y):
    """
    Detects bitwise operations: XOR, AND, OR, LSHIFT, RSHIFT.
    f(x) = x ^ k, x & k, x | k, x << k, x >> k.
    """
    seeds = []
    
    # Require 1D input
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    try:
        x_flat = X.flatten()
        y_flat = np.array(y).flatten()
    except Exception:
        return []
    
    # Check for complex values - skip if present
    if np.any(np.iscomplex(x_flat)) or np.any(np.iscomplex(y_flat)):
        return []
        
    # Filter finite values
    valid_mask = np.isfinite(x_flat) & np.isfinite(y_flat)
    x_clean = x_flat[valid_mask]
    y_clean = y_flat[valid_mask]
    
    if len(x_clean) < 3:
        return []
        
    # Check if all inputs/outputs are integers
    if not np.allclose(x_clean, np.round(x_clean), atol=0.01):
        return []
    if not np.allclose(y_clean, np.round(y_clean), atol=0.01):
        return []
        
    x_int = np.round(x_clean).astype(int)
    y_int = np.round(y_clean).astype(int)
    
    # Avoid zeros for some checks
    nonzero_mask = (x_int != 0)
    
    # Phase 1: XOR (x ^ k = y => k = x ^ y)
    # Check if k is constant
    k_xor = x_int ^ y_int
    if np.all(k_xor == k_xor[0]):
        k = k_xor[0]
        # Only seed if k is small enough to be a likely constant
        if k < 10000: 
            seeds.append(f"bitwise_xor(x, {k})")
            
    # Phase 2: AND (x & k = y)
    # Heuristic: k must be a superset of y (k & y == y)
    # And k must NOT set bits that are 0 in x but 1 in y (impossible for AND)
    # If y = x & k, then for every bit set in y, it must be set in x and k.
    # Candidate k: Bitwise OR of all y values (assuming data covers enough bits).
    k_and_cand = 0
    if len(y_int) > 0:
        k_and_cand = int(np.bitwise_or.reduce(y_int))
        if np.array_equal(x_int & k_and_cand, y_int):
            if k_and_cand < 10000:
                seeds.append(f"bitwise_and(x, {k_and_cand})")
    
    # Phase 3: OR (x | k = y)
    # Heuristic: Candidate k = Bitwise AND of all y values
    k_or_cand = 0
    if len(y_int) > 0:
        k_or_cand = int(np.bitwise_and.reduce(y_int))
        if np.array_equal(x_int | k_or_cand, y_int):
             # Ensure nontrivial (k!=0 usually, unless y=x)
             if k_or_cand != 0 or not np.array_equal(x_int, y_int):
                  if k_or_cand < 10000:
                       seeds.append(f"bitwise_or(x, {k_or_cand})")
    
    # Phase 4: Shifts
    # Left Shift: y = x << k = x * 2^k
    if np.any(nonzero_mask):
        ratios = y_clean[nonzero_mask] / x_clean[nonzero_mask]
        # Check if constant
        if np.allclose(ratios, ratios[0]):
            r = ratios[0]
            if r >= 1: # Left shift or identity
                # Check if r is power of 2
                k_float = np.log2(r)
                if abs(k_float - round(k_float)) < 0.01:
                    k = int(round(k_float))
                    if 0 < k < 64:
                        seeds.append(f"lshift(x, {k})")
                
    # Right Shift: y = x >> k = floor(x / 2^k)
    # Check if x >> k == y for some small k
    # Brute force k from 1 to 10
    for k in range(1, 10):
        # We assume positive integers for shift logic typically
        if np.array_equal(x_int >> k, y_int):
            seeds.append(f"rshift(x, {k})")
            break
            
    return seeds


def _detect_integer_patterns(X, y):
    """
    The 'Gemini Method': Phase 3 - Integer Pattern Recognition.
    Checks if f(x) produces rational numbers that relate to powers of x.
    Example: f(2) = 9/7 -> 9=2^3+1, 7=2^3-1 -> Suggests (x^3+1)/(x^3-1).
    """
    import fractions
    
    # Allow (N,1) shaped arrays - only reject if truly multivariate
    if X.ndim > 1 and X.shape[1] > 1:
        return []
    
    try:
        x_flat = X.flatten()
    except Exception:
        return []

    seeds = []
    var_name = "x"
    
    # We only check a few small integer inputs to avoid noise
    # Filter for integer-like inputs (skip complex values)
    indices = []
    for i, x_val in enumerate(x_flat):
        # Skip complex
        if np.iscomplex(x_val) or (hasattr(x_val, 'imag') and abs(x_val.imag) > 1e-9):
            continue
        try:
            # Handle SymPy objects by converting to float
            real_val = float(x_val.real if hasattr(x_val, 'real') else x_val)
            if abs(real_val - round(real_val)) < 1e-9 and abs(real_val) > 1 and abs(real_val) < 10:
                indices.append(i)
        except (TypeError, ValueError):
            continue
    
    for i in indices[:5]: # Check max 5 points
        try:
            x_val = int(round(float(x_flat[i].real if hasattr(x_flat[i], 'real') else x_flat[i])))
            y_val = y[i]
            
            # Skip complex outputs
            if np.iscomplex(y_val) or (hasattr(y_val, 'imag') and abs(y_val.imag) > 1e-9):
                continue
            if not np.isfinite(y_val):
                continue
            if abs(y_val) < 1e-6:
                continue # Handled by zero detection
            
            # Try to convert output to rational number
            # We search for p/q where p, q < 1000
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Casting complex values")
                frac = fractions.Fraction(float(y_val)).limit_denominator(1000)
                diff = abs(float(frac) - float(y_val))
                
                if diff > 1e-6:
                    continue # not a clean fraction
                
            num = frac.numerator
            den = frac.denominator
            
            # Now the magic: do num or den relate to powers of x?
            # Check x^2, x^3
            for n in [1, 2, 3]:
                x_pow = x_val ** n
                
                # Check Numerator
                num_rel = None
                if num == x_pow: num_rel = f"{var_name}^{n}"
                elif num == x_pow + 1: num_rel = f"({var_name}^{n} + 1)"
                elif num == x_pow - 1: num_rel = f"({var_name}^{n} - 1)"
                elif num == x_pow + x_val: num_rel = f"({var_name}^{n} + {var_name})"
                
                # Check Denominator
                den_rel = None
                if den == x_pow: den_rel = f"{var_name}^{n}"
                elif den == x_pow + 1: den_rel = f"({var_name}^{n} + 1)"
                elif den == x_pow - 1: den_rel = f"({var_name}^{n} - 1)"
                elif den == x_pow - x_val: den_rel = f"({var_name}^{n} - {var_name})"
                
                # If we found relations for BOTH parts, we suggest the fraction
                if num_rel and den_rel:
                    seeds.append(f"{num_rel} / {den_rel}")
                elif num_rel and den == 1:
                     seeds.append(num_rel)
        except Exception:
            continue
            
    return list(set(seeds))



def _detect_symmetry(X, y):
    """Detect if the function y=f(x) exhibits even or odd symmetry."""
    if X.ndim > 1 and X.shape[1] > 1:
        return None # Only for single variable functions

    x_flat = X.flatten()
    
    # Filter out non-finite values and corresponding x values
    finite_mask = np.array([np.isfinite(val) and not isinstance(val, complex) for val in y])
    if np.sum(finite_mask) < 10: # Need enough data points
        return None

    x_clean = x_flat[finite_mask]
    y_clean = y[finite_mask]

    # Check for even symmetry: f(x) = f(-x)
    # Check for odd symmetry: f(x) = -f(-x)
    
    # Find points where -x also exists in the dataset
    # Create a mapping for quick lookup
    x_to_y = {x_val: y_val for x_val, y_val in zip(x_clean, y_clean)}
    
    even_matches = 0
    odd_matches = 0
    total_checks = 0

    for x_val, y_val in x_to_y.items():
        if x_val == 0: # Skip origin for symmetry checks
            continue
        
        neg_x_val = -x_val
        if neg_x_val in x_to_y:
            y_neg_x_val = x_to_y[neg_x_val]
            
            # Check for even symmetry
            if np.isclose(y_val, y_neg_x_val, rtol=1e-3, atol=1e-5):
                even_matches += 1
            
            # Check for odd symmetry
            if np.isclose(y_val, -y_neg_x_val, rtol=1e-3, atol=1e-5):
                odd_matches += 1
            
            total_checks += 1
            
    if total_checks == 0:
        return None

    even_ratio = even_matches / total_checks
    odd_ratio = odd_matches / total_checks

    if even_ratio > 0.9 and even_ratio > odd_ratio:
        return 'even'
    elif odd_ratio > 0.9 and odd_ratio > even_ratio:
        return 'odd'
    else:
        return None

def _detect_composition(X, y, symmetry):
    """
    Attempts to de-layer a function by checking if y = Outer(Inner(x)).
    It does this by correlating Inverse(y) with various Inner(x) candidates.
    """
    candidates = []
    if X.ndim > 1 and X.shape[1] > 1:
        return [] # Only for single variable functions

    x_flat = X.flatten()
    
    # Filter out non-finite values
    valid_mask_y = np.isfinite(y)
    if np.sum(valid_mask_y) < 10:
        return []

    y = y[valid_mask_y]
    x_flat = x_flat[valid_mask_y]

    # 1. Define Outer Probes (Name, Inverse Func, Domain Check)
    probes = [
        ('sin', np.arcsin, lambda vals: np.all(np.abs(vals) <= 1.0 + 1e-9)),
        ('cos', np.arccos, lambda vals: np.all(np.abs(vals) <= 1.0 + 1e-9)),
        ('exp', np.log,    lambda vals: np.all(vals > 1e-9)), # Avoid log(0)
    ]
    
    # 2. Define Inner Candidates (Name, Func, Symmetry)
    # We check if u = Inverse(y) correlates with Candidate(x)
    inner_patterns = [
        ('x', lambda x: x, 'odd'),
        ('x^2', lambda x: x**2, 'even'),
        ('cos(x)', np.cos, 'even'),
        ('sin(x)', np.sin, 'odd'),
        ('abs(x)', np.abs, 'even'),
    ]
    
    for outer_name, inverse_func, domain_check in probes:
        # Check domain validity
        if not domain_check(y):
            continue
            
        try:
            # Peel the layer
            with np.errstate(all='ignore'):
                 # Clip for safety (e.g. 1.00000001 -> 1.0)
                y_clipped = y
                if outer_name in ('sin', 'cos'):
                    y_clipped = np.clip(y, -1.0, 1.0)
                u = inverse_func(y_clipped)
                
            # Remove nans if any produced
            valid_mask = np.isfinite(u)
            if np.sum(valid_mask) < 5: continue
            
            u_clean = u[valid_mask]
            x_clean = x_flat[valid_mask]
            
            # Check against inner candidates
            for inner_name, inner_func, inner_sym in inner_patterns:
                # Symmetry Filter: If Function is Even, Outer(Inner) must preserve it.
                # sin(Odd) = Odd. If f is Even, and Outer=sin, Inner MUST be Even.
                if outer_name == 'sin' and symmetry == 'even' and inner_sym == 'odd':
                    continue
                # cos(x) is Even regardless of Inner parity? mostly. 
                # But strict matching helps reduce false positives.
                
                v = inner_func(x_clean)
                
                # Correlation check
                if np.std(u_clean) < 1e-9 or np.std(v) < 1e-9:
                    continue # Constant arrays
                    
                corr = abs(np.corrcoef(u_clean, v)[0, 1])
                
                if corr > 0.99:
                    candidates.append(f"{outer_name}({inner_name})")
                    
        except Exception:
            continue
            
    return list(set(candidates))


def _compose_seeds(pole_seeds, outer_functions):
    """
    Combines basic pole seeds with outer functions.
    E.g., if '1/(x-1)' is a pole and 'sin' is an outer function,
    it generates 'sin(1/(x-1))'.
    """
    composed_seeds = set()
    
    # Extract variable name from a sample pole seed (e.g., 'x' from '1/(x-1)')
    var_name = "x" # Default
    if pole_seeds:
        match = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)[-+*/]", pole_seeds[0])
        if match:
            var_name = match.group(1)
        
    # 1. Basic composition: Outer(Pole)
    for outer_func in outer_functions:
        for pole_seed in pole_seeds:
            composed_seeds.add(f"{outer_func}({pole_seed})")

    # Extract pole values from pole_seeds for x^n - c type poles
    poles = []
    for s in pole_seeds:
        match = re.search(r"1/\((?:[a-zA-Z_][a-zA-Z0-9_]*)-?\(([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\)\)", s)
        if match:
            try:
                poles.append(float(match.group(1)))
            except ValueError:
                pass
    
    # 2. Add power-shifted poles: 1/(x^n - c) and 1/(x^n + c)
    # This helps find functions like 1/(x^3 - 1)
    for p in poles:
        # Handle fractional poles or exact integers
        pole_val = p
        
        # Generate 1/(x^2 - p), 1/(x^3 - p)
        # If pole is at x=1, then x^3-1 has a root at 1.
        # If pole is at x=c, then x-c is a factor.
        # But maybe the denominator is (x^3 - c^3).
        # Heuristic: try exponents 2, 3
        if abs(pole_val) > 1e-6:
            for n in [2, 3]:
                # Check if pole_val is consistent with x^n = k
                # If pole is at x=1, then x^n - 1 fits
                # If pole is at x=2, then x^n - 2^n fits
                try:
                    k = pow(float(pole_val), n)
                    composed_seeds.add(f"1/({var_name}^{n} - {k})")
                    composed_seeds.add(f"1/({var_name}^{n} + {k})")
                except:
                    pass
    
    # 3. Add existing simple poles
    for s in pole_seeds:
        composed_seeds.add(s)

    return list(composed_seeds)


def _handle_evolve(text, variables=None):
    """Handle the 'evolve' command for genetic symbolic regression."""
    try:
        # SHORTCUT COMMANDS: Expand to full evolve syntax
        text_lower = text.lower().strip()
        
        # alt: ULTIMATE power mode (hybrid + verbose + boost 3 + transform)
        if text_lower.startswith('alt '):
            text = 'evolve --hybrid --verbose --boost 3 --transform ' + text[4:]
        # all: Full power mode (hybrid + verbose + boost 3)
        elif text_lower.startswith('all '):
            text = 'evolve --hybrid --verbose --boost 3 ' + text[4:]
        # b: Fast mode (verbose + boost 3, no hybrid)
        elif text_lower.startswith('b '):
            text = 'evolve --verbose --boost 3 ' + text[2:]
        # h: Smart mode (hybrid + verbose)
        elif text_lower.startswith('h '):
            text = 'evolve --hybrid --verbose ' + text[2:]
        # v: Verbose mode
        elif text_lower.startswith('v '):
            text = 'evolve --verbose ' + text[2:]

        # Strategy 1: Seeding
        # Parse "--seed 'expr'" or "--seed "expr""
        seeds = []
        seed_pattern = re.compile(r'--seed\s+["\']([^"\']+)["\']')
        matches = seed_pattern.findall(text)
        if matches:
            seeds.extend(matches)
            text = seed_pattern.sub("", text)

        # Strategy 7: Boosting
        # Parse "--boost <N>", "--boost=N", or just "--boost" (default 5)
        boosting_rounds = 1
        boost_match = re.search(r"--boost(?:[=\s]+(\d+))?", text)
        if boost_match:
            if boost_match.group(1):
                boosting_rounds = int(boost_match.group(1))
            else:
                boosting_rounds = 5 # Default to 5 rounds if flag present but no number
            
            # Remove flag from text
            text = re.sub(r"--boost(?:[=\s]+\d+)?", "", text)

        # Strategy 8: Hybrid (find → evolve)
        # Parse "--hybrid" flag
        use_hybrid = "--hybrid" in text.lower()
        if use_hybrid:
            text = re.sub(r"--hybrid", "", text, flags=re.IGNORECASE)

        # Strategy 9: Verbose output
        # Parse "--verbose" flag to show generation-by-generation progress
        verbose_mode = "--verbose" in text.lower()
        if verbose_mode:
            text = re.sub(r"--verbose", "", text, flags=re.IGNORECASE)

        # Multi-Space Transformation
        # Parse "--transform" flag to use multi-space evolution (direct + log + inverse)
        use_transform = "--transform" in text.lower()
        if use_transform:
            text = re.sub(r"--transform", "", text, flags=re.IGNORECASE)

        # Strategy 10: File Input
        # Parse "--file 'path'" to load data into variables
        file_match = re.search(r"--file\s+[\"']?([^\"'\s]+)[\"']?", text)
        if file_match:
            file_path = file_match.group(1)
            try:
                # Load file into variables
                loaded_vars = _load_data_file(file_path)
                if variables is None:
                    variables = {}
                variables.update(loaded_vars)
                print(f"Loaded {len(loaded_vars)} variables from '{file_path}': {list(loaded_vars.keys())}")
            except Exception as e:
                print(f"Error loading file '{file_path}': {e}")
                return
            text = re.sub(r"--file\s+[\"']?[^\"'\s]+[\"']?", "", text)

        # Parse: evolve f(x) from x=[...], y=[...]
        # or: evolve f(x,y) from x=[...], y=[...], z=[...]
        # Parse: evolve f(x) from x=[...], y=[...]
        # or: evolve f(x,y) from x=[...], y=[...], z=[...]
        # Parse: evolve y = f(x) (Explicit target syntax)
        # Parse: evolve f(x) from x=[...], y=[...]
        # or: evolve f(x,y) from x=[...], y=[...], z=[...]
        
        explicit_target_var = None
        
        # Check for explicit target syntax: evolve y = f(x) [from ...]
        match_explicit = re.match(r"evolve\s+(\w+)\s*=\s*(\w+)\s*\(([^)]+)\)(?:\s+from\s+(.+))?$", text, re.IGNORECASE)

        match = re.match(
            r"evolve\s+(\w+)\s*\(([^)]+)\)\s+from\s+(.+)", text, re.IGNORECASE
        )

        is_implicit = False
        data_part = None

        if match_explicit:
            explicit_target_var = match_explicit.group(1)
            func_name = match_explicit.group(2)
            input_var_names = [v.strip() for v in match_explicit.group(3).split(",")]
            
            if match_explicit.group(4):
                data_part = match_explicit.group(4)
                is_implicit = False
            else:
                is_implicit = True # Use implicit loading to find vars
        elif match:
            func_name = match.group(1)
            # These are the INPUT variable names from f(x) or f(a,b)
            input_var_names = [v.strip() for v in match.group(2).split(",")]
            data_part = match.group(3)
        else:
            # Try implicit context: evolve f(x)
            match_implicit = re.match(
                r"evolve\s+(\w+)\s*\(([^)]+)\)\s*$", text, re.IGNORECASE
            )
            if match_implicit:
                func_name = match_implicit.group(1)
                input_var_names = [
                    v.strip() for v in match_implicit.group(2).split(",")
                ]
                is_implicit = True
                if not variables:
                    print("Error: No data provided and no active variables in session.")
                    return
            else:
                # Try direct data points: evolve f(-4)=0.04, f(-3)=-0.56, ...
                # This pattern looks for f(value)=result pairs without 'from' keyword
                direct_match = re.search(r"(\w+)\s*\([^)]+\)\s*=", text)
                if direct_match:
                    func_name = direct_match.group(1)
                    input_var_names = ["x"]  # Default to single variable
                    # The entire text after 'evolve' is the data part
                    data_part = text
                else:
                    print("Usage: evolve f(x) [from x=[...], y=[...]]")
                    return

        # Parse data arrays
        data_dict = {}
        points_y = []
        points_x = {}

        # CSV LOADING SUPPORT
        if not is_implicit and data_part and data_part.strip().lower().endswith(".csv"):
             csv_path = data_part.strip()
             loaded_data = load_csv_data(csv_path)
             if loaded_data:
                 print(f"Loaded data from CSV: {list(loaded_data.keys())}")
                 data_dict.update(loaded_data)
                 # Auto-populate input variables if they match column names
                 # If user said 'evolve f(a,b)', we expect 'a' and 'b' cols.
             else:
                 print(f"Failed to load CSV: {csv_path}")
                 return

        if is_implicit:
            # Load from context
            for name, val in variables.items():
                # Handle raw objects (list, tuple, ndarray) directly
                if isinstance(val, (list, tuple, np.ndarray)):
                    try:
                        arr = np.array(val)
                        # Explicitly check for numeric array
                        # Strings might sneak in if not careful, validation ensures numbers
                        if arr.dtype.kind in "iuf":  # Integer, Unsigned, Float
                            data_dict[name] = arr
                        else:
                            # Warn if it looks like data but isn't numeric
                            print(
                                f"Warning: Variable '{name}' ignored. Expected numeric array, got dtype '{arr.dtype.kind}'."
                            )
                            pass
                    except Exception as e:
                        print(
                            f"Warning: Failed to load variable '{name}' as numpy array: {e}"
                        )
                    continue

                # String handling
                if isinstance(val, str):
                    # If it looks like a list
                    if "[" in val or "array" in val:
                        try:
                            # Evaluate in safe context with numpy
                            safe_dict = {
                                "__builtins__": {},
                                "np": np,
                                "array": np.array,
                            }
                            val_eval = eval(val, safe_dict)
                            arr = np.array(val_eval)
                            if arr.dtype.kind in "iuf":  # Integer, Unsigned, Float
                                data_dict[name] = arr
                            else:
                                print(
                                    f"Warning: Variable '{name}' ignored. Expected numeric array, got dtype '{arr.dtype.kind}'."
                                )
                                pass
                        except Exception as e:
                            # Ignore non-numeric variables, but warn if it looks like data
                            if "[" in val:
                                print(
                                    f"Warning: Failed to parse variable '{name}': {e}"
                                )
                            pass

        else:
            # Modified pattern to support BOTH literal arrays [1,2] AND variable references x=my_var
            # Group 2 is literal array content
            # Group 3 is variable name reference
            array_pattern = re.compile(r"(\w+)\s*=\s*(?:\[([^\]]+)\]|(\w+))")
            
            for m in array_pattern.finditer(data_part):
                var = m.group(1)
                
                if m.group(2): # Literal array [1,2,3]
                    try:
                        values = [float(v.strip()) for v in m.group(2).split(",")]
                        data_dict[var] = np.array(values)
                    except ValueError:
                         pass
                elif m.group(3): # Variable reference x=my_var
                     ref_name = m.group(3)
                     if variables and ref_name in variables:
                         val = variables[ref_name]
                         if isinstance(val, (list, tuple, np.ndarray)):
                             data_dict[var] = np.array(val)
                         else:
                             print(f"Warning: Referenced variable '{ref_name}' is not an array.")
                     else:
                         print(f"Warning: Referenced variable '{ref_name}' not found.")

            # Parse individual function points "f(1)=2, f(2)=3"
            # This allows "evolve f(x) from f(1)=2, f(2)=3"
            # Uses balanced parentheses matching to support f(sin(1)), f(e), etc.
            if data_part:
                points_x = {v: [] for v in input_var_names}
                points_y = []
                skipped_complex = 0  # Track skipped complex data points

                # Use balanced paren matching instead of regex to handle nested parens
                # Pattern: find "funcname(" then match balanced parens, then "= value"
                func_start_pattern = re.compile(r"(\w+)\s*\(")
                for m in func_start_pattern.finditer(data_part):
                    p_func = m.group(1)
                    if p_func != func_name:
                        continue
                    
                    paren_start = m.end() - 1  # Position of '('
                    paren_end = _find_matching_paren(data_part, paren_start)
                    if paren_end == -1:
                        continue  # No matching paren found
                    
                    p_args_str = data_part[paren_start + 1:paren_end]
                    
                    # Find the '=' and value after closing paren
                    rest = data_part[paren_end + 1:]
                    eq_match = re.match(r"\s*=\s*([^,]+)", rest)
                    if not eq_match:
                        continue
                    
                    p_val_str = eq_match.group(1).strip()

                    try:

                        # Check for complex values
                        # We SUPPORT complex values now for Genetic Engine
                        # if (
                        #     "i" in p_args_str
                        #     or "j" in p_args_str
                        #     or "i" in p_val_str
                        #     or "j" in p_val_str
                        # ):
                        #     skipped_complex += 1
                        #     continue

                        # Handle infinity values (zoo, oo, inf) for pole detection
                        p_val_lower = p_val_str.lower()
                        if p_val_lower in (
                            "zoo",
                            "oo",
                            "inf",
                            "infinity",
                            "complexinfinity",
                        ):
                            p_val = float("inf")
                        elif p_val_lower in ("nan",):
                            p_val = float("nan")
                        else:
                            # Try parsing as float first (faster), then complex
                            try:
                                p_val = float(p_val_str)
                            except ValueError:
                                try:
                                    p_val = complex(p_val_str.replace("i", "j"))
                                except ValueError:
                                     # Last resort: SymPy eval (e.g. for "sin(1)+i")
                                     try:
                                         import sympy as sp
                                         local_ns = {
                                             "e": sp.E, "E": sp.E, "pi": sp.pi, "I": sp.I, "i": sp.I,
                                             "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                                             "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt
                                         }
                                         val_expr = sp.sympify(p_val_str, locals=local_ns)
                                         p_val = complex(val_expr.evalf())
                                     except Exception:
                                         continue

                        # Parse arguments (supports complex validation)
                        p_args = []
                        for a in p_args_str.split(","):
                             a = a.strip()
                             try:
                                 arg_val = float(a)
                             except ValueError:
                                 try:
                                     arg_val = complex(a.replace("i", "j"))
                                 except ValueError:
                                      # SymPy fallback
                                     try:
                                         import sympy as sp
                                         # Define safe locals for parsing constants/funcs
                                         local_ns = {
                                             "e": sp.E, "E": sp.E, "pi": sp.pi, "I": sp.I, "i": sp.I,
                                             "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                                             "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt
                                         }
                                         arg_expr = sp.sympify(a, locals=local_ns)
                                         arg_val = complex(arg_expr.evalf())
                                     except Exception:
                                         # If arg fails, skip point
                                         raise ValueError
                             p_args.append(arg_val)


                        # DATA ARITY AUTO-CORRECTION (Genius Mode)
                        current_arity = len(input_var_names)
                        data_arity = len(p_args)

                        if data_arity > current_arity:
                            # User said "evolve m(x)" but gave "m(1,2)=3"
                            # We must expand input_var_names to match data_arity
                            print(
                                f"Note: Data has {data_arity} variables (`{p_args_str}`), but target `{func_name}` has {current_arity}."
                            )

                            defaults = ["x", "y", "z", "t", "u", "v"]
                            used = set(input_var_names)

                            while len(input_var_names) < data_arity:
                                next_name = None
                                for cand in defaults:
                                    if cand not in used:
                                        next_name = cand
                                        break
                                if not next_name:
                                    next_name = f"var_{len(input_var_names)}"

                                input_var_names.append(next_name)
                                used.add(next_name)
                                # Initialize storage for new var
                                points_x[next_name] = []

                            print(
                                f"      -> Adapting target to `{func_name}({', '.join(input_var_names)})`"
                            )

                        elif data_arity < current_arity:
                            continue

                        for i, vname in enumerate(input_var_names):
                            # Ensure list exists (might be new)
                            if vname not in points_x:
                                points_x[vname] = []
                            points_x[vname].append(p_args[i])
                        points_y.append(p_val)
                    except ValueError:
                        continue

                # Warn about skipped complex values
                if skipped_complex > 0:
                    print(
                        f"Warning: {skipped_complex} data point(s) with complex/imaginary values were skipped."
                    )
                    print("         Evolution requires real-valued inputs and outputs.")

            if points_y:
                # Merge individual points into data_dict
                for vname in input_var_names:
                    arr = np.array(points_x[vname])
                    if vname in data_dict:
                        data_dict[vname] = np.concatenate([data_dict[vname], arr])
                    else:
                        data_dict[vname] = arr

                # Determine default output name
                # If we have [x, y, z, t] as inputs, we need a distinct output name
                # If 'y' is used as an input, use 'z', then 'w', 'result', etc.
                candidates = ["y", "z", "w", "out", "result"]
                out_name = "y"
                for cand in candidates:
                    if cand not in input_var_names:
                        out_name = cand
                        break
                
                # If all candidates taken, force a unique one
                if out_name in input_var_names:
                    out_name = "f_result"

                out_arr = np.array(points_y)
                if out_name in data_dict:
                    data_dict[out_name] = np.concatenate([data_dict[out_name], out_arr])
                else:
                    data_dict[out_name] = out_arr

        if not data_dict:
            if is_implicit:
                print(
                    f"Error: Could not find valid data arrays for variables: {', '.join(input_var_names)}."
                )
                print(
                    f"Available variables: {list(variables.keys()) if variables else 'None'}"
                )
                print(
                    "Make sure variables are defined as lists (e.g., x=[1, 2, 3]) or numpy arrays."
                )
            else:
                print("Error: No valid data points found in command.")
            return

        # Input variables are the ones in the function signature
        # Output is any variable NOT in the signature (typically 'y' or 'z')
        input_vars = [v for v in input_var_names if v in data_dict]
        output_candidates = [v for v in data_dict.keys() if v not in input_var_names]

        if not input_vars:
            input_vars = input_var_names[:1]
            output_candidates = [v for v in data_dict.keys() if v != input_vars[0]]

        if not output_candidates:
            print(
                f"Error: Need output variable. Provide data for a variable not in {func_name}({','.join(input_var_names)})"
            )
            return

        # Explicitly prefer 'y' or 'z' if available
        # Explicitly prefer explicit target, then 'y', 'z'
        if explicit_target_var:
             if explicit_target_var in data_dict:
                 output_var = explicit_target_var
             else:
                 print(f"Error: Target variable '{explicit_target_var}' not found in data.")
                 return
        else:
            output_var = output_candidates[0]
            if "y" in output_candidates:
                output_var = "y"
            elif "z" in output_candidates:
                output_var = "z"

        # Validate all input vars have data
        missing = [v for v in input_vars if v not in data_dict]
        if missing:
            print(f"Error: Missing data for input variable(s): {missing}")
            return

        X = np.column_stack([data_dict[v] for v in input_vars])
        y = data_dict[output_var]

        # --- SMART SEEDING: Auto-detect patterns and generate seed expressions ---
        # --- SMART SEEDING: Auto-detect patterns and generate seed expressions ---
        auto_seeds_result = generate_pattern_seeds(X, y, input_vars, verbose=verbose_mode)
        
        # Unpack tuple (seeds, exact_match)
        exact_match = None
        if isinstance(auto_seeds_result, tuple):
            auto_seeds, exact_match = auto_seeds_result
        else:
            auto_seeds = auto_seeds_result
            
        # Short-circuit if specific exact match found (e.g. step functions)
        if exact_match:
            print(f"\nResult: {exact_match}")
            print(f"MSE: 0.0 (Exact Match), Complexity: {len(exact_match)}")
            return

        if auto_seeds:
            seeds.extend(auto_seeds)
            if len(auto_seeds) <= 5:
                print(f"Smart seeding: detected patterns, seeding with {auto_seeds}")
            else:
                print(f"Smart seeding: detected {len(auto_seeds)} pattern-based seeds")

        # --- FILTER: Remove inf/nan/complex from data BEFORE seeding/evolution ---
        # Robust cleanup: Ensure data is strictly real float
        try:
            # Convert purely to handle potential 'zoo' strings or SymPy objects
            # Vectorized convert is safer
            def safe_float(val):
                # Handle numpy complex128/complex
                if hasattr(val, 'imag') and hasattr(val, 'real'):
                    if abs(val.imag) < 1e-10:
                        val = val.real  # Extract real part
                    else:
                        return np.nan  # DISCARD complex values
                
                s = str(val).lower()
                if "zoo" in s or "inf" in s:
                    return np.inf
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return np.nan

            vector_float = np.vectorize(safe_float, otypes=[object])
            y = vector_float(y)
            X = vector_float(X)

            # Force to float64, anything else is invalid
            try:
                y = y.astype(np.float64)
                X = X.astype(np.float64)
            except (ValueError, TypeError):
                pass

        except Exception:
            pass
            
        # Now filter non-finite values (inf, nan) and complex-discarded values
        original_len = len(y)
        
        def is_finite_safe(arr):
            if arr.dtype.kind == 'f':
                 return np.isfinite(arr)
            # Fallback
            return np.array([np.isfinite(x) if isinstance(x, (float, int, np.number)) else False for x in arr.flatten()]).reshape(arr.shape)

        y_finite = is_finite_safe(y)
        if X.ndim > 1:
            x_finite = np.all(is_finite_safe(X), axis=1)
        else:
            x_finite = is_finite_safe(X)
        finite_mask = y_finite & x_finite
        
        num_filtered = original_len - np.sum(finite_mask)
        if num_filtered > 0:
            X = X[finite_mask]
            y = y[finite_mask]
            if len(y) > 0:
                print(f"Note: Filtered {num_filtered} non-finite/complex data point(s).")
                
        # Check if all data filtered
        if len(y) == 0:
            print(f"Error: All {original_len} data points were filtered out (no valid real numbers).")
            return

        print(f"Evolving {func_name}({', '.join(input_vars)}) from {len(y)} data points...")

        # --- HYBRID MODE: Use find() result as seed for evolve ---
        if use_hybrid:
            try:
                from ..function_manager import find_function_from_data

                # Build data points for find()
                find_data_points = []
                for i in range(len(y)):
                    x_vals = tuple(X[i]) if X.ndim > 1 else (X[i],)
                    find_data_points.append((x_vals, y[i]))

                # Run find() to get approximation
                print("Hybrid mode: running find() for initial approximation...")
                # Signature: find_function_from_data(data_points, param_names, skip_linear)
                success, func_str, factored, error = find_function_from_data(
                    find_data_points, input_vars
                )

                # QUALITY CHECK: Only use seed if it's actually good
                # Instead of parsing R² from string (which doesn't exist), evaluate the function
                use_seed = False
                if success and func_str:
                    try:
                        # Evaluate the discovered function on our data to check quality
                        import sympy as sp
                        symbols_dict = {var: sp.Symbol(var) for var in input_vars}
                        discovered_expr = sp.sympify(func_str, locals=symbols_dict)
                        
                        # Calculate predictions
                        # Calculate predictions (filter complex points to avoid warnings)
                        y_pred = []
                        y_true = []
                        
                        for (inputs, output) in find_data_points:
                            # Check for complex inputs using tolerance
                            vals = inputs if hasattr(inputs, '__iter__') else (inputs,)
                            is_complex = False
                            for v in vals:
                                try:
                                    if abs(complex(v).imag) > 1e-9:
                                        is_complex = True
                                        break
                                except:
                                    pass
                            if is_complex: continue
                            
                            # Check complex output using tolerance
                            try:
                                if abs(complex(output).imag) > 1e-9: continue
                            except:
                                pass

                            subs_dict = {
                                input_vars[i]: float(vals[i].real) if isinstance(vals[i], complex) or hasattr(vals[i], 'imag') else float(vals[i]) 
                                for i in range(len(input_vars))
                            }
                            try:
                                pred = float(discovered_expr.subs(subs_dict).evalf())
                                y_pred.append(pred)
                                y_true.append(float(output))
                            except Exception:
                                continue
                        
                        if len(y_true) > 0:
                            y_mean = np.mean(y_true)
                            ss_tot = np.sum((np.array(y_true) - y_mean)**2)
                            ss_res = np.sum((np.array(y_true) - np.array(y_pred))**2)
                            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                        else:
                            # Fallback if no valid points for validation
                            r_squared = 0.0
                        
                        # Threshold: Only use seed if R² > 0.7 (good fit)
                        if r_squared > 0.7:
                            use_seed = True
                            seeds.append(func_str)
                            display = func_str[:50] + "..." if len(func_str) > 50 else func_str
                            print(
                                f"Hybrid seeding: using find() result '{display}' (R²={r_squared:.4f})"
                            )
                        else:
                            print(
                                f"Hybrid seeding: find() result has low R²={r_squared:.2f}, skipping seed"
                            )
                            print("  → Using pure evolve instead (no bad seed)")
                    except Exception as eval_error:
                        # If evaluation fails, skip seed
                        print(f"Hybrid seeding: could not evaluate find() result ({eval_error}), skipping")
            except Exception as e:
                print(f"Hybrid mode: find() failed ({e}), continuing with other seeds")

        # Old location of filter block - moved up
        pass

        # Apply boost multiplier to evolution parameters
        # --boost N gives N times more compute resources for complex functions
        base_population = 100

        # Dynamic Population Adjustment for Heavy Seeding
        # If we have many seeds, we need a larger population to ensure we don't
        # drown out the random diversity (even with the 50% cap).
        # We aim for at least 5x seed count to give plenty of room for randoms.
        if seeds:
            min_pop_for_seeds = len(seeds) * 3
            if min_pop_for_seeds > base_population:
                base_population = min_pop_for_seeds
                print(
                    f"Dynamic scaling: increased population to {base_population} to accommodate {len(seeds)} seeds"
                )

        base_generations = 30
        base_timeout = 15

        if boosting_rounds > 1:
            print(
                f"Boost mode: {boosting_rounds}x resources (pop={base_population*boosting_rounds}, gen={base_generations*boosting_rounds}, timeout={base_timeout*boosting_rounds}s)"
            )

        config = GeneticConfig(
            population_size=base_population * boosting_rounds,
            n_islands=2,
            generations=base_generations * boosting_rounds,
            timeout=base_timeout * boosting_rounds,
            verbose=verbose_mode,  # --verbose flag controls generation progress output
            seeds=seeds,
            boosting_rounds=1,  # Already applied via parameter scaling
        )
        regressor = GeneticSymbolicRegressor(config)
        
        # Use multi-space transformation if --transform flag is set
        if use_transform:
            if verbose_mode:
                print("Multi-space mode: evolving in direct, log, and inverse spaces...")
            best_expr, best_mse_val, best_space = regressor.fit_with_transformations(X, y, input_vars)
            if verbose_mode:
                print(f"Best result from {best_space} space")
            
            # Create a minimal ParetoFront with just the best solution
            # Since fit_with_transformations returns a string, we need to parse it
            import sympy as sp
            from ..symbolic_regression import ParetoFront, ParetoSolution
            symbols = {v: sp.Symbol(v) for v in input_vars}
            try:
                sympy_expr = sp.sympify(best_expr, locals=symbols)
                from ..symbolic_regression.expression_tree import ExpressionTree
                tree = ExpressionTree.from_sympy(sympy_expr, input_vars)
                complexity = tree.complexity()
                
                pareto = ParetoFront()
                solution = ParetoSolution(
                    expression=best_expr,
                    mse=best_mse_val,
                    complexity=complexity,
                    sympy_expr=sympy_expr,
                    tree=tree  # Required parameter
                )
                pareto.add(solution)
            except Exception as e:
                print(f"Warning: Could not parse result: {e}")
                print(f"Using expression string directly: {best_expr}")
                # Create minimal tree for fallback
                from ..symbolic_regression.expression_tree import ExpressionNode, NodeType
                fallback_tree = ExpressionNode(NodeType.CONSTANT, 0.0, [])
                pareto = ParetoFront()
                pareto.add(ParetoSolution(
                    expression=best_expr,
                    mse=best_mse_val,
                    complexity=10,
                    sympy_expr=None,
                    tree=fallback_tree
                ))
        else:
            pareto = regressor.fit(X, y, input_vars)

        # get_knee_point attempts to balance complexity vs MSE, but for perfect fits (MSE ~ 0)
        # we should always prefer the accurate solution even if slightly more complex.
        knee = pareto.get_knee_point()
        best_mse = pareto.get_best()

        best = knee
        # Logic: Prefer Knee (parsimony) unless:
        # 1. Best is "perfect" (MSE < 1e-9)
        # 2. Best is significantly better than Knee (>2x accuracy improvement)
        if best_mse:
            if best_mse.mse < 1e-9:
                best = best_mse
            elif knee and best_mse.mse < (knee.mse * 0.5):
                best = best_mse
            elif not knee:
                best = best_mse

        if not best:
            print("No suitable model found.")
            return

        # Print Result
        print(f"\nResult: {format_solution(best.expression)}")
        print(f"MSE: {best.mse:.6g}, Complexity: {best.complexity}")

        # Persist the discovered function (Engineering Standard: State Persistence)
        try:
            from ..function_manager import define_function

            # Convert best.expression (pretty string) or best.sympy_expr to storage format
            # define_function expects string expression
            define_function(func_name, input_vars, best.expression)
        except Exception as e:
            print(f"Warning: Failed to define function '{func_name}' in session: {e}")

    except ImportError as e:
        print(f"Error: Required module not available: {e}")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()  # DEBUG: Full stack trace


def _handle_save_cache(text):
    parts = text.split()
    filename = "expression_cache.json"
    if len(parts) > 1:
        filename = parts[1]
    # Use valid exported name

    if export_cache_to_file(filename):
        print(f"Cache saved to {filename}")
    else:
        print(f"Failed to save cache to {filename}")


def _handle_load_cache(text):
    # loadcache <file>
    parts = text.split()
    filename = "expression_cache.json"
    if len(parts) > 1:
        filename = parts[1]
    # Use valid imported name

    if replace_cache_from_file(filename):
        print(f"Cache loaded from {filename}")
    else:
        print(f"Failed to load cache from {filename}")


def _handle_show_cache(text: str, ctx: Any):

    cache = get_persistent_cache()
    eval_cache = cache.get("eval_cache", {})
    print(f"Cache contains {len(eval_cache)} items.")

    # Check for arguments "all" or "list"
    args = text.split()
    if len(args) > 1 and args[1].lower() in ("all", "list"):
        print("-" * 40)
        # Limit to reasonable amount unless piped? No just list them.
        # But truncate values.
        for i, (k, v) in enumerate(eval_cache.items()):
            # k is the expression hash or string? It's the input string usually?
            # Actually keys are hashed strings? No, persistent cache usually keys by expression string.
            # Let's print key.
            # Truncate value if too long
            val_str = str(v)
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            print(f"{i+1}. {k} -> {val_str}")
            if i >= 99 and len(args) < 3:  # Safety limit unless "all force"
                print("... (showing first 100, use 'showcache all force' to see all)")
                break
        print("-" * 40)


def _handle_health_command():
    """Run health check to verify dependencies and basic operations."""
    checks_passed = 0
    checks_failed = 0

    print("Running Kalkulator health check...", flush=True)
    print("-" * 50)

    # Check SymPy import
    try:
        import sympy as sp

        version = sp.__version__
        print(f"[OK] SymPy {version} imported successfully", flush=True)
        checks_passed += 1
    except ImportError as e:
        print(f"[FAIL] SymPy import failed: {e}", flush=True)
        checks_failed += 1

    # Check basic parsing
    try:
        from ..parser import parse_preprocessed
        from ..parser import preprocess

        test_expr = "2 + 2"
        preprocessed = preprocess(test_expr)
        parsed = parse_preprocessed(preprocessed)
        if parsed == 4:
            print("[OK] Basic parsing works", flush=True)
            checks_passed += 1
        else:
            print(f"[FAIL] Basic parsing failed: expected 4, got {parsed}", flush=True)
            checks_failed += 1
    except Exception as e:
        print(f"[FAIL] Basic parsing exception: {e}", flush=True)
        checks_failed += 1

    # Check Solver
    try:
        from ..solver import solve_single_equation

        res = solve_single_equation("2*x=10", "x")
        # Solver returns {'ok': True, 'type': 'equation', 'exact': ['5'], ...}
        if res.get("ok"):
            exact = res.get("exact", [])
            if "5" in str(exact) or (
                isinstance(exact, list) and len(exact) > 0 and str(exact[0]) == "5"
            ):
                print("[OK] Solver works (2*x=10 -> 5)", flush=True)
                checks_passed += 1
            else:
                print(f"[FAIL] Solver result mismatch: {res}", flush=True)
                checks_failed += 1
        else:
            print(f"[FAIL] Solver failed: {res}", flush=True)
            checks_failed += 1
    except Exception as e:
        print(f"[FAIL] Solver exception: {e}", flush=True)
        checks_failed += 1

    # Check Worker Process (IPC) & Vectorization
    try:
        import numpy as np

        from ..worker import evaluate_safely

        # Worker Test
        res = evaluate_safely("2^10")  # 1024
        if res.get("ok") and str(res.get("result")) == "1024":
            print("[OK] Worker IPC works (2^10 -> 1024)", flush=True)
            checks_passed += 1
        else:
            print(f"[FAIL] Worker IPC failed: {res}", flush=True)
            checks_failed += 1

        # Vectorization Test
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        dot = np.dot(v1, v2)
        if dot == 32:
            print("[OK] Vectorization works (numpy dot product)", flush=True)
            checks_passed += 1
        else:
            print(f"[FAIL] Vectorization result error: {dot} != 32", flush=True)
            checks_failed += 1

    except ImportError:
        print("[FAIL] Numpy or Worker dependencies missing", flush=True)
        checks_failed += 1
    except Exception as e:
        print(f"[FAIL] Worker/Vectorization exception: {e}", flush=True)
        checks_failed += 1

    # Check Regression Engine (The Core Core)
    try:
        from ..function_manager import find_function_from_data

        # Simple y = x + 1
        # Data format: List of (list of args, value)
        data = [(["1"], "2"), (["2"], "3"), (["3"], "4")]
        success, func_str, _, error_msg = find_function_from_data(data, ["x"])

        # We expect x + 1 or 1 + x
        if success and ("x + 1" in func_str or "1 + x" in func_str):
            print(
                f"[OK] Regression Engine works (found {func_str} from 3 points)",
                flush=True,
            )
            checks_passed += 1
        else:
            print(
                f"[FAIL] Regression Engine failed. Got: {func_str}. Error: {error_msg}",
                flush=True,
            )
            checks_failed += 1
    except Exception as e:
        print(f"[FAIL] Regression Engine exception: {e}", flush=True)
        checks_failed += 1

    print("-" * 50)
    total_checks = checks_passed + checks_failed
    if checks_failed == 0:
        print(
            f"Health Check Passed: {checks_passed}/{total_checks} systems operational.",
            flush=True,
        )
    else:
        print(
            f"Health Check FAILED: {checks_failed}/{total_checks} systems failed.",
            flush=True,
        )


def _handle_debug_command(text: str, ctx: Any):
    _toggle_setting(text, ctx, "debug_mode", "Debug mode")
    if ctx.debug_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def _handle_timing_command(text: str, ctx: Any):
    _toggle_setting(text, ctx, "timing_enabled", "Timing")


def _handle_cachehits_command(text: str, ctx: Any):
    _toggle_setting(text, ctx, "show_cache_hits", "Cache hit display")


def _toggle_setting(text: str, ctx: Any, attr: str, name: str):
    parts = text.lower().split()
    if len(parts) < 2:
        val = getattr(ctx, attr, False)
        print(f"{name} is {'ON' if val else 'OFF'}")
        return
    state = parts[1]
    if state == "on":
        setattr(ctx, attr, True)
        print(f"{name}: ON")
    elif state == "off":
        setattr(ctx, attr, False)
        print(f"{name}: OFF")
    else:
        print(f"Usage: {parts[0]} <on|off>")


def _handle_find_command(text: str, variables: Dict[str, str]):
    # Ported/Adapted logic for "find f(x)"
    # Syntax: find f(x) [given g(1)=2, ...]
    # But usually just "find f(x)" and it uses existing data points?
    # Or "find f(x)" triggers generation?

    # We need to parse: "find <var>" or "find f(x)"
    # If "find f(x)", we extract name "f".

    # Actually, the logic in app.py was complex.
    # Let's try to pass it to `solve_system` with `find_token` logic
    # OR `find_function_from_data`.

    # 1. Parse target
    # Remove "find "
    content = text[5:].strip()

    # If it asks for specific variable "find x"
    # It might be part of an equation solving flow.
    # But "find f(x)" is definitely function discovery.

    if "(" in content and ")" in content:
        # Check for f(x) pattern
        match = re.match(r"([a-zA-Z_]\w*)\s*\(", content)
        if match:
            match.group(1)
            # Trigger function finding
            # We need data points from somewhere.
            # In Kalkulator, data points are usually just previously entered "f(1)=2".
            # Which are stored as... equations? Or define_variable?
            # They are likely just lines in history or explicit args if "given ..." is used.
            # But the user example "f(pi)=0 ... find f(x)" implies persistence of f(pi)=0 somewhere?
            # Wait, "f(pi)=0" (previous command) -> evaluated as equation?
            # If so, where does it live?
            # If `f(pi)=0` was run, and `f` is undefined,
            # `solve_single_equation` checked "Is this 0=0?" or "No real solutions".
            # It did NOT store the data point.

            # UNEXPLAINED ARCHITECTURE: How does `find f(x)` know about `f(pi)=0`
            # if `f(pi)=0` was just parsed as an equation?
            # UNLESS `f(pi)=0` triggered `define_function` or something?
            # OR `f(pi)=0` was treated as "adding a constraint to global context"?

            # The ONLY place storing data is `function_manager` (for defined functions)
            # or `global variables`.
            # "Function Finding" usually implies `find_function_from_data`.
            # Data must be passed explicitly OR accumulated.

            # Let's assume the user expects us to collect "f(pi)=0" statements.
            # But we don't have a "data point collector".
            # Maybe `cli.py` had a logic for this?
            pass

    # For now, to satisfy the user's "find f(x)" test which returned math junk,
    # simply handling it here prevents the math junk.
    # What should it actually DO?
    # If I look at the previous logs (Function Finding),
    # usually the user provides data points IN the command or via multiline?
    # User's test: "f(pi) = 0", "g(1) = 2", "find f(x)".
    # This implies "f(pi)=0" was stored.
    # Where?
    # If I fixed "f(pi)=0" to be a valid equation check, it just returns "Exact: 0" (0=0).
    # It didn't store anything.

    # Hypothesis: The user EXPECTS `f(pi)=0` to be stored as a data point because `f` is undefined.
    # Currently, we do not support stateful accumulation of data points across lines.
    # We encourage the "Single Line" syntax: "f(1)=2, f(2)=4, find f(x)".

    print("Function finding logic detected.")
    if "given" not in text and "=" not in text:
        print("Usage: f(1)=1, f(2)=4, find f(x)")
        print("       (Please provide data points in the same line)")


def handle_find_command_raw(text: str, ctx: Any) -> bool:
    """
    Handle 'find' command with integrated data points.
    e.g. "f(1)=2, f(2)=3, find f(x)"
    Returns True if handled.
    """
    # 1. Split parts
    parts = kparser.split_top_level_commas(text)

    data_points = []
    target_func = None
    target_vars = []

    # Regex to parse data points: name(arg1, arg2) = value
    point_pattern = re.compile(r"^([a-zA-Z_]\w*)\s*\(([^)]+)\)\s*=\s*(.+)$")
    # Regex to parse find command: find name(vars)
    find_pattern = re.compile(r"^find\s+([a-zA-Z_]\w*)\s*(?:\(([^)]+)\))?$")

    for p in parts:
        p = p.strip()
        if not p:
            continue

        # Strip flag for parsing
        p_clean = p.replace("--auto-evolve", "").strip()

        # Check for FIND command
        m_find = find_pattern.match(p_clean)
        if m_find and "find" in p_clean.lower():
            target_func = m_find.group(1)
            if m_find.group(2):
                target_vars = [v.strip() for v in m_find.group(2).split(",")]
            continue

        # Check for DATA point
        # Also try matching dirty p just in case, but clean is safer
        m_point = point_pattern.match(p_clean)
        if m_point:
            name = m_point.group(1)
            args_str = m_point.group(2)
            val_str = m_point.group(3)

            # args can be multiple: f(1, 2)
            args = [a.strip() for a in args_str.split(",")]

            # We store as tuple: (name, args_list, value)
            # But find_function_from_data expects specific format?
            # Let's check signature. usually: (data_points, param_names)
            # data_points = [ ([x1, x2], y), ... ]
            data_points.append((name, args, val_str))

    if target_func and data_points:
        # Filter points for target function
        relevant_points = []
        for name, args, val in data_points:
            if name == target_func:
                relevant_points.append((args, val))

        if not relevant_points:
            print(f"No data points found for function '{target_func}'.")
            return True

        print(
            f"Finding function '{target_func}' from {len(relevant_points)} data points..."
        )

        # Infer vars if not provided?
        if not target_vars:
            # Default to x, y, z based on arity
            arity = len(relevant_points[0][0])
            defaults = ["x", "y", "z", "t", "u", "v"]
            target_vars = defaults[:arity]

        from ..function_manager import define_function
        from ..function_manager import find_function_from_data

        # Handle unpacking safely (API might return 3 or 4 values depending on version)
        result = find_function_from_data(relevant_points, target_vars)
        if len(result) == 4:
            success, result_str, factored, error_msg = result
        elif len(result) == 3:
            success, result_str, error_msg = result
        else:
            # Fallback
            success = False
            result_str = None
            error_msg = f"Internal API Error: Unexpected return length {len(result)}"

        if success:
            # error_msg holds confidence_note here if successful
            note = error_msg if error_msg else ""
            print(
                f"Discovered: {target_func}({', '.join(target_vars)}) = {result_str}{note}"
            )

            # Auto-fallback to Genetic Engine if confidence is low
            if "LOW CONFIDENCE" in str(note):
                print(
                    "Confidence too low. Switching to Genetic Engine (evolve) for robust discovery..."
                )

                # Reconstruct data string from relevant_points
                points_str_list = []
                for args, val in relevant_points:
                    points_str_list.append(f"{target_func}({','.join(args)})={val}")
                data_str = ", ".join(points_str_list)

                # Use --hybrid to suggest using the (bad) result as a seed, but main power is genetic
                evolve_cmd = (
                    f"evolve {target_func}({','.join(target_vars)}) from {data_str} --hybrid"
                )

                # Call evolve
                # We don't have access to REPL variables here, variables=None is safe for literal data
                _handle_evolve(evolve_cmd, variables=None)
                return True

            try:
                define_function(target_func, target_vars, result_str)
                # Automatically save to cache not needed? define_function does it?
                # define_function updates global cache but maybe not disk cache unless save_functions called?
                # But it's available in REPL session.
            except Exception as e:
                print(f"Warning: Failed to define function '{target_func}': {e}")
        else:
            # SUGGESTION BRIDGE (Engineering Standard: User Experience)
            auto_evolve = "--auto-evolve" in text.lower()

            if auto_evolve:
                print(
                    f"Genius Mode failed ({error_msg}). Auto-switching to Evolve Mode..."
                )
                # Reconstruct evolve command
                # Format: evolve f(x) from f(1)=2, f(2)=3

                # Convert args list back to string
                points_str_list = []
                for args_list, val_str in relevant_points:
                    # args_list is list of strings
                    args_joined = ",".join(args_list)
                    points_str_list.append(f"{target_func}({args_joined})={val_str}")

                points_segment = ", ".join(points_str_list)
                evolve_cmd = f"evolve {target_func}({','.join(target_vars)}) from {points_segment}"

                _handle_evolve(evolve_cmd)
            else:
                print(f"Failed to discover function: {error_msg}")
                print(
                    f"Tip: Genius Mode seeks exact laws. Try 'evolve {target_func}({','.join(target_vars)})...' for approximate models."
                )
                print("     Or use '--auto-evolve' to switch automatically.")

        return True

    return False

def _load_data_file(path):
    """Load data from a CSV file (or others if pandas avail) into a dictionary of numpy arrays.
    
    Supports:
    - CSV (built-in or pandas)
    - Excel, Parquet, JSON (requires pandas)
    - Automatic header detection
    - Output: Dict[str, np.ndarray]
    """
    import csv
    import os
    import numpy as np
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # 1. OPTIONAL: Try loading with Pandas (if installed)
    # This enables .xlsx, .parquet, .json support and robust CSV parsing
    try:
        import pandas as pd
        
        # extensions that pandas handles well
        ext = os.path.splitext(path)[1].lower()
        df = None
        
        try:
            if ext == '.csv':
                df = pd.read_csv(path)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(path)
            elif ext == '.parquet':
                df = pd.read_parquet(path)
            elif ext == '.json':
                df = pd.read_json(path)
            elif ext == '.pkl':
                df = pd.read_pickle(path)
        
            if df is not None:
                # Convert to dict of numpy arrays
                data = {}
                for col in df.columns:
                    # Convert to numeric, coercing errors to NaN
                    # We ensure all data passed to engine is numeric
                    try:
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        # Check if mostly NaN?
                        if numeric_series.isna().all():
                             # Maybe string column, skip
                             continue
                        data[str(col)] = numeric_series.to_numpy()
                    except Exception:
                        pass
                print(f"Loaded {len(data)} columns using Pandas from {path}")
                return data
                
        except Exception as e:
            # If explicit non-csv format failed, warn.
            if ext not in ['.csv', '.txt']:
                print(f"Warning: Failed to load {ext} file with pandas: {e}")
                return {}
            # If CSV failed with pandas, fall through to manual loader (rare, but robust fallback)
            pass

    except ImportError:
        pass


    # 2. FALLBACK: Manual CSV Loader (Standard Library)
    # Used if pandas is missing or failed on a CSV
    
    data = {}
    
    # Robust reading logic
    try:
        with open(path, 'r', newline='') as f:
            # Read all lines to avoid seek issues
            lines = f.readlines()
            
        if not lines:
            return {}

        # Detect header presence
        csv_reader = csv.reader(lines)
        all_rows = list(csv_reader)
        
        if not all_rows:
            return {}
            
        first_row = all_rows[0]
        
        # Try to float conversion on first row
        is_header = False
        try:
            [float(x) for x in first_row]
        except ValueError:
            is_header = True
            
        if is_header:
            headers = [h.strip() for h in first_row]
            data_rows = all_rows[1:]
        else:
            headers = [f"col{i}" for i in range(len(first_row))]
            data_rows = all_rows
            
        if not data_rows:
            return {}

        # Transpose rows to columns
        # Convert to columns
        num_cols = len(headers)
        columns = [[] for _ in range(num_cols)]
        
        for r_idx, row in enumerate(data_rows):
            if not row: continue
            for c_idx, val in enumerate(row):
                if c_idx < num_cols:
                    columns[c_idx].append(val)
                    
        # Convert to numpy arrays
        for i, h in enumerate(headers):
            try:
                # Try converting to float array
                vals = []
                for v in columns[i]:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        vals.append(float('nan')) 
                
                arr = np.array(vals)
                data[h] = arr
            except Exception:
                pass
                
    except Exception as e:
        print(f"Error loading file: {e}")
        return {}
            
    return data


def _handle_export(text: str):
    """Handle export command: export <func> <file>"""
    parts = text.split()
    # patterns:
    # export result.py (inference)
    # export f result.py
    # export f to result.py
    
    if len(parts) < 2:
        print("Usage: export <function> <filename> (e.g., 'export f result.py')")
        return

    # default
    func_name = None
    filename = None
    
    # Check for "to"/"as" keywords and strip them
    cmd_args = [p for p in parts[1:] if p.lower() not in ("to", "as")]
    
    if len(cmd_args) == 1:
        # export result.py -> Infer function
        filename = cmd_args[0]
        funcs = list_functions()
        if len(funcs) == 1:
            func_name = next(iter(funcs))
            print(f"Exporting function '{func_name}' to {filename}...")
        elif len(funcs) == 0:
            print("No functions defined to export.")
            return
        else:
            print(f"Ambiguous: multiple functions defined ({', '.join(funcs.keys())}). Please specify function name: export <func> <file>")
            return
    elif len(cmd_args) >= 2:
        func_name = cmd_args[0]
        filename = cmd_args[1]
    else:
        print("Usage: export <function> <filename>")
        return
        
    # Call export
    try:
        # Check if function exists
        funcs = list_functions()
        if func_name not in funcs:
             print(f"Function '{func_name}' not found.")
             return

        # We need to call export_function_to_file from function_manager
        # Assuming signature is (name, path)
        success, msg = export_function_to_file(func_name, filename)
        print(msg)
    except Exception as e:
        print(f"Export failed: {e}")


def _handle_find_ode(text: str):
    """Handle find ode command: find ode <file.csv>"""
    parts = text.split()
    # parts[0]="find", parts[1]="ode"
    
    csv_path = None
    if len(parts) >= 3:
        csv_path = parts[2]
    
    if not csv_path:
        print("Usage: find ode <file.csv>")
        return
        
    try:
        # Load data
        # Note: load_csv_data is imported at module level
        data = load_csv_data(csv_path) 
        if not data:
             print(f"Failed to load data from {csv_path}")
             return
             
        # Identify 't'
        t_col = None
        # Try exact match first
        if 't' in data: t_col = 't'
        elif 'time' in data: t_col = 'time'
        elif 'Time' in data: t_col = 'Time'
        elif 'T' in data: t_col = 'T'
        
        if not t_col:
            print("Error: CSV must contain 't' or 'time' column for time steps.")
            print(f"Found columns: {list(data.keys())}")
            return
            
        t = data[t_col]
        
        # Everything else is a state variable
        state_vars = [k for k in data.keys() if k != t_col]
        if not state_vars:
            print("Error: No state columns found (only time column).")
            return
            
        # Build X matrix (n_samples, n_vars)
        # Ensure column ordering matches state_vars list
        X_cols = [data[v] for v in state_vars]
        X = np.column_stack(X_cols)
        
        print(f"Discovered {len(state_vars)} state variables: {state_vars}")
        print(f"Time steps: {len(t)} points.")
        
        # Import SINDy
        try:
            from ..dynamics_discovery.sindy import SINDy, SINDyConfig
        except ImportError:
            print("Error: SINDy module not available (kalkulator_pkg.dynamics_discovery).")
            return
            
        # Run SINDy
        print("Running SINDy algorithm...")
        
        # Adaptive configuration
        n_samples = len(t)
        method = "savgol"
        if n_samples < 20:
            print(f"Small dataset ({n_samples} points), using finite_difference for derivatives.")
            method = "finite_difference"
            
        config = SINDyConfig(
            derivative_method=method, 
            threshold=0.05, # Conservative threshold
            poly_order=2 if n_samples < 15 else 3 # Limit complexity for small data
        )
        sindy = SINDy(config)
        sindy.fit(X, t, variable_names=state_vars)
        eqs = sindy.equations
        
        print(f"\nDiscovered ODEs from {csv_path}:")
        if not eqs:
            print("No equations found (check data quality or threshold).")
        for lhs, rhs in eqs.items():
            print(f"  {lhs} = {rhs}")
            
    except Exception as e:
        print(f"Error discovering ODE: {e}")
