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
from ..symbolic_regression.expression_tree import symbolify_constants
from ..utils.formatting import format_solution
from ..utils.formatting import print_result_pretty
from ..utils.formatting import print_result_pretty
from ..worker import clear_caches
from ..symbolic_regression.forensic_analysis import generate_pattern_seeds

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
    "plot",
    "alt",
    "altv",
    "all",
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

    # Shortcut commands route to evolve: alt, all, b, h, v, altv
    if raw_lower.startswith(("alt ", "all ", "b ", "h ", "v ", "altv ")):
        _handle_evolve(text, variables)
        return True

    # ODE discovery shortcut: 'ode f(...)' is equivalent to 'alt --discover-ode f(...)'
    if raw_lower.startswith("ode "):
        text = text[4:]  # Remove 'ode ' prefix
        text = "--discover-ode " + text  # Add the flag
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

    if raw_lower.startswith("plot"):
        _handle_plot_command(text, variables)
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



# Forensic Analysis functions have been moved to 'kalkulator_pkg/symbolic_regression/forensic_analysis.py'




# [Orphaned definitions deleted]
def _handle_evolve(text, variables=None):
    """Handle the 'evolve' command for genetic symbolic regression."""
    try:
        # SHORTCUT COMMANDS: Expand to full evolve syntax
        text_lower = text.lower().strip()
        
        # altv: SUPER-VERBOSE mode (like alt but with detailed analysis logging)
        # Shows EXACTLY what the engine is thinking for tech-savvy users
        if text_lower.startswith('altv '):
            text = 'evolve --hybrid --verbose --super-verbose --boost 3 --transform ' + text[5:]
        # alt: ULTIMATE power mode (hybrid + verbose + boost 3 + transform)
        elif text_lower.startswith('alt '):
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

        # Super-Verbose Mode: Detailed analysis logging for tech-savvy users
        # Shows EXACTLY what the engine is thinking at each step
        super_verbose = "--super-verbose" in text.lower() or "-sv" in text
        if super_verbose:
            text = re.sub(r"--super-verbose", "", text, flags=re.IGNORECASE)
            text = re.sub(r"-sv\b", "", text)

        # Multi-Space Transformation
        # Parse "--transform" flag to use multi-space evolution (direct + log + inverse)
        use_transform = "--transform" in text.lower()
        if use_transform:
            text = re.sub(r"--transform", "", text, flags=re.IGNORECASE)

        # High-Precision Mode
        # Parse "--high-precision" or "--hp" flag for arbitrary-precision arithmetic
        high_precision_mode = "--high-precision" in text.lower() or "--hp" in text.lower()
        if high_precision_mode:
            text = re.sub(r"--high-precision", "", text, flags=re.IGNORECASE)
            text = re.sub(r"--hp\b", "", text, flags=re.IGNORECASE)
            print("   [High-Precision Mode] Using arbitrary-precision arithmetic (50+ digits)")

        # Constraint-Based Search
        # Parse "--ban func1,func2,..." to restrict operator search space
        banned_operators = []
        ban_match = re.search(r'--ban\s+([a-zA-Z0-9_,]+)', text)
        if ban_match:
            banned_str = ban_match.group(1)
            banned_operators = [f.strip().lower() for f in banned_str.split(',') if f.strip()]
            text = re.sub(r'--ban\s+[a-zA-Z0-9_,]+', '', text)
            print(f"   [Constraint] Banned functions: {banned_operators}")

        # Polynomial Mode: Ban all transcendentals, force pure polynomial evolution
        # This enables Taylor series discovery for functions like sin(x)
        polynomial_taylor_seeds = []
        use_polynomial = "--polynomial" in text.lower()
        if use_polynomial:
            text = re.sub(r"--polynomial", "", text, flags=re.IGNORECASE)
            # Ban all transcendental and special functions
            polynomial_banned = [
                'sin', 'cos', 'tan', 'exp', 'log', 'sqrt',
                'bessel_j0', 'gamma', 'prime_pi',
                'bitwise_xor', 'bitwise_and', 'bitwise_or', 'lshift', 'rshift',
                'floor', 'ceil', 'frac'
            ]
            banned_operators.extend(polynomial_banned)
            print(f"   [Polynomial Mode] Forcing pure polynomial search")
            
            # Taylor Series Templates for common transcendentals
            # sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
            # cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720
            # exp(x) ≈ 1 + x + x²/2 + x³/6
            # sinh(x) ≈ x + x³/6 + x⁵/120
            # cosh(x) ≈ 1 + x²/2 + x⁴/24
            polynomial_taylor_seeds = [
                # Sine Taylor (odd, oscillatory)
                'x - x**3/6',
                'x - x**3/6 + x**5/120',
                'x - x**3/6 + x**5/120 - x**7/5040',
                # Cosine Taylor (even, oscillatory)
                '1 - x**2/2',
                '1 - x**2/2 + x**4/24',
                '1 - x**2/2 + x**4/24 - x**6/720',
                # Exponential Taylor
                '1 + x + x**2/2',
                '1 + x + x**2/2 + x**3/6',
                # Sinh Taylor (odd, exponential growth)
                'x + x**3/6',
                'x + x**3/6 + x**5/120',
                # Cosh Taylor (even, exponential growth)
                '1 + x**2/2',
                '1 + x**2/2 + x**4/24',
                # Generic polynomials
                'x + a*x**3',
                'x + a*x**3 + b*x**5',
                '1 + a*x**2',
                '1 + a*x**2 + b*x**4',
            ]
            seeds.extend(polynomial_taylor_seeds)
            print(f"   [Polynomial Mode] Seeding with {len(polynomial_taylor_seeds)} Taylor templates")

        # ODE Discovery Mode: Discover differential equations instead of curve-fitting
        # Parse "--discover-ode" flag to find relationships like y'' + y = 0
        use_discover_ode = "--discover-ode" in text.lower()
        if use_discover_ode:
            text = re.sub(r"--discover-ode", "", text, flags=re.IGNORECASE)
            print(f"   [ODE Discovery Mode] Will search for differential equations")

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
                # Try direct data points: evolve f(-4)=0.04, f(-3)=-0.56, ..., find f(x)
                # This pattern looks for f(value)=result pairs without 'from' keyword
                direct_match = re.search(r"(\w+)\s*\([^)]+\)\s*=", text)
                if direct_match:
                    func_name = direct_match.group(1)
                    
                    # Try to extract variable names from "find func(var1, var2)" clause
                    find_match = re.search(r"find\s+(\w+)\s*\(([^)]+)\)", text, re.IGNORECASE)
                    if find_match and find_match.group(1) == func_name:
                        # Extract variable names from find clause
                        input_var_names = [v.strip() for v in find_match.group(2).split(",")]
                    else:
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
                        # Use robust parsing utility
                        # This handles floats, complex numbers, pi, e, inf, nan, etc.
                        try:
                            from ..utils.parsing import eval_to_float
                        except ImportError:
                            # Fallback if module not found (e.g. during refactor)
                            def eval_to_float(v): return float(v)

                        p_val = eval_to_float(p_val_str)

                        # Parse arguments
                        p_args = []
                        for a in p_args_str.split(","):
                            a = a.strip()
                            arg_val = eval_to_float(a)
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
            beautified_match = symbolify_constants(exact_match)
            print(f"\nResult: {beautified_match}")
            print(f"MSE: 0.0 (Exact Match), Complexity: {len(beautified_match)}")
            return

        if auto_seeds:
            seeds.extend(auto_seeds)
            if len(auto_seeds) <= 5:
                print(f"Smart seeding: detected patterns, seeding with {auto_seeds}")
            else:
                print(f"Smart seeding: detected {len(auto_seeds)} pattern-based seeds")

        # --- FILTER: Remove inf/nan/zoo from data BEFORE seeding/evolution ---
        # Robust cleanup: Handle potential 'zoo' strings or SymPy objects
        # NOTE: Complex values ARE supported and should NOT be filtered
        try:
            def safe_convert(val):
                # Handle complex values - KEEP them (they're supported!)
                if isinstance(val, complex) or (hasattr(val, 'imag') and abs(val.imag) > 1e-10):
                    return val  # Keep complex values
                
                # Handle numpy complex128/complex
                if hasattr(val, 'imag') and hasattr(val, 'real'):
                    if abs(val.imag) < 1e-10:
                        val = val.real  # Extract real part for near-real values
                
                s = str(val).lower()
                if "zoo" in s or "inf" in s:
                    return np.inf
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return np.nan

            vector_convert = np.vectorize(safe_convert, otypes=[object])
            y = vector_convert(y)
            X = vector_convert(X)

            # Convert to complex64 if any complex values, else float64
            has_complex = any(isinstance(v, complex) for v in y.flatten()) or \
                          any(isinstance(v, complex) for v in X.flatten())
            if has_complex:
                try:
                    y = y.astype(np.complex128)
                    X = X.astype(np.complex128)
                except (ValueError, TypeError):
                    pass
            else:
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
            # Handle complex arrays - check both real and imaginary parts
            if np.iscomplexobj(arr):
                return np.isfinite(arr.real) & np.isfinite(arr.imag)
            if arr.dtype.kind == 'f':
                return np.isfinite(arr)
            # Fallback for mixed types
            def check_item(x):
                if isinstance(x, complex):
                    return np.isfinite(x.real) and np.isfinite(x.imag)
                elif isinstance(x, (float, int, np.number)):
                    return np.isfinite(x)
                return False
            return np.array([check_item(x) for x in arr.flatten()]).reshape(arr.shape)

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
        
        # --- ROBUST OUTLIER FILTERING (IQR-Based) ---
        # Remove extreme outliers that could corrupt evolution
        # This helps when input data has precision errors near poles (e.g., f(0.0005) wrong by 10 orders of magnitude)
        try:
            if len(y) >= 10 and not np.iscomplexobj(y):
                y_real = np.real(y) if np.iscomplexobj(y) else y.astype(float)
                q1 = np.percentile(y_real, 25)
                q3 = np.percentile(y_real, 75)
                iqr = q3 - q1
                
                # Use a relaxed 3*IQR fence (broader than 1.5*IQR)
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                outlier_mask = (y_real >= lower_bound) & (y_real <= upper_bound)
                num_outliers = np.sum(~outlier_mask)
                
                if num_outliers > 0 and num_outliers < len(y) * 0.3:  # Don't filter if too many outliers
                    X = X[outlier_mask]
                    y = y[outlier_mask]
                    print(f"Note: Filtered {num_outliers} outlier point(s) using IQR method.")
        except Exception:
            pass  # If outlier detection fails, continue with original data

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

                # Check if data has ACTUAL complex values (non-zero imaginary parts)
                # Note: np.iscomplexobj only checks dtype, not actual values!
                # sqrt(pi) is real but may be stored in complex128 array
                # Checks:
                # 1. We need Real data for Rational Analysis (scipy.optimize/numpy.polyfit don't like complex)
                # 2. But we shouldn't just GIVE UP if there are a few complex points (e.g. f(i)).
                #    We should filter them out and run Rational Analysis on the real subset.
                
                # Filter for Real-only points
                find_data_points_real = []
                count_complex_skipped = 0
                
                for i in range(len(y)):
                    # Check X realness
                    x_row = X[i] if X.ndim > 1 else np.array([X[i]])
                    if np.any(np.abs(np.imag(x_row)) > 1e-9):
                        count_complex_skipped += 1
                        continue
                        
                    # Check Y realness
                    if np.abs(np.imag(y[i])) > 1e-9:
                        count_complex_skipped += 1
                        continue
                        
                    # Add to real subset
                    x_vals = tuple(x_row.real) # extracting real part explicitly
                    find_data_points_real.append((x_vals, float(y[i].real)))

                if count_complex_skipped > 0 and len(find_data_points_real) < 5:
                     # Too few real points to run rational analysis reliably
                     print(f"Hybrid mode: skipping find() (only {len(find_data_points_real)} real points found, need 5+)")
                     success = False
                     func_str = None
                else:
                    if count_complex_skipped > 0:
                        print(f"Hybrid mode: filtering {count_complex_skipped} complex points, running find() on {len(find_data_points_real)} real points...")
                    else:
                        print("Hybrid mode: running find() for initial approximation...")
                    
                    # Super-verbose: Show input data statistics
                    if super_verbose:
                        y_vals_real = [p[1] for p in find_data_points_real]
                        x_vals_real = [p[0][0] for p in find_data_points_real]
                        print(f"\n[SV] INPUT DATA ANALYSIS:")
                        print(f"     Points: {len(find_data_points_real)}")
                        print(f"     X range: [{min(x_vals_real):.4g}, {max(x_vals_real):.4g}]")
                        print(f"     Y range: [{min(y_vals_real):.4g}, {max(y_vals_real):.4g}]")
                        print(f"     Y mean: {np.mean(y_vals_real):.4g}, std: {np.std(y_vals_real):.4g}")
                        
                    import warnings
                    with warnings.catch_warnings():
                         warnings.simplefilter("ignore")
                         # Run on the filtered real dataset
                         # Signature: find_function_from_data(data_points, param_names, skip_linear)
                         success, func_str, factored, error = find_function_from_data(
                             find_data_points_real, input_vars, verbose=super_verbose
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
                                pred_val = discovered_expr.subs(subs_dict).evalf()
                                pred = float(complex(pred_val).real)
                                y_pred.append(pred)
                                y_true.append(float(complex(output).real))
                            except Exception:
                                continue
                        
                        if len(y_true) > 0:
                            y_mean = np.mean(y_true)
                            ss_tot = np.sum((np.array(y_true) - y_mean)**2)
                            ss_res = np.sum((np.array(y_true) - np.array(y_pred))**2)
                            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                            mse_val = ss_res / len(y_true)
                        else:
                            # Fallback if no valid points for validation
                            r_squared = 0.0
                            mse_val = 1e9
                        
                        
                        # Threshold: Only use seed if R² > 0.7 (good fit)
                        # OR if it's a decent fit (R² > 0.4) with low MSE (useful for rational functions with poles)
                        # Relaxed to 0.05 to handle SVD linearization approximations
                        if r_squared > 0.7 or (r_squared > 0.4 and mse_val < 0.05):
                            use_seed = True
                            seeds.append(func_str)
                            display = func_str[:50] + "..." if len(func_str) > 50 else func_str
                            print(
                                f"Hybrid seeding: using find() result '{display}' (R²={r_squared:.4f}, MSE={mse_val:.6f})"
                            )
                            
                            # EARLY RETURN: If find() result is EXCELLENT (MSE < 0.01), skip evolution
                            # This handles cases where SVD finds perfect rational functions that
                            # the Genetic Engine can't parse or improve
                            if mse_val < 0.01:
                                print(f"\n🎯 find() discovered an excellent solution (MSE={mse_val:.6f})")
                                print(f"   Skipping evolution and returning directly.")
                                
                                # Format and return the result
                                from ..symbolic_regression.expression_tree import symbolify_constants
                                from ..utils.formatting import format_solution
                                beautified = symbolify_constants(func_str)
                                print(f"\nResult: {format_solution(beautified)}")
                                print(f"MSE: {mse_val:.6g}, Complexity: ~{len(func_str)//5}")
                                
                                # Persist the function
                                try:
                                    from ..function_manager import define_function
                                    define_function(func_name, input_vars, beautified)
                                except Exception:
                                    pass
                                return
                        else:
                            print(
                                f"Hybrid seeding: find() result has low R²={r_squared:.2f} (MSE={mse_val:.6f}), skipping seed"
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
            high_precision=high_precision_mode,  # Use arbitrary-precision arithmetic
        )
        
        # Apply operator bans if specified
        if banned_operators:
            original_ops = config.operators.copy()
            config.operators = [op for op in config.operators if op.lower() not in banned_operators]
            removed = set(original_ops) - set(config.operators)
            if removed:
                print(f"   [Constraint] Remaining arsenal: {config.operators}")
            
            # Also filter seeds that contain banned operators
            original_seed_count = len(config.seeds)
            filtered_seeds = []
            for seed in config.seeds:
                seed_lower = seed.lower()
                contains_banned = any(ban in seed_lower for ban in banned_operators)
                if not contains_banned:
                    filtered_seeds.append(seed)
            config.seeds = filtered_seeds
            if len(filtered_seeds) < original_seed_count:
                print(f"   [Constraint] Filtered {original_seed_count - len(filtered_seeds)} seeds containing banned operators")
        
        # === ODE DISCOVERY MODE ===
        # If --discover-ode flag is set, use ODE discovery instead of standard regression
        if use_discover_ode:
            from ..symbolic_regression.ode_discovery import ODEDiscoveryEngine, ODEConfig
            
            ode_config = ODEConfig(
                population_size=200,
                generations=50,
                verbose=verbose_mode,
                parsimony_coefficient=0.01
            )
            
            ode_engine = ODEDiscoveryEngine(ode_config)
            ode_str, residual = ode_engine.fit(X[:, 0], y)  # Single variable only for now
            
            print(f"\n=== ODE Discovery Result ===")
            print(f"Discovered: {ode_str}")
            print(f"Residual: {residual:.6e}")
            
            # Human-friendly interpretation
            print(f"\n📖 Interpretation:")
            if "y''" in ode_str and "y'" not in ode_str.replace("y''", ""):
                # Contains y'' but not y' (standalone)
                if "+ y" in ode_str or "y +" in ode_str:
                    print("   This is Simple Harmonic Motion: acceleration = -position")
                    print("   → The function oscillates like a wave (sin, cos)")
                    print("   → Physical examples: pendulum, spring, vibration")
                elif "- y" in ode_str or "y -" in ode_str:
                    print("   This is exponential: acceleration = position")
                    print("   → The function grows/decays exponentially (exp, cosh, sinh)")
            elif "y'" in ode_str and "y''" not in ode_str:
                if "+ y" in ode_str or "y +" in ode_str:
                    print("   This is exponential decay: rate = -value")
                    print("   → The function decays over time (e^(-x))")
                elif "- y" in ode_str or "y -" in ode_str:
                    print("   This is exponential growth: rate = value")
                    print("   → The function grows exponentially (e^x)")
            else:
                print("   This describes how the function changes with its derivatives.")
            return
        
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

        # Print Result (with symbolic constant beautification)
        from ..symbolic_regression.expression_tree import symbolify_constants
        from ..utils.formatting import format_solution
        beautified_expr = symbolify_constants(best.expression)
        print(f"\nResult: {format_solution(beautified_expr)}")
        print(f"MSE: {best.mse:.6g}, Complexity: {best.complexity}")

        # === AUTO ODE DISCOVERY ===
        # Silently run ODE discovery and show if it finds meaningful physics
        try:
            from ..symbolic_regression.ode_discovery import ODEDiscoveryEngine, ODEConfig
            from ..symbolic_regression.numerical_diff import check_even_spacing
            
            # Only run if we have enough data and it's roughly evenly spaced
            if len(y) >= 10:
                is_even, _ = check_even_spacing(X[:, 0])
                # Also allow approximately even spacing
                if is_even or len(y) >= 15:
                    ode_config = ODEConfig(
                        population_size=100,
                        generations=20,
                        verbose=False,  # Silent
                        parsimony_coefficient=0.01
                    )
                    ode_engine = ODEDiscoveryEngine(ode_config)
                    
                    # Try linear ODE first (y'' + y = 0 style)
                    ode_str, residual = ode_engine.fit(X[:, 0], y)
                    
                    # Always try autonomous ODE (y' = G(y)) and pick the better one
                    auto_ode_str, auto_residual = ode_engine.discover_autonomous_ode(X[:, 0], y)
                    if auto_residual < residual:
                        ode_str = auto_ode_str
                        residual = auto_residual
                    
                    # Only show if residual is low (good fit)
                    if residual < 0.1:
                        print(f"\n📖 Underlying Physics:")
                        print(f"   ODE: {ode_str}")
                        # Add interpretation based on ODE pattern
                        # Check for autonomous ODE first (y' = ...)
                        if ode_str.startswith("y' = "):
                            rhs = ode_str[5:]  # Get the G(y) part
                            if "y**2" in rhs or "y*y" in rhs or "(1 - y)" in rhs:
                                print("   → Logistic Growth (population with carrying capacity)")
                            elif "y" in rhs and ("*" not in rhs or rhs.count("y") == 1):
                                print("   → Exponential dynamics")
                            else:
                                print("   → Autonomous ODE (rate depends on state)")
                        else:
                            # Linear ODE interpretation
                            has_ypp = "y''" in ode_str
                            has_yp = "y'" in ode_str and "y''" not in ode_str
                            
                            if has_ypp:
                                if ("y + y''" in ode_str or "y'' + y" in ode_str):
                                    print("   → Simple Harmonic Motion (oscillating wave: sin, cos)")
                                elif ("y - y''" in ode_str or "y'' - y" in ode_str or 
                                      "-y + y''" in ode_str or "y'' + -y" in ode_str):
                                    print("   → Exponential/Hyperbolic (exp, cosh, sinh)")
                                else:
                                    print("   → Second-order dynamics")
                            elif has_yp:
                                if ("y' - y" in ode_str or "y - y'" in ode_str or 
                                    "-y + y'" in ode_str):
                                    print("   → Exponential growth (rate = value)")
                                elif ("y' + y" in ode_str or "y + y'" in ode_str):
                                    print("   → Exponential decay (rate = -value)")
        except Exception:
            pass  # Silently fail if ODE discovery doesn't work

        # Persist the discovered function (Engineering Standard: State Persistence)
        try:
            from ..function_manager import define_function

            # Convert best.expression (pretty string) or best.sympy_expr to storage format
            # define_function expects string expression - use beautified version
            define_function(func_name, input_vars, beautified_expr)
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
                f"Discovered: {target_func}({', '.join(target_vars)}) = {result_str} {note}"
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


def _ascii_plot(x, y, width=80, height=20):
    """Draw a basic ASCII plot of y vs x."""
    if len(x) != len(y) or len(x) == 0:
        print("No data to plot.")
        return

    # Handle NaN/Inf
    mask = np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(y) == 0:
        print("Function evaluated to non-finite values only.")
        return

    min_y, max_y = np.min(y), np.max(y)
    if min_y == max_y:
        print(f"Constant function: y = {min_y}")
        return

    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Resample y to width (if x is not linspace, this is approx)
    # We assume x is sorted and spans the width

    # Normalized Y to 0..(height-1)
    y_norm = (y - min_y) / (max_y - min_y)
    y_indices = (y_norm * (height - 1)).astype(int)

    for col, row_idx in enumerate(y_indices):
        if 0 <= col < width:
            row = height - 1 - row_idx # Flip Y
            if 0 <= row < height:
                grid[row][col] = '*'

    # Draw
    print(f"\nPlotting range x: [{x[0]:.2f}, {x[-1]:.2f}]")
    print("-" * (width + 2))
    for row in grid:
        print("|" + "".join(row) + "|")
    print("-" * (width + 2))
    print(f"Y range: [{min_y:.4f}, {max_y:.4f}]")


def _handle_plot_command(text: str, variables: Dict[str, str]):
    """Handle plot <expr>"""
    parts = text.split(" ", 1)
    if len(parts) < 2:
        print("Usage: plot <expression> (e.g., 'plot sin(x)', 'plot x^2')")
        return

    expr_str = parts[1].strip()

    # Check for implicit y=
    if "=" in expr_str:
        # assume y=... take rhs
        expr_str = expr_str.split("=", 1)[1].strip()

    # Preprocess (handle implicit mul, etc.)
    try:
        # Use parser preprocessing but we evaluated via numpy
        expr_processed = kparser.preprocess(expr_str)
    except Exception as e:
        print(f"Error parsing expression: {e}")
        return

    # Substitute variables (excluding x)
    plot_vars = variables.copy()
    if 'x' in plot_vars:
        # print(f"Note: Ignoring global x={plot_vars['x']} for plotting")
        del plot_vars['x']

    sorted_vars = sorted(plot_vars.keys(), key=len, reverse=True)
    for var in sorted_vars:
         # Use regex for safe word replacement
         pattern = r"\b" + re.escape(var) + r"\b"
         expr_processed = re.sub(pattern, f"({plot_vars[var]})", expr_processed)

    # Evaluate
    # Range [-10, 10]
    x = np.linspace(-10, 10, 80)

    # Build safe local dict for numpy evaluation
    safe_locals = {"x": x, "np": np}
    # Add numpy math functions
    for name in dir(np):
        if not name.startswith("_"):
            safe_locals[name] = getattr(np, name)

    # Also add standard names mapped to numpy
    safe_locals["sin"] = np.sin
    safe_locals["cos"] = np.cos
    safe_locals["tan"] = np.tan
    safe_locals["exp"] = np.exp
    safe_locals["log"] = np.log
    safe_locals["sqrt"] = np.sqrt
    safe_locals["pi"] = np.pi

    try:
        # Use SymPy for safe evaluation instead of raw eval()
        import sympy as sp
        from sympy import lambdify
        
        x_sym = sp.Symbol('x')
        # Parse expression safely with SymPy
        expr = sp.sympify(expr_processed, locals={'x': x_sym, 'pi': sp.pi, 'e': sp.E})
        
        # Convert to numpy function for vectorized evaluation
        f = lambdify(x_sym, expr, modules=['numpy'])
        y = f(x)

        # Check if result is scalar (constant function)
        if np.isscalar(y):
            y = np.full_like(x, y)
        elif isinstance(y, (list, tuple)):
            y = np.array(y)

        # Try Matplotlib first
        try:
            import matplotlib.pyplot as plt
            
            # Simple check to ensure we can show a window
            # (In some environments basic import works but backend fails)
             
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, label=expr_str)
            plt.title(f"Plot of {expr_str}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axhline(0, color='black', linewidth=0.5)
            plt.axvline(0, color='black', linewidth=0.5)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Check backend interactivity
            if plt.get_backend().lower() == 'agg':
                # Headless backend - save to file
                filename = "plot_output.png"
                plt.savefig(filename)
                print(f"Backend is non-interactive (Agg). Saved plot to '{filename}'.")
            else:
                try:
                    plt.show()
                    print("Displayed plot in regular window.")
                except UserWarning:
                    # Fallback for "FigureCanvasAgg is non-interactive" warning that wasn't caught by backend check
                    filename = "plot_output.png"
                    plt.savefig(filename)
                    print(f"Interactive window not available. Saved plot to '{filename}'.")
            return
        except ImportError:

            print("Matplotlib not installed. Falling back to ASCII plot.")
            print("Tip: Run `pip install matplotlib` for high-quality plots.")
        except Exception as e:
            print(f"Matplotlib error: {e}. Falling back to ASCII plot.")
            
        _ascii_plot(x, y)


    except Exception as e:
        print(f"Error evaluating plot: {e}\n(Make sure expression is valid numpy syntax)")


def _detect_modulo_patterns(X, y, verbose: bool = False):
    """
    Detects if f(x) = x % T (sawtooth/modulo pattern).
    
    Uses heuristic:
    1. Finds zero values in y (or near-zero).
    2. Checks if x values at these zeros are evenly spaced (Period T).
    3. Verifies if f(x) ≈ x % T between zeros.
    """
    seeds = []
    
    # Require 1D input
    if X.ndim > 1 and X.shape[1] > 1:
        return []
        
    x_flat = X.flatten()
    y_flat = np.array(y).flatten()
    
    # Filter out complex numbers (can't use round() on them)
    real_mask = np.array([np.isreal(x) and np.isreal(yv) and np.isfinite(np.real(x)) 
                          for x, yv in zip(x_flat, y_flat)])
    if np.sum(real_mask) < 3:
        return []
    x_flat = np.real(x_flat[real_mask])
    y_flat = np.real(y_flat[real_mask])
    
    # Sort
    idx = np.argsort(x_flat)
    x_sorted = x_flat[idx]
    y_sorted = y_flat[idx]
    
    # 1. Find zeros (roots)
    # Allow small tolerance
    zeros_mask = np.abs(y_sorted) < 1e-3
    x_zeros = x_sorted[zeros_mask]
    
    if len(x_zeros) < 3:
        # Not enough zeros to establish periodicity
        return []
        
    # 2. Analyze spacing (differences between consecutive zeros)
    diffs = np.diff(x_zeros)
    
    # Filter out tiny diffs (duplicate points)
    diffs = diffs[diffs > 1e-4]
    
    if len(diffs) == 0:
        return []
        
    # Check if diffs are consistent (multiples of some period T)
    # Taking the median diff as candidate period T
    # Note: If we have missing zeros, some diffs might be 2T, 3T.
    # So we look for the GCD or smallest common gap.
    
    # Simple approach: Mode or Median
    # Round diffs to avoid float noise
    diffs_rounded = np.round(diffs, 3)
    vals, counts = np.unique(diffs_rounded, return_counts=True)
    best_T = vals[np.argmax(counts)]
    
    # Calculate consistency
    # We expect most diffs to be integer multiples of best_T
    is_periodic = True
    for d in diffs:
        ratio = d / best_T
        if abs(ratio - round(ratio)) > 0.05:
            is_periodic = False
            break
            
    if not is_periodic:
        return []
        
    # 3. Verify function shape: f(x) ≈ x % T
    # We check a few non-zero points
    matches = 0
    checks = 0
    failed = False
    
    for i in range(len(x_sorted)):
        xi = x_sorted[i]
        yi = y_sorted[i]
        
        # Skip the zeros we used to find T
        if abs(yi) < 1e-3:
            continue
            
        # Expected: xi % best_T
        # Note: numpy fmod vs mod semantics for negative numbers
        expected = xi % best_T
        
        # Correction for float precision near the reset point
        # e.g. 2.999 % 1.5 -> 1.499, but observed might be near 0 if slightly over
        if abs(expected - best_T) < 1e-3:
            expected = 0.0
            
        if abs(yi - expected) < 1e-2:
            matches += 1
        else:
            # Tolerant check
            if abs(yi - expected) > 0.1: # Gross mismatch
                failed = True
                break
        checks += 1
        
    if not failed and checks > 0 and matches >= checks * 0.8:
        if verbose:
            print(f"   Forensic Analysis: Checking Modulo pattern...")
            print(f"      → Detected periodic zeros with period T={best_T}")
            print(f"      → Pattern: f(x) = x % {best_T}")
            print(f"      → Match rate: {matches}/{checks} checked points")
        
        seeds.append(f"mod(x, {best_T})")
        # Also try "x % T" (operator syntax)
        seeds.append(f"x % {best_T}") 
        
    return seeds
