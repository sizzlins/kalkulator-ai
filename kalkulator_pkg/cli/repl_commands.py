"""
Command handlers for the Kalkulator CLI.
Extracted from app.py to enforce Rule 4 (Small Units).
"""

import logging
import re
import warnings
from typing import Any
from typing import Dict
import json

import kalkulator_pkg.parser as kparser
import numpy as np
import scipy.special


from ..cache_manager import export_cache_to_file
from ..cache_manager import get_persistent_cache
from ..cache_manager import replace_cache_from_file
from ..function_manager import get_builtin_names
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
from ..worker import evaluate_safely
from ..symbolic_regression.forensic_analysis import generate_pattern_seeds
from ..heuristics import detect_smoothness

logger = logging.getLogger(__name__)



# Regex Constants
# regex anchored to start/end combined with negated classes prevents ReDoS.
POINT_PATTERN = re.compile(r"^([a-zA-Z_]\w*)\s*\(([^)]+)\)\s*=\s*(.+)$")
FIND_PATTERN = re.compile(r"^find\s+([a-zA-Z_]\w*)\s*(?:\(([^)]+)\))?$")
SEED_PATTERN = re.compile(r'--seed\s+["\']([^"\']+)["\']')
BOOST_PATTERN = re.compile(r"--boost(?:[=\s]+(\d+))?")
BAN_PATTERN = re.compile(r'--ban\s+([a-zA-Z0-9_,]+)')
FILE_PATTERN = re.compile(r"--file\s+[\"']?([^\"'\s]+)[\"']?")
EVOLVE_EXPLICIT_PATTERN = re.compile(r"evolve\s+(\w+)\s*=\s*(\w+)\s*\(([^)]+)\)(?:\s+from\s+(.+))?$", re.IGNORECASE)
EVOLVE_PATTERN = re.compile(r"evolve\s+(\w+)\s*\(([^)]+)\)\s+from\s+(.+)", re.IGNORECASE)
EVOLVE_IMPLICIT_PATTERN = re.compile(r"evolve\s+(\w+)\s*\(([^)]+)\)\s*$", re.IGNORECASE)
DIRECT_POINT_PATTERN = re.compile(r"(\w+)\s*\([^)]+\)\s*=")
ARRAY_ASSIGN_PATTERN = re.compile(r"(\w+)\s*=\s*(?:\[([^\]]+)\]|(\w+))")
FUNC_START_PATTERN = re.compile(r"(\w+)\s*\(")
FIND_FUNC_CLAUSE_PATTERN = re.compile(r"find\s+(\w+)\s*\(([^)]+)\)", re.IGNORECASE)

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
    "altvd",
    "all",
    "all4",
    "alld4",
    "alt4",
    "altv4",
    "altvd4",
    "call",
    "callset",
    "callm",
    "callr",
    "callrm",
    "genes",
    "ban",
    "unban",
}



# Dynamic Shortcut Pattern: Matches alt/all commands with optional digits (e.g., altvd, altvd4, altvd12)
DYNAMIC_SHORTCUT_PATTERN = re.compile(r"^(alt|altv|altvd|all|alld|alt4|altv4|altvd4|alld4|alrvd|alrv)(\d*)\s+(.*)", re.IGNORECASE)

def handle_command(text: str, ctx: Any, variables: Dict[str, str]) -> bool:
    """
    Attempt to handle the input text as a command.
    Returns True if handled, False otherwise.
    """
    raw_lower = text.lower().strip()

    # === Function Persistence Commands ===
    if raw_lower in ("save", "savefunction", "savefunctions"):
        success, msg = save_functions(ctx)
        print(msg)
        return True

    if raw_lower in ("loadfunction", "loadfunctions"):
        success, msg = load_functions(ctx)
        print(msg)
        return True

    if raw_lower in ("clearfunction", "clearfunctions"):
        clear_functions(ctx)
        print("Functions cleared from current session.")
        return True

    if raw_lower in ("clearsavefunction", "clearsavefunctions"):
        success, msg = clear_saved_functions() # Assuming no context needed for clear_saved (disk only)
        print(msg)
        return True

    if raw_lower in ("showfunction", "showfunctions", "list"):
        _handle_show_functions(ctx)
        return True

    if raw_lower.startswith("debug"):
        _handle_debug_command(text, ctx)
        return True

    if raw_lower == "health":
        _handle_health_command(text, ctx)
        return True

    if raw_lower.startswith("timing"):
        _handle_timing_command(text, ctx)
        return True

    if raw_lower.startswith("cachehits"):
        _handle_cachehits_command(text, ctx)
        _handle_cachehits_command(text, ctx)
        return True

    if raw_lower.startswith("ban "):
        _handle_ban_command(text, ctx)
        return True

    if raw_lower.startswith("unban ") or raw_lower == "unban":
        _handle_unban_command(text, ctx)
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
        _handle_evolve(text, ctx, variables)
        return True

    # === ALT CALL SHORTCUT ===
    # Check for 'alt[flags] call[r] <func>' pattern
    # e.g. "altvd call f" -> "altvd f(1)=2, f(2)=4..."
    # We intercept this BEFORE the generic shortcut expander
    # Check for Dynamic Shortcuts (altvd12, alt4, etc.)
    shortcut_match = DYNAMIC_SHORTCUT_PATTERN.match(text.strip())
    if shortcut_match:
        base_cmd = shortcut_match.group(1).lower()
        boost_suffix = shortcut_match.group(2)
        rest_args = shortcut_match.group(3)
        
        # Define flags mapping
        # Note: 4-series aliases (alt4, etc.) are just base aliases + explicit default boost 4
        # We handle boost variable separately below
        
        flags = ""
        default_boost = 3
        
        if "altvd" in base_cmd or "alrvd" in base_cmd:
            flags = "--hybrid --verbose --super-verbose --debug --transform"
        elif "altv" in base_cmd or "alrv" in base_cmd:
            flags = "--hybrid --verbose --transform"
        elif "alt" in base_cmd:
            # Upgrade "alt" to be verbose as requested by user ("alt... does not show logs")
            # Formerly: flags = "--hybrid --transform"
            flags = "--hybrid --verbose --transform"
        elif "alld" in base_cmd:
             flags = "--verbose --transform"
        elif "all" in base_cmd:
             flags = "--verbose"
             
        # Handle 4-series defaults (if explicit 4 in name)
        if "4" in base_cmd:
            default_boost = 4
            
        # Determine actual boost
        if boost_suffix:
            # Explicit number overrides everything (e.g. altvd12 -> boost 12)
            boost_val = boost_suffix
        else:
            boost_val = str(default_boost)
            
        # Construct new command
        # Append flags and boost to the arguments
        new_text = f"{rest_args} {flags} --boost {boost_val}"
        
        if "--debug" in flags:
            print(f"[Debug] Shortcut '{base_cmd}{boost_suffix}' expanded to: {new_text}")
            
        _handle_evolve(new_text, ctx, variables)
        return True

    # ODE discovery shortcut: 'ode f(...)' is equivalent to 'alt --discover-ode f(...)'
    if raw_lower.startswith("ode "):
        text = text[4:]  # Remove 'ode ' prefix
        text = "--discover-ode " + text  # Add the flag
        _handle_evolve(text, ctx, variables)
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

    if raw_lower.startswith("callset"):
        _handle_callset_command(text, ctx)
        return True

    if raw_lower.startswith("callm"):
        _handle_callm_command(text, ctx)
        return True

    if raw_lower.startswith("callrm"):
        _handle_callrm_command(text, ctx)
        return True

    if raw_lower.startswith("callr"):
        _handle_callr_command(text, ctx)
        return True

    if raw_lower.startswith("call"):
        _handle_call_command(text, ctx, multiline=False)
        return True

    # === Gene Bank Command ===
    if raw_lower.startswith("genes"):
        _handle_genes_command(text, ctx)
        return True

    return False


def _substitute_vars(text: str, variables: Dict[str, str], exclude: set[str] | None = None) -> str:
    # Helper to substitute vars before command execution
    # exclude: Set of variable names to skip (protect from shadowing)
    if exclude is None:
        exclude = set()
        
    sorted_vars = sorted(variables.keys(), key=len, reverse=True)
    for var in sorted_vars:
        if var in exclude:
            continue
        if var in text:
            pattern = r"\b" + re.escape(var) + r"\b"
            text = re.sub(pattern, f"({variables[var]})", text)
    return text


def _handle_show_functions(ctx: Any):
    # ... (Keep existing implementation, jumping to _handle_solve_command) ...
    """Display user-defined functions and categorized built-in functions."""
    funcs = list_functions(ctx)
    
    # === User-Defined Functions ===
    print("=" * 60)
    print("USER-DEFINED FUNCTIONS")
    print("=" * 60)
    if funcs:
        for name in sorted(funcs.keys()):
            params, body = funcs[name]
            param_str = ", ".join(params) if params else ""
            print(f"  {name}({param_str}) = {body}")
    else:
        print("  (none)")
    
    # === Built-in Functions by Category ===
    print("\n" + "=" * 60)
    print("BUILT-IN FUNCTIONS")
    print("=" * 60)
    
    # Categorize builtins
    categories = {
        "Trigonometric": ["sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan", "atan2", "arcsin", "arccos", "arctan"],
        "Hyperbolic": ["sinh", "cosh", "tanh", "asinh", "acosh", "atanh"],
        "Exponential/Log": ["exp", "log", "ln", "log2", "log10"],
        "Power/Root": ["sqrt", "cbrt", "root", "abs", "Abs", "sign"],
        "Rounding": ["floor", "ceil", "ceiling", "round", "frac"],
        "Special": ["gamma", "factorial", "binomial", "fibonacci", "lucas", "erf", "sinc", "besselj", "prime_pi", "primepi", "LambertW"],
        "Piecewise": ["Heaviside", "heaviside", "Max", "max", "Min", "min", "Piecewise"],
        "Comparison": ["Eq", "Ne", "Lt", "Le", "Gt", "Ge"],
        "Bitwise": ["bitwise_and", "bitwise_or", "bitwise_xor", "lshift", "rshift"],
        "Algebra": ["gcd", "lcm", "Mod", "mod", "expand", "factor", "simplify"],
        "Calculus": ["diff", "integrate", "limit"],
        "Linear Algebra": ["Matrix", "matrix", "det", "inv"],
        "Constants": ["pi", "e", "E", "I", "oo", "nan", "zoo"],
        "Special Eval": ["locked", "neg", "AccumBounds"],
    }
    
    builtins_set = set(get_builtin_names())
    displayed = set()
    
    for cat_name, cat_funcs in categories.items():
        matching = [f for f in cat_funcs if f in builtins_set]
        if matching:
            print(f"\n  {cat_name}:")
            line = "    "
            for f in sorted(matching, key=str.lower):
                entry = f"{f}()"
                if len(line) + len(entry) + 2 > 76:
                    print(line.rstrip(", "))
                    line = "    "
                line += entry + ", "
                displayed.add(f)
            if line.strip():
                print(line.rstrip(", "))
    
    # Show any remaining (miscellaneous)
    remaining = builtins_set - displayed
    if remaining:
        print(f"\n  Other:")
        line = "    "
        for f in sorted(remaining, key=str.lower):
            entry = f"{f}()"
            if len(line) + len(entry) + 2 > 76:
                print(line.rstrip(", "))
                line = "    "
            line += entry + ", "
        if line.strip():
            print(line.rstrip(", "))
    
    print()


def _handle_solve_command(text: str, variables: Dict[str, str]):
    # Format: solve x^2 - 1 = 0 [, var]
    # v4.3 Fix: Variable Shadowing
    # Explicitly protect likely target variables from substitution.
    
    # 1. Parse command
    base_text = text[6:].strip() # remove 'solve '
    
    # Check for explicit variable: "eq, x"
    target_var = None
    if "," in base_text:
        # Split by last comma
        start, end = base_text.rsplit(",", 1)
        # Verify 'end' looks like a variable
        candidate = end.strip()
        if candidate.isidentifier() or (len(candidate) == 1 and candidate.isalpha()):
            eq_str = start.strip()
            target_var = candidate
        else:
            eq_str = base_text
    else:
        eq_str = base_text
        
    print(f"Solving equation: {eq_str}")
    
    # 2. Determine Exclusion Set (Variables to NOT substitute)
    exclude = set()
    if target_var:
        exclude.add(target_var)
    else:
        # Auto-detect protection
        # If common variables appear in equation, protect them
        # Limit to single-letter vars or known math vars to avoid protecting parameters
        common_vars = ["x", "y", "z", "t", "n", "a", "b", "c"]
        
        # Simple tokenization by word boundary to find usage
        # This prevents "exp" matching "x"
        tokens = set(re.findall(r"\b[a-zA-Z_]\w*\b", eq_str))
        
        for v in common_vars:
            if v in tokens:
                exclude.add(v)
                
    # 3. Substitute (with exclusion)
    eq_str_subbed = _substitute_vars(eq_str, variables, exclude=exclude)
    
    if eq_str != eq_str_subbed:
        # Inform user if substitutions happened
        # Filter out excluded to only show what WAS substituted
        actually_subbed = []
        for v in variables:
            if v not in exclude and v in eq_str:
                actually_subbed.append(v)
        if actually_subbed:
           print(f"Note: Substituted variables: {', '.join(actually_subbed)}")

    # 4. Solve
    res = solve_single_equation(eq_str_subbed, target_var)

    # Check for "Contradiction" - Fallback hint
    if res.get("type") == "identity_or_contradiction" and "Contradiction" in str(res.get("result", "")):
        if eq_str != eq_str_subbed:
             print("Note: If you meant to solve for a variable that was substituted, try clearing it or specifying it explicitly (e.g. 'solve eq, x').")

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


def _matches_ban(func_lower: str, ban_token: str) -> bool:
    """Check if a function string violates a ban token.
    
    Handles:
    - Stripping parens from ban tokens: 'sqrt()' → checks for 'sqrt'
    - Semantic equivalences: sqrt ↔ x**0.5, pow ↔ ** ↔ ^
    - Unicode symbols: √ → sqrt
    """
    ban = ban_token.lower().strip()
    
    # Strip trailing parens: "sqrt()" → "sqrt"
    if ban.endswith("()"):
        ban = ban[:-2]
    # Strip wrapping parens with content: "√(x)" → "√"
    import re as _re
    ban = _re.sub(r'\(.*?\)', '', ban).strip()
    # Unicode normalization: √ → sqrt
    ban = ban.replace('√', 'sqrt')
    
    # Direct substring check
    if ban and ban in func_lower:
        return True
    
    # Semantic equivalences
    SQRT_FORMS = {'sqrt', '**0.5', '**(0.5)', '**(1/2)', '**0.50'}
    POW_FORMS = {'pow', '**', '^'}
    
    if ban in {'sqrt', 'x**0.5'}:
        return any(form in func_lower for form in SQRT_FORMS)
    if ban in POW_FORMS:
        return any(form in func_lower for form in POW_FORMS)
    
    return False


# [Orphaned definitions deleted]
def _handle_ban_command(text: str, ctx: Any):
    """
    Handle 'ban' command to persistently exclude operators from evolution.
    Syntax: ban <func1>, <func2>, ...
    """
    if not ctx:
        print("Error: Context not available for persistent banning.")
        return

    # Extract arguments (remove 'ban ' prefix)
    args = text[3:].strip()
    if not args:
        # If no args, just list currently banned
        if ctx.banned_operators:
             print(f"Currently banned operators: {sorted(list(ctx.banned_operators))}")
        else:
             print("No operators are currently banned.")
        return

    # Parse comma-separated list
    new_bans = [op.strip().lower() for op in args.split(",") if op.strip()]
    
    # helper validation against known operators? 
    # For now, just add them (engine filters unknown ones anyway)
    ctx.banned_operators.update(new_bans)
    print(f"Banned: {sorted(list(ctx.banned_operators))}")


def _handle_unban_command(text: str, ctx: Any):
    """
    Handle 'unban' command to remove exclusions.
    Syntax: unban <func> | unban all
    """
    if not ctx:
        print("Error: Context not available.")
        return

    args = text[5:].strip()
    if not args:
        print("Usage: unban <function_name> | unban all")
        return

    if args.lower() == "all":
        ctx.banned_operators.clear()
        print("All bans cleared.")
        return

    to_remove = [op.strip().lower() for op in args.split(",") if op.strip()]
    missing = []
    removed = []
    
    for op in to_remove:
        if op in ctx.banned_operators:
            ctx.banned_operators.remove(op)
            removed.append(op)
        else:
            missing.append(op)
            
    if removed:
        print(f"Unbanned: {removed}")
    if missing:
        print(f"Not found in ban list: {missing}")
    
    if ctx.banned_operators:
        print(f"Remaining bans: {sorted(list(ctx.banned_operators))}")
    else:
        print("No operators are currently banned.")


def _handle_evolve(text: str, ctx, variables=None):
    from kalkulator_pkg.cli.handlers.evolution import handle_evolve
    return handle_evolve(text, ctx, variables)
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
    """Display cached items with readable formatting."""
    cache = get_persistent_cache()
    eval_cache = cache.get("eval_cache", {})
    subexpr_cache = cache.get("subexpr_cache", {})
    
    total = len(eval_cache) + len(subexpr_cache)
    
    # Parse arguments
    args = text.lower().split()
    show_count_only = "count" in args or "summary" in args
    show_all = "all" in args
    
    print("=" * 70)
    print(f"CACHE CONTENTS ({total} total items)")
    print("=" * 70)
    
    if show_count_only:
        print(f"  Eval cache: {len(eval_cache)} items")
        print(f"  Subexpr cache: {len(subexpr_cache)} items")
        print("\nUse 'showcache' to see recent items, or 'showcache all' for everything.")
        return
    
    def _clean_key(k: str) -> str:
        """Remove context hash prefix if present."""
        if ":" in k and len(k.split(":")[0]) == 32: # Simple heuristic for md5 hash
            return k.split(":", 1)[1]
        return k

    def _clean_val(v: Any) -> tuple[str, str | None]:
        """Extract result and time from cache value."""
        res_str = ""
        time_str = None
        
        # Handle dict (new format) vs string (old format)
        data = v
        if isinstance(v, str):
            try:
                data = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                # Raw string (legacy)
                return str(v), None

        if isinstance(data, dict):
            # Extract result
            if "result" in data:
                 raw_res = data["result"]
                 # Sometimes result itself is double-encoded JSON
                 if isinstance(raw_res, str) and raw_res.startswith("{"):
                     try:
                         inner = json.loads(raw_res)
                         res_str = str(inner.get("result", raw_res))
                     except:
                         res_str = raw_res
                 else:
                     res_str = str(raw_res)
            elif "value" in data:
                res_str = str(data["value"])
            else:
                res_str = json.dumps(data) # Fallback
            
            # Extract timing
            if "time" in data:
                try:
                    t = float(data["time"])
                    if t < 0.001:
                        time_str = "<1ms"
                    else:
                        time_str = f"{t*1000:.1f}ms"
                except:
                    pass
        else:
            res_str = str(data)
            
        return res_str, time_str

    # === Eval Cache ===
    eval_items = list(eval_cache.items())
    # Sort by key for stability, or by time order if possible? 
    # Current list order corresponds to insertion order in Py3.7+
    
    print(f"\n  EVAL CACHE ({len(eval_items)} items):")
    print(f"  {'EXPRESSION':<40} | {'RESULT':<25}")
    print("  " + "-" * 68)
    
    limit = 50 if not show_all else len(eval_items)
    # improved: show MOST RECENT items first (reversed key order?)
    # usually insertion order is oldest first. Let's show END of list first?
    # No, 'showcache' usually implies listing. Let's stick to standard slice.
    
    display_items = eval_items[-limit:] if not show_all else eval_items
    if not show_all and len(eval_items) > limit:
         print(f"  ... ({len(eval_items) - limit} older items hidden) ...")

    for k, v in display_items:
        key_display = _clean_key(str(k))
        val_display, time_display = _clean_val(v)
        
        # Truncate for table
        if len(key_display) > 38:
            key_display = key_display[:35] + "..."
        if len(val_display) > 23:
            val_display = val_display[:20] + "..."
            
        line = f"  {key_display:<40} | {val_display:<25}"
        if time_display:
            line += f" ({time_display})"
        print(line)
        
    if not eval_items:
        print("  (empty)")

    # === Subexpr Cache ===
    sub_items = list(subexpr_cache.items())
    print(f"\n  SUBEXPR CACHE ({len(sub_items)} items):")
    print(f"  {'SUB-EXPRESSION':<40} | {'VALUE':<25}")
    print("  " + "-" * 68)
    
    limit_sub = 20 if not show_all else len(sub_items)
    display_sub = sub_items[-limit_sub:] if not show_all else sub_items
    if not show_all and len(sub_items) > limit_sub:
         print(f"  ... ({len(sub_items) - limit_sub} older items hidden) ...")

    for k, v in display_sub:
        key_display = _clean_key(str(k))
        val_display, time_display = _clean_val(v)
        
        if len(key_display) > 38:
            key_display = key_display[:35] + "..."
        if len(val_display) > 23:
            val_display = val_display[:20] + "..."
            
        line = f"  {key_display:<40} | {val_display:<25}"
        if time_display:
            line += f" ({time_display})"
        print(line)

    if not sub_items:
        print("  (empty)")

    print("\n" + "=" * 70)
    print("Commands: 'clearcache', 'showcache all', 'savecache'")
    print("=" * 70)


def _handle_health_command(text: str, ctx: Any):
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
        res = evaluate_safely("2**10")  # 1024
        
        # Check if result is a dict and has 'result' key
        val = res
        if isinstance(res, dict) and res.get("ok"):
            val = res.get("result")
            
        if str(val) == "1024":
            print("[OK] Worker IPC works (2**10 -> 1024)", flush=True)
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
        # Fix: Pass context as first argument
        success, func_str, _, error_msg = find_function_from_data(ctx, data, ["x"])

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


def _handle_genes_command(text: str, ctx: Any):
    """Handle 'genes' command for Gene Bank management.
    
    Usage:
        genes          - List all saved genes
        genes delete N - Delete gene at index N
        genes clear    - Clear all genes
    """
    try:
        from kalkulator_pkg.symbolic_regression.gene_bank import get_gene_bank
        bank = get_gene_bank()
    except ImportError:
        print("Error: Gene Bank module not found.")
        return
    except Exception as e:
        print(f"Error loading Gene Bank: {e}")
        return
    
    parts = text.strip().lower().split()
    
    # genes (list all)
    if len(parts) == 1:
        genes = bank.list_genes()
        if not genes:
            print("Gene Bank is empty. Run successful function discoveries to populate it.")
            return
        
        print(f"\nGENE BANK ({len(genes)} cached functions)")
        print("─" * 76)
        print(f" {'ID':<4} {'Expression':<32} {'Vars':<6} {'Compl.':<8} {'MSE':<10} {'Status':<10}")
        print("─" * 76)
        
        for g in genes:
            expr_str = g['expression']
            mse = g['fitness']
            
            # Prettify expression
            pretty_expr = _prettify_gene_expression(expr_str)
            
            # Truncate if too long
            if len(pretty_expr) > 30:
                pretty_expr = pretty_expr[:27] + "..."
                
            # Determine status
            if mse < 1e-30:
                status = "⭐ Exact"
            elif mse < 1e-6:
                status = "🎯 Precise"
            else:
                status = "〰️ Approx" # Wave for approximation
                
            # Vars count
            n_vars = g.get('n_vars', 1)
            meas_vars = str(n_vars)
            
            # Complexity
            complexity = g.get('complexity', 0.0)
            
            # MSE string
            if mse < 1e-99:
                mse_str = "0.00e+00"
            else:
                mse_str = f"{mse:.2e}"
            
            print(f" [{g['id']}]  {pretty_expr:<32} {meas_vars:<6} {complexity:<8} {mse_str:<10} {status}")
            
        print("─" * 76)
        return
    
    # genes delete N
    if len(parts) >= 2 and parts[1] == "delete":
        if len(parts) < 3:
            print("Usage: genes delete <index>")
            return
        try:
            idx = int(parts[2])
            if bank.delete(idx):
                print(f"Deleted gene at index {idx}.")
            else:
                print(f"Invalid index: {idx}. Use 'genes' to see available indices.")
        except ValueError:
            print("Error: Index must be an integer.")
        return
    
    # genes clear
    if len(parts) >= 2 and parts[1] == "clear":
        bank.clear()
        print("Gene Bank cleared.")
        return
    
    print("Usage: genes | genes delete <index> | genes clear")


def _prettify_gene_expression(expr_str: str) -> str:
    """Prettify a gene expression for display.
    
    1. Mapping: v0->x, v1->y, v2->z
    2. Snap Fractions: 50018.../100... -> 0.50 (only if ugly)
    3. Formatting: Spacing around operators
    """
    import re
    
    # 1. Variable Mapping
    # Simple replacement is safe because v0, v1 do not appear inside other words usually
    expr_str = expr_str.replace("v0", "x")
    expr_str = expr_str.replace("v1", "y")
    expr_str = expr_str.replace("v2", "z")
    
    # 2. Smart Fraction Snapping
    # Detect patterns like digits/digits where length > 3
    def replace_fraction(match):
        numer = match.group(1)
        denom = match.group(2)
        
        # Only snap if it looks "ugly" (long numerator or denominator)
        if len(numer) > 3 or len(denom) > 3:
            try:
                val = float(numer) / float(denom)
                # If close to simple float, format nicely
                return f"{val:.2f}"
            except:
                return match.group(0)
        return match.group(0)
        
    expr_str = re.sub(r'(\d+)/(\d+)', replace_fraction, expr_str)
    
    # 3. Clean up power operator
    expr_str = expr_str.replace("**", "^")
    
    # 4. Spacing (optional, but looks nicer)
    # Don't add spaces inside function calls, just around top-level ops mostly
    # Simple heuristic: space around + - * / if not already spaced
    # But SymPy output usually is tight.
    # Let's just fix the big fraction mess primarily.
    
    # Also fix "0.50" to "0.5" if it ends in 0
    # expr_str = re.sub(r'(\.\d*?)0+\b', r'\1', expr_str) # Strip trailing zeros? maybe later
    
    return expr_str


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
    # Regex to parse find command: find name(vars)

    for p in parts:
        p = p.strip()
        if not p:
            continue

        # Strip flag for parsing
        p_clean = p.replace("--auto-evolve", "").strip()

        # Check for FIND command
        m_find = FIND_PATTERN.match(p_clean)
        if m_find and "find" in p_clean.lower():
            target_func = m_find.group(1)
            if m_find.group(2):
                target_vars = [v.strip() for v in m_find.group(2).split(",")]
            continue

        # Check for DATA point
        # Also try matching dirty p just in case, but clean is safer
        m_point = POINT_PATTERN.match(p_clean)
        if m_point:
            name = m_point.group(1)
            args_str = m_point.group(2)
            val_str = m_point.group(3)

            # args can be multiple: f(1, 2)
            args = [a.strip() for a in args_str.split(",")]

            # PASTE ERROR DETECTION: Check if val_str contains another function call
            # e.g., "26.4f(0)=12.2" from paste error "f(0.35)=26.4f(0)=12.2"
            concat_match = re.search(r'^([0-9.\-+eE]+)([a-zA-Z_]\w*\s*\([^)]+\)\s*=.+)$', val_str)
            if concat_match:
                # Split the concatenated value into: actual value + next function call
                actual_val = concat_match.group(1)
                remaining = concat_match.group(2)
                
                # Store the first point with corrected value
                data_points.append((name, args, actual_val))
                
                # Recursively parse the remaining concatenated part
                m_remaining = POINT_PATTERN.match(remaining)
                if m_remaining:
                    r_name = m_remaining.group(1)
                    r_args = [a.strip() for a in m_remaining.group(2).split(",")]
                    r_val = m_remaining.group(3)
                    data_points.append((r_name, r_args, r_val))
            else:
                # Normal case - store as tuple: (name, args_list, value)
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

        # Validate points structure
        valid_points = []
        for i, pt in enumerate(relevant_points):
            if not isinstance(pt, (tuple, list)):
                print(f"Skipping malformed point #{i}: {pt} (not tuple/list)")
                continue
            if len(pt) != 2:
                print(f"Skipping malformed point #{i}: {pt} (length {len(pt)})")
                continue
            valid_points.append(pt)
        relevant_points = valid_points

        # Handle unpacking safely (API might return 3 or 4 values depending on version)
        import logging
        try:
             # Fix: Pass context as first argument as per signature in function_manager.py
             result = find_function_from_data(ctx, relevant_points, target_vars)
        except ValueError as e:
             # Last ditch catch for the exact error we saw
             print(f"Regression Engine Crash: {e}. Data sample: {relevant_points[:3]}")
             return True
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
                # Fix: Pass ctx to _handle_evolve
                _handle_evolve(evolve_cmd, ctx, variables=None)
                return True

            try:
                # Fix: Pass context as first argument
                define_function(ctx, target_func, target_vars, result_str)
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

                _handle_evolve(evolve_cmd, ctx)
            else:
                print(f"Failed to discover function: {error_msg}")
                print(
                    f"Tip: Genius Mode seeks exact laws. Try 'evolve {target_func}({','.join(target_vars)})...' for approximate models."
                )
                print("     Or use '--auto-evolve' to switch automatically.")

        return True



    return False


# --- Call Command Logic (Batch Evaluation) ---

# Pre-defined sets
_CALL_SETS = {
    "default": [
        # Integers
        -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        # Fractions/Decimals
        0.5, 0.3, 0.25, 0.22, 0.21, 0.18, 0.15, 0.12, 0.09, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02, 
        0.018, 0.015, 0.012, 0.01, 0.009, 0.008, 0.007, 0.006, 0.004, 0.003, 0.002, 
        0.005, 0.0005, 0.0015, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.001,
        -0.5, -0.3, -0.25, -0.22, -0.21, -0.18, -0.15, -0.12, -0.09, -0.08, -0.07, -0.06, -0.04, -0.03, -0.02,
        -0.018, -0.015, -0.012, -0.01, -0.009, -0.008, -0.007, -0.006, -0.004, -0.003, -0.002, 
        -0.005, -0.0005, -0.0015, -0.0008, -0.0006, -0.0004, -0.0002, -0.0001, -0.001,
        # Special Values
        "e", "pi", "sin(1)", "sin(pi)", "sqrt(2)", "sqrt(5)", "2*pi", "log(10)", "cos(0)",
        "1/3", "-3/4", "3/4", "-1/3",
        # Custom from request
        -2.5, 0.99, -0.99, 12.345, -12.345, 19.9, -19.9, 15.5, -15.5, 3.333, -3.333,
        # More Integers/Decimals from massive list
        4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0, 
        2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0,
        -4.5, -4.4, -4.3, -4.2, -4.1, -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0,
        -2.9, -2.8, -2.7, -2.6, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.0,
        # Signs
        "-e", "-pi", "-sin(1)", "-sin(pi)", "-sqrt(2)", "-sqrt(5)", "-2*pi", "-log(10)", "-cos(0)",
    ]
}

def _handle_callset_command(text: str, ctx: Any):
    """
    Handle 'callset <name> <val1>, <val2>...' to define a custom set.
    """
    parts = text.split(maxsplit=2)
    if len(parts) < 3:
        print("Usage: callset <name> <val1>, <val2>, ...")
        return

    set_name = parts[1].strip()
    values_str = parts[2]
    
    # Check if this overrides default
    if set_name == "default":
        print("Warning: Overriding 'default' set.")

    # Parse values using robust splitter (handles tuples like (1,2))
    from ..parser import split_top_level_commas
    
    raw_values = split_top_level_commas(values_str)
    parsed_values = []
    
    for v in raw_values:
        v = v.strip()
        if not v:
            continue
            
        # Check for Tuple syntax: (a, b, ...)
        if v.startswith("(") and v.endswith(")"):
            # Strip parens
            inner = v[1:-1]
            # Split items
            items_str = split_top_level_commas(inner)
            # Evaluate each item
            tuple_items = []
            for item in items_str:
                res = evaluate_safely(item)
                if res.get("ok"):
                    # We store the result (string or number), likely string from worker
                    tuple_items.append(res.get("result"))
                else:
                    tuple_items.append(item) # Fallback
            parsed_values.append(tuple(tuple_items))
        else:
            # Single value
            res = evaluate_safely(v)
            if res.get("ok"):
                parsed_values.append(res.get("result"))
            else:
                parsed_values.append(v)
    
    _CALL_SETS[set_name] = parsed_values
    print(f"Set '{set_name}' defined with {len(parsed_values)} values.")


def _handle_callm_command(text: str, ctx: Any):
    """
    Handle 'callm <func> [set_name]' command (Multi-line output).
    """
    # Just redirect to call command with multiline=True
    # Replace 'callm' with 'call' for parsing transparency if needed, or just pass parts
    _handle_call_command(text.replace("callm", "call", 1), ctx, multiline=True)


def _get_call_results(ctx: Any, func_name: str, set_name: str = "default", randomize: bool = False, count: int = None) -> list[str]:
    """
    Helper to generate list of 'f(x)=y' strings for a given function and set.
    Make sure to update signature to accept count.
    """
    if set_name not in _CALL_SETS:
        return []
        
    inputs = _CALL_SETS[set_name]
    results = []
    import random
    
    # Determine function arity
    from ..function_manager import list_functions
    try:
        funcs = list_functions(ctx)
        if func_name in funcs:
            params, _ = funcs[func_name]
            arity = len(params)
        else:
            # Check if it's a valid builtin (lazy load check)
            from ..config import ALLOWED_SYMPY_NAMES
            if func_name in ALLOWED_SYMPY_NAMES:
                arity = 1 # Assumption for builtins like sin/cos
            else:
                # Function not found - abort
                print(f"Error: Function '{func_name}' is not defined.")
                return []
    except Exception:
        arity = 1

    N = len(inputs)
    target_count = count if count is not None else N
    
    for k in range(target_count):
        # Select base value/index
        if randomize:
            idx = random.randint(0, N-1)
            val = inputs[idx]
        else:
            idx = k % N
            val = inputs[idx]

        # Prepare arguments
        if isinstance(val, (list, tuple)):
            # Explicit tuple overrides randomization (user intent)
            args_list = val
        else:
            if randomize and arity > 1:
                # Randomize all arguments by sampling from the input set
                args_list = [random.choice(inputs) for _ in range(arity)]
            elif arity > 1:
                # Deterministic Mixing (Shifted Rotation)
                # Arg 0: inputs[idx]
                # Arg 1: inputs[(idx+1)%N] ...
                args_list = [inputs[(idx + j) % N] for j in range(arity)]
            else:
                 # Arity 1: just val
                 args_list = [val]

        args_str = ", ".join(str(v) for v in args_list)
        expr = f"{func_name}({args_str})"
        
        res = evaluate_safely(expr)
        if res.get("ok"):
            # Ensure "result" is used for output, usually "2.0" or symbolic
            results.append(f"{expr} = {res.get('result')}")
        else:
            # Skip errors or include? For altvd generation, we probably want valid points.
            pass
            
    return results

def _handle_call_command(text: str, ctx: Any, multiline: bool = False, randomize: bool = False):
    """
    Handle 'call <func> [set_name]' command.
    Display results directly.
    Args:
        randomize: If True, randomly samples inputs from set (breaks symmetry)
    """
    parts = text.split(maxsplit=2)
    if len(parts) < 2:
        cmd = "callr" if randomize else "call"
        print(f"Usage: {cmd} <function_name> [set_name]")
        return
        
    func_name = parts[1].strip()
    
    # Determine input set and optional count
    set_name = "default"
    count_arg = None
    
    # args after function name: [set_name?, count?]
    extra_args = parts[2:]
    
    if extra_args and extra_args[-1].isdigit():
        try:
            count_arg = int(extra_args[-1])
            extra_args.pop() # Remove count from args
        except ValueError:
            pass
            
    if extra_args:
        set_name = extra_args[0].strip()
        
    if set_name not in _CALL_SETS:
        print(f"Error: Unknown set '{set_name}'. Use 'callset {set_name} ...' to define it.")
        available = ", ".join(_CALL_SETS.keys())
        print(f"Available sets: {available}")
        return
        
    inputs = _CALL_SETS[set_name]
    
    # Check Arity
    from ..function_manager import list_functions
    try:
        funcs = list_functions(ctx)
        if func_name in funcs:
            params, _ = funcs[func_name]
            arity = len(params)
        else:
            from ..config import ALLOWED_SYMPY_NAMES
            if func_name in ALLOWED_SYMPY_NAMES:
                 arity = 1
            else:
                 print(f"Error: Function '{func_name}' is not defined.")
                 return
    except Exception:
        arity = 1

    
    
    # Logic to determine loop parameters
    # If explicit count, use it. Else use set length.
    target_count = count_arg if count_arg is not None else len(inputs)
    
    action_verb = "Randomized calling" if randomize else "Calling"
    if multiline:
        print(f"{action_verb} '{func_name}' with set '{set_name}' ({target_count} inputs)...")
    
    count = 0
    results = []
    import random
    
    N = len(inputs)
    
    for k in range(target_count):
        # Determine value and index based on mode
        if randomize:
             # Fully random selection for base value
             idx = random.randint(0, N-1)
             val = inputs[idx]
        else:
             # Sequential selection
             idx = k % N
             val = inputs[idx]
    
        # Prepare arguments
        if isinstance(val, (list, tuple)):
            args_list = val
        else:
            if randomize and arity > 1:
                # Randomize all arguments by sampling from the input set
                # Note: val above is just one sample. We might regenerate args_list purely randomly.
                # Consistent with previous 'callr' which picked random choices for all args
                args_list = [random.choice(inputs) for _ in range(arity)]
            elif arity > 1:
                # Deterministic Mixing (Shifted Rotation)
                # Arg 0: inputs[idx]
                # Arg 1: inputs[(idx+1)%N] ...
                # Use idx from outer loop
                args_list = [inputs[(idx + j) % N] for j in range(arity)]
            else:
                 # Arity 1: just val
                 args_list = [val]

        args_str = ", ".join(str(v) for v in args_list)
        expr = f"{func_name}({args_str})"

        res = evaluate_safely(expr)
        
        if res.get("ok"):
            output = f"{expr} = {res.get('result')}"
            if multiline:
                print(output)
            else:
                results.append(output)
            count += 1
        else:
            output = f"{expr} = Error: {res.get('error')}"
            if multiline:
                print(output)
            else:
                results.append(output)
            
    if not multiline:
        # Print all in one line (comma separated)
        print(", ".join(results))



def _handle_callr_command(text: str, ctx: Any):
    """Handle 'callr <func> [set_name]' command (Randomized inputs)."""
    # Use standard handler with randomize=True
    # Fix command name for parsing
    _handle_call_command(text.replace("callr", "call", 1), ctx, multiline=False, randomize=True)


def _handle_callrm_command(text: str, ctx: Any):
    """Handle 'callrm <func> [set_name] [count]' command (Randomized inputs, multiline)."""
    # Use standard handler with randomize=True, multiline=True
    _handle_call_command(text.replace("callrm", "call", 1), ctx, multiline=True, randomize=True)


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
            from ..experimental.dynamics_discovery.sindy import SINDy, SINDyConfig
        except ImportError:
            print("Error: SINDy module not available (kalkulator_pkg.experimental.dynamics_discovery).")
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
        # SECURITY: Use safe AST-based parser instead of sympify (which uses eval)
        import sympy as sp
        from ..parser import safe_sympy_parse
        from ..config import ALLOWED_SYMPY_NAMES
        
        x_sym = sp.Symbol('x')
        # Build local dict with x symbol
        local_dict = {**ALLOWED_SYMPY_NAMES, 'x': x_sym}
        
        # Parse expression safely with AST-based parser
        expr = safe_sympy_parse(expr_processed, local_dict=local_dict)
        
        # NOTE: We still use lambdify here for plotting performance.
        # This is acceptable because:
        # 1. The expression has already been validated by safe_sympy_parse
        # 2. Plotting is read-only visualization, not code execution
        # 3. Any dangerous constructs were blocked at parse time
        from sympy import lambdify
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
            print(f"      -> Detected periodic zeros with period T={best_T}")
            print(f"      -> Pattern: f(x) = x % {best_T}")
            print(f"      -> Match rate: {matches}/{checks} checked points")
        
        seeds.append(f"mod(x, {best_T})")
        # Also try "x % T" (operator syntax)
        seeds.append(f"x % {best_T}") 
        
    return seeds
                                                                                                                        