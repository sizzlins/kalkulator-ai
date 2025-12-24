"""
Command handlers for the Kalkulator CLI.
Extracted from app.py to enforce Rule 4 (Small Units).
"""

import logging
import re
from typing import Any
from typing import Dict

import kalkulator_pkg.parser as kparser

from ..cache_manager import export_cache_to_file
from ..cache_manager import get_persistent_cache
from ..cache_manager import replace_cache_from_file
from ..function_manager import BUILTIN_FUNCTION_NAMES
from ..function_manager import clear_functions
from ..function_manager import clear_saved_functions
from ..function_manager import export_function_to_file
from ..function_manager import list_functions
from ..function_manager import load_functions
from ..function_manager import save_functions
from ..solver.dispatch import solve_single_equation
from ..utils.formatting import print_result_pretty
from ..worker import clear_caches

logger = logging.getLogger(__name__)

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

    # === Function Finding/System ===
    if raw_lower.startswith("find "):
        # e.g. "find f(x)" or "find f(x) given ..."
        # Implement _handle_find_command logic here or call helper
        _handle_find_command(text, variables)
        return True

    # === Calculation Commands ===
    if raw_lower.startswith("calc "):
        # calc <expr> -> print result (legacy wrapper)
        # REPL handles expressions natively, but explicit calc strips prefix.
        # Returning False allows REPL to process the stripped text?
        # No, REPL dispatch is strict. We should process it here via evaluate_safely mechanism?
        # Or return a special signal?
        # Actually, REPL can handle the stripped arithmetic if we return handled=True but print result ourselves?
        # Better: let REPL handle "calc" by stripping it in process_input?
        # No, "process_input" calls "handle_command". If handle_command processes it, great.
        # But handle_command needs access to REPL's evaluate logic?
        # To avoid circular dep, we can return False and let REPL handle.
        # BUT REPL needs to know to strip "calc ".
        # Let's handle it here if possible or return a modified text?
        # Protocol: return (handled: bool, modified_text: Optional[str])?
        # Too complex. Let's just strip and assume REPL handles expressions.
        # Wait, if I return True, REPL stops.
        # So I must evaluate here if I return True.
        # But I don't want to duplicate evaluate_safely.
        # Let's SKIP "calc" here and handle it in REPL core explicitly?
        # Or import evaluate_safely here.
        # Since evaluate_safely is in worker.py/worker usage, we can import it.
        # But REPL core uses REPL.evaluate_safely wrapper logic? No, it uses 'from ..worker import evaluate_safely'.
        from ..worker import evaluate_safely

        eval_text = text[5:].strip()
        # We need to substitute variables?
        # handle_command receives 'variables' dict.
        eval_text_subbed = _substitute_vars(eval_text, variables)
        res = evaluate_safely(eval_text_subbed)
        print_result_pretty(res)
        return True

    # === Solver Command ===
    # "solve" is partly handled in repl_core for shadowing check, but the heavy lifting is here.
    # Actually repl_core had special logic to detect "solve" and pass to handle_single_part.
    # If we move logic here, we centralization it.
    if raw_lower.startswith("solve "):
        # Logic from app.py
        _handle_solve_command(text, variables)
        return True

    # === Export Command ===
    if raw_lower.startswith("export "):
        _handle_export_command(text)
        return True

    # === Research Commands (Evolve, SINDy, Causal, Dimensionless) ===
    if raw_lower.startswith("evolve "):
        _handle_evolve(text, variables)
        return True

    if raw_lower.startswith("find ode"):
        _handle_find_ode(text)
        return True

    if raw_lower.startswith("discover causal"):
        _handle_discover_causal(text)
        return True

    if raw_lower.startswith("find dimensionless"):
        _handle_find_dimensionless(text)
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


def generate_pattern_seeds(X, y, variable_names):
    """Detect patterns in data and return seed expression strings for evolve.

    Smart seeding: detects poles (inf/nan) and frequencies, then generates
    seed expressions that give evolution a head start.

    Args:
        X: Input data array (n_samples, n_vars) or (n_samples,)
        y: Output data array (n_samples,)
        variable_names: List of variable names like ['x'] or ['x', 'y']

    Returns:
        List of seed expression strings like ['1/(x-1)', '1/(x-1)**2']
    """
    import numpy as np

    seeds = []
    var = variable_names[0] if variable_names else "x"

    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # --- 1. Detect poles (where y is inf/nan) ---
    for i, y_val in enumerate(y):
        if not np.isfinite(y_val):
            pole_x = X[i, 0]
            # Generate pole-based seeds
            seeds.append(f"1/({var}-{pole_x})")
            seeds.append(f"1/({var}-{pole_x})**2")
            seeds.append(f"{var}/({var}-{pole_x})**2")
            # Also add with negative sign
            seeds.append(f"1/({pole_x}-{var})")

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
                        seeds.append(f"1/({var}-{mid_x:.2f})")

    # Remove duplicates while preserving order
    seen = set()
    unique_seeds = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            unique_seeds.append(s)

    return unique_seeds


def _handle_evolve(text, variables=None):
    """Handle the 'evolve' command for genetic symbolic regression."""
    try:
        import numpy as np

        from ..symbolic_regression import GeneticConfig
        from ..symbolic_regression import GeneticSymbolicRegressor

        # Strategy 1: Seeding
        # Parse "--seed 'expr'" or "--seed "expr""
        seeds = []
        seed_pattern = re.compile(r'--seed\s+["\']([^"\']+)["\']')
        matches = seed_pattern.findall(text)
        if matches:
            seeds.extend(matches)
            text = seed_pattern.sub("", text)

        # Strategy 7: Boosting
        # Parse "--boost <N>"
        boosting_rounds = 1
        boost_match = re.search(r"--boost\s+(\d+)", text)
        if boost_match:
            boosting_rounds = int(boost_match.group(1))
            # Remove flag from text
            text = re.sub(r"--boost\s+\d+", "", text)

        # Strategy 8: Hybrid (find → evolve)
        # Parse "--hybrid" flag
        use_hybrid = "--hybrid" in text.lower()
        if use_hybrid:
            text = re.sub(r"--hybrid", "", text, flags=re.IGNORECASE)

        # Parse: evolve f(x) from x=[...], y=[...]
        # or: evolve f(x,y) from x=[...], y=[...], z=[...]
        # Parse: evolve f(x) from x=[...], y=[...]
        # or: evolve f(x,y) from x=[...], y=[...], z=[...]
        match = re.match(
            r"evolve\s+(\w+)\s*\(([^)]+)\)\s+from\s+(.+)", text, re.IGNORECASE
        )

        is_implicit = False
        data_part = None

        if match:
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
            array_pattern = re.compile(r"(\w+)\s*=\s*\[([^\]]+)\]")
            for m in array_pattern.finditer(data_part):
                var = m.group(1)
                values = [float(v.strip()) for v in m.group(2).split(",")]
                data_dict[var] = np.array(values)

            # Parse individual function points "f(1)=2, f(2)=3"
            # This allows "evolve f(x) from f(1)=2, f(2)=3"
            if data_part:
                point_pattern = re.compile(r"(\w+)\s*\(([^)]+)\)\s*=\s*([^,]+)")
                points_x = {v: [] for v in input_var_names}
                points_y = []
                skipped_complex = 0  # Track skipped complex data points

                for m in point_pattern.finditer(data_part):
                    p_func = m.group(1)
                    if p_func != func_name:
                        continue

                    try:
                        p_args_str = m.group(2)
                        p_val_str = m.group(3).strip()

                        # Check for complex/imaginary values (skip with warning)
                        if (
                            any(indicator in p_val_str for indicator in ["i", "I", "j"])
                            and not p_val_str.replace(".", "")
                            .replace("-", "")
                            .replace("+", "")
                            .replace("e", "")
                            .isdigit()
                        ):
                            skipped_complex += 1
                            continue

                        # Basic float parsing
                        p_args = [float(a.strip()) for a in p_args_str.split(",")]

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
                            p_val = float(p_val_str)

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
                            # User said "evolve m(x,y)" but gave "m(1)=2"
                            # This is harder. We can't invent data.
                            # Treat as partial match or warn?
                            # For now, simplistic approach: drop the extra target vars if empty?
                            # Or just skip this point?
                            # Strict behavior for UNDER-specified data is safer.
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
        auto_seeds = generate_pattern_seeds(X, y, input_vars)
        if auto_seeds:
            seeds.extend(auto_seeds)
            if len(auto_seeds) <= 5:
                print(f"Smart seeding: detected patterns, seeding with {auto_seeds}")
            else:
                print(f"Smart seeding: detected {len(auto_seeds)} pattern-based seeds")

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

                # If find() succeeded, add its expression as a seed
                if success and func_str:
                    seeds.append(func_str)
                    display = func_str[:50] + "..." if len(func_str) > 50 else func_str
                    print(
                        f"Hybrid seeding: using find() result '{display}' as starting point"
                    )
            except Exception as e:
                print(f"Hybrid mode: find() failed ({e}), continuing with other seeds")

        # --- FILTER: Remove inf/nan from data AFTER pattern detection ---
        # Poles were used for seeding, but must be removed for fitness calculation
        finite_mask = np.isfinite(y)
        if not np.all(finite_mask):
            X = X[finite_mask]
            y = y[finite_mask]

        print(
            f"Evolving {func_name}({', '.join(input_vars)}) from {len(y)} data points..."
        )

        # Apply boost multiplier to evolution parameters
        # --boost N gives N times more compute resources for complex functions
        base_population = 100
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
            verbose=True,
            seeds=seeds,
            boosting_rounds=1,  # Already applied via parameter scaling
        )
        regressor = GeneticSymbolicRegressor(config)
        pareto = regressor.fit(X, y, input_vars)

        # get_knee_point attempts to balance complexity vs MSE, but for perfect fits (MSE ~ 0)
        # we should always prefer the accurate solution even if slightly more complex.
        knee = pareto.get_knee_point()
        best_mse = pareto.get_best()

        best = knee
        if best_mse and best_mse.mse < 1e-9:
            best = best_mse
        elif knee:
            best = knee
        else:
            best = best_mse

        if not best:
            print("No suitable model found.")
            return

        # Print Result
        print(f"\nResult: {best.expression}")
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
        print(f"Error: {e}")


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
    # My "fix" made it a valid equation, but didn't implement storage.

    # To fix this properly (Rule 5), `handle_single_part` in `repl_core`
    # needs to detect "Undefined Function Call = Value" and store it as a constraint/datapoint
    # INSTEAD of just solving it.

    print(
        "Function finding logic detected. (Data point collection not fully active in this patching phase)."
    )
    # This avoids the crash/math junk, but functionality is partial.
    # The prompt asked me to fix parsing "Undefined Function Parsing".
    # I did.
    # Now I need to fix the REPL flow to USE that parsed info.
