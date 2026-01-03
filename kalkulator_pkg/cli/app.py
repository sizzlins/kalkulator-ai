from __future__ import annotations

import argparse
import json
import logging
import math
from math import gcd
from typing import Any

import sympy as sp

import kalkulator_pkg.parser as kparser

from .commands import handle_debug_command
from .context import ReplContext

# Helper for Python < 3.9 compatibility
try:
    from math import lcm as _math_lcm
except ImportError:

    def _math_lcm(*args: int) -> int:
        if not args:
            return 1
        res = args[0]
        for v in args[1:]:
            res = abs(res * v) // gcd(res, v)
        return res


from ..config import VAR_NAME_RE
from ..config import VERSION
from ..parser import format_superscript
from ..parser import split_top_level_commas
from ..solver import solve_inequality
from ..solver import solve_single_equation
from ..solver import solve_system
from ..types import ParseError
from ..types import ValidationError
from ..worker import evaluate_safely

_logger = logging.getLogger(__name__)

from ..utils.formatting import format_inverse_solutions as _format_inverse_solutions
from ..utils.formatting import (
    format_number_no_trailing_zeros as _format_number_no_trailing_zeros,
)
from ..utils.formatting import format_special_values as _format_special_values
from ..utils.formatting import print_result_pretty
from ..utils.numeric import (
    compute_integerized_equation as _compute_integerized_equation,
)
from ..utils.numeric import extended_gcd as _extended_gcd
from ..utils.numeric import find_integer_solutions_for_linear
from ..utils.numeric import (
    parse_target_with_ambiguity_detection as _parse_target_with_ambiguity_detection,
)
from ..utils.numeric import (
    solve_modulo_system_if_applicable as _solve_modulo_system_if_applicable,
)


def _health_check() -> int:
    """Run health check to verify dependencies and basic operations.

    Returns:
        Exit code (0 for success, non-zero for failures)
    """
    checks_passed = 0
    checks_failed = 0

    print("Running Kalkulator health check...")
    print("-" * 50)

    # Check SymPy import
    try:
        import sympy as sp

        version = sp.__version__
        print(f"[OK] SymPy {version} imported successfully")
        checks_passed += 1
    except ImportError as e:
        print(f"[FAIL] SymPy import failed: {e}")
        checks_failed += 1

    # Check basic parsing
    try:
        from ..parser import parse_preprocessed
        from ..parser import preprocess

        test_expr = "2 + 2"
        preprocessed = preprocess(test_expr)
        parsed = parse_preprocessed(preprocessed)
        if parsed == 4:
            print("[OK] Basic parsing works")
            checks_passed += 1
        else:
            print(f"[FAIL] Basic parsing failed: expected 4, got {parsed}")
            checks_failed += 1
    except Exception as e:
        print(f"[FAIL] Parsing check failed: {e}")
        checks_failed += 1

    # Check solving
    try:
        from ..solver import solve_single_equation

        result = solve_single_equation("x + 1 = 0")
        if result.get("ok") and result.get("exact") == ["-1"]:
            print("[OK] Basic solving works")
            checks_passed += 1
        else:
            print(f"[FAIL] Solving check failed: {result}")
            checks_failed += 1
    except Exception as e:
        print(f"[FAIL] Solving check failed: {e}")
        checks_failed += 1

    # Check worker (if available)
    try:
        from ..worker import evaluate_safely

        result = evaluate_safely("3 * 3")
        if result.get("ok") and result.get("result") == "9":
            print("[OK] Worker evaluation works")
            checks_passed += 1
        else:
            print(f"[FAIL] Worker check failed: {result}")
            checks_failed += 1
    except Exception as e:
        print(f"[WARN] Worker check skipped: {e}")

    # Check optional dependencies
    try:
        import numpy

        print(f"[OK] NumPy {numpy.__version__} available")
        checks_passed += 1
    except ImportError:
        print("[WARN] NumPy not available (plotting features limited)")
        print("  To install: pip install numpy")
        print("  Or install all optional dependencies: pip install numpy matplotlib")

    try:
        import matplotlib

        print(f"[OK] Matplotlib {matplotlib.__version__} available")
        checks_passed += 1
    except ImportError:
        print("[WARN] Matplotlib not available (plotting features limited)")
        print("  To install: pip install matplotlib")
        print("  Or install all optional dependencies: pip install numpy matplotlib")

    # Check Windows-specific limitations
    try:
        import sys

        if sys.platform == "win32":
            try:
                import resource  # noqa: F401

                print("[OK] Resource limits available (Unix-like behavior)")
                checks_passed += 1
            except ImportError:
                print(
                    "[INFO] Resource limits unavailable on Windows (expected limitation)"
                )
                print(
                    "  This is normal on Windows - the 'resource' module is Unix-only."
                )
                print(
                    "  Resource limits are not required for normal calculator operation."
                )
    except Exception:
        pass

    print("-" * 50)
    print(f"Results: {checks_passed} passed, {checks_failed} failed")

    if checks_failed > 0:
        print("\n[WARN] Some health checks failed. Core functionality may be impaired.")
        return 1

    print("\n[OK] All health checks passed!")
    return 0


def repl_loop(output_format: str = "human") -> None:
    """
    Legacy entry point delegating to the new modular REPL class.
    Refactored to enforce Engineering Standard #4 (Small Units) and #5 (Wall Rule).
    """
    from .repl_core import REPL

    # Initialize context (global vs local handling logic has been moved)
    repl = REPL()
    repl.start()
    print("Goodbye.")


def print_help_text() -> None:
    """Print help text for REPL commands."""
    from ..config import VERSION

    help_text = f"""kalkulator-ai v{VERSION}

COMMANDS
  help      Show commands
  quit      Exit
  health    System check

MATH
  diff(y,x)      Differentiate
  integrate(y,x) Integrate
  factor(y)      Factor
  expand(y)      Expand
  solve <eq>     Solve equation
  plot <expr>    Plot function

DISCOVERY
  f(1)=1, f(2)=4...       Find function from data points
    --auto-evolve         Auto-switch to evolve if exact finding fails
  evolve f(x) from...     Genetic algorithm discovery
    --hybrid              Seed with find() result
    --boost N             N× resources (pop/gen/timeout)
  find ode                Find ODE (SINDy)
  benchmark               Run benchmarks

SHORTCUTS (for evolve)
  all f(...)    Full power: --hybrid --verbose --boost 3
  b f(...)      Fast mode:  --verbose --boost 3
  h f(...)      Smart mode: --hybrid --verbose
  v f(...)      Verbose:    --verbose

DATA
  clearcache     Clear cache
  showcache      List cache
  savecache      Save cache
  loadcache      Load cache
  export         Export to file

SETTINGS
  debug [on|off]    Debug mode
  timing [on|off]   Timing stats
  cachehits [on|off] Show hits

Full Docs: https://github.com/sizzlins/kalkulator-ai
"""
    print(help_text)


def main_entry(argv: list[str] | None = None) -> int:
    """
    Main entry point for Kalkulator CLI.

    Args:
        argv: Optional command-line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Load persistent cache on startup
    try:
        from ..cache_manager import load_persistent_cache

        load_persistent_cache()  # Initialize cache
    except ImportError:
        pass  # Cache manager not available, continue without persistent cache

    parser = argparse.ArgumentParser(prog="kalkulator")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expr", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--worker-solve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--payload", type=str, help=argparse.SUPPRESS)
    parser.add_argument(
        "-e",
        "--eval",
        type=str,
        help="Evaluate one expression and exit (non-interactive)",
        dest="eval_expr",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Emit JSON for machine parsing (deprecated, use --format json)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "human"],
        default="human",
        help="Output format: json (machine-readable) or human (human-readable)",
    )
    parser.add_argument(
        "-t", "--timeout", type=int, help="Override worker timeout (seconds)"
    )
    parser.add_argument(
        "--no-numeric-fallback",
        action="store_true",
        help="Disable numeric root-finding fallback",
    )
    parser.add_argument(
        "-p", "--precision", type=int, help="Set output precision (significant digits)"
    )
    parser.add_argument(
        "-v", "--version", action="store_true", help="Show program version"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument("--log-file", type=str, help="Write logs to file")
    parser.add_argument(
        "--cache-size", type=int, help="Set parse/eval cache size (default: 1024/2048)"
    )
    parser.add_argument(
        "--max-nsolve-guesses",
        type=int,
        help="Set maximum nsolve guesses for numeric root finding (default: 36)",
    )
    parser.add_argument(
        "--worker-mode",
        type=str,
        choices=["pool", "single", "subprocess"],
        help="Worker execution mode (default: pool)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["auto", "symbolic", "numeric"],
        help="Solver method (default: auto)",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check and verify dependencies",
    )
    args = parser.parse_args(argv)

    # Determine output format
    output_format = args.format
    if args.json:  # Backward compatibility for deprecated -j flag
        output_format = "json"

    # Setup logging
    try:
        from ..logging_config import setup_logging

        setup_logging(level=args.log_level, log_file=args.log_file)
    except ImportError:
        pass  # Logging optional
    # Apply CLI configuration overrides
    # Note: Direct module variable modification is used for simplicity.
    # Future improvement: Use dependency injection or a configuration object.
    import kalkulator_pkg.config as _config
    import kalkulator_pkg.worker as _worker_module

    if args.timeout and args.timeout > 0:
        _worker_module.WORKER_TIMEOUT = int(args.timeout)
    if args.no_numeric_fallback:
        _config.NUMERIC_FALLBACK_ENABLED = False
    if args.precision and args.precision > 0:
        _config.OUTPUT_PRECISION = int(args.precision)
    if args.cache_size and args.cache_size > 0:
        _config.CACHE_SIZE_PARSE = int(args.cache_size)
        _config.CACHE_SIZE_EVAL = int(args.cache_size * 2)
    if args.max_nsolve_guesses and args.max_nsolve_guesses > 0:
        _config.MAX_NSOLVE_GUESSES = int(args.max_nsolve_guesses)
    if args.worker_mode:
        if args.worker_mode == "subprocess":
            _config.ENABLE_PERSISTENT_WORKER = False
        elif args.worker_mode == "single":
            _config.WORKER_POOL_SIZE = 1
        # "pool" is default
    if args.method:
        _config.SOLVER_METHOD = args.method
    if args.worker:
        from ..worker import worker_evaluate

        out = worker_evaluate(args.expr or "")
        print(json.dumps(out))
        return 0
    if args.worker_solve:
        from ..worker import _worker_solve_dispatch

        try:
            payload = json.loads(args.payload or "{}")
        except (json.JSONDecodeError, ValueError, TypeError):
            # Invalid JSON - use empty dict
            payload = {}
        print(json.dumps(_worker_solve_dispatch(payload)))
        return 0
    if args.version:
        print(VERSION)
        return 0
    if args.eval_expr:
        expr = args.eval_expr.strip()
        # Remove ">>>" prompt if present
        if expr.startswith(">>>"):
            expr = expr[3:].strip()
        # Handle empty input or just "="
        if not expr or expr == "=":
            print(
                "Error: Empty input. Please enter a valid expression, equation, or command."
            )
            return 1
        import re

        # Check for function finding command FIRST (before other processing)
        is_find_command = False
        find_func_cmd = None

        # Check for explicit "find" keyword
        if "find" in expr.lower():
            try:
                from ..function_manager import find_function_from_data
                from ..function_manager import parse_find_function_command

                find_func_cmd = parse_find_function_command(expr)
                if find_func_cmd is not None:
                    is_find_command = True
            except Exception:
                pass

        # If no explicit "find" keyword, check for multiple function assignments
        if not is_find_command:
            try:
                from ..function_manager import find_function_from_data  # noqa: F811
                from ..function_manager import parse_find_function_command

                # Count function assignment patterns: func_name(args) = value
                func_assignment_pattern = (
                    r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]+\)\s*=\s*[^,]+"
                )
                matches = list(re.finditer(func_assignment_pattern, expr))
                # If we have 2 or more such patterns, treat as function finding command
                if len(matches) >= 2:
                    # Extract function name from first match
                    func_name = matches[0].group(1)
                    # Check if all matches use the same function name
                    if all(m.group(1) == func_name for m in matches):
                        # Infer parameter names from the first match
                        first_match = matches[0]
                        args_match = re.search(
                            rf"{re.escape(func_name)}\s*\(([^)]+)\)",
                            first_match.group(0),
                        )
                        if args_match:
                            args_str = args_match.group(1)
                            # Parse arguments to count them
                            arg_list = split_top_level_commas(args_str)
                            # Generate parameter names: x, y, z, ... or x1, x2, x3, ...
                            param_names = []
                            param_chars = "xyzuvwrst"
                            for i, _ in enumerate(arg_list):
                                if i < len(param_chars):
                                    param_names.append(param_chars[i])
                                else:
                                    param_names.append(f"x{i+1}")
                            # Create a modified expression with "find" keyword for parsing
                            expr_with_find = (
                                expr.rstrip(",").strip()
                                + f", find {func_name}({', '.join(param_names)})"
                            )
                            find_func_cmd = parse_find_function_command(expr_with_find)
                            if find_func_cmd is not None:
                                is_find_command = True
                                expr = expr_with_find
            except Exception:
                pass

        # Process function finding if detected
        if is_find_command and find_func_cmd is not None:
            try:
                # split_top_level_commas is already imported at module level
                import sympy as sp

                from ..function_manager import find_function_from_data  # noqa: F811

                func_name, param_names = find_func_cmd
                find_pattern = rf"find\s+{re.escape(func_name)}\s*\([^)]*\)"
                data_str = re.sub(find_pattern, "", expr, flags=re.IGNORECASE).strip()
                data_str = data_str.rstrip(",").strip()

                # First, process any function definitions in data_str
                # e.g., "f(x,y)=x^2+y^2, find f(x,y) = 5" should define f first
                parts_to_process = split_top_level_commas(data_str)
                remaining_parts = []
                for part in parts_to_process:
                    part = part.strip()
                    if not part:
                        continue
                    # Try to parse as function definition
                    try:
                        from ..function_manager import ValidationError
                        from ..function_manager import define_function
                        from ..function_manager import parse_function_definition

                        func_def = parse_function_definition(part)
                        if func_def is not None:
                            def_func_name, def_params, def_body = func_def
                            try:
                                define_function(def_func_name, def_params, def_body)
                                params_str = ", ".join(def_params) if def_params else ""
                                print(
                                    f"Function '{def_func_name}({params_str})' defined as: {def_body}"
                                )
                            except ValidationError as e:
                                print(f"Error defining function: {e.message}")
                        else:
                            # Not a function definition, keep for further processing
                            remaining_parts.append(part)
                    except Exception:
                        remaining_parts.append(part)

                # Update data_str with remaining parts
                data_str = ", ".join(remaining_parts)

                # Check if this is an inverse solving case (e.g., "find f(x,y) = 5")
                # vs function finding case (e.g., "f(1)=1, f(2)=4, find f(x)")
                is_inverse_solve = False
                target_value = None

                # If data_str starts with "=" it's likely an inverse solve: "find f(x,y) = 5"
                if data_str.strip().startswith("="):
                    is_inverse_solve = True
                    target_value = data_str.strip()[1:].strip()
                elif not any(
                    f"{func_name}(" in part for part in split_top_level_commas(data_str)
                ):
                    # No function calls with this func_name, likely inverse solve
                    is_inverse_solve = True
                    # Check if there's an equals sign
                    if "=" in data_str:
                        target_value = data_str.split("=", 1)[1].strip()
                    else:
                        target_value = data_str.strip()

                if is_inverse_solve and target_value:
                    # Check if function is defined for inverse solving
                    from ..function_manager import _function_registry

                    if func_name in _function_registry:
                        from ..solver import solve_inverse_function

                        result = solve_inverse_function(
                            func_name, target_value, param_names
                        )
                        _format_inverse_solutions(
                            result, func_name, param_names, target_value
                        )
                        if result["ok"]:
                            return 0
                        else:
                            return 1
                    else:
                        print(
                            f"Error: Function '{func_name}' is not defined. Define it first or provide data points."
                        )
                        return 1

                # Parse data points
                data_points = []
                parts = split_top_level_commas(data_str)
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    pattern = rf"{re.escape(func_name)}\s*\(([^)]+)\)\s*=\s*(.+)"
                    match = re.match(pattern, part)
                    if match:
                        args_str = match.group(1)
                        value_str = match.group(2).strip()

                        # Check if value_str is a simple numeric string (preserve for exact precision)
                        value_str_preserved = None
                        try:
                            float(value_str)
                            if not any(
                                op in value_str
                                for op in [
                                    "+",
                                    "-",
                                    "*",
                                    "/",
                                    "(",
                                    ")",
                                    "^",
                                    "**",
                                    "sqrt",
                                    "sin",
                                    "cos",
                                    "exp",
                                    "log",
                                    "pi",
                                    "e",
                                ]
                            ):
                                value_str_preserved = value_str
                        except (ValueError, TypeError):
                            pass

                        # Parse arguments
                        args = []
                        arg_strings_preserved = []
                        for arg in split_top_level_commas(args_str):
                            try:
                                arg_stripped = arg.strip()
                                arg_str_preserved = None
                                try:
                                    float(arg_stripped)
                                    if not any(
                                        op in arg_stripped
                                        for op in [
                                            "+",
                                            "-",
                                            "*",
                                            "/",
                                            "(",
                                            ")",
                                            "^",
                                            "**",
                                            "sqrt",
                                            "sin",
                                            "cos",
                                            "exp",
                                            "log",
                                            "pi",
                                            "e",
                                        ]
                                    ):
                                        arg_str_preserved = arg_stripped
                                except (ValueError, TypeError):
                                    pass

                                arg_strings_preserved.append(arg_str_preserved)
                                arg_expr = kparser.parse_preprocessed(arg_stripped)
                                # Ensure constants are treated as such
                                arg_expr = arg_expr.subs(
                                    {sp.Symbol("pi"): sp.pi, sp.Symbol("e"): sp.E}
                                )
                                try:
                                    arg_val = float(sp.N(arg_expr))
                                except (ValueError, TypeError):
                                    arg_val = arg_expr
                                args.append(arg_val)
                                print(
                                    f"DEBUG: Successfully parsed {arg_stripped} -> {arg_val}",
                                    flush=True,
                                )
                            except Exception as e:
                                print(
                                    f"DEBUG: Failed to parse arg {arg_stripped}: {e}",
                                    flush=True,
                                )
                                break

                        if len(args) == len(param_names):
                            try:
                                if value_str_preserved:
                                    value = value_str_preserved
                                else:
                                    value_expr = kparser.parse_preprocessed(value_str)
                                    try:
                                        value = float(sp.N(value_expr))
                                    except (ValueError, TypeError):
                                        value = value_expr

                                # Preserve strings for exact precision
                                final_args = []
                                for i, arg_val in enumerate(args):
                                    if (
                                        i < len(arg_strings_preserved)
                                        and arg_strings_preserved[i]
                                    ):
                                        final_args.append(arg_strings_preserved[i])
                                    elif isinstance(arg_val, str):
                                        final_args.append(arg_val)
                                    elif isinstance(arg_val, (int, float)):
                                        final_args.append(arg_val)
                                    else:
                                        try:
                                            if isinstance(
                                                arg_val,
                                                (sp.Rational, sp.Integer, sp.Float),
                                            ):
                                                final_args.append(str(arg_val))
                                            else:
                                                final_args.append(float(sp.N(arg_val)))
                                        except (ValueError, TypeError):
                                            final_args.append(arg_val)

                                if value_str_preserved:
                                    final_value = value_str_preserved
                                elif isinstance(value, str):
                                    final_value = value
                                elif isinstance(value, (int, float)):
                                    if isinstance(value, float):
                                        final_value = (
                                            format(value, ".15f")
                                            .rstrip("0")
                                            .rstrip(".")
                                        )
                                    else:
                                        final_value = str(value)
                                else:
                                    try:
                                        if isinstance(
                                            value, (sp.Rational, sp.Integer, sp.Float)
                                        ):
                                            final_value = str(value)
                                        else:
                                            final_value = (
                                                str(value)
                                                if hasattr(value, "__str__")
                                                else float(sp.N(value))
                                            )
                                    except (ValueError, TypeError):
                                        final_value = value

                                data_points.append((final_args, final_value))
                            except Exception:
                                pass

                if data_points:
                    success, func_str, factored_form, error_msg = (
                        find_function_from_data(data_points, param_names)
                    )
                    if success:
                        params_str = ", ".join(param_names)
                        print(f"{func_name}({params_str}) = {func_str}")
                        if factored_form:
                            print(
                                f"Equivalent: {func_name}({params_str}) = {factored_form}"
                            )
                        print(
                            f"Function '{func_name}' is now available. You can call it like: {func_name}(values)"
                        )
                        return 0
                    else:
                        print(f"Error: Error finding function: {error_msg}")
                        return 1
                else:
                    print("Error: No valid data points found for function finding")
                    return 1
            except Exception as e:
                _logger.exception("Error processing function finding in --eval")
                print(f"Error: Failed to process function finding: {e}")
                return 1

        # Continue with normal processing if not a function finding command
        find_tokens = re.findall(r"\bfind\s+(\w+)\b", expr, re.IGNORECASE)
        find = find_tokens[0] if find_tokens else None
        raw_no_find = re.sub(r"\bfind\s+\w+\b", "", expr, flags=re.IGNORECASE).strip()
        if any(op in raw_no_find for op in ("<", ">", "<=", ">=")):
            res = solve_inequality(raw_no_find, find)
        elif "=" in raw_no_find:
            parts = split_top_level_commas(raw_no_find)
            if len(parts) > 1:
                # Check if all assignments are to the same variable
                # This handles cases like "x = 1 % 2, x=3 % 6, x=3 % 7" where we want to evaluate each expression
                all_assign_same = all(
                    "=" in p and VAR_NAME_RE.match(p.split("=", 1)[0].strip())
                    for p in parts
                )
                if all_assign_same:
                    assigned_vars_main = [
                        p.split("=", 1)[0].strip() for p in parts if "=" in p
                    ]
                    if (
                        len(assigned_vars_main) > 1
                        and len(set(assigned_vars_main)) == 1
                    ):
                        # All assignments are to the same variable
                        var = assigned_vars_main[0]
                        # Try to solve as system of congruences first
                        solved, exit_code = _solve_modulo_system_if_applicable(
                            parts, var, output_format
                        )
                        if solved:
                            # Save cache after evaluation
                            try:
                                from ..cache_manager import save_cache_to_disk

                                save_cache_to_disk()
                            except ImportError:
                                pass
                            return exit_code

                        # If not all modulo or CRT solving failed, evaluate each expression separately
                        for p in parts:
                            if "=" in p:
                                left, right = p.split("=", 1)
                                var = left.strip()
                                rhs = right.strip() or "0"
                                # Evaluate the RHS expression (like "1 % 2")
                                eva = evaluate_safely(rhs)
                                if not eva.get("ok"):
                                    print(
                                        f"Error evaluating '{var} = {rhs}': {eva.get('error')}"
                                    )
                                    continue
                                try:
                                    # Format and print the result
                                    val_str = eva.get("result", "")
                                    approx_str = eva.get("approx", "")
                                    if output_format == "json":
                                        print(
                                            json.dumps(
                                                {
                                                    "ok": True,
                                                    "result": val_str,
                                                    "variable": var,
                                                }
                                            )
                                        )
                                    else:
                                        if approx_str:
                                            print(f"{var} = {val_str}")
                                            if approx_str != val_str:
                                                print(f"  Decimal: {approx_str}")
                                        else:
                                            print(f"{var} = {val_str}")
                                except Exception as e:
                                    print(
                                        f"Error formatting result for '{var} = {rhs}': {e}"
                                    )
                                    continue
                        # Save cache after evaluation
                        try:
                            from ..cache_manager import save_cache_to_disk

                            save_cache_to_disk()
                        except ImportError:
                            pass
                        return 0

                res = solve_system(raw_no_find, find)
            else:
                res = solve_single_equation(parts[0], find)
        else:
            eva = evaluate_safely(raw_no_find)
            if not eva.get("ok"):
                res = {"ok": False, "error": eva.get("error")}
            else:
                res = {
                    "ok": True,
                    "type": "value",
                    "result": eva.get("result"),
                    "approx": eva.get("approx"),
                }
        print_result_pretty(res, output_format=output_format)
        # Save cache after evaluation (periodic save)
        try:
            from ..cache_manager import save_cache_to_disk

            save_cache_to_disk()
        except ImportError:
            pass
        return 0

    # Add health check command
    if hasattr(args, "health_check") and args.health_check:
        return _health_check()

    try:
        repl_loop(output_format=output_format)
    finally:
        # Ensure worker processes are stopped on exit
        try:
            from ..worker import _WORKER_MANAGER

            _WORKER_MANAGER.stop()
        except Exception:
            pass
        # Save persistent cache on exit
        try:
            from ..cache_manager import save_cache_to_disk

            save_cache_to_disk()
        except ImportError:
            pass
    return 0


if __name__ == "__main__":
    """Allow running the CLI module directly with python -m kalkulator_pkg.cli"""
    import sys

    sys.exit(main_entry())
