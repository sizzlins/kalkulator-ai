"""
Evolution handler for Genetic Symbolic Regression.
Extracted from repl_commands.py.
"""
import logging
import re
import warnings
import ast
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import scipy.special
import sympy as sp

import kalkulator_pkg.parser as kparser
from kalkulator_pkg.cli.arg_schemas import parse_evolve_flags
from kalkulator_pkg.symbolic_regression.expression_tree import symbolify_constants, ExpressionTree, ExpressionNode, NodeType
from kalkulator_pkg.symbolic_regression import GeneticConfig, GeneticSymbolicRegressor, ParetoFront, ParetoSolution
from kalkulator_pkg.symbolic_regression.forensic_analysis import generate_pattern_seeds
from kalkulator_pkg.symbolic_regression.ode_discovery import ODEDiscoveryEngine, ODEConfig
from kalkulator_pkg.symbolic_regression.numerical_diff import check_even_spacing
from kalkulator_pkg.symbolic_regression.gene_bank import get_gene_bank
from kalkulator_pkg.heuristics import detect_smoothness

from kalkulator_pkg.utils.data_loading import load_csv_data
from kalkulator_pkg.utils.parsing import eval_to_float
from kalkulator_pkg.utils.formatting import format_solution, print_result_pretty
from kalkulator_pkg.worker import evaluate_safely
from kalkulator_pkg.function_manager import find_function_from_data, define_function
from kalkulator_pkg.sympy_defs import ALLOWED_SYMPY_NAMES
from kalkulator_pkg.cli.handlers.constraints import matches_ban

logger = logging.getLogger(__name__)

# Regex Constants
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


def handle_evolve(text: str, ctx: Any, variables: Dict[str, str] | None = None) -> None:
    """Handle the 'evolve' command for genetic symbolic regression."""
    try:
        # SHORTCUT COMMANDS: Expand to full evolve syntax
        text_lower = text.lower().strip()
        
        # Mapping of shortcut -> flags
        shortcuts = {
            # 4-Series (Boost 4)
            'altvd4': '--super-verbose --transform --debug --boost 4',
            'altv4':  '--super-verbose --transform --boost 4',
            'alld4':  '--verbose --transform --boost 4',
            'alt4':   '--verbose --transform --boost 4',
            'all4':   '--verbose --boost 4',
            
            # Standard Series (Boost 3 / Default)
            'altvd': '--hybrid --verbose --super-verbose --debug --boost 3 --transform', # DEBUG
            'altv':  '--hybrid --verbose --boost 3 --transform', # FORENSIC
            'altd':  '--hybrid --verbose --debug --boost 3 --transform', # DEBUG (Inferred)
            'alt':   '--hybrid --verbose --boost 3 --transform', # ULTIMATE
            
            'all':   '--verbose --boost 3', # Full Power
            
            # Modes
            'b':     '--verbose --boost 3', # Fast
            'h':     '--hybrid --verbose',  # Smart
            'v':     '--verbose',           # Verbose
            'ode':   '--discover-ode',      # ODE
        }
        
        parts = text.split(' ', 1)
        cmd = parts[0].lower()
        if cmd in shortcuts:
            rest = parts[1] if len(parts) > 1 else ""
            text = f"evolve {shortcuts[cmd]} {rest}"

        # === Structured Flag Extraction (Phase 2 Refactor) ===
        try:
            config, text = parse_evolve_flags(text)
        except Exception as e:
            print(f"Error parsing command flags: {e}")
            print("Usage: evolve f(x) from f(1)=2, f(2)=4 [--boost N] [--verbose] [--seed 'expr']")
            return

        seeds = list(config.seeds)  # Mutable copy for downstream additions

        # Merge REPL-level bans (from 'ban' command) into config
        if ctx and hasattr(ctx, 'banned_operators') and ctx.banned_operators:
            config.banned.extend(b for b in ctx.banned_operators if b not in config.banned)
            
        # Strategy 2: Callr (Random Call) Data Generation
        if ' callr ' in text:
            match = re.search(r' callr\s+([a-zA-Z_]\w*)\s+(\d+)', text)
            if match:
                f_name = match.group(1)
                count = int(match.group(2))
                val = None
                
                # 1. Check Function Registry (User-defined functions)
                if ctx.function_registry and f_name in ctx.function_registry:
                    params, body_expr = ctx.function_registry[f_name]
                    val = f"{f_name}({','.join(params)})={body_expr}"
                    
                # 2. Check Variables
                elif variables and f_name in variables:
                     val = variables[f_name]
                
                if val:
                     import random
                     generated_data = []
                     
                     arity = 1
                     if "(" in val and ")" in val and "=" in val:
                         lhs = val.split("=")[0]
                         params = lhs[lhs.find("(")+1:lhs.find(")")].split(",")
                         arity = len([p for p in params if p.strip()])
                     
                     print(f"   [AltCall] Randomized generation from '{f_name}' on set 'default' ({count} points)...")
                     
                     success_count = 0
                     
                     # Helper to call function from context
                     def _call_func(args):
                         arg_str = ", ".join(map(str, args))
                         stmt = f"{f_name}({arg_str})"
                         res = evaluate_safely(stmt, allowed_functions=variables.keys() if variables else [])
                         return res.get('result')

                     # CROSS-HAIR SAMPLING
                     axis_count = max(20, count // 5)
                     if arity >= 2:
                         for i in range(axis_count):
                             args = []
                             zero_idx = i % arity
                             for j in range(arity):
                                 if j == zero_idx:
                                     args.append(0)
                                 else:
                                     args.append(round(random.uniform(0.5, 5), 4))
                             
                             res = _call_func(args)
                             if res:
                                 arg_str = ", ".join(map(str, args))
                                 generated_data.append(f"{f_name}({arg_str})={res}")
                                 success_count += 1
                         
                         # POWER LAW REFERENCE POINTS
                         ref_count = 0
                         for ref_y in [1, 4, 9]:
                             for _ in range(4):
                                 args = [round(random.uniform(1, 5), 4), ref_y]
                                 res = _call_func(args)
                                 if res:
                                     arg_str = ", ".join(map(str, args))
                                     generated_data.append(f"{f_name}({arg_str})={res}")
                                     ref_count += 1
                         
                         if generated_data:
                             print(f"   [CrossHair] Generated {axis_count} axis + {ref_count} reference points")
                     
                     # Random sampling
                     remaining = count - success_count
                     for _ in range(remaining):
                         args = []
                         for _ in range(arity):
                             if random.random() < 0.2:
                                 args.append(random.randint(-10, 10))
                             else:
                                 args.append(round(random.uniform(-5, 5), 4))
                                 
                         res = _call_func(args)
                         if res:
                             arg_str = ", ".join(map(str, args))
                             generated_data.append(f"{f_name}({arg_str})={res}")
                             success_count += 1
                     
                     if generated_data:
                         print(f"      {', '.join(generated_data)}")
                     
                     if success_count > 0:
                         from_clause = ", ".join(generated_data)
                         text = text.replace(match.group(0), f" from {from_clause}", 1)
                         
                         if f_name not in text.split("evolve")[1].split("from")[0]:
                              vnames = ['x','y','z','t','u','v'][:arity]
                              if not vnames: vnames = ['x']
                              target = f"{f_name}({','.join(vnames)})"
                              text = text.replace("evolve ", f"evolve {target} ", 1)
                     else:
                         print("   [AltCall] Failed to generate valid data points.")
                else:
                    print(f"   [AltCall] Function '{f_name}' not defined.")

        # === Flag Assignments ===
        boosting_rounds = config.boost
        use_hybrid = config.hybrid
        verbose_mode = config.verbose
        super_verbose = config.super_verbose
        use_transform = config.transform
        high_precision_mode = config.high_precision
        use_debug = config.debug
        use_discover_ode = config.discover_ode
        banned_operators = list(config.banned)

        if use_debug:
            logging.getLogger().setLevel(logging.DEBUG)
            print("   [Debug Mode] Enabled full debug logging")

        if high_precision_mode:
            print("   [High-Precision Mode] Using arbitrary-precision arithmetic (50+ digits)")

        if use_discover_ode:
            print(f"   [ODE Discovery Mode] Will search for differential equations")

        if banned_operators:
            print(f"   [Constraint] Banned functions: {banned_operators}")

        # Polynomial Mode
        use_polynomial = config.polynomial
        if use_polynomial:
            polynomial_banned = [
                'sin', 'cos', 'tan', 'exp', 'log', 'sqrt',
                'bessel_j0', 'gamma', 'prime_pi',
                'bitwise_xor', 'bitwise_and', 'bitwise_or', 'lshift', 'rshift',
                'floor', 'ceil', 'frac'
            ]
            banned_operators.extend(polynomial_banned)
            print(f"   [Polynomial Mode] Forcing pure polynomial search")

            polynomial_taylor_seeds = [
                'x - x**3/6', 'x - x**3/6 + x**5/120',
                'x - x**3/6 + x**5/120 - x**7/5040',
                '1 - x**2/2', '1 - x**2/2 + x**4/24',
                '1 - x**2/2 + x**4/24 - x**6/720',
                '1 + x + x**2/2', '1 + x + x**2/2 + x**3/6',
                'x + x**3/6', 'x + x**3/6 + x**5/120',
                '1 + x**2/2', '1 + x**2/2 + x**4/24',
                'x + a*x**3', 'x + a*x**3 + b*x**5',
                '1 + a*x**2', '1 + a*x**2 + b*x**4',
            ]
            seeds.extend(polynomial_taylor_seeds)
            print(f"   [Polynomial Mode] Seeding with {len(polynomial_taylor_seeds)} Taylor templates")

        # File Input
        if config.file_path:
            try:
                loaded_vars = load_csv_data(config.file_path)
                if variables is None:
                    variables = {}
                variables.update(loaded_vars)
                print(f"Loaded {len(loaded_vars)} variables from '{config.file_path}': {list(loaded_vars.keys())}")
            except Exception as e:
                print(f"Error loading file '{config.file_path}': {e}")
                return

        explicit_target_var = None
        match_explicit = EVOLVE_EXPLICIT_PATTERN.match(text)
        match = EVOLVE_PATTERN.match(text)

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
                is_implicit = True
        elif match:
            func_name = match.group(1)
            input_var_names = [v.strip() for v in match.group(2).split(",")]
            data_part = match.group(3)
        else:
            match_implicit = EVOLVE_IMPLICIT_PATTERN.match(text)
            if match_implicit:
                func_name = match_implicit.group(1)
                input_var_names = [v.strip() for v in match_implicit.group(2).split(",")]
                is_implicit = True
                if not variables:
                    print("Error: No data provided and no active variables in session.")
                    return
            else:
                direct_match = DIRECT_POINT_PATTERN.search(text)
                if direct_match:
                    func_name = direct_match.group(1)
                    find_match = FIND_FUNC_CLAUSE_PATTERN.search(text)
                    if find_match and find_match.group(1) == func_name:
                        input_var_names = [v.strip() for v in find_match.group(2).split(",")]
                    else:
                        input_var_names = ["x"]
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
             else:
                 print(f"Failed to load CSV: {csv_path}")
                 return

        if is_implicit:
            for name, val in variables.items():
                if isinstance(val, (list, tuple, np.ndarray)):
                    try:
                        arr = np.array(val)
                        if arr.dtype.kind in "iuf":
                            data_dict[name] = arr
                        else:
                            print(f"Warning: Variable '{name}' ignored. Expected numeric array, got dtype '{arr.dtype.kind}'.")
                    except Exception as e:
                        print(f"Warning: Failed to load variable '{name}' as numpy array: {e}")
                    continue

                if isinstance(val, str):
                    if "[" in val or "array" in val:
                        try:
                            import ast
                            cleaned = val.strip()
                            if "array(" in cleaned:
                                start = cleaned.find("[")
                                end = cleaned.rfind("]")
                                if start != -1 and end != -1:
                                    cleaned = cleaned[start:end+1]
                            val_parsed = ast.literal_eval(cleaned)
                            arr = np.array(val_parsed)
                            if arr.dtype.kind in "iuf":
                                data_dict[name] = arr
                            else:
                                print(f"Warning: Variable '{name}' ignored. Expected numeric array, got dtype '{arr.dtype.kind}'.")
                        except Exception as e:
                            if "[" in val:
                                print(f"Warning: Failed to parse variable '{name}': {e}")
                            pass

        else:
            for m in ARRAY_ASSIGN_PATTERN.finditer(data_part):
                var = m.group(1)
                
                if m.group(2):
                    try:
                        values = [float(v.strip()) for v in m.group(2).split(",")]
                        data_dict[var] = np.array(values)
                    except ValueError:
                         pass
                elif m.group(3):
                     ref_name = m.group(3)
                     if variables and ref_name in variables:
                         val = variables[ref_name]
                         if isinstance(val, (list, tuple, np.ndarray)):
                             data_dict[var] = np.array(val)
                         else:
                             print(f"Warning: Referenced variable '{ref_name}' is not an array.")
                     else:
                         print(f"Warning: Referenced variable '{ref_name}' not found.")

            if data_part:
                points_x = {v: [] for v in input_var_names}
                points_y = []
                skipped_complex = 0

                for m in FUNC_START_PATTERN.finditer(data_part):
                    p_func = m.group(1)
                    if p_func != func_name:
                        continue
                    
                    paren_start = m.end() - 1
                    paren_end = _find_matching_paren(data_part, paren_start)
                    if paren_end == -1:
                        continue
                    
                    p_args_str = data_part[paren_start + 1:paren_end]
                    rest = data_part[paren_end + 1:]
                    eq_match = re.match(r"\s*=\s*([^,]+)", rest)
                    if not eq_match:
                        continue
                    
                    p_val_str = eq_match.group(1).strip()

                    try:
                        p_val = eval_to_float(p_val_str)
                        p_args = []
                        for a in p_args_str.split(","):
                            arg_val = eval_to_float(a.strip())
                            p_args.append(arg_val)

                        current_arity = len(input_var_names)
                        data_arity = len(p_args)

                        if data_arity > current_arity:
                            print(f"Note: Data has {data_arity} variables (`{p_args_str}`), but target `{func_name}` has {current_arity}.")
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
                                points_x[next_name] = []
                            print(f"      -> Adapting target to `{func_name}({', '.join(input_var_names)})`")

                        elif data_arity < current_arity:
                            continue

                        for i, vname in enumerate(input_var_names):
                            if vname not in points_x:
                                points_x[vname] = []
                            points_x[vname].append(p_args[i])
                        points_y.append(p_val)
                    except ValueError:
                        continue

                if skipped_complex > 0:
                    print(f"Warning: {skipped_complex} data point(s) with complex/imaginary values were skipped.")
                    print("         Evolution requires real-valued inputs and outputs.")

            if points_y:
                for vname in input_var_names:
                    arr = np.array(points_x[vname])
                    if vname in data_dict:
                        data_dict[vname] = np.concatenate([data_dict[vname], arr])
                    else:
                        data_dict[vname] = arr

                candidates = ["y", "z", "w", "out", "result"]
                out_name = "y"
                for cand in candidates:
                    if cand not in input_var_names:
                        out_name = cand
                        break
                
                if out_name in input_var_names:
                    out_name = "f_result"

                out_arr = np.array(points_y)
                if out_name in data_dict:
                    data_dict[out_name] = np.concatenate([data_dict[out_name], out_arr])
                else:
                    data_dict[out_name] = out_arr

        if not data_dict:
            if is_implicit:
                print(f"Error: Could not find valid data arrays for variables: {', '.join(input_var_names)}.")
                print(f"Available variables: {list(variables.keys()) if variables else 'None'}")
            else:
                print("Error: No valid data points found in command.")
            return

        input_vars = [v for v in input_var_names if v in data_dict]
        output_candidates = [v for v in data_dict.keys() if v not in input_var_names]

        if not input_vars:
            input_vars = input_var_names[:1]
            output_candidates = [v for v in data_dict.keys() if v != input_vars[0]]

        if not output_candidates:
            print(f"Error: Need output variable. Provide data for a variable not in {func_name}({','.join(input_var_names)})")
            return

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

        missing = [v for v in input_vars if v not in data_dict]
        if missing:
            print(f"Error: Missing data for input variable(s): {missing}")
            return

        X = np.column_stack([data_dict[v] for v in input_vars])
        y = data_dict[output_var]

        # --- FILTER: Remove inf/nan/zoo ---
        try:
            def safe_convert(val):
                if isinstance(val, complex) or (hasattr(val, 'imag') and abs(val.imag) > 1e-10):
                    return val
                if hasattr(val, 'imag') and hasattr(val, 'real'):
                    if abs(val.imag) < 1e-10:
                        val = val.real
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
            
        original_len = len(y)
        
        def is_finite_safe(arr):
            if np.iscomplexobj(arr):
                return np.isfinite(arr.real) & np.isfinite(arr.imag)
            if arr.dtype.kind == 'f':
                return np.isfinite(arr)
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
                
        if len(y) == 0:
            print(f"Error: All {original_len} data points were filtered out (no valid real numbers).")
            return
        
        # --- ROBUST OUTLIER FILTERING ---
        try:
            y_real_check = np.real(y) if np.iscomplexobj(y) else y.astype(float)
            y_round_check = np.round(y_real_check)
            mse_int_check = np.mean((y_real_check - y_round_check)**2)
            
            y_abs = np.abs(y_real_check[np.isfinite(y_real_check)])
            skip_iqr_dynamic = False
            if len(y_abs) >= 5:
                y_99 = np.percentile(y_abs, 99)
                y_1 = np.percentile(y_abs, 1) + 1e-9
                dynamic_range = y_99 / y_1
                if dynamic_range > 1e5:
                    print(f"Note: High dynamic range ({dynamic_range:.1e}) detected - skipping outlier filter.")
                    skip_iqr_dynamic = True
            
            if mse_int_check < 0.01:
                print("Note: Discrete values detected. Outlier filtering disabled to preserve step/jump data.")
            elif skip_iqr_dynamic:
                pass
            elif len(y) >= 10 and not np.iscomplexobj(y):
                y_real = np.real(y) if np.iscomplexobj(y) else y.astype(float)
                q1 = np.percentile(y_real, 25)
                q3 = np.percentile(y_real, 75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                outlier_mask = (y_real >= lower_bound) & (y_real <= upper_bound)
                num_outliers = np.sum(~outlier_mask)
                if num_outliers > 0 and num_outliers < len(y) * 0.3:
                    print(f"Warning: Detected {num_outliers} IQR outliers, but KEEPING them to avoid data loss on step functions.")
        except Exception:
            pass

        # --- SMART SEEDING ---
        try:
            if 'seeds' not in locals(): seeds = []
            auto_seeds_result = generate_pattern_seeds(ctx, X, y, input_vars, verbose=verbose_mode)
            exact_match = None
            if isinstance(auto_seeds_result, tuple):
                auto_seeds, exact_match = auto_seeds_result
            else:
                auto_seeds = auto_seeds_result
                
            if exact_match:
                if type(exact_match).__name__ == 'RecurrenceResult':
                    if verbose_mode:
                        print(f"Exact match is a discrete recurrence. Bypassing continuous validation.")
                    beautified_match = str(exact_match)
                    print(f"\nResult: {beautified_match}")
                    print(f"MSE: 0.000000e+00 (Exact Discrete Match), Complexity: {len(beautified_match)}")
                    # Pass exact_match specifically to retain 'RecurrenceResult' subclass
                    define_function(ctx, func_name, input_vars, exact_match)
                    return
                try:
                    beautified_match = symbolify_constants(exact_match)
                    try:
                        preprocessed_match = kparser.preprocess_expression(exact_match)
                        local_dict = {v: sp.Symbol(v) for v in input_vars}
                        expr_obj = kparser.safe_sympy_parse(preprocessed_match, local_dict=local_dict)
                        def _v_primepi(x): 
                            try: return float(sp.primepi(int(x))) 
                            except: return 0.0
                        def _v_prime(x): 
                            try: return float(sp.prime(int(x))) 
                            except: return 0.0
                        
                        custom_modules = [{
                            "primepi": np.vectorize(_v_primepi), "prime_pi": np.vectorize(_v_primepi), 
                            "ith_prime": np.vectorize(_v_prime), "prime": np.vectorize(_v_prime),
                            "SafePrime": np.vectorize(_v_prime),
                            "trunc": np.trunc, "locked": lambda x: x, "factorial": lambda x: scipy.special.gamma(x + 1),
                            "lshift": np.left_shift, "rshift": np.right_shift,
                            "bitwise_and": np.bitwise_and, "bitwise_or": np.bitwise_or, "bitwise_xor": np.bitwise_xor,
                        }, "numpy", "scipy"]
                        f_lamb = sp.lambdify(input_vars, expr_obj, modules=custom_modules)
                        
                        if len(input_vars) == 1:
                              with warnings.catch_warnings():
                                  warnings.simplefilter("ignore", RuntimeWarning)
                                  # Cast to complex128 when y is complex, so sqrt(neg) works
                                  x_input = X.flatten().astype(np.complex128) if np.iscomplexobj(y) else X.flatten()
                                  y_pred = f_lamb(x_input)
                        else:
                              with warnings.catch_warnings():
                                  warnings.simplefilter("ignore", RuntimeWarning)
                                  if np.iscomplexobj(y):
                                      y_pred = f_lamb(*(col.astype(np.complex128) for col in X.T))
                                  else:
                                      y_pred = f_lamb(*X.T)
                            
                        if np.isscalar(y_pred) or (hasattr(y_pred, 'shape') and y_pred.shape == ()):
                            y_pred = np.full_like(y, y_pred)
                            
                        mse = float(np.real(np.mean(np.abs(y - y_pred)**2)))
                        
                        if (not np.isnan(mse)) and mse < 1e-9:
                            # Check ban list before accepting exact match
                            _exact_is_banned = False
                            if banned_operators:
                                match_lower = exact_match.lower()
                                _exact_is_banned = any(matches_ban(match_lower, b) for b in banned_operators)
                            
                            if _exact_is_banned:
                                print(f"\n⛔ Exact match '{beautified_match}' contains a banned operator.")
                                print(f"   Adding as seed and proceeding to evolution...")
                                if auto_seeds is None: auto_seeds = []
                                auto_seeds.append(exact_match)
                            else:
                                print(f"\nResult: {beautified_match}")
                                print(f"MSE: {mse:.6e} (Exact Match), Complexity: {len(beautified_match)}")
                                define_function(ctx, func_name, input_vars, beautified_match)
                                return
                        else:
                            print(f"\nResult: {beautified_match}")
                            print(f"MSE: {mse:.6f} (Heuristic Match - Continuing Evolution)")
                            if auto_seeds is None: auto_seeds = []
                            auto_seeds.append(exact_match)
                    except Exception as e:
                        if verbose_mode: print(f"Exact match validation failed: {e}")
                        if auto_seeds is None: auto_seeds = []
                        auto_seeds.append(exact_match)
                except Exception as e:
                    if verbose_mode: print(f"Exact match check failed: {e}")
                    pass

            if auto_seeds:
                seeds.extend(auto_seeds)
                if len(auto_seeds) <= 5:
                    print(f"Smart seeding: detected patterns, seeding with {auto_seeds}")
                else:
                    print(f"Smart seeding: detected {len(auto_seeds)} pattern-based seeds")
        except Exception as e:
             if verbose_mode: print(f"Smart seeding error: {e}")

        print(f"Evolving {func_name}({', '.join(input_vars)}) from {len(y)} data points...")

        # --- HYBRID MODE ---
        if use_hybrid:
            try:
                success = False
                func_str = None
                find_data_points = []
                for i in range(len(y)):
                    x_vals = tuple(X[i]) if X.ndim > 1 else (X[i],)
                    find_data_points.append((x_vals, y[i]))

                find_data_points_real = []
                count_complex_skipped = 0
                for i in range(len(y)):
                    x_row = X[i] if X.ndim > 1 else np.array([X[i]])
                    if np.any(np.abs(np.imag(x_row)) > 1e-9):
                        count_complex_skipped += 1
                        continue
                    if np.abs(np.imag(y[i])) > 1e-9:
                        count_complex_skipped += 1
                        continue
                    x_vals = tuple(x_row.real)
                    find_data_points_real.append((x_vals, float(y[i].real)))

                if count_complex_skipped > 0 and len(find_data_points_real) < 5:
                     print(f"Hybrid mode: skipping find() (only {len(find_data_points_real)} real points found, need 5+)")
                     success = False
                else:
                    if count_complex_skipped > 0:
                        print(f"Hybrid mode: filtering {count_complex_skipped} complex points just for find(). Evolutionary engine will keep ALL points.")
                        if find_data_points_real:
                            # Do NOT overwrite X and y here, it amputates data for the genetic engine!
                            pass
                    else:
                        print("Hybrid mode: running find() for initial approximation...")
                    
                    if super_verbose:
                        y_vals_real = [p[1] for p in find_data_points_real]
                        x_vals_real = [p[0][0] for p in find_data_points_real]
                        print(f"\n[SV] INPUT DATA ANALYSIS:")
                        print(f"     Points: {len(find_data_points_real)}")
                        print(f"     X range: [{min(x_vals_real):.4g}, {max(x_vals_real):.4g}]")
                        print(f"     Y range: [{min(y_vals_real):.4g}, {max(y_vals_real):.4g}]")
                    
                    with warnings.catch_warnings():
                         warnings.simplefilter("ignore")
                         success, func_str, factored, error = find_function_from_data(
                             ctx, find_data_points_real, input_vars, verbose=super_verbose
                         )

                use_seed = False
                if success and func_str:
                    try:
                        local_ns = {var: sp.Symbol(var) for var in input_vars}
                        local_ns.update(ALLOWED_SYMPY_NAMES)
                        local_ns.update({"pi": sp.pi, "e": sp.E, "E": sp.E})
                        preprocessed_func = kparser.preprocess_expression(func_str)
                        discovered_expr = kparser.safe_sympy_parse(preprocessed_func, local_dict=local_ns)
                        
                        y_pred = []
                        y_true = []
                        for (inputs, output) in find_data_points:
                            vals = inputs if hasattr(inputs, '__iter__') else (inputs,)
                            is_complex = False
                            for v in vals:
                                try:
                                    if abs(complex(v).imag) > 1e-9: is_complex = True
                                except: pass
                            if is_complex: continue
                            try:
                                if abs(complex(output).imag) > 1e-9: continue
                            except: pass

                            subs_dict = {
                                input_vars[i]: float(vals[i].real) if isinstance(vals[i], complex) or hasattr(vals[i], 'imag') else float(vals[i]) 
                                for i in range(len(input_vars))
                            }
                            try:
                                pred_val = discovered_expr.subs(subs_dict).evalf()
                                pred = float(complex(pred_val).real)
                                y_pred.append(pred)
                                y_true.append(float(complex(output).real))
                            except Exception: continue
                        
                        if len(y_true) > 0:
                            y_mean = np.mean(y_true)
                            ss_tot = np.sum((np.array(y_true) - y_mean)**2)
                            ss_res = np.sum((np.array(y_true) - np.array(y_pred))**2)
                            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                            mse_val = ss_res / len(y_true)
                        else:
                            r_squared = 0.0; mse_val = 1e9
                        
                        if r_squared > 0.7 or (r_squared > 0.4 and mse_val < 0.05):
                            use_seed = True
                            seeds.append(func_str)
                            display = func_str[:50] + "..." if len(func_str) > 50 else func_str
                            print(f"Hybrid seeding: using find() result '{display}' (R²={r_squared:.4f}, MSE={mse_val:.6f})")
                            
                            if mse_val < 1e-7:
                                _effective_banned = set(banned_operators)
                                if ctx and hasattr(ctx, 'banned_operators'):
                                    _effective_banned.update(ctx.banned_operators)
                                
                                _find_is_banned = False
                                if _effective_banned:
                                    func_lower = func_str.lower()
                                    _find_is_banned = any(matches_ban(func_lower, b) for b in _effective_banned)
                                
                                if _find_is_banned:
                                    print(f"\n⛔ find() found '{func_str}' but it contains a banned operator.")
                                    print(f"   Proceeding to evolution with constrained operator set...")
                                    use_seed = False
                                else:
                                    print(f"\n🎯 find() discovered a good solution (MSE={mse_val:.6f})")
                                    print(f"   Skipping evolution and returning directly.")
                                    beautified = symbolify_constants(func_str)
                                    print(f"\nResult: {format_solution(beautified)}")
                                    print(f"MSE: {mse_val:.6g}, Complexity: ~{len(func_str)//5}")
                                    try:
                                        define_function(ctx, func_name, input_vars, beautified)
                                    except Exception as e:
                                        print(f"Error defining function: {e}")
                                    try:
                                        bank = get_gene_bank()
                                        local_dict = {v: sp.Symbol(v) for v in input_vars}
                                        expr = sp.sympify(beautified, locals=local_dict)
                                        is_constant = expr.is_number or expr.is_Number
                                        is_single_var = expr in local_dict.values()
                                        is_linear = len(expr.free_symbols) == 1 and expr.is_polynomial() and sp.degree(expr) <= 1
                                        if not is_constant and not is_single_var and not is_linear:
                                            class HeuristicResult:
                                                def __init__(self, sympy_expr, complexity):
                                                    self._expr = sympy_expr
                                                    self._complexity = complexity
                                                def to_sympy(self): return self._expr
                                                def complexity(self): return self._complexity
                                                def to_pretty_string(self): return str(self._expr)
                                            mock_tree = HeuristicResult(expr, len(str(expr)) // 3)
                                            r2 = 1.0 - mse_val if mse_val < 1 else 0.99
                                            saved = bank.add(mock_tree, mse_val, r2)
                                            if saved: print(f"[GeneBank] Saved: {beautified}")
                                    except Exception: pass
                                    return
                        else:
                            display = func_str[:50] + "..." if len(func_str) > 50 else func_str
                            print(f"Hybrid seeding: find() result '{display}' has low R²={r_squared:.2f} (MSE={mse_val:.6f}), skipping seed")
                            print("  -> Using pure evolve instead (no bad seed)")
                    except Exception as eval_error:
                        print(f"Hybrid seeding: could not evaluate find() result ({eval_error}), skipping")
            except Exception as e:
                print(f"Hybrid mode: find() failed ({e}), continuing with other seeds")

        # --- SEED SANITIZATION ---
        if seeds and len(seeds) > 0:
            clean_seeds = []
            symbols_dict = {var: sp.Symbol(var) for var in input_vars}
            for seed in seeds:
                try:
                    preprocessed_seed = kparser.preprocess_expression(seed)
                    expr = kparser.safe_sympy_parse(preprocessed_seed, local_dict=symbols_dict)
                    
                    def _v_primepi(x): 
                        try: return float(sp.primepi(int(x))) 
                        except: return 0.0
                    def _v_prime(x): 
                        try: return float(sp.prime(int(x))) 
                        except: return 0.0
                    def _v_mobius(x):
                        try: return float(sp.mobius(int(x)))
                        except: return 0.0
                    def _v_omega(x):
                        try: return float(len(sp.primefactors(int(x))))
                        except: return 0.0
                    def _v_primeomega(x):
                        try: return float(sp.primeomega(int(x)))
                        except: return 0.0
                    custom_modules = [{
                        "primepi": np.vectorize(_v_primepi), "prime_pi": np.vectorize(_v_primepi), 
                        "ith_prime": np.vectorize(_v_prime), "prime": np.vectorize(_v_prime), 
                        "moebius": np.vectorize(_v_mobius), "omega": np.vectorize(_v_omega), "big_omega": np.vectorize(_v_primeomega),
                        "trunc": np.trunc, "locked": lambda x: x, "factorial": lambda x: scipy.special.gamma(x + 1),
                        "lshift": np.left_shift, "rshift": np.right_shift,
                        "bitwise_and": np.bitwise_and, "bitwise_or": np.bitwise_or, "bitwise_xor": np.bitwise_xor,
                    }, "numpy", "scipy"]
                    f_lamb = sp.lambdify(input_vars, expr, modules=custom_modules)
                    
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            if len(input_vars) == 1:
                                preds = f_lamb(X.flatten())
                            else:
                                preds = f_lamb(*X.T)
                        preds_arr = np.array(preds)
                    except Exception as eval_e:
                         if verbose_mode:
                             print(f"   [Sanitizer] Warning: Seed '{seed}' evaluation failed ({eval_e}), but keeping it.")
                    
                    clean_seeds.append(preprocessed_seed)
                except Exception as e:
                    if verbose_mode:
                         print(f"   [Sanitizer] Discarding broken seed '{seed}': {e}")
            
            if len(clean_seeds) < len(seeds):
                 print(f"   [Sanitizer] Removed {len(seeds) - len(clean_seeds)} toxic seeds.")
            # ALWAYS use preprocessed seeds (fixes ^ -> ** conversion)
            seeds = clean_seeds

        base_population = config.pop if config.pop is not None else 100
        if seeds:
            min_pop_for_seeds = len(seeds) * 3
            if min_pop_for_seeds > base_population:
                base_population = min_pop_for_seeds
                print(f"Dynamic scaling: increased population to {base_population} to accommodate {len(seeds)} seeds")

        base_generations = config.gen if config.gen is not None else 30
        base_timeout = 15

        final_population = base_population * boosting_rounds if config.pop is None else base_population
        final_generations = base_generations * boosting_rounds if config.gen is None else base_generations
        final_timeout = base_timeout * boosting_rounds

        if boosting_rounds > 1:
            print(f"Boost mode: {boosting_rounds}x resources (pop={final_population}, gen={final_generations}, timeout={final_timeout}s)")

        is_smooth = detect_smoothness(X.tolist(), y.tolist(), verbose=verbose_mode)
        allow_bitwise_ops = not is_smooth
        if verbose_mode:
            if is_smooth:
                print("[Safety] Data appears SMOOTH/CONTINUOUS. Disabling bitwise operators.")
            else:
                print("[Safety] Data appears DISCRETE/STEPPED. Allowing bitwise operators.")

        genetic_config = GeneticConfig(
            population_size=final_population,
            n_islands=2,
            generations=final_generations,
            timeout=final_timeout,
            verbose=verbose_mode,
            seeds=seeds,
            boosting_rounds=1,
            high_precision=high_precision_mode,
            operators=["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "pow", "sqrt", "lambertw", "lshift", "rshift", "bitwise_and", "bitwise_or", "bitwise_xor", "factorial", "abs", "neg", "tanh", "max", "min", "square", "cube"],
            allow_bitwise=allow_bitwise_ops,
        )
        
        effective_banned = set(banned_operators)
        if ctx and hasattr(ctx, 'banned_operators'):
             effective_banned.update(ctx.banned_operators)
        
        if effective_banned:
            original_ops = genetic_config.operators.copy()
            genetic_config.operators = [op for op in genetic_config.operators if op.lower() not in effective_banned]
            removed = set(original_ops) - set(genetic_config.operators)
            if removed:
                print(f"   [Constraint] Banned from arsenal: {removed}")
            
            original_seed_count = len(genetic_config.seeds)
            filtered_seeds = []
            for seed in genetic_config.seeds:
                seed_lower = seed.lower()
                contains_banned = any(ban in seed_lower for ban in effective_banned)
                if not contains_banned:
                    filtered_seeds.append(seed)
            genetic_config.seeds = filtered_seeds
            if len(filtered_seeds) < original_seed_count:
                print(f"   [Constraint] Filtered {original_seed_count - len(filtered_seeds)} seeds containing banned operators")

        # === ODE DISCOVERY MODE ===
        if use_discover_ode:
            ode_config = ODEConfig(
                population_size=200, generations=50, verbose=verbose_mode, parsimony_coefficient=0.01
            )
            ode_engine = ODEDiscoveryEngine(ode_config)
            ode_str, residual = ode_engine.fit(X[:, 0], y)
            
            print(f"\n=== ODE Discovery Result ===")
            print(f"Discovered: {ode_str}")
            print(f"Residual: {residual:.6e}")
            print(f"\n📖 Interpretation:")
            if "y''" in ode_str and "y'" not in ode_str.replace("y''", ""):
                if "+ y" in ode_str or "y +" in ode_str:
                    print("   This is Simple Harmonic Motion: acceleration = -position")
                    print("   -> The function oscillates like a wave (sin, cos)")
                    print("   -> Physical examples: pendulum, spring, vibration")
                elif "- y" in ode_str or "y -" in ode_str:
                    print("   This is exponential: acceleration = position")
                    print("   -> The function grows/decays exponentially (exp, cosh, sinh)")
            elif "y'" in ode_str and "y''" not in ode_str:
                if "+ y" in ode_str or "y +" in ode_str:
                    print("   This is exponential decay: rate = -value")
                    print("   -> The function decays over time (e^(-x))")
                elif "- y" in ode_str or "y -" in ode_str:
                    print("   This is exponential growth: rate = value")
                    print("   -> The function grows exponentially (e^x)")
            else:
                print("   This describes how the function changes with its derivatives.")
            return
        
        print("Loading genetic evolution engine...", flush=True)
        regressor = GeneticSymbolicRegressor(genetic_config)
        
        if use_transform:
            if verbose_mode:
                print("Multi-space mode: evolving in direct, log, and inverse spaces...")
            best_expr, best_mse_val, best_space = regressor.fit_with_transformations(X, y, input_vars)
            if verbose_mode:
                print(f"Best result from {best_space} space")
            
            try:
                symbols = {v: sp.Symbol(v) for v in input_vars}
                sympy_expr = kparser.safe_sympy_parse(best_expr, local_dict=symbols)
                tree = ExpressionTree.from_sympy(sympy_expr, input_vars)
                complexity = tree.complexity()
                pareto = ParetoFront()
                solution = ParetoSolution(expression=best_expr, mse=best_mse_val, complexity=complexity, sympy_expr=sympy_expr, tree=tree)
                pareto.add(solution)
            except Exception as e:
                print(f"Warning: Could not parse result: {e}")
                print(f"Using expression string directly: {best_expr}")
                fallback_tree = ExpressionNode(NodeType.CONSTANT, 0.0, [])
                pareto = ParetoFront()
                pareto.add(ParetoSolution(expression=best_expr, mse=best_mse_val, complexity=10, sympy_expr=None, tree=fallback_tree))
        else:
            pareto = regressor.fit(X, y, input_vars)

        if pareto is None:
            print("Error: Genetic engine failed to find any candidate solutions.")
            return

        knee = pareto.get_knee_point()
        best_mse = pareto.get_best()

        best = knee
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

        beautified_expr = symbolify_constants(best.expression)
        print(f"\nResult: {format_solution(beautified_expr)}")
        print(f"MSE: {best.mse:.6g}, Complexity: {best.complexity}")

        # === AUTO ODE DISCOVERY ===
        try:
             # Only run if we have enough data and it's roughly evenly spaced
            if len(y) >= 10:
                is_even, _ = check_even_spacing(X[:, 0])
                if X.shape[1] == 1 and (is_even or len(y) >= 15):
                    ode_config = ODEConfig(population_size=100, generations=20, verbose=False, parsimony_coefficient=0.01)
                    ode_engine = ODEDiscoveryEngine(ode_config)
                    
                    ode_str, residual = ode_engine.fit(X[:, 0], y)
                    auto_ode_str, auto_residual = ode_engine.discover_autonomous_ode(X[:, 0], y)
                    if auto_residual < residual:
                        ode_str = auto_ode_str
                        residual = auto_residual
                    
                    if residual < 0.1:
                        print(f"\n📖 Underlying Physics:")
                        print(f"   ODE: {ode_str}")
                        if ode_str.startswith("y' = "):
                            rhs = ode_str[5:]
                            if "y**2" in rhs or "y*y" in rhs or "(1 - y)" in rhs:
                                print("   -> Logistic Growth (population with carrying capacity)")
                            elif "y" in rhs and ("*" not in rhs or rhs.count("y") == 1):
                                print("   -> Exponential dynamics")
                            else:
                                print("   -> Autonomous ODE (rate depends on state)")
                        else:
                            has_ypp = "y''" in ode_str
                            has_yp = "y'" in ode_str and "y''" not in ode_str
                            if has_ypp:
                                if ("y + y''" in ode_str or "y'' + y" in ode_str):
                                    print("   -> Simple Harmonic Motion (oscillating wave: sin, cos)")
                                elif ("y - y''" in ode_str or "y'' - y" in ode_str or  "-y + y''" in ode_str or "y'' + -y" in ode_str):
                                    print("   -> Exponential/Hyperbolic (exp, cosh, sinh)")
                                else:
                                    print("   -> Second-order dynamics")
                            elif has_yp:
                                if ("y' - y" in ode_str or "y - y'" in ode_str or "-y + y'" in ode_str):
                                    print("   -> Exponential growth (rate = value)")
                                elif ("y' + y" in ode_str or "y + y'" in ode_str):
                                    print("   -> Exponential decay (rate = -value)")
        except Exception:
            pass

        try:
            define_function(ctx, func_name, input_vars, beautified_expr)
        except Exception as e:
            print(f"Warning: Failed to define function '{func_name}' in session: {e}")

    except ImportError as e:
        print(f"Error: Required module not available: {e}")
    except Exception as e:
        print(f"Error: {e}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            traceback.print_exc()
