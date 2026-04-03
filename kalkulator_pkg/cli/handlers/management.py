"""
Handler functions for management commands (save/load/show, cache, genes, export).
Part of the CLI refactoring to decompose repl_commands.py.
"""
import json
import logging
from typing import Any, Dict

from ...function_manager import (
    save_functions, load_functions, clear_functions, clear_saved_functions,
    list_functions, get_builtin_names, export_function_to_file
)
from ...cache_manager import (
    get_persistent_cache, export_cache_to_file, replace_cache_from_file
)

logger = logging.getLogger(__name__)


def handle_save_functions_command(ctx: Any) -> bool:
    """Handle 'save' command."""
    success, msg = save_functions(ctx)
    print(msg)
    return True


def handle_load_functions_command(ctx: Any) -> bool:
    """Handle 'loadfunction' command."""
    success, msg = load_functions(ctx)
    print(msg)
    return True


def handle_clear_functions_command(ctx: Any) -> bool:
    """Handle 'clearfunction' command."""
    clear_functions(ctx)
    print("Functions cleared from current session.")
    return True


def handle_clear_saved_functions_command() -> bool:
    """Handle 'clearsavefunction' command."""
    success, msg = clear_saved_functions()
    print(msg)
    return True


def handle_show_functions(ctx: Any):
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
            print(f"\\n  {cat_name}:")
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
        print(f"\\n  Other:")
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


def handle_export_command(text: str):
    import re
    export_match = re.match(r"export\\s+(\\w+)\\s+to\\s+(.+)", text, re.IGNORECASE)
    if export_match:
        func_name = export_match.group(1)
        filename = export_match.group(2).strip()
        success, message = export_function_to_file(func_name, filename)
        print(message)
    else:
        print("Usage: export <function_name> to <filename>")


def handle_save_cache(text):
    parts = text.split()
    filename = "expression_cache.json"
    if len(parts) > 1:
        filename = parts[1]
    # Use valid exported name

    if export_cache_to_file(filename):
        print(f"Cache saved to {filename}")
    else:
        print(f"Failed to save cache to {filename}")


def handle_load_cache(text):
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


def handle_show_cache(text: str, ctx: Any):
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
        print("\\nUse 'showcache' to see recent items, or 'showcache all' for everything.")
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
    
    print(f"\\n  EVAL CACHE ({len(eval_items)} items):")
    print(f"  {'EXPRESSION':<40} | {'RESULT':<25}")
    print("  " + "-" * 68)
    
    limit = 50 if not show_all else len(eval_items)
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
    print(f"\\n  SUBEXPR CACHE ({len(sub_items)} items):")
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

    print("\\n" + "=" * 70)
    print("Commands: 'clearcache', 'showcache all', 'savecache'")
    print("=" * 70)


def _prettify_gene_expression(expr_str: str) -> str:
    """Prettify a gene expression for display."""
    import re
    
    # 1. Variable Mapping
    expr_str = expr_str.replace("v0", "x")
    expr_str = expr_str.replace("v1", "y")
    expr_str = expr_str.replace("v2", "z")
    
    # 2. Smart Fraction Snapping
    def replace_fraction(match):
        numer = match.group(1)
        denom = match.group(2)
        if len(numer) > 3 or len(denom) > 3:
            try:
                val = float(numer) / float(denom)
                return f"{val:.2f}"
            except:
                return match.group(0)
        return match.group(0)
        
    expr_str = re.sub(r'(\\d+)/(\\d+)', replace_fraction, expr_str)
    
    # 3. Clean up power operator
    expr_str = expr_str.replace("**", "^")
    
    return expr_str


def handle_genes_command(text: str, ctx: Any):
    """Handle 'genes' command for Gene Bank management."""
    try:
        from ...symbolic_regression.gene_bank import get_gene_bank
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
        
        print(f"\\nGENE BANK ({len(genes)} cached functions)")
        print("─" * 76)
        print(f" {'ID':<4} {'Expression':<32} {'Vars':<6} {'Compl.':<8} {'MSE':<10} {'Status':<10}")
        print("─" * 76)
        
        for g in genes:
            expr_str = g['expression']
            mse = g['fitness']
            pretty_expr = _prettify_gene_expression(expr_str)
            if len(pretty_expr) > 30:
                pretty_expr = pretty_expr[:27] + "..."
                
            if mse < 1e-30:
                status = "⭐ Exact"
            elif mse < 1e-6:
                status = "🎯 Precise"
            else:
                status = "〰️ Approx"
                
            n_vars = g.get('n_vars', 1)
            meas_vars = str(n_vars)
            complexity = g.get('complexity', 0.0)
            
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
