import json
import logging
import re
from typing import Any

import sympy as sp

from kalkulator_pkg.parser import format_inequality_solution
from kalkulator_pkg.parser import format_number

logger = logging.getLogger(__name__)


def format_number_no_trailing_zeros(num_str: str) -> str:
    """Format a number string by removing trailing zeros and decimal point if not needed."""
    try:
        # Try to parse as float
        num = float(num_str)
        # If it's an integer, return as integer string
        if num.is_integer():
            return str(int(num))
        # Otherwise, remove trailing zeros
        return str(num).rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        # If parsing fails, return original string
        return num_str


def format_inverse_solutions(
    result: dict, func_name: str, param_names: list, target_value: str
) -> None:
    """Format and print inverse solutions with proper domain classification."""
    if not result.get("ok"):
        print(f"Error: {result.get('error')}")
        return

    print(
        f"\nInverse solutions for {func_name}({', '.join(param_names)}) = {target_value}:"
    )
    domains = result.get("domains", {})

    # Summary header
    integers = domains.get("integers")
    rationals = domains.get("rationals")
    reals = domains.get("reals")
    complex_sols = domains.get("complex")
    parametric = domains.get("parametric")

    int_count = integers.get("count", 0) if isinstance(integers, dict) else 0
    rat_count = len(rationals) if rationals else 0
    real_count = len(reals) if reals else 0

    print("\n  Summary:")
    print(f"    Integer solutions: {int_count if int_count else 'None'}")
    print(f"    Rational solutions: {rat_count if rat_count else 'None'}")
    if parametric:
        print("    Real solutions: continuous (see parametric form)")
    elif real_count:
        print(f"    Real solutions: {real_count}")
    else:
        print("    Real solutions: None")

    # 1. Integer solutions (2 per line for compactness)
    if isinstance(integers, dict) and int_count > 0:
        sols = integers.get("solutions", [])
        print(f"\n  Integers (exact) [{int_count} solutions]:")
        if len(param_names) == 2:
            # Format pairs as (x, y) - 2 per line
            pairs = [(s.get("x", "?"), s.get("y", "?")) for s in sols]
            lines = [f"({x:>3}, {y:>3})" for x, y in pairs]
            for i in range(0, len(lines), 2):
                row = "    " + lines[i]
                if i + 1 < len(lines):
                    row += "   " + lines[i + 1]
                print(row)
        else:
            for sol in sols:
                print(f"    x = {sol.get('x', '?')}")
    else:
        print("\n  Integers: None")

    # 2. Rational solutions
    if rationals:
        print("\n  Rationals (exact):")
        for sol in rationals:
            exact = sol.get("exact", "?")
            numeric = sol.get("numeric")
            if numeric:
                print(f"    x = {exact}  ≈ {numeric}")
            else:
                print(f"    x = {exact}")
    else:
        print("\n  Rationals: None")

    # 3. Real solutions (irrational) - skip if parametric covers it for 2-var
    if reals and len(param_names) == 1:
        print("\n  Reals (exact):")
        for sol in reals:
            exact = sol.get("exact", "?")
            numeric = sol.get("numeric")
            if numeric:
                print(f"    x = {exact}  ≈ {numeric}")
            else:
                print(f"    x = {exact}")

    # 4. Parametric form (for 2-variable) - primary real representation
    if parametric:
        print("\n  Reals (parametric):")
        print(f"    {parametric.get('form', '')},  {parametric.get('parameter', '')}")

    # 5. Algebraic form (only for 2-variable, combined with general)
    general = domains.get("general")
    if general and isinstance(general, dict) and len(param_names) == 2:
        forms = general.get("forms", [])
        note = general.get("note", "")
        if forms:
            print("\n  Reals (algebraic):")
            # Combine ± forms
            if len(forms) == 2 and "sqrt" in forms[0] and "sqrt" in forms[1]:
                # Extract the sqrt expression
                import re

                match = re.search(r"sqrt\(([^)]+)\)", forms[0])
                if match:
                    inner = match.group(1)
                    sym = param_names[0]
                    print(f"    {sym} = ±sqrt({inner})")
                else:
                    for form in forms:
                        print(f"    {form}")
            else:
                for form in forms:
                    print(f"    {form}")
            if note:
                print(f"    ({note})")

    # 6. Complex solutions (only show if different from reals)
    if complex_sols and len(param_names) == 1:
        print("\n  Complex:")
        for sol in complex_sols:
            exact = sol.get("exact", "?")
            numeric = sol.get("numeric")
            if numeric:
                print(f"    x = {exact}  ≈ {numeric}")
            else:
                print(f"    x = {exact}")


def find_pi_fraction_form(
    num_val: float, max_denominator: int = 10000, tolerance: float = 1e-8
) -> str | None:
    """Find if a number is close to a rational multiple of π and return the fraction form.
    
    Only returns π fractions with "clean" denominators (1, 2, 3, 4, 6, 8, 12) to avoid
    absurd results like -20576π/5199 for random decimals.
    """
    # Whitelist of recognizable denominators for π fractions
    # These are the common denominators found in trigonometry: π/2, π/3, π/4, π/6, etc.
    CLEAN_PI_DENOMINATORS = {1, 2, 3, 4, 6, 8, 12}
    
    try:
        # Skip conversion for exactly zero or very small numbers close to zero
        if abs(num_val) < 1e-10:
            return None

        pi_val = float(sp.pi.evalf())
        # Divide by π to get the coefficient
        coeff = num_val / pi_val

        # Calculate relative tolerance based on magnitude
        abs_coeff = abs(coeff)
        rel_tol = max(abs_coeff * tolerance, 1e-10)  # At least 1e-10

        # Try to find rational approximation
        try:
            # Use SymPy's Rational with limit_denominator
            rat = sp.Rational(sp.N(coeff)).limit_denominator(max_denominator)

            # Check if the approximation is close enough
            pi_mult = float(rat) * pi_val
            diff = abs(num_val - pi_mult)

            # Use relative error check
            if num_val != 0:
                rel_error = diff / abs(num_val)
            else:
                rel_error = diff

            if rel_error < rel_tol or diff < 1e-10:
                # Format as (numerator/denominator)*pi
                num_val_int = int(rat.numerator)
                den_val_int = int(rat.denominator)

                # Only accept clean denominators to avoid absurd fractions
                if den_val_int not in CLEAN_PI_DENOMINATORS:
                    return None  # Reject: denominator not recognizable

                if den_val_int == 1:
                    if num_val_int == 1:
                        return "pi"
                    elif num_val_int == -1:
                        return "-pi"
                    else:
                        return f"{num_val_int}*pi"
                else:
                    return f"({num_val_int}/{den_val_int})*pi"
        except (ValueError, TypeError, AttributeError, OverflowError):
            pass

        return None
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def convert_to_pi_fraction(num_str: str, tolerance: float = 1e-6) -> str:
    """Convert a decimal number to fractional π form if it's close to a rational multiple of π."""
    try:
        num = float(num_str)
        # Divide by π to get the coefficient
        pi_val = float(sp.pi.evalf())
        coeff = num / pi_val

        # Try to find a rational approximation
        try:
            # Find rational approximation with max denominator
            max_denom = 1000
            rat = sp.Rational(sp.N(coeff)).limit_denominator(max_denom)

            # Check if the approximation is close enough
            pi_mult = float(rat) * pi_val
            if abs(num - pi_mult) < tolerance:
                # Format as (numerator)π/(denominator)
                num_val = int(rat.numerator)
                den_val = int(rat.denominator)

                if den_val == 1:
                    if num_val == 1:
                        return "π"
                    elif num_val == -1:
                        return "-π"
                    else:
                        return f"{num_val}π"
                else:
                    if num_val < 0:
                        if abs(num_val) == 1:
                            return f"-π/{den_val}"
                        else:
                            return f"{num_val}π/{den_val}"
                    else:
                        if num_val == 1:
                            return f"π/{den_val}"
                        else:
                            return f"{num_val}π/{den_val}"
        except (ValueError, TypeError, AttributeError):
            pass

        # If conversion fails, return formatted number
        return format_number_no_trailing_zeros(num_str)
    except (ValueError, TypeError):
        # If parsing fails, return original string
        return num_str


def format_special_values(val_str: str) -> str:
    """Format special values like zoo and oo*I to user-friendly strings."""
    if not val_str:
        return val_str

    # Handle zoo (complex infinity / division by zero)
    if val_str == "zoo" or val_str.strip() == "zoo":
        return "undefined"

    # Handle complex infinity
    if val_str == "oo*I" or val_str == "I*oo":
        return "i*∞"

    # Handle imaginary unit I standalone
    if val_str == "I" or val_str.strip() == "I":
        return "√(-1)"

    # Handle expressions with I
    if val_str == "1.0*I" or val_str == "I*1.0" or val_str == "1*I" or val_str == "I*1":
        return "√(-1)"

    return val_str


def pretty_print_expression(expr: str) -> str:
    """Normalize expression with spaces around operators for readability.

    Transforms 'e^(i*pi)' -> 'e^(i * pi)'
    Transforms '2+3*4' -> '2 + 3 * 4'

    Engineering Standards:
    - Simple Control Flow: Linear regex replacements
    - No Magic: Explicit transformations
    """
    assert isinstance(expr, str), "Expression must be a string"

    result = expr

    # Add spaces around * (but not **)
    result = re.sub(r"(?<!\*)\*(?!\*)", " * ", result)

    # Add spaces around + (but not in exponents like e+10)
    result = re.sub(r"(?<![eE])\+", " + ", result)

    # Add spaces around - (but not unary minus or scientific notation)
    result = re.sub(r"(?<=[\w)])\-(?=[\w(])", " - ", result)

    # Add spaces around /
    result = re.sub(r"/", " / ", result)

    # Clean up multiple spaces
    result = re.sub(r"\s+", " ", result)

    # Trim and return
    return result.strip()


def print_result_pretty(
    res: dict[str, Any], output_format: str = "human", expression: str | None = None
) -> None:
    """Print result in specified format.

    Args:
        res: Result dictionary with 'ok', 'type', 'result' keys
        output_format: 'human' or 'json'
        expression: Optional original expression to display alongside result
    """
    if output_format == "json":
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if not res.get("ok"):
        print("Error:", res.get("error"))
        return
    typ = res.get("type", "value")
    if typ == "equation":
        exact_sols = res.get("exact", [])
        approx_sols = res.get("approx", [])

        # Try to convert exact solutions to π fractions if they look like decimals
        exact_formatted = []
        all_are_exact = True  # Track if all results are clean integers or π fractions
        any_pi_fractions = False  # Track if we found any π fractions
        
        for sol in exact_sols:
            formatted = format_solution(sol)
            is_exact = False
            
            # Check if this is effectively an integer
            try:
                num = float(sol)
                if num.is_integer():
                    formatted = str(int(num))  # Format as clean integer (29.0 → 29)
                    is_exact = True
                else:
                    # Try to convert to π fraction (e.g., 0.785398 → π/4)
                    pi_form = find_pi_fraction_form(num)
                    if pi_form:
                        # Format nicely: (1/4)*pi → π/4, (-3/4)*pi → -3π/4
                        if pi_form == "pi":
                            formatted = "π"
                        elif pi_form == "-pi":
                            formatted = "-π"
                        elif "*pi" in pi_form:
                            # Handle (n/d)*pi format
                            coeff = pi_form.replace("*pi", "")
                            if coeff.startswith("(") and coeff.endswith(")"):
                                # (n/d)*pi → nπ/d
                                inner = coeff[1:-1]  # "n/d"
                                if "/" in inner:
                                    num_part, den_part = inner.split("/")
                                    if num_part == "1":
                                        formatted = f"π/{den_part}"
                                    elif num_part == "-1":
                                        formatted = f"-π/{den_part}"
                                    else:
                                        formatted = f"{num_part}π/{den_part}"
                                else:
                                    formatted = f"{inner}π"
                            else:
                                # n*pi → nπ
                                formatted = f"{coeff}π"
                        else:
                            formatted = pi_form.replace("pi", "π")
                        is_exact = True
                        any_pi_fractions = True
                    else:
                        is_exact = False
            except (ValueError, TypeError):
                # Not a simple number - could be symbolic or complex
                sol_str = str(sol)
                if '.' in sol_str and not sol_str.rstrip('0').endswith('.'):
                    is_exact = False
                else:
                    # Symbolic expression (sqrt, etc.) - treat as exact
                    is_exact = True
            
            if not is_exact:
                all_are_exact = False
            exact_formatted.append(formatted)

        if exact_formatted:
            # Dynamic label: "Exact" for integers/symbols/π-fractions, "Decimal" for floats
            label = "Exact" if all_are_exact else "Decimal"
            
            try:
                print(f"{label}:", ", ".join(exact_formatted))
            except UnicodeEncodeError:
                # Fallback: print without Unicode characters
                exact_formatted_safe = []
                for item in exact_formatted:
                    try:
                        # Try to encode to check if it's safe
                        item.encode("ascii")
                        exact_formatted_safe.append(item)
                    except UnicodeEncodeError:
                        # Replace Unicode characters with ASCII equivalents
                        safe_item = item.replace("π", "pi").replace("≈", "approx")
                        exact_formatted_safe.append(safe_item)
                print(f"{label}:", ", ".join(exact_formatted_safe))

        # Show Approx only when result contains symbols (π, √, i, /)
        # - Hide for purely numeric results (Decimal: 12.345...) - Approx adds no value
        # - Show for symbolic results (Exact: π/4) - user needs decimal reference
        has_symbols = any(
            any(c in val for c in "π√/")
            for val in exact_formatted
        )
        if approx_sols and has_symbols:
            approx_display = ", ".join(
                format_number(approx_val)
                for approx_val in approx_sols
                if approx_val is not None
            )
            if approx_display:
                print("Approx:", approx_display)
    elif typ == "multi_isolate":
        sols = res.get("solutions", {})
        approx = res.get("approx", {})
        for var, sol_list in sols.items():
            if isinstance(sol_list, (list, tuple)):
                formatted = ", ".join(
                    format_solution(solution) for solution in sol_list
                )
            else:
                formatted = format_solution(sol_list)
            print(f"{var} = {formatted}")
            approx_list = approx.get(var)
            if approx_list:
                approx_display = ", ".join(
                    format_number(approx_val)
                    for approx_val in approx_list
                    if approx_val is not None
                )
                if approx_display:
                    print(f"  Decimal: {approx_display}")
    elif typ == "inequality":
        for k, v in res.get("solutions", {}).items():
            formatted_v = format_inequality_solution(str(v))
            print(f"Solution for {k}: {formatted_v}")
    elif typ == "pell":
        solution_str = res.get("solution", "")
        # Handle Unicode characters for Windows console compatibility
        try:
            print(f"Solution: {solution_str}")
        except UnicodeEncodeError:
            safe_sol = solution_str.replace("π", "pi").replace("≈", "approx")
            print(f"Solution: {safe_sol}")

        # Show specific solutions if available
        if "first_solutions" in res:
            print("First few solutions (x, y):")
            for sol in res["first_solutions"]:
                print(f"  ({sol[0]}, {sol[1]})")
    elif typ == "congruence_system":
        # Handled inside the solver logic really, but here for completeness
        print(f"Solution: {res.get('solution')}")
    elif typ == "system":
        solutions = res.get("solutions", [])
        if not solutions:
            print("No solutions found.")
        else:
            for idx, sol in enumerate(solutions):
                if len(solutions) > 1:
                    print(f"\nSolution {idx + 1}:")

                # Format each variable assignment in the solution
                # sol is a dict like {'x': '6', 'y': '4'}
                parts = []
                for var, val in sol.items():
                    formatted = format_solution(str(val))
                    parts.append(f"{var} = {formatted}")
                print(", ".join(parts))
    else:
        # Default value handling
        formatted_val = format_solution(str(res.get("result", "")))
        try:
            if expression:
                pretty_expr = pretty_print_expression(expression)
                print(f"Result: {pretty_expr} = {formatted_val}")
            else:
                print(f"Result: {formatted_val}")
        except UnicodeEncodeError:
            if expression:
                pretty_expr = pretty_print_expression(expression)
                print(
                    f"Result: {pretty_expr} = {formatted_val.encode('ascii', 'replace').decode()}"
                )
            else:
                print(f"Result: {formatted_val.encode('ascii', 'replace').decode()}")


def format_solution(val: Any) -> str:
    """
    Format a solution value (string, number, or SymPy object) into a 'pretty' string.

    Philosophy:
    - readability > precision for display (but keep precision reasonably high)
    - standard math notation (x^2 not x**2)
    - remove unnecessary noise (.0)

    Transformations:
    1. x**y -> x^y
    2. sqrt(x) -> √x (or √(x))
    3. 1.0 -> 1 (if integer)
    4. 2*x -> 2x (implicit multiplication for number-variable)
    5. I -> i (imaginary unit)
    """
    # Try to convert string values to SymPy for numeric evaluation
    # This handles cases like "-4*log(4) - 4*I*pi" from worker results
    if isinstance(val, str):
        try:
            # Check if string contains symbolic functions that should be evaluated
            if any(
                func in val
                for func in ["log(", "sin(", "cos(", "exp(", "sqrt(", "*I*", "*pi"]
            ):
                parsed = sp.sympify(val)
                if hasattr(parsed, "free_symbols") and not parsed.free_symbols:
                    # No free symbols - evaluate numerically
                    val = parsed.evalf()
        except Exception:
            # If parsing fails, keep original string
            pass

    # Apply algebraic simplifications first (if it's a SymPy expression)
    if hasattr(val, "subs") or isinstance(val, sp.Expr):
        try:
            val = simplify_exponential_bases(val)
        except Exception:
            # Fallback if simplification fails
            pass

        # Force numeric evaluation for expressions with no free symbols
        # This ensures f(-4) returns -5.545... not -4*log(4)
        try:
            if hasattr(val, "free_symbols") and not val.free_symbols:
                # No free symbols - this is a numeric value, evaluate it
                val = val.evalf()
        except Exception:
            pass

    s = str(val)

    # 1. Replace '**' with '^'
    s = s.replace("**", "^")

    # 2. Replace 'sqrt' with '√'
    # Simple replacement first: sqrt(x) -> √x.
    # SymPy usually outputs sqrt(expr). We'll map 'sqrt(' to '√('.
    s = s.replace("sqrt(", "√(")

    # 3. Handle floats ending in .0
    # Regex to find numbers like "123.0" not followed by other digits
    # matches 12.0, -5.0 but not 12.05
    s = re.sub(r"(\d+)\.0(?!\d)", r"\1", s)

    # 4. Implicit multiplication checks (SAFE ONLY)
    # 2*x -> 2x
    # digit followed by * followed by letter
    s = re.sub(r"(\d)\*([a-zA-Z])", r"\1\2", s)

    # 5. Paren multiplication
    # (expr)*(expr) -> (expr)(expr)
    s = s.replace(")*(", ")(")

    # 6. Imaginary unit
    # SymPy uses 'I'. Math notation usually 'i'.
    # Only replace isolated I, not "Image" or "III".
    # SymPy output usually puts I at the end or coefficient * I.
    # We'll validly assume 'I' as a standalone token is imaginary unit.
    s = re.sub(r"(?<![a-zA-Z])I(?![a-zA-Z])", "i", s)

    # 7. Variable-Paren multiplication
    # x*(... -> x(...
    # letter followed by * followed by (
    s = re.sub(r"([a-zA-Z])\*\((?=.+)", r"\1(", s)

    # 8. Remove confusing re() notation for real-valued expressions
    # re(x) -> x (real part of x, which equals x for real inputs)
    s = re.sub(r"re\(([^)]+)\)", r"\1", s)

    return s


def simplify_exponential_bases(expr: sp.Expr) -> sp.Expr:
    """
    Transform exp(c*x) -> (base)^x where base = exp(c) is a clean integer/rational.

    Example:
        exp(0.693147... * x) -> 2^x
        exp(1.098612... * x) -> 3^x

    Engineering Standards:
    - Bounded Logic: Uses SymPy's structural replacement (bounded complexity)
    - No Magic: Explicit mathematical transformation
    """
    if not isinstance(expr, sp.Expr):
        return expr

    # Engineering Standards: Explicit Recursion for Bounded Logic and No Magic

    # 1. Base case: Atomic or non-Expr objects
    if not isinstance(expr, sp.Basic):
        return expr

    # 2. Recursively simplify arguments first (Bottom-Up)
    new_args = [simplify_exponential_bases(arg) for arg in expr.args]

    # Reconstruct expression with simplified arguments if they changed
    if new_args != list(expr.args):
        try:
            expr = expr.func(*new_args)
        except Exception:
            # Fallback for weird SymPy internal structures
            pass

    # 3. Transform current node if it matches pattern: exp(c*x) or exp(c)
    if expr.func == sp.exp:
        arg = expr.args[0]

        # Check for exp(c * x) pattern
        if arg.is_Mul:
            coeffs = [a for a in arg.args if a.is_number]
            non_coeffs = [a for a in arg.args if not a.is_number]

            if coeffs and non_coeffs:
                # Combine all numeric coefficients
                total_coeff = sp.Mul(*coeffs)
                remaining = sp.Mul(*non_coeffs)

                # Check if exp(total_coeff) is close to an integer/simple rational
                try:
                    val = sp.exp(total_coeff)
                    val_f = float(val.evalf())

                    # Check for integer closeness (tolerance 1e-9)
                    if abs(val_f - round(val_f)) < 1e-9:
                        base = int(round(val_f))
                        if base > 1:
                            return sp.Pow(base, remaining)

                    # Additional check for 1/integer (e.g. 0.5^x)
                    if abs(val_f) > 1e-9:  # Avoid division by zero
                        inv_val_f = 1.0 / val_f
                        if abs(inv_val_f - round(inv_val_f)) < 1e-9:
                            inv_base = int(round(inv_val_f))
                            if inv_base > 1:
                                return sp.Pow(sp.Rational(1, inv_base), remaining)
                except Exception:
                    pass

        # Check for simple exp(c) -> integer pattern
        elif arg.is_number:
            try:
                val_f = float(expr.evalf())
                if abs(val_f - round(val_f)) < 1e-9:
                    base = int(round(val_f))
                    return sp.Integer(base)
            except Exception:
                pass

    return expr
