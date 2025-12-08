
import json
import logging
import sympy as sp
import re
from typing import Any

from kalkulator_pkg.parser import (
    format_inequality_solution,
    format_number,
    format_solution,
    format_superscript,
)

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
    """Find if a number is close to a rational multiple of π and return the fraction form."""
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


def print_result_pretty(res: dict[str, Any], output_format: str = "human") -> None:
    """Print result in specified format."""
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
        for sol in exact_sols:
            # π-fraction conversion (Casio-style) disabled
            exact_formatted.append(format_solution(sol))

        if exact_formatted:
            try:
                print("Exact:", ", ".join(exact_formatted))
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
                print("Exact:", ", ".join(exact_formatted_safe))

        if approx_sols:
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
    else:
        # Default value handling
        formatted_val = format_solution(str(res.get("result", "")))
        try:
            print(f"Result: {formatted_val}")
        except UnicodeEncodeError:
            print(f"Result: {formatted_val.encode('ascii', 'replace').decode()}")
