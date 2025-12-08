from __future__ import annotations

import argparse
import json
import logging
import math
import re  # <--- ADDED: For parsing 'find' command
from fractions import Fraction
from math import gcd
from typing import Any

import sympy as sp

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


from .config import VAR_NAME_RE, VERSION
from .parser import (
    format_inequality_solution,
    format_number,
    format_solution,
    format_superscript,
    parse_preprocessed,
    split_top_level_commas,
)
from .solver import solve_inequality, solve_single_equation, solve_system
from .types import ParseError, ValidationError
from .worker import evaluate_safely

logger = logging.getLogger(__name__)


def _parse_target_with_ambiguity_detection(
    target_str: str, max_small_denominator: int = 12
) -> tuple[sp.Rational, sp.Rational | None, bool]:
    """Parse target string with ambiguity detection for simpler rationals.

    Args:
        target_str: Target value as string (e.g., "65.083", "781/12")
        max_small_denominator: Maximum denominator to check for simpler rationals

    Returns:
        Tuple (target_rational, simpler_rational_or_none, was_simplified)
    """
    from decimal import Decimal

    # Parse as exact Fraction first
    try:
        # Try parsing as fraction if it contains '/'
        if "/" in target_str:
            parts = target_str.split("/")
            if len(parts) == 2:
                num = int(parts[0].strip())
                den = int(parts[1].strip())
                literal_frac = sp.Rational(num, den)
                return (literal_frac, None, False)

        # Parse as decimal string
        decimal_val = Decimal(target_str.strip())
        literal_frac = sp.Rational(str(decimal_val))

        # Try to find simpler rational approximation
        # Check small denominators: 2, 3, 4, 6, 12, etc.
        frac = Fraction(literal_frac.numerator, literal_frac.denominator)
        simpler_frac = frac.limit_denominator(max_small_denominator)

        # Check if simpler and very close
        if simpler_frac.denominator < literal_frac.denominator:
            diff = abs(
                float(
                    literal_frac
                    - sp.Rational(simpler_frac.numerator, simpler_frac.denominator)
                )
            )

            # Use practical tolerance: absolute diff < 1e-3 for detecting repeating decimals
            # This handles cases like 65.083 ≈ 781/12 (diff ≈ 0.0003)
            tolerance = 1e-3
            if diff <= tolerance:
                simpler_rational = sp.Rational(
                    simpler_frac.numerator, simpler_frac.denominator
                )
                return (simpler_rational, literal_frac, True)

        return (literal_frac, None, False)
    except (ValueError, TypeError, ImportError):
        # Fallback: try parsing with SymPy
        try:
            from .parser import parse_preprocessed

            target_expr = parse_preprocessed(target_str)
            if isinstance(target_expr, (sp.Float, float)):
                target_expr = sp.Rational(str(target_expr))
            return (target_expr, None, False)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot parse target '{target_str}'") from None


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm.

    Returns (s, t, g) such that s*a + t*b = g = gcd(a, b).
    """
    if b == 0:
        return (1, 0, a)
    else:
        s1, t1, g = _extended_gcd(b, a % b)
        s = t1
        t = s1 - (a // b) * t1
        return (s, t, g)


def _compute_integerized_equation(
    coeffs: list[sp.Rational], target: sp.Rational, L_func: int
) -> tuple[int, int, int, int] | None:
    """Compute integerized form of linear equation A*x + B*y = C for integer solution finding.

    For equation: coeff_x*x + coeff_y*y + const = target
    Rearranged: coeff_x*x + coeff_y*y = target - const

    Uses L_func (LCM of function coefficient denominators) and computes L_total = lcm(L_func, dF)
    where dF is the denominator of target. Returns integer coefficients (L_total*A, L_total*B, L_total*C).

    Args:
        coeffs: List of coefficients [A, B, const] as SymPy Rationals
        target: Target value as SymPy Rational
        L_func: LCM of denominators of function coefficients (A, B, C)

    Returns:
        Tuple (a_int, b_int, c_int, L_total) of integer coefficients and total LCM, or None if not linear
    """
    if len(coeffs) != 3:
        return None

    coeff_x, coeff_y, const = coeffs[0], coeffs[1], coeffs[2]

    # Check if coefficients are rational
    if not (
        isinstance(coeff_x, sp.Rational)
        and isinstance(coeff_y, sp.Rational)
        and isinstance(const, sp.Rational)
        and isinstance(target, sp.Rational)
    ):
        return None

    # Compute L_total = lcm(L_func, denominator(target))
    dF = target.denominator

    def lcm(a: int, b: int) -> int:
        return abs(a * b) // gcd(a, b) if a and b else 0

    L_total = lcm(L_func, dF)

    # Compute integer coefficients: a_total = A * L_total, etc.
    a_total = int(coeff_x * L_total)
    b_total = int(coeff_y * L_total)
    const_total = int(const * L_total)

    # Compute RHS: L_total * target - const_total
    # This ensures RHS is integer
    RHS_total_int = int(L_total * target) - const_total

    # Verify RHS is integer (should always be true, but check for safety)
    if not isinstance(RHS_total_int, int):
        raise ValueError(f"RHS is not integer: {L_total * target - const}")

    # Simplify by dividing by GCD of all coefficients (a_total, b_total, RHS_total_int)
    g_all = gcd(gcd(abs(a_total), abs(b_total)), abs(RHS_total_int))
    if g_all > 1:
        a_total //= g_all
        b_total //= g_all
        RHS_total_int //= g_all
        # Note: We don't adjust L_total here - it's the original L_total used

    return (a_total, b_total, RHS_total_int, L_total)


def find_integer_solutions_for_linear(equation, x, y):
    """
    Return a list of integer solution parameterizations for linear equations in x,y.
    - Accepts sp.Eq or expression (==0).
    - Returns a list of sympy solutions (the same format as sp.diophantine set items)
      or an empty list if none. Raises ValueError on non-linear / non-numeric coeffs.
    """
    # Normalize equation -> expr = lhs - rhs
    if isinstance(equation, sp.Equality):
        expr = sp.simplify(equation.lhs - equation.rhs)
    else:
        expr = sp.simplify(equation)

    # Ensure expr is linear in x,y
    poly = sp.Poly(expr, x, y)
    if poly.total_degree() > 1:
        raise ValueError(f"Equation is not linear in {x} and {y}")

    # Extract coefficients from expr = a*x + b*y + c  (i.e. expr should be 0 when satisfied)
    a = sp.simplify(poly.coeff_monomial(x))
    b = sp.simplify(poly.coeff_monomial(y))
    c = sp.simplify(poly.coeff_monomial(1))  # constant term in expr

    # Check coefficients are numeric (no free symbols left other than x,y were removed by Poly)
    for coef in (a, b, c):
        if coef.free_symbols:
            raise ValueError("Non-numeric coefficient found: %r" % coef)

    # Convert coefficients to rationals and scale to integer coefficients:
    def denom_of(sympy_number):
        t = sp.together(sympy_number)
        if t.is_Rational:
            return int(t.q)
        if t.is_integer:
            return 1
        # fallback try nsimplify
        try:
            r = sp.nsimplify(t)
            if r.is_Rational:
                return int(r.q)
        except Exception:
            pass
        raise ValueError("Coefficient is not rational/integer: %r" % sympy_number)

    denoms = [denom_of(v) for v in (a, b, c)]
    lcm_den = _math_lcm(*denoms) if denoms else 1

    a_int = sp.Integer(sp.simplify(a * lcm_den))
    b_int = sp.Integer(sp.simplify(b * lcm_den))
    c_int = sp.Integer(sp.simplify(c * lcm_den))

    # expr = a*x + b*y + c = 0  =>  a*x + b*y = -c
    A = int(a_int)
    B = int(b_int)
    C = int(-c_int)

    # Use gcd test for existence
    g = math.gcd(A, B)
    if g == 0:
        # degenerate: A=B=0 -> either no solutions or all integers if C==0
        if C == 0:
            # all integer pairs are solutions; return a sentinel
            return [{"all_integer_pairs": True}]
        else:
            return []

    if C % g != 0:
        return []

    # Use sympy.diophantine for general solution
    try:
        sols = list(sp.diophantine(A * x + B * y - C))
        return sols
    except Exception:
        logger.exception("diophantine failed")
        # fallback: return empty and let caller decide
        return []


def _solve_modulo_system_if_applicable(
    parts: list[str], var: str, output_format: str = "human"
) -> tuple[bool, int]:
    """Check if parts form a system of congruences and solve it using CRT.

    Args:
        parts: List of assignment strings (e.g., ["x = 1 % 2", "x = 3 % 6"])
        var: Variable name
        output_format: Output format ("human" or "json")

    Returns:
        Tuple (solved, exit_code) where solved is True if solved as congruence system
    """
    # Check if all RHS expressions are modulo operations (like "1 % 2")
    congruences = []
    all_modulo = True

    for p in parts:
        if "=" in p:
            left, right = p.split("=", 1)
            rhs = right.strip() or "0"
            # Check if RHS is a modulo expression (pattern: number % number)
            # Allow for optional whitespace and handle both integer and float-like patterns
            modulo_match = re.match(r"^\s*(-?\d+)\s*%(?:\s*(\d+)\s*)?$", rhs)
            if modulo_match:
                remainder_str = modulo_match.group(1)
                modulus_str = modulo_match.group(2)  # Modulus can be None if just 'X %'

                if modulus_str is None:
                    # Handle cases like 'X %', which should not be treated as congruence
                    all_modulo = False
                    break

                try:
                    remainder = int(remainder_str)
                    modulus = int(modulus_str)
                    if modulus > 0:
                        congruences.append((remainder, modulus))
                    else:
                        all_modulo = False
                        break
                except ValueError:
                    # This happens if remainder_str or modulus_str are not valid integers
                    all_modulo = False
                    break
            else:
                all_modulo = False
                break

    # If all are modulo expressions, solve as system of congruences
    if all_modulo and len(congruences) > 1:
        try:
            from .solver import solve_system_of_congruences

            solution = solve_system_of_congruences(congruences)
            if solution is not None:
                k, m = solution
                if output_format == "json":
                    print(
                        json.dumps(
                            {
                                "ok": True,
                                "type": "congruence_system",
                                "solution": f"{var} == {k} (mod {m})",  # Use == instead of ≡ for JSON compatibility
                                "remainder": k,
                                "modulus": m,
                            }
                        )
                    )
                else:
                    # Use ASCII-safe representation for Windows compatibility
                    try:
                        print(f"Solution: {var} ≡ {k} (mod {m})")
                    except UnicodeEncodeError:
                        print(f"Solution: {var} == {k} (mod {m})")
                return (True, 0)
            else:
                # System is inconsistent
                if output_format == "json":
                    print(
                        json.dumps(
                            {
                                "ok": False,
                                "error": "System of congruences is inconsistent (no solution exists)",
                            }
                        )
                    )
                else:
                    print(
                        "Error: System of congruences is inconsistent (no solution exists)"
                    )
                return (True, 1)
        except Exception as e:
            # Log the exception for debugging, but fall through to individual evaluation
            logger.debug(f"Error solving system of congruences: {e}", exc_info=True)
            pass

    return (False, 0)


def _format_number_no_trailing_zeros(num_str: str) -> str:
    """Format a number string by removing trailing zeros and decimal point if not needed.

    Args:
        num_str: Number string (e.g., "2.00000000000000", "1.5")

    Returns:
        Formatted string (e.g., "2", "1.5")
    """
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


def _format_inverse_solutions(
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


def _find_pi_fraction_form(
    num_val: float, max_denominator: int = 10000, tolerance: float = 1e-8
) -> str | None:
    """Find if a number is close to a rational multiple of π and return the fraction form.

    This implements Casio-style automatic π-fraction conversion.
    Returns form like "(156158413/3600)*pi" if found, None otherwise.

    Args:
        num_val: Numeric value to check
        max_denominator: Maximum denominator to search for (higher = more accurate but slower)
        tolerance: Relative tolerance for matching (relative to magnitude of num_val)

    Returns:
        String in form "(numerator/denominator)*pi" if found, None otherwise
    """
    try:
        # Skip conversion for exactly zero or very small numbers close to zero
        # This avoids silly results like "0*pi" for 1-1=0
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


def _convert_to_pi_fraction(num_str: str, tolerance: float = 1e-6) -> str:
    """Convert a decimal number to fractional π form if it's close to a rational multiple of π.

    Args:
        num_str: Decimal number string (e.g., "-11.7809724509620")
        tolerance: Tolerance for matching rational multiples of π

    Returns:
        String in form like "-15π/4" if convertible, otherwise returns formatted number
    """
    try:
        num = float(num_str)
        # Divide by π to get the coefficient
        pi_val = float(sp.pi.evalf())
        coeff = num / pi_val

        # Try to find a rational approximation
        # Use SymPy's Rational approximation
        try:
            # Find rational approximation with max denominator
            # Limit denominator to avoid huge fractions
            max_denom = 1000
            rat = sp.Rational(sp.N(coeff)).limit_denominator(max_denom)

            # Check if the approximation is close enough
            pi_mult = float(rat) * pi_val
            if abs(num - pi_mult) < tolerance:
                # Format as (numerator)π/(denominator)
                num_val = int(rat.numerator)
                den_val = int(rat.denominator)

                if den_val == 1:
                    # Integer multiple of π
                    if num_val == 1:
                        return "π"
                    elif num_val == -1:
                        return "-π"
                    else:
                        return f"{num_val}π"
                else:
                    # Fractional multiple
                    if num_val < 0:
                        # Negative fraction
                        if abs(num_val) == 1:
                            return f"-π/{den_val}"
                        else:
                            return f"{num_val}π/{den_val}"
                    else:
                        # Positive fraction
                        if num_val == 1:
                            return f"π/{den_val}"
                        else:
                            return f"{num_val}π/{den_val}"
        except (ValueError, TypeError, AttributeError):
            pass

        # If conversion fails, return formatted number
        return _format_number_no_trailing_zeros(num_str)
    except (ValueError, TypeError):
        # If parsing fails, return original string
        return num_str


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
        from .parser import parse_preprocessed, preprocess

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
        from .solver import solve_single_equation

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
        from .worker import evaluate_safely

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


def print_result_pretty(res: dict[str, Any], output_format: str = "human") -> None:
    """Print result in specified format.

    Args:
        res: Result dictionary
        output_format: "json" for JSON output, "human" for human-readable
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
            print("Pell parametric solution:")
            print(solution_str)
        except UnicodeEncodeError:
            # Fallback: replace Unicode with ASCII
            safe_solution = solution_str.replace("\u221a", "sqrt").replace(
                "\u00b2", "^2"
            )
            print("Pell parametric solution:")
            print(safe_solution)
    elif typ == "identity_or_contradiction":
        print(res.get("result"))
    elif typ == "system":
        sols = res.get("solutions", [])
        if not sols:
            print("No solutions found.")
        else:
            print("System Solutions:")
            for _, sol in enumerate(sols):
                # sol is a dict like {'a': '2', 'b': '5'}
                formatted_parts = []
                for var, val in sol.items():
                    val_fmt = _format_special_values(format_solution(val))
                    formatted_parts.append(f"{var} = {val_fmt}")
                print(f"  {', '.join(formatted_parts)}")
    elif typ == "evaluation":
        # Handle evaluation results (e.g., "2+2=")
        exact_list = res.get("exact", [])
        approx_list = res.get("approx", [])
        if exact_list:
            val = exact_list[0] if exact_list[0] else ""
            print(_format_special_values(val))
        if approx_list and approx_list[0]:
            print(f"Decimal: {_format_special_values(str(approx_list[0]))}")
    elif typ == "value":
        res_str = res.get("result")
        approx = res.get("approx")
        if res_str is None:
            print(res)
            return
        try:
            formatted_res = _format_special_values(str(res_str))
            print(f"{formatted_res}")
        except (UnicodeEncodeError, OSError):
            # Handle encoding errors on Windows console
            try:
                # Try printing without formatting
                print(str(res_str))
            except (UnicodeEncodeError, OSError):
                # Last resort: print raw representation
                print(repr(res_str))
        try:
            parsed = parse_preprocessed(res_str)
            expanded = sp.expand(parsed)
            if str(expanded) != str(parsed):
                print(f"Expanded: {format_solution(expanded)}")
        except (ParseError, ValidationError, ValueError, TypeError, AttributeError):
            # Expected errors for some expressions - silently skip expansion
            pass
        if approx:
            print("Decimal:", approx)
    else:
        print(res)


def _format_special_values(val_str: str) -> str:
    """Format special values like zoo and oo*I to user-friendly strings.

    Args:
        val_str: String representation of value

    Returns:
        Formatted string
    """
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

    # Handle expressions with I (preserve them as-is for now, but could enhance)
    # For example "1.0*I" should become "√(-1)"
    if val_str == "1.0*I" or val_str == "I*1.0" or val_str == "1*I" or val_str == "I*1":
        return "√(-1)"

    return val_str


def repl_loop(output_format: str = "human") -> None:
    """Interactive REPL loop with graceful interrupt handling."""

    global logger  # Use module-level logger

    try:
        import readline  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        # readline not available on Windows - that's fine
        pass

    # Pre-warm worker processes to avoid startup delay on first calculation
    try:
        from .worker import warmup_workers

        warmup_workers()
    except ImportError:
        pass

    # Ensure evaluate_safely and parse_preprocessed are always available in this function scope
    # Import them here to avoid any scoping issues in nested blocks
    import sympy as sp  # Import locally to avoid scoping issues with later local imports

    from .parser import parse_preprocessed as _parse_preprocessed
    from .worker import evaluate_safely as _evaluate_safely

    print("kalkulator-ai — type 'help' for commands, 'quit' to exit.")
    _current_req_id = None  # Track current request for cancellation
    _timing_enabled = False  # Track whether timing is enabled
    _cache_hits_enabled = False  # Track whether cache hit display is enabled
    _cache_hits_tracking: list[tuple[str, str]] = []  # Track cache hits: [(expr, type)]

    def signal_handler(signum: Any, frame: Any) -> None:
        """Handle interrupt signal gracefully."""
        nonlocal _current_req_id
        if _current_req_id:
            from .worker import cancel_current_request

            cancel_current_request(_current_req_id)
            print("\n[Cancelling request...]")
        else:
            print("\n[Press Ctrl+C again to exit]")

    # Register signal handler for graceful interrupt (Unix)
    try:
        import signal

        signal.signal(signal.SIGINT, signal_handler)
    except (ImportError, AttributeError):
        # Windows doesn't support signal.SIGINT the same way
        pass

    pending_commands = []

    while True:
        try:
            if pending_commands:
                raw = pending_commands.pop(0)
                # Echo the command being processed if it came from a split line
                # print(f">>> {raw}")
            else:
                raw_input = input(">>> ").strip()
                if not raw_input:
                    continue

                # Split input by top-level commas to handle multiple commands
                # e.g. "f(x)=2x, g(x)=x/2, f(g(5))"
                # BUT: Don't split if input contains "find" keyword with variable assignments
                # e.g. "a=2, b=a+3, find b" should NOT be split
                # ALSO: Don't split if ALL parts look like chained simple variable assignments
                # e.g. "a=1, x=a+1" should NOT be split
                from .parser import split_top_level_commas

                # Check if this appears to be a variable-finding command (contains "find" and assignments)
                has_find_keyword = "find" in raw_input.lower()
                has_assignments = "=" in raw_input

                # Check if ALL parts look like simple variable assignments (var = expr)
                # This pattern: single lowercase letter or word followed by = and expression
                import re

                parts_check = split_top_level_commas(raw_input)
                simple_assignment_pattern = re.compile(
                    r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.+$"
                )
                all_simple_assignments = all(
                    simple_assignment_pattern.match(p.strip())
                    and not p.strip().startswith("f(")
                    and "(" not in p.split("=")[0]
                    for p in parts_check
                    if p.strip()
                )

                # Check for function finding pattern: multiple func(numeric) = value
                # with same function name (e.g., "g(1) = -4, g(2) = 0, g(0) = -6")
                func_finding_pattern = re.compile(
                    r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]+)\)\s*=\s*(.+)$"
                )
                is_function_finding = False
                if len(parts_check) >= 2:
                    func_names_found = set()
                    numeric_args_count = 0
                    for p in parts_check:
                        m = func_finding_pattern.match(p.strip())
                        if m:
                            func_names_found.add(m.group(1))
                            # Check if argument contains at least one digit (numeric data point)
                            if re.search(r"\d", m.group(2)):
                                numeric_args_count += 1
                    # If all parts match the pattern, same function name, and have numeric args
                    if len(func_names_found) == 1 and numeric_args_count >= 2:
                        is_function_finding = True

                # Check for research-grade commands that should NOT be split
                raw_lower_check = raw_input.lower().strip()
                is_research_command = (
                    raw_lower_check.startswith("evolve ")
                    or raw_lower_check.startswith("find ode")
                    or raw_lower_check.startswith("discover causal")
                    or raw_lower_check.startswith("find dimensionless")
                    or raw_lower_check.startswith("benchmark")
                )

                if is_research_command:
                    # Don't split - these commands use array syntax with commas
                    raw = raw_input
                elif has_find_keyword and has_assignments:
                    # Don't split - process entire input together for variable finding
                    raw = raw_input
                elif all_simple_assignments and len(parts_check) > 1:
                    # Don't split - process as chained assignments (a=1, x=a+1)
                    raw = raw_input
                elif is_function_finding:
                    # Don't split - process entire input as function finding data points
                    raw = raw_input
                else:
                    parts = split_top_level_commas(raw_input)

                    if len(parts) > 1:
                        raw = parts[0].strip()
                        # Add remaining parts to pending queue
                        for part in parts[1:]:
                            if part.strip():
                                pending_commands.append(part.strip())
                    else:
                        raw = raw_input

            _current_req_id = None  # Clear on new input

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            # Stop worker processes gracefully
            try:
                from .worker import _WORKER_MANAGER

                _WORKER_MANAGER.stop()
            except Exception:
                pass
            # Save persistent cache on shutdown
            try:
                from .cache_manager import save_cache_to_disk

                save_cache_to_disk()
            except ImportError:
                pass
            break

        if not raw:
            continue

        # Typo detection for REPL commands
        # Check if input looks like a typo of a known command (single word, no math operators)
        raw_lower = raw.lower().strip()
        # Only check if it's a single word (no spaces) or has minimal spaces, and no math operators
        has_math_ops = any(
            op in raw for op in ["+", "-", "*", "/", "=", "(", ")", "^", "<", ">"]
        )
        is_single_word = " " not in raw_lower or raw_lower.count(" ") <= 1

        if raw_lower and not has_math_ops and is_single_word:
            from .parser import REPL_COMMANDS

            def levenshtein_distance(s1: str, s2: str) -> int:
                """Calculate Levenshtein distance between two strings."""
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                if len(s2) == 0:
                    return len(s1)
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                return previous_row[-1]

            # Check all known commands and their variations
            all_commands = list(REPL_COMMANDS)
            all_commands.extend(
                ["clear cache", "show cache", "save cache", "load cache"]
            )

            best_match = None
            best_distance = float("inf")
            for cmd in all_commands:
                cmd_lower = cmd.lower()
                # Calculate distance
                distance = levenshtein_distance(raw_lower, cmd_lower)
                # Check if it's a reasonable match (within 2 edits or 30% of length)
                max_allowed = max(2, min(3, len(raw_lower) // 3))
                if distance < best_distance and distance <= max_allowed:
                    # Also check if it starts with a significant prefix match
                    min_prefix_len = min(3, len(raw_lower), len(cmd_lower))
                    if (
                        raw_lower[:min_prefix_len] == cmd_lower[:min_prefix_len]
                        or distance <= 2
                    ):
                        best_distance = distance
                        best_match = cmd

            # If we found a close match, suggest it
            if best_match and best_distance > 0:
                print(f'Did you mean "{best_match}"?')
                continue

        # Export command: export <function_name> to <filename>
        # Generates Python file from discovered function
        if raw_lower.startswith("export "):
            import re

            export_match = re.match(r"export\s+(\w+)\s+to\s+(.+)", raw, re.IGNORECASE)
            if export_match:
                func_name = export_match.group(1)
                filename = export_match.group(2).strip()

                from .function_manager import export_function_to_file

                success, message = export_function_to_file(func_name, filename)
                print(message)
                continue
            else:
                print("Usage: export <function_name> to <filename>")
                print("Example: export V to volume_lib.py")
                continue

        # =====================================================================
        # RESEARCH-GRADE SYMBOLIC REGRESSION COMMANDS
        # =====================================================================

        # Evolve command: Use genetic programming to discover equations
        # Usage: evolve f(x) from x=[1,2,3,4], y=[2,4,6,8]
        if raw_lower.startswith("evolve "):
            import re

            try:
                import numpy as np

                from .symbolic_regression import GeneticConfig, GeneticSymbolicRegressor

                # Parse: evolve f(x) from x=[...], y=[...]
                # or: evolve f(x,y) from x=[...], y=[...], z=[...]
                evolve_match = re.match(
                    r"evolve\s+(\w+)\s*\(([^)]+)\)\s+from\s+(.+)", raw, re.IGNORECASE
                )

                if evolve_match:
                    func_name = evolve_match.group(1)
                    # These are the INPUT variable names from f(x) or f(a,b)
                    input_var_names = [
                        v.strip() for v in evolve_match.group(2).split(",")
                    ]
                    data_part = evolve_match.group(3)

                    # Parse data arrays
                    data_dict = {}
                    array_pattern = re.compile(r"(\w+)\s*=\s*\[([^\]]+)\]")
                    for match in array_pattern.finditer(data_part):
                        var = match.group(1)
                        values = [float(v.strip()) for v in match.group(2).split(",")]
                        data_dict[var] = np.array(values)

                    if not data_dict:
                        print(
                            "Error: No data arrays found. Use format: x=[1,2,3], y=[4,5,6]"
                        )
                        continue

                    # Input variables are the ones in the function signature
                    # Output is any variable NOT in the signature (typically 'y' or 'z')
                    input_vars = [v for v in input_var_names if v in data_dict]
                    output_candidates = [
                        v for v in data_dict.keys() if v not in input_var_names
                    ]

                    if not input_vars:
                        # Maybe user provided x,y in data but only x in function signature
                        # Use all vars from signature that exist in data
                        input_vars = input_var_names[:1]  # Use first var as input
                        output_candidates = [
                            v for v in data_dict.keys() if v != input_vars[0]
                        ]

                    if not output_candidates:
                        print(
                            f"Error: Need output variable. Provide data for a variable not in {func_name}({','.join(input_var_names)})"
                        )
                        print(
                            "  Example: evolve f(x) from x=[1,2,3], y=[2,4,6]  (y is output)"
                        )
                        continue

                    output_var = output_candidates[
                        0
                    ]  # First non-input variable is output

                    # Validate all input vars have data
                    missing = [v for v in input_vars if v not in data_dict]
                    if missing:
                        print(f"Error: Missing data for input variable(s): {missing}")
                        continue

                    X = np.column_stack([data_dict[v] for v in input_vars])
                    y = data_dict[output_var]

                    print(
                        f"Evolving {func_name}({', '.join(input_vars)}) from {len(y)} data points..."
                    )

                    config = GeneticConfig(
                        population_size=200,
                        n_islands=2,
                        generations=50,
                        timeout=30,
                        verbose=True,
                    )
                    regressor = GeneticSymbolicRegressor(config)
                    pareto = regressor.fit(X, y, input_vars)

                    best = pareto.get_knee_point() or pareto.get_best()
                    if best:
                        print(
                            f"\nDiscovered: {func_name}({', '.join(input_vars)}) = {best.expression}"
                        )
                        print(f"MSE: {best.mse:.6e}")
                        print(f"Complexity: {best.complexity} nodes")
                    else:
                        print("No solution found")
                else:
                    print("Usage: evolve f(x) from x=[1,2,3], y=[2,4,6]")
                    print("       evolve g(a,b) from a=[...], b=[...], z=[...]")
            except ImportError as e:
                print(f"Error: Required module not available: {e}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        # Find ODE command: Discover differential equations using SINDy
        # Usage: find ode from t=[...], x=[...], v=[...]
        if raw_lower.startswith("find ode"):
            import re

            try:
                import numpy as np

                from .dynamics_discovery import SINDy, SINDyConfig

                # Parse: find ode from t=[...], x=[...], v=[...]
                ode_match = re.match(r"find\s+ode\s+from\s+(.+)", raw, re.IGNORECASE)

                if ode_match:
                    data_part = ode_match.group(1)

                    # Parse data arrays
                    data_dict = {}
                    array_pattern = re.compile(r"(\w+)\s*=\s*\[([^\]]+)\]")
                    for match in array_pattern.finditer(data_part):
                        var = match.group(1)
                        values = [float(v.strip()) for v in match.group(2).split(",")]
                        data_dict[var] = np.array(values)

                    if "t" not in data_dict:
                        print("Error: Time variable 't' is required")
                        continue

                    t = data_dict.pop("t")
                    state_vars = list(data_dict.keys())
                    X = np.column_stack([data_dict[v] for v in state_vars])

                    print(f"Discovering ODEs for {state_vars}...")

                    if len(t) < 10:
                        print(
                            f"Warning: Only {len(t)} data points provided. SINDy requires more data for reliable results."
                        )
                        print("  With very few points, complex models will overfit.")
                        print(
                            "  Restricting to LINEAR models only (poly_order=1) and higher threshold..."
                        )
                        sindy = SINDy(SINDyConfig(threshold=0.5, poly_order=1))
                    else:
                        sindy = SINDy(SINDyConfig(threshold=0.1, poly_order=3))

                    sindy.fit(X, t, variable_names=state_vars)

                    print("\nDiscovered equations:")
                    sindy.print_equations()
                else:
                    print("Usage: find ode from t=[0,0.1,0.2,...], x=[...], v=[...]")
            except ImportError as e:
                print(f"Error: Required module not available: {e}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        # Discover causal graph command
        # Usage: discover causal graph from x=[...], y=[...], z=[...]
        if raw_lower.startswith("discover causal"):
            import re

            try:
                import numpy as np

                from .causal_discovery import PCAlgorithm

                # Parse: discover causal graph from x=[...], y=[...], z=[...]
                causal_match = re.match(
                    r"discover\s+causal\s+(?:graph\s+)?from\s+(.+)", raw, re.IGNORECASE
                )

                if causal_match:
                    data_part = causal_match.group(1)

                    # Parse alpha parameter if present
                    alpha = 0.05
                    alpha_match = re.search(r"alpha\s*=\s*([\d.]+)", data_part)
                    if alpha_match:
                        alpha = float(alpha_match.group(1))
                        data_part = re.sub(r"alpha\s*=\s*[\d.]+,?\s*", "", data_part)

                    # Parse data arrays
                    data_dict = {}
                    array_pattern = re.compile(r"(\w+)\s*=\s*\[([^\]]+)\]")
                    for match in array_pattern.finditer(data_part):
                        var = match.group(1)
                        values = [float(v.strip()) for v in match.group(2).split(",")]
                        data_dict[var] = np.array(values)

                    if len(data_dict) < 2:
                        print("Error: At least 2 variables required")
                        continue

                    var_names = list(data_dict.keys())
                    X = np.column_stack([data_dict[v] for v in var_names])

                    print(f"Discovering causal structure for {var_names}...")

                    pc = PCAlgorithm(alpha=alpha)
                    graph = pc.fit(X, variable_names=var_names)

                    print(graph)
                else:
                    print("Usage: discover causal graph from x=[...], y=[...], z=[...]")
                    print("       discover causal from x=[...], y=[...], alpha=0.01")
            except ImportError as e:
                print(f"Error: Required module not available: {e}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        # Dimensionless groups / Buckingham Pi
        # Usage: find dimensionless from F=force, rho=density, v=velocity, L=length
        if raw_lower.startswith("find dimensionless"):
            import re

            try:
                from .dimensional_analysis import (
                    ACCELERATION,
                    DENSITY,
                    ENERGY,
                    FORCE,
                    FREQUENCY,
                    LENGTH,
                    MASS,
                    POWER,
                    PRESSURE,
                    TIME,
                    VELOCITY,
                    Dimension,
                    find_dimensionless_groups,
                    format_pi_group,
                )

                dim_map = {
                    "mass": MASS,
                    "m": MASS,
                    "length": LENGTH,
                    "l": LENGTH,
                    "time": TIME,
                    "t": TIME,
                    "force": FORCE,
                    "f": FORCE,
                    "velocity": VELOCITY,
                    "v": VELOCITY,
                    "density": DENSITY,
                    "rho": DENSITY,
                    "energy": ENERGY,
                    "e": ENERGY,
                    "power": POWER,
                    "p": POWER,
                    "pressure": PRESSURE,
                    "acceleration": ACCELERATION,
                    "a": ACCELERATION,
                    "frequency": FREQUENCY,
                    "freq": FREQUENCY,
                }

                dim_match = re.match(
                    r"find\s+dimensionless\s+from\s+(.+)", raw, re.IGNORECASE
                )

                if dim_match:
                    data_part = dim_match.group(1)

                    quantities = []
                    for item in data_part.split(","):
                        item = item.strip()
                        if "=" in item:
                            var, dim_name = item.split("=", 1)
                            var = var.strip()
                            dim_name = dim_name.strip().lower()
                            if dim_name in dim_map:
                                quantities.append((var, dim_map[dim_name]))
                            else:
                                print(f"Unknown dimension: {dim_name}")
                                print(
                                    f"Available: {', '.join(sorted(set(dim_map.keys())))}"
                                )
                                continue

                    if quantities:
                        print(
                            f"Finding dimensionless groups for: {[q[0] for q in quantities]}"
                        )
                        groups = find_dimensionless_groups(quantities)

                        if groups:
                            print("\nDimensionless groups (Buckingham Pi):")
                            for i, g in enumerate(groups, 1):
                                print(f"  Π{i} = {format_pi_group(g)}")
                        else:
                            print("No dimensionless groups found")
                else:
                    print(
                        "Usage: find dimensionless from F=force, rho=density, v=velocity, L=length"
                    )
            except ImportError as e:
                print(f"Error: Required module not available: {e}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        # Quick benchmark command
        if raw_lower.startswith("benchmark"):
            try:
                from .benchmarks import FEYNMAN_EQUATIONS, quick_benchmark

                parts = raw.split()
                n_eq = 5
                if len(parts) > 1:
                    try:
                        n_eq = int(parts[1])
                    except ValueError:
                        pass

                print(f"Running benchmark on {n_eq} Feynman equations...")
                print(f"(Total available: {len(FEYNMAN_EQUATIONS)} equations)")

                suite = quick_benchmark(method="lasso", n_equations=n_eq, timeout=10)
                print(f"\nSuccess rate: {suite.success_rate:.1%}")
            except ImportError as e:
                print(f"Error: Required module not available: {e}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if raw.lower() in ("clearcache", "clear cache"):
            from .worker import clear_caches

            clear_caches()
            print("Caches cleared.")
            continue
        if raw.lower().startswith("showcache") or raw.lower() in (
            "show cache",
            "cache",
        ):
            try:
                from .cache_manager import get_persistent_cache

                # Parse command arguments
                parts = raw.lower().split()
                show_all = len(parts) > 1 and (parts[1] in ("all", "--all"))

                cache = get_persistent_cache()
                eval_cache = cache.get("eval_cache", {})
                subexpr_cache = cache.get("subexpr_cache", {})

                print("\n=== Cache Status ===")
                print(f"Evaluation cache entries: {len(eval_cache)}")
                print(f"Sub-expression cache entries: {len(subexpr_cache)}")
                print(f"Total entries: {len(eval_cache) + len(subexpr_cache)}")

                if subexpr_cache:
                    if show_all:
                        print("\n=== Sub-expression Cache (all entries) ===")
                        for _i, (expr, entry) in enumerate(subexpr_cache.items()):
                            # Support both old format (string) and new format (dict)
                            if isinstance(entry, dict):
                                value = entry.get("value", "")
                                cache_time = entry.get("time")
                                value_formatted = _format_number_no_trailing_zeros(
                                    value
                                )
                                if cache_time is not None:
                                    print(
                                        f"  {expr} → {value_formatted:20} [time: {cache_time:.4f}s]"
                                    )
                                else:
                                    print(f"  {expr} → {value_formatted}")
                            else:
                                # Old format
                                value_formatted = _format_number_no_trailing_zeros(
                                    entry
                                )
                                print(f"  {expr} → {value_formatted}")
                    else:
                        print("\n=== Sub-expression Cache (showing first 20) ===")
                        for _i, (expr, entry) in enumerate(
                            list(subexpr_cache.items())[:20]
                        ):
                            # Support both old format (string) and new format (dict)
                            if isinstance(entry, dict):
                                value = entry.get("value", "")
                                cache_time = entry.get("time")
                                value_formatted = _format_number_no_trailing_zeros(
                                    value
                                )
                                if cache_time is not None:
                                    print(
                                        f"  {expr} → {value_formatted:20} [time: {cache_time:.4f}s]"
                                    )
                                else:
                                    print(f"  {expr} → {value_formatted}")
                            else:
                                # Old format
                                value_formatted = _format_number_no_trailing_zeros(
                                    entry
                                )
                                print(f"  {expr} → {value_formatted}")
                        if len(subexpr_cache) > 20:
                            print(f"  ... and {len(subexpr_cache) - 20} more entries")
                            print("  Use 'showcache all' to see all entries")

                if eval_cache:
                    if show_all:
                        print("\n=== Evaluation Cache (all entries) ===")
                        for _i, (expr, entry) in enumerate(eval_cache.items()):
                            try:
                                import json

                                # Extract expression from hash:expression format
                                display_expr = expr
                                if ":" in expr:
                                    # Format is "hash:expression", extract just the expression
                                    parts = expr.split(":", 1)
                                    if (
                                        len(parts) == 2 and len(parts[0]) == 32
                                    ):  # MD5 hash is 32 chars
                                        display_expr = parts[1]

                                # Support both old format (string) and new format (dict)
                                if isinstance(entry, dict):
                                    result_json = entry.get("result", "{}")
                                    cache_time = entry.get("time")
                                else:
                                    result_json = entry  # Old format
                                    cache_time = None

                                result_data = json.loads(result_json)
                                result_str = result_data.get("result", "N/A")
                                result_formatted = _format_number_no_trailing_zeros(
                                    result_str
                                )
                                if len(result_formatted) > 50:
                                    result_formatted = result_formatted[:47] + "..."
                                if cache_time is not None:
                                    print(
                                        f"  {display_expr[:50]:50} → {result_formatted:20} [time: {cache_time:.4f}s]"
                                    )
                                else:
                                    print(
                                        f"  {display_expr[:50]:50} → {result_formatted}"
                                    )
                            except (json.JSONDecodeError, KeyError):
                                print(f"  {display_expr[:50]:50} → [cached result]")
                    else:
                        print("\n=== Evaluation Cache (showing first 10) ===")
                        for _i, (expr, entry) in enumerate(
                            list(eval_cache.items())[:10]
                        ):
                            try:
                                import json

                                # Extract expression from hash:expression format
                                display_expr = expr
                                if ":" in expr:
                                    # Format is "hash:expression", extract just the expression
                                    parts = expr.split(":", 1)
                                    if (
                                        len(parts) == 2 and len(parts[0]) == 32
                                    ):  # MD5 hash is 32 chars
                                        display_expr = parts[1]

                                # Support both old format (string) and new format (dict)
                                if isinstance(entry, dict):
                                    result_json = entry.get("result", "{}")
                                    cache_time = entry.get("time")
                                else:
                                    result_json = entry  # Old format
                                    cache_time = None

                                result_data = json.loads(result_json)
                                result_str = result_data.get("result", "N/A")
                                result_formatted = _format_number_no_trailing_zeros(
                                    result_str
                                )
                                if len(result_formatted) > 50:
                                    result_formatted = result_formatted[:47] + "..."
                                if cache_time is not None:
                                    print(
                                        f"  {display_expr[:50]:50} → {result_formatted:20} [time: {cache_time:.4f}s]"
                                    )
                                else:
                                    print(
                                        f"  {display_expr[:50]:50} → {result_formatted}"
                                    )
                            except (json.JSONDecodeError, KeyError):
                                print(f"  {display_expr[:50]:50} → [cached result]")
                        if len(eval_cache) > 10:
                            print(f"  ... and {len(eval_cache) - 10} more entries")
                            print("  Use 'showcache all' to see all entries")
                else:
                    print("\nNo cached entries yet.")
            except ImportError:
                print("Cache manager not available.")
            continue
        if raw.lower().startswith("savecache"):
            try:
                from .cache_manager import export_cache_to_file, get_persistent_cache

                parts = raw.split(None, 1)
                if len(parts) > 1:
                    file_path = parts[1].strip()
                    # Remove quotes if present
                    if (file_path.startswith('"') and file_path.endswith('"')) or (
                        file_path.startswith("'") and file_path.endswith("'")
                    ):
                        file_path = file_path[1:-1]
                else:
                    # Default to cache_backup.json in current directory
                    file_path = "cache_backup.json"

                cache = get_persistent_cache()
                total_entries = len(cache.get("eval_cache", {})) + len(
                    cache.get("subexpr_cache", {})
                )

                if export_cache_to_file(file_path):
                    print(f"Cache exported successfully to: {file_path}")
                    print(f"  ({total_entries} total entries saved)")
                else:
                    print(f"Error: Failed to export cache to {file_path}")
            except Exception as e:
                print(f"Error saving cache: {e}")
            continue
        if raw.lower().startswith("loadcache"):
            try:
                import os  # noqa: F811

                from .cache_manager import (
                    get_persistent_cache,
                    import_cache_from_file,
                    replace_cache_from_file,
                )

                parts = raw.split()
                replace_mode = False
                file_path = None

                # Parse arguments: loadcache [replace] <file_path>
                if len(parts) > 1:
                    if parts[1].lower() == "replace":
                        replace_mode = True
                        if len(parts) > 2:
                            file_path = parts[2].strip()
                            if (
                                file_path.startswith('"') and file_path.endswith('"')
                            ) or (
                                file_path.startswith("'") and file_path.endswith("'")
                            ):
                                file_path = file_path[1:-1]
                    else:
                        file_path = parts[1].strip()
                        if (file_path.startswith('"') and file_path.endswith('"')) or (
                            file_path.startswith("'") and file_path.endswith("'")
                        ):
                            file_path = file_path[1:-1]

                if not file_path:
                    # Default to cache_backup.json in current directory
                    file_path = "cache_backup.json"

                if not os.path.exists(file_path):
                    print(f"Error: File not found: {file_path}")
                    continue

                if replace_mode:
                    if replace_cache_from_file(file_path):
                        cache = get_persistent_cache()
                        total_entries = len(cache.get("eval_cache", {})) + len(
                            cache.get("subexpr_cache", {})
                        )
                        print(f"Cache replaced from: {file_path}")
                        print(f"  ({total_entries} total entries loaded)")
                    else:
                        print(f"Error: Failed to replace cache from {file_path}")
                else:
                    if import_cache_from_file(file_path):
                        cache = get_persistent_cache()
                        total_entries = len(cache.get("eval_cache", {})) + len(
                            cache.get("subexpr_cache", {})
                        )
                        print(f"Cache merged from: {file_path}")
                        print(f"  ({total_entries} total entries in cache after merge)")
                    else:
                        print(f"Error: Failed to import cache from {file_path}")
            except Exception as e:
                print(f"Error loading cache: {e}")
            continue
        if raw.lower().startswith("timing"):
            parts = raw.lower().split()
            if len(parts) == 1 or (
                len(parts) == 2 and parts[1] in ("on", "enable", "1")
            ):
                _timing_enabled = True
                print("Timing enabled. Calculation time will be displayed.")
            elif len(parts) == 2 and parts[1] in ("off", "disable", "0"):
                _timing_enabled = False
                print("Timing disabled.")
            else:
                print("Usage: timing [on|off]")
            continue
        if raw.lower().startswith("cachehits") or raw.lower().startswith("cache-hits"):
            parts = raw.lower().split()
            if len(parts) == 1 or (
                len(parts) == 2 and parts[1] in ("on", "enable", "1")
            ):
                _cache_hits_enabled = True
                print(
                    "Cache hits tracking enabled. Will show which expressions used cache."
                )
            elif len(parts) == 2 and parts[1] in ("off", "disable", "0"):
                _cache_hits_enabled = False
                print("Cache hits tracking disabled.")
            else:
                print("Usage: cachehits [on|off]")
            continue
        if raw.lower() in ("showcachehits", "show-cache-hits", "cachehits show"):
            try:
                from .cache_manager import clear_cache_hits, get_cache_hits

                hits = get_cache_hits()
                if hits:
                    print("\n=== Cache Hits (recent computations) ===")
                    for expr, cache_type in hits:
                        cache_type_name = (
                            "evaluation" if cache_type == "eval" else "sub-expression"
                        )
                        print(f"  {expr} (from {cache_type_name} cache)")
                    clear_cache_hits()  # Clear after showing
                else:
                    print("No cache hits recorded. Run a computation first.")
            except ImportError:
                print("Cache manager not available.")
            continue
        if raw.startswith("--eval"):
            parts = raw.split(None, 1)
            if len(parts) == 1:
                print("Usage: --eval <expression>")
                continue
            expr = parts[1].strip()
            if (expr.startswith('"') and expr.endswith('"')) or (
                expr.startswith("'") and expr.endswith("'")
            ):
                expr = expr[1:-1]

            # Check for function finding command FIRST (before other processing)
            # This includes both explicit "find" keyword and multiple function assignments
            is_find_command = False
            find_func_cmd = None

            # Check for explicit "find" keyword
            if "find" in expr.lower():

                try:
                    from .function_manager import (
                        find_function_from_data,
                        parse_find_function_command,
                    )

                    find_func_cmd = parse_find_function_command(expr)

                    if find_func_cmd is not None:
                        is_find_command = True
                except Exception as e:
                    logger.exception("Error parsing find function command")
                    # If "find" is in the expression but parsing failed, don't try to parse as normal expression
                    print(f"Error: Failed to parse function finding command: {e}")
                    continue

            # If no explicit "find" keyword, check for multiple function assignments (like main REPL)
            if not is_find_command:
                try:
                    from .function_manager import (  # noqa: F811
                        find_function_from_data,
                        parse_find_function_command,
                    )

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
                                find_func_cmd = parse_find_function_command(
                                    expr_with_find
                                )
                                if find_func_cmd is not None:
                                    is_find_command = True
                                    # Use the modified expression for parsing
                                    expr = expr_with_find
                except Exception:
                    logger.exception(
                        "Error detecting function finding from multiple assignments"
                    )

            # Process function finding if detected
            if is_find_command and find_func_cmd is not None:
                try:
                    from .function_manager import (  # noqa: F811
                        define_function,
                        find_function_from_data,
                        parse_function_definition,
                    )

                    func_name, param_names = find_func_cmd
                    find_pattern = rf"find\s+{re.escape(func_name)}\s*\([^)]*\)"
                    data_str = re.sub(
                        find_pattern, "", expr, flags=re.IGNORECASE
                    ).strip()
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

                    # Update data_str with remaining parts
                    data_str = ", ".join(remaining_parts)

                    # Parse data points using the same logic as main REPL
                    data_points = []

                    # Check if this is an inverse solving case (e.g., "find f(x,y) = 5")
                    # vs function finding case (e.g., "f(1)=1, f(2)=4, find f(x)")
                    is_inverse_solve = False
                    target_value = None

                    # If data_str starts with "=" it's likely an inverse solve: "find f(x,y) = 5"

                    if data_str.strip().startswith("="):
                        is_inverse_solve = True
                        target_value = data_str.strip()[1:].strip()

                    elif not any(
                        f"{func_name}(" in part
                        for part in split_top_level_commas(data_str)
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
                        from .function_manager import _function_registry

                        if func_name in _function_registry:
                            from .solver import solve_inverse_function

                            result = solve_inverse_function(
                                func_name, target_value, param_names
                            )
                            _format_inverse_solutions(
                                result, func_name, param_names, target_value
                            )
                            continue  # Skip further processing
                        else:
                            print(
                                f"Error: Function '{func_name}' is not defined. Define it first or provide data points."
                            )
                            continue

                    # Otherwise, parse as data points for function finding
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

                                    arg_expr = _parse_preprocessed(arg_stripped)
                                    try:
                                        arg_val = float(sp.N(arg_expr))
                                    except (ValueError, TypeError):
                                        arg_val = arg_expr
                                    args.append(arg_val)
                                except Exception:
                                    break

                            if len(args) == len(param_names):
                                try:
                                    if value_str_preserved:
                                        value = value_str_preserved
                                    else:
                                        value_expr = _parse_preprocessed(value_str)
                                        try:
                                            value = float(sp.N(value_expr))
                                        except (ValueError, TypeError):
                                            value = value_expr

                                    # Preserve strings for exact precision (same as main REPL)
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
                                                    final_args.append(
                                                        float(sp.N(arg_val))
                                                    )
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
                                                value,
                                                (sp.Rational, sp.Integer, sp.Float),
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
                                except Exception as e:
                                    logger.debug(
                                        f"Error parsing value '{value_str}': {e}"
                                    )

                    if data_points:
                        success, func_str, factored_form, error_msg = (
                            find_function_from_data(data_points, param_names)
                        )
                        if success:
                            if output_format == "json":
                                result = {"ok": True, "function": func_str}
                                if factored_form and isinstance(factored_form, str):
                                    result["factored_form"] = factored_form
                                print(json.dumps(result))
                            else:
                                params_str = ", ".join(param_names)
                                # Display the function string as-is (it's already in a readable format)
                                # Note: We used to remove * for display, but that made it unparseable
                                # The function string is already human-readable
                                print(f"{func_name}({params_str}) = {func_str}")
                                if factored_form and isinstance(factored_form, str):
                                    print(
                                        f"Equivalent: {func_name}({params_str}) = {factored_form}"
                                    )
                                print(
                                    f"Function '{func_name}' is now available. You can call it like: {func_name}(values)"
                                )
                        else:
                            print(f"Error: Error finding function: {error_msg}")
                    else:
                        print("Error: No valid data points found for function finding")
                    continue  # Skip further processing
                except Exception as e:
                    logger.exception("Error processing function finding in eval mode")
                    print(f"Error: Failed to process function finding: {e}")
                    continue

            try:
                import time

                start_time = time.perf_counter()
                # Check for unterminated quotes before processing
                if expr.count('"') % 2 != 0 or expr.count("'") % 2 != 0:
                    print(
                        "Error: Unmatched quotes detected. Check that all opening quotes have matching closing quotes."
                    )
                    if _timing_enabled:
                        print(f"[Time: {time.perf_counter() - start_time:.4f}s]")
                    continue

                # Check for assignment with equation mixed (e.g., "a=(expr=0)")
                if "," in expr:
                    parts = split_top_level_commas(expr)
                    for part in parts:
                        if "=" in part:
                            # Check if this looks like an assignment with nested equation
                            assign_parts = part.split("=", 1)
                            if len(assign_parts) == 2 and "=" in assign_parts[1]:
                                print(
                                    f"Error: Cannot use assignment '=' inside another assignment. Found: '{part}'"
                                )
                                print(
                                    "Hint: Separate assignments and equations. Example: Use 'a = expression' then solve 'equation = 0' separately."
                                )
                                if _timing_enabled:
                                    print(
                                        f"[Time: {time.perf_counter() - start_time:.4f}s]"
                                    )
                                continue

                # Check for function definition or mixed definitions/calls in --eval mode
                if "," in expr:
                    parts = split_top_level_commas(expr)
                    handled_all = True
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        # Check if it's a function definition
                        try:
                            from .function_manager import (
                                define_function,
                                parse_function_definition,
                            )

                            func_def = parse_function_definition(part)
                            if func_def is not None:
                                func_name, params, body = func_def
                                try:
                                    define_function(func_name, params, body)
                                    params_str = ", ".join(params) if params else ""
                                    if output_format == "json":
                                        print(
                                            json.dumps(
                                                {
                                                    "ok": True,
                                                    "function_defined": func_name,
                                                    "params": params,
                                                    "body": body,
                                                }
                                            )
                                        )
                                    else:
                                        print(
                                            f"Function '{func_name}({params_str})' defined as: {body}"
                                        )
                                    continue
                                except ValidationError as e:
                                    print(f"Error: {e.message}")
                                    handled_all = False
                                    break
                        except Exception:
                            pass

                        # If not a function definition, evaluate it
                        eva = _evaluate_safely(part)
                        elapsed = time.perf_counter() - start_time
                        if eva.get("ok"):
                            res_str = eva.get("result")
                            if output_format == "json":
                                print(json.dumps({"ok": True, "result": res_str}))
                            else:
                                formatted_res = _format_special_values(res_str)
                                print(f"{part} = {format_superscript(formatted_res)}")
                        else:
                            print(f"Error: {eva.get('error', 'Unknown error')}")
                            handled_all = False

                    if handled_all:
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                        continue

                # Check for single function definition in --eval mode
                try:
                    from .function_manager import (
                        define_function,
                        parse_function_definition,
                    )

                    func_def = parse_function_definition(expr)
                    if func_def is not None:
                        func_name, params, body = func_def
                        try:
                            define_function(func_name, params, body)
                            params_str = ", ".join(params) if params else ""
                            if output_format == "json":
                                print(
                                    json.dumps(
                                        {
                                            "ok": True,
                                            "function_defined": func_name,
                                            "params": params,
                                            "body": body,
                                        }
                                    )
                                )
                            else:
                                print(
                                    f"Function '{func_name}({params_str})' defined as: {body}"
                                )
                            if _timing_enabled:
                                print(
                                    f"[Time: {time.perf_counter() - start_time:.4f}s]"
                                )
                            continue
                        except ValidationError as e:
                            print(f"Error: {e.message}")
                            if _timing_enabled:
                                print(
                                    f"[Time: {time.perf_counter() - start_time:.4f}s]"
                                )
                            continue
                except Exception:
                    pass

                if any(op in expr for op in ("<", ">", "<=", ">=")):
                    res = solve_inequality(expr, None)
                    elapsed = time.perf_counter() - start_time
                    print_result_pretty(res, output_format=output_format)
                    if _timing_enabled:
                        print(f"[Time: {elapsed:.4f}s]")
                elif "=" in expr:
                    # Clear cache hits tracking before new computation
                    if _cache_hits_enabled:
                        try:
                            from .cache_manager import clear_cache_hits

                            clear_cache_hits()
                        except ImportError:
                            pass

                    # Check if all assignments are to the same variable FIRST
                    # This handles cases like "x = 1 % 2, x=3 % 6, x=3 % 7" where we want to evaluate each expression
                    if "," in expr:
                        parts_eval = split_top_level_commas(expr)
                        all_assign_same_var_eval = all(
                            "=" in p and VAR_NAME_RE.match(p.split("=", 1)[0].strip())
                            for p in parts_eval
                        )
                        if all_assign_same_var_eval and len(parts_eval) > 1:
                            assigned_vars_eval = [
                                p.split("=", 1)[0].strip()
                                for p in parts_eval
                                if "=" in p
                            ]
                            if (
                                len(assigned_vars_eval) > 1
                                and len(set(assigned_vars_eval)) == 1
                            ):
                                # All assignments are to the same variable
                                var = assigned_vars_eval[0]
                                # Try to solve as system of congruences first
                                solved, exit_code = _solve_modulo_system_if_applicable(
                                    parts_eval, var, output_format
                                )
                                if solved:
                                    if _timing_enabled:
                                        print(
                                            f"[Time: {time.perf_counter() - start_time:.4f}s]"
                                        )
                                    return exit_code

                                # If not all modulo or CRT solving failed, evaluate each expression separately
                                for p in parts_eval:
                                    if "=" in p:
                                        left, right = p.split("=", 1)
                                        var = left.strip()
                                        rhs = right.strip() or "0"
                                        # Evaluate the RHS expression (like "1 % 2")
                                        res = _evaluate_safely(rhs)
                                        if not res.get("ok"):
                                            print(
                                                f"Error evaluating '{var} = {rhs}': {res.get('error')}"
                                            )
                                            continue
                                        try:
                                            # Format and print the result
                                            val_str = res.get("result", "")
                                            approx_str = res.get("approx", "")
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
                                                        print(
                                                            f"  Decimal: {approx_str}"
                                                        )
                                                else:
                                                    print(f"{var} = {val_str}")
                                        except Exception as e:
                                            print(
                                                f"Error formatting result for '{var} = {rhs}': {e}"
                                            )
                                            continue
                                if _timing_enabled:
                                    print(
                                        f"[Time: {time.perf_counter() - start_time:.4f}s]"
                                    )
                                continue

                    # Check for function finding patterns BEFORE equation solving (to avoid π-style output)
                    # This handles both single and comma-separated function finding patterns
                    func_finding_pattern_eval = (
                        r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]+\)\s*=\s*[^,]+"
                    )
                    if re.search(func_finding_pattern_eval, expr):
                        matches_eval = list(
                            re.finditer(func_finding_pattern_eval, expr)
                        )
                        has_numeric_args_eval = False
                        func_groups_eval = {}
                        for m in matches_eval:
                            func_name_eval = m.group(1)
                            args_match_eval = re.search(
                                rf"{re.escape(func_name_eval)}\s*\(([^)]+)\)",
                                m.group(0),
                            )
                            if args_match_eval:
                                args_str_eval = args_match_eval.group(1).strip()
                                if re.search(r"[-+]?\d", args_str_eval):
                                    has_numeric_args_eval = True
                                    value_match_eval = re.search(
                                        r"=\s*(.+)", m.group(0)
                                    )
                                    if value_match_eval:
                                        value_str_eval = value_match_eval.group(
                                            1
                                        ).strip()
                                        if func_name_eval not in func_groups_eval:
                                            func_groups_eval[func_name_eval] = []
                                        func_groups_eval[func_name_eval].append(
                                            (args_str_eval, value_str_eval)
                                        )

                        if has_numeric_args_eval and func_groups_eval:
                            # Process each function finding pattern
                            try:
                                from .function_manager import (
                                    define_function,
                                    find_function_from_data,
                                )

                                processed_any = False
                                for (
                                    func_name_eval,
                                    data_points_list,
                                ) in func_groups_eval.items():
                                    # Process all data points for this function
                                    param_names_eval = ["x"]
                                    data_points_eval = []
                                    for (
                                        args_str_eval,
                                        value_str_eval,
                                    ) in data_points_list:
                                        # Parse arguments
                                        arg_list_eval = []
                                        for arg in args_str_eval.split(","):
                                            try:
                                                arg_list_eval.append(float(arg.strip()))
                                            except ValueError:
                                                arg_list_eval.append(arg.strip())

                                        # Parse value
                                        try:
                                            value_eval = float(value_str_eval)
                                        except ValueError:
                                            value_eval = value_str_eval

                                        data_points_eval.append(
                                            (
                                                (
                                                    [arg_list_eval[0]]
                                                    if len(arg_list_eval) == 1
                                                    else arg_list_eval
                                                ),
                                                value_eval,
                                            )
                                        )

                                    # Find function from all data points
                                    success, func_str, factored_form, error_msg = (
                                        find_function_from_data(
                                            data_points_eval, param_names_eval
                                        )
                                    )
                                    if success:
                                        print(f"{func_name_eval}(x) = {func_str}")
                                        if factored_form:
                                            print(
                                                f"Equivalent: {func_name_eval}(x) = {factored_form}"
                                            )
                                        try:
                                            define_function(
                                                func_name_eval,
                                                param_names_eval,
                                                func_str,
                                            )
                                            print(
                                                f"Function '{func_name_eval}' is now available."
                                            )
                                            processed_any = True
                                        except Exception as e:
                                            print(
                                                f"Warning: Could not define function automatically: {e}"
                                            )
                                    else:
                                        print(
                                            f"Error finding {func_name_eval}: {error_msg}"
                                        )

                                # After processing all function finding patterns, check if there are remaining parts to evaluate
                                if processed_any:
                                    # Remove processed function finding patterns and evaluate remaining parts
                                    remaining_expr = expr
                                    for func_name_eval in func_groups_eval.keys():
                                        # Remove all patterns for this function
                                        remaining_expr = re.sub(
                                            rf"{re.escape(func_name_eval)}\s*\([^)]+\)\s*=\s*[^,]+",
                                            "",
                                            remaining_expr,
                                        )
                                    remaining_expr = (
                                        re.sub(r",\s*,+", ",", remaining_expr)
                                        .strip(",")
                                        .strip()
                                    )

                                    if remaining_expr:
                                        # Evaluate remaining parts (like f(g(2)))
                                        # split_top_level_commas is already imported at module level
                                        remaining_parts = split_top_level_commas(
                                            remaining_expr
                                        )
                                        for part in remaining_parts:
                                            part = part.strip()
                                            if part:
                                                try:
                                                    eva = _evaluate_safely(part)
                                                    if eva.get("ok"):
                                                        print(
                                                            f"{part} = {format_superscript(eva.get('result'))}"
                                                        )
                                                    else:
                                                        print(
                                                            f"Error: {eva.get('error', 'Unknown error')}"
                                                        )
                                                except Exception as e:
                                                    print(
                                                        f"Error evaluating '{part}': {e}"
                                                    )
                                    continue
                            except Exception as e:
                                print(f"Error processing function finding: {e}")
                                # Fall through to normal processing
                                pass

                        # If we only found one function finding pattern, process it
                        if (
                            has_numeric_args_eval
                            and func_name_eval
                            and args_str_eval
                            and value_str_eval
                            and len(matches_eval) == 1
                        ):
                            # Process as function finding
                            try:
                                from .function_manager import (
                                    define_function,
                                    find_function_from_data,
                                )

                                # Parse arguments
                                arg_list_eval = []
                                for arg in args_str_eval.split(","):
                                    try:
                                        arg_list_eval.append(float(arg.strip()))
                                    except ValueError:
                                        arg_list_eval.append(arg.strip())

                                # Parse value
                                try:
                                    value_eval = float(value_str_eval)
                                except ValueError:
                                    value_eval = value_str_eval

                                # Find function
                                param_names_eval = ["x"]
                                data_points_eval = [
                                    (
                                        (
                                            [arg_list_eval[0]]
                                            if len(arg_list_eval) == 1
                                            else arg_list_eval
                                        ),
                                        value_eval,
                                    )
                                ]
                                success, func_str, factored_form, error_msg = (
                                    find_function_from_data(
                                        data_points_eval, param_names_eval
                                    )
                                )
                                if success:
                                    print(f"{func_name_eval}(x) = {func_str}")
                                    if factored_form:
                                        print(
                                            f"Equivalent: {func_name_eval}(x) = {factored_form}"
                                        )
                                    try:
                                        define_function(
                                            func_name_eval, param_names_eval, func_str
                                        )
                                        print(
                                            f"Function '{func_name_eval}' is now available."
                                        )
                                    except Exception as e:
                                        print(
                                            f"Warning: Could not define function automatically: {e}"
                                        )
                                    continue
                                else:
                                    print(f"Error: {error_msg}")
                                    continue
                            except Exception as e:
                                print(f"Error processing function finding: {e}")
                                continue

                    # Check again for same-variable assignments before solving as system
                    pts = split_top_level_commas(expr)
                    if len(pts) > 1:
                        # Check if all assignments are to the same variable (re-check here in case earlier check missed it)
                        all_assign_check = all(
                            "=" in p and VAR_NAME_RE.match(p.split("=", 1)[0].strip())
                            for p in pts
                        )
                        if all_assign_check:
                            assigned_vars_check = [
                                p.split("=", 1)[0].strip() for p in pts if "=" in p
                            ]
                            if (
                                len(assigned_vars_check) > 1
                                and len(set(assigned_vars_check)) == 1
                            ):
                                # All assignments are to the same variable
                                var = assigned_vars_check[0]
                                # Try to solve as system of congruences first
                                solved, exit_code = _solve_modulo_system_if_applicable(
                                    pts, var, output_format
                                )
                                if solved:
                                    if _timing_enabled:
                                        print(
                                            f"[Time: {time.perf_counter() - start_time:.4f}s]"
                                        )
                                    return exit_code

                                # If not all modulo or CRT solving failed, evaluate each expression separately
                                for p in pts:
                                    if "=" in p:
                                        left, right = p.split("=", 1)
                                        var = left.strip()
                                        rhs = right.strip() or "0"
                                        # Evaluate the RHS expression (like "1 % 2")
                                        res = _evaluate_safely(rhs)
                                        if not res.get("ok"):
                                            print(
                                                f"Error evaluating '{var} = {rhs}': {res.get('error')}"
                                            )
                                            continue
                                        try:
                                            # Format and print the result
                                            val_str = res.get("result", "")
                                            approx_str = res.get("approx", "")
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
                                                        print(
                                                            f"  Decimal: {approx_str}"
                                                        )
                                                else:
                                                    print(f"{var} = {val_str}")
                                        except Exception as e:
                                            print(
                                                f"Error formatting result for '{var} = {rhs}': {e}"
                                            )
                                            continue
                                if _timing_enabled:
                                    print(
                                        f"[Time: {time.perf_counter() - start_time:.4f}s]"
                                    )
                                continue

                        res = solve_system(expr, None)
                        elapsed = time.perf_counter() - start_time
                        print_result_pretty(res, output_format=output_format)
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                        # Show cache hits if enabled
                        if _cache_hits_enabled:
                            try:
                                from .cache_manager import get_cache_hits

                                hits = get_cache_hits()
                                if hits:
                                    print("\n[Cache hits used:]")
                                    for hit_expr, cache_type in hits:
                                        cache_type_name = (
                                            "eval"
                                            if cache_type == "eval"
                                            else "subexpr"
                                        )
                                        print(f"  {hit_expr} ({cache_type_name})")
                            except ImportError:
                                pass
                    else:
                        res = solve_single_equation(expr, None)
                        elapsed = time.perf_counter() - start_time
                        print_result_pretty(res, output_format=output_format)
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                        # Show cache hits if enabled
                        if _cache_hits_enabled:
                            try:
                                from .cache_manager import get_cache_hits

                                hits = get_cache_hits()
                                if hits:
                                    print("\n[Cache hits used:]")
                                    for hit_expr, cache_type in hits:
                                        cache_type_name = (
                                            "eval"
                                            if cache_type == "eval"
                                            else "subexpr"
                                        )
                                        print(f"  {hit_expr} ({cache_type_name})")
                            except ImportError:
                                pass
                else:
                    # Note: evaluate_safely clears cache hits at the start and captures them
                    # during preprocessing and evaluation, then returns them in the result
                    eva = _evaluate_safely(expr)
                    elapsed = time.perf_counter() - start_time
                    # Show cache hits if enabled
                    if _cache_hits_enabled:
                        try:
                            # Get cache hits from result (evaluate_safely captures and attaches them)
                            cache_hits = eva.get("cache_hits", [])
                            if cache_hits:
                                print("\n[Cache hits used:]")
                                for hit_expr, cache_type in cache_hits:
                                    cache_type_name = (
                                        "eval" if cache_type == "eval" else "subexpr"
                                    )
                                    print(f"  {hit_expr} ({cache_type_name})")
                        except Exception:
                            # Don't let cache hits display failure break the evaluation
                            # Silently continue - cache hits are optional
                            pass
                    if not eva.get("ok"):
                        error_msg = str(
                            eva.get("error", "Unknown error") or "Unknown error"
                        )
                        error_code = eva.get("error_code", "UNKNOWN_ERROR")

                        # Provide helpful hints based on error code
                        if error_code == "COMMAND_IN_EXPRESSION":
                            print(f"Error: {error_msg}")
                            print(
                                "Hint: Commands must be entered on separate lines. "
                                "Each command or expression should be on its own line."
                            )
                        elif error_code == "SYNTAX_ERROR":
                            if (
                                "unterminated" in error_msg.lower()
                                or "unmatched" in error_msg.lower()
                            ):
                                print(f"Error: {error_msg}")
                                print(
                                    "Hint: Check that all quotes, parentheses, and brackets are properly matched."
                                )
                            else:
                                print(f"Error: {error_msg}")
                        elif error_code == "PARSE_ERROR":
                            print(f"{error_msg}")
                            # Additional hints already included in error message from worker
                        elif error_code == "INCOMPLETE_EXPRESSION":
                            print(f"{error_msg}")
                        elif error_code == "EMPTY_INPUT":
                            print(f"{error_msg}")
                        elif error_code == "INVALID_NAME":
                            print(f"{error_msg}")
                        else:
                            print(f"Error: {error_msg}")

                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                    else:
                        res_str = eva.get("result")
                        # Format to remove trailing zeros
                        res_str_formatted = _format_number_no_trailing_zeros(res_str)
                        print(f"{expr} = {format_superscript(res_str_formatted)}")
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                        try:
                            parsed = _parse_preprocessed(res_str)
                            expanded = sp.expand(parsed)
                            if str(expanded) != str(parsed):
                                print(f"Expanded: {format_solution(expanded)}")
                        except (
                            ParseError,
                            ValidationError,
                            ValueError,
                            TypeError,
                            AttributeError,
                        ):
                            # Expected errors for some expressions - silently skip expansion
                            pass
                        if eva.get("approx"):
                            approx_formatted = _format_number_no_trailing_zeros(
                                eva.get("approx")
                            )
                            # π-fraction conversion (Casio-style) disabled
                            print(f"Decimal: {approx_formatted}")
            except Exception as e:
                # Log full error but show clean message to user
                try:
                    from .logging_config import get_logger

                    logger = get_logger("cli")
                    logger.error(f"Error handling --eval in REPL: {e}", exc_info=True)
                except ImportError:
                    pass

                # Show clean error message without traceback
                error_str = str(e)
                if (
                    "TokenError" in str(type(e).__name__)
                    or "unterminated" in error_str.lower()
                ):
                    print(
                        "Error: Syntax error detected. Check that all quotes and parentheses are properly matched."
                    )
                elif "assign" in error_str.lower():
                    print(f"Error: {error_str}")
                    print(
                        "Hint: Use '==' for equations and '=' for assignments. Don't mix them in a single expression."
                    )
                else:
                    print(f"Error: {error_str}")
            continue
        cmd = raw.lower()
        if cmd in ("quit", "exit"):
            print("Goodbye.")
            break
        if cmd in ("help", "?", "--help"):
            print_help_text()
            continue
        if cmd in ("health", "healthcheck", "health-check"):
            exit_code = _health_check()
            continue

        # Plot command: plot <expression> [variable] [x_min] [x_max] [--save filename]
        if raw.lower().startswith("plot"):
            try:
                from .plotting import plot_function

                parts = raw.split(None, 1)
                if len(parts) < 2:
                    print(
                        "Usage: plot <expression> [variable=x] [x_min=-10] [x_max=10] [--save filename]"
                    )
                    print("Example: plot x^2")
                    print("Example: plot sin(x), x_min=-pi, x_max=pi")
                    print("Example: plot x^2, --save plot.png")
                    continue

                # Parse the expression and optional parameters
                expr_part = parts[1].strip()

                # Default values
                variable = "x"
                x_min = -10.0
                x_max = 10.0
                save_file = None
                points = 100

                # Check for --save flag
                if "--save" in expr_part:
                    save_idx = expr_part.find("--save")
                    save_part = expr_part[save_idx:].split(None, 1)
                    if len(save_part) > 1:
                        save_file = save_part[1].strip()
                        # Remove quotes if present
                        if (save_file.startswith('"') and save_file.endswith('"')) or (
                            save_file.startswith("'") and save_file.endswith("'")
                        ):
                            save_file = save_file[1:-1]
                    expr_part = expr_part[:save_idx].strip()

                # Parse comma-separated parameters
                expr_parts = [p.strip() for p in expr_part.split(",")]
                expression = expr_parts[0]

                # Parse optional parameters
                for param in expr_parts[1:]:
                    if "=" in param:
                        key, value = param.split("=", 1)
                        key = key.strip().lower()
                        value = value.strip()
                        try:
                            if key == "variable":
                                variable = value
                            elif key == "x_min":
                                # Evaluate expression (e.g., "-pi", "2*pi") before converting to float
                                try:
                                    from .parser import parse_preprocessed

                                    eval_result = _evaluate_safely(value)
                                    if eval_result.get("ok"):
                                        parsed_value = parse_preprocessed(
                                            eval_result["result"]
                                        )
                                        x_min = float(sp.N(parsed_value))
                                    else:
                                        # Fallback: try direct float conversion
                                        x_min = float(value)
                                except Exception:
                                    # Fallback: try direct float conversion
                                    x_min = float(value)
                            elif key == "x_max":
                                # Evaluate expression (e.g., "pi", "2*pi") before converting to float
                                try:
                                    from .parser import parse_preprocessed

                                    eval_result = _evaluate_safely(value)
                                    if eval_result.get("ok"):
                                        parsed_value = parse_preprocessed(
                                            eval_result["result"]
                                        )
                                        x_max = float(sp.N(parsed_value))
                                    else:
                                        # Fallback: try direct float conversion
                                        x_max = float(value)
                                except Exception:
                                    # Fallback: try direct float conversion
                                    x_max = float(value)
                            elif key == "points":
                                points = int(value)
                            else:
                                print(f"Warning: Unknown parameter '{key}', ignoring")
                        except ValueError:
                            print(
                                f"Error: Invalid value for parameter '{key}': {value}"
                            )
                            print(
                                "  Hint: Use mathematical expressions like 'pi', '-pi', '2*pi', etc."
                            )
                            continue

                # Plot the function (with save support if requested)
                if save_file:
                    # Import here to avoid issues if matplotlib not available
                    try:
                        # Set non-GUI backend for saving (no Tkinter needed)
                        import matplotlib

                        matplotlib.use("Agg")  # Non-GUI backend
                        import matplotlib.pyplot as plt
                        import numpy as np

                        from .plotting import HAS_MATPLOTLIB

                        if not HAS_MATPLOTLIB:
                            print(
                                "Error: Plotting requires matplotlib. Install with: pip install matplotlib"
                            )
                            continue

                        # Evaluate and plot manually to keep figure for saving
                        eval_result = _evaluate_safely(expression)
                        if not eval_result.get("ok"):
                            print(f"Error: {eval_result.get('error', 'Unknown error')}")
                            continue

                        from .parser import parse_preprocessed

                        expr = parse_preprocessed(eval_result["result"])
                        var_sym = sp.symbols(variable)
                        f = sp.lambdify(var_sym, expr, "numpy")

                        x_vals = np.linspace(x_min, x_max, points)
                        try:
                            y_vals = f(x_vals)
                        except (ValueError, TypeError, ZeroDivisionError):
                            y_vals = []
                            for x in x_vals:
                                try:
                                    y = float(sp.N(expr.subs(var_sym, x)))
                                    y_vals.append(y)
                                except (
                                    ValueError,
                                    TypeError,
                                    ZeroDivisionError,
                                    OverflowError,
                                ):
                                    y_vals.append(np.nan)
                            y_vals = np.array(y_vals)

                        # Create and save plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(
                            x_vals,
                            y_vals,
                            linewidth=2,
                            color="#2E86AB",
                            label=f"f({variable}) = {expression}",
                        )
                        ax.set_xlabel(variable, fontsize=12, fontweight="bold")
                        ax.set_ylabel(f"f({variable})", fontsize=12, fontweight="bold")
                        ax.set_title(
                            f"Plot of {expression}", fontsize=14, fontweight="bold"
                        )
                        ax.grid(True, alpha=0.3, linestyle="--")
                        ax.axhline(
                            y=0, color="k", linewidth=0.8, linestyle="-", alpha=0.3
                        )
                        ax.axvline(
                            x=0, color="k", linewidth=0.8, linestyle="-", alpha=0.3
                        )
                        ax.legend(loc="best", fontsize=10)
                        plt.tight_layout()

                        plt.savefig(save_file, dpi=150, bbox_inches="tight")
                        plt.close(fig)  # Close figure to free memory
                        print(f"Plot saved to: {save_file}")

                        # Try to open the file automatically
                        try:
                            import os
                            import subprocess
                            import sys

                            if sys.platform == "win32":
                                os.startfile(save_file)
                                print("Plot opened in default viewer.")
                            elif sys.platform == "darwin":
                                subprocess.run(["open", save_file], check=True)
                                print("Plot opened in default viewer.")
                            else:
                                subprocess.run(["xdg-open", save_file], check=True)
                                print("Plot opened in default viewer.")
                        except Exception:
                            # Silently fail - opening is a convenience feature
                            pass
                    except Exception as e:
                        print(f"Error saving plot: {e}")
                else:
                    # Regular plot (tries to open window, falls back to file if GUI unavailable)

                    result = plot_function(
                        expression,
                        variable=variable,
                        x_min=x_min,
                        x_max=x_max,
                        points=points,
                        ascii=False,
                    )

                    if result.ok:
                        print(result.result)
                    else:
                        print(f"Error: {result.error}")
                        if (
                            "Tcl" in str(result.error)
                            or "Tkinter" in str(result.error)
                            or "init.tcl" in str(result.error)
                        ):
                            print(
                                "\nTip: GUI plotting is not available. Try saving the plot instead:"
                            )
                            print(f"  plot {expression}, --save plot.png")
            except ImportError:
                print(
                    "Error: Plotting requires matplotlib. Install with: pip install matplotlib"
                )
            except Exception as e:
                print(f"Error plotting: {e}")
            continue

        # Check for function finding command FIRST (e.g., find f(x,y) with data points)
        # This must come before any other processing to avoid parsing "find" as an expression
        # Define is_find_command outside try block so it's accessible in exception handler

        # Normalize common typos: "= =" -> "=", multiple spaces around =
        raw = re.sub(r"=\s*=", "=", raw)  # Fix double equals typo
        raw = re.sub(r"\s*=\s*", " = ", raw)  # Normalize spaces around =
        raw = re.sub(r"\s+", " ", raw)  # Collapse multiple spaces
        raw = raw.strip()

        is_find_command = "find" in raw.lower()
        find_func_cmd = None

        try:
            from .function_manager import (
                define_function,
                find_function_from_data,
                parse_find_function_command,
            )

            # split_top_level_commas is already imported at module level
            # Check if input contains "find" keyword OR multiple function assignment patterns
            # Pattern: f(...)=..., f(...)=..., ... (suggests function finding)
            # (is_find_command already set above)
            # If no "find" keyword, check for pattern of multiple f(...)=... assignments
            if not is_find_command:
                # Count function assignment patterns: func_name(args) = value
                func_assignment_pattern = (
                    r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]+\)\s*=\s*[^,]+"
                )
                matches = list(re.finditer(func_assignment_pattern, raw))
                # If we have 1 or more such patterns, check if it's function finding or function definition
                # Function finding: f(1)=2 (numeric arguments - data points)
                # Function definition: f(x)=2*x (symbolic arguments - parameters)
                if len(matches) >= 1:
                    # Check if arguments contain numeric values (function finding) or are purely symbolic (function definition)
                    has_numeric_args = False
                    for m in matches:
                        func_name_check = m.group(1)
                        args_match_check = re.search(
                            rf"{re.escape(func_name_check)}\s*\(([^)]+)\)", m.group(0)
                        )
                        if args_match_check:
                            args_str_check = args_match_check.group(1).strip()
                            # Check if any argument contains a digit or numeric operator
                            # This indicates numeric data points, not symbolic parameters
                            if re.search(r"[-+]?\d", args_str_check):
                                has_numeric_args = True
                                break

                    # Only treat as function finding if arguments contain numeric values
                    if has_numeric_args:
                        # Group matches by function name to handle multiple functions
                        func_groups = {}
                        for m in matches:
                            func_name = m.group(1)
                            if func_name not in func_groups:
                                func_groups[func_name] = []
                            func_groups[func_name].append(m)

                        # If all matches use the same function name, process as single function finding
                        if len(func_groups) == 1:
                            func_name = list(func_groups.keys())[0]
                            matches_for_func = func_groups[func_name]
                            # Infer parameter names from the first match
                            first_match = matches_for_func[0]
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
                                # Create a modified input with "find" keyword
                                param_str = ", ".join(param_names)
                                raw_with_find = f"{raw}, find {func_name}({param_str})"
                                # Parse this as a find command
                                find_func_cmd = parse_find_function_command(
                                    raw_with_find
                                )
                                if find_func_cmd is not None:
                                    is_find_command = True
                                    # Replace raw with the modified version
                                    raw = raw_with_find
                                    # Continue to process this function finding (no break needed - not in a loop)
                    else:
                        # Multiple different function names - process each comma-separated part individually
                        # Split by commas and handle each part that matches function finding pattern
                        # Import here to avoid scoping issues
                        from .parser import split_top_level_commas as _split_commas

                        parts = _split_commas(raw)
                        function_finding_parts = []
                        evaluation_parts = []

                        for part in parts:
                            part = part.strip()
                            if not part:
                                continue

                            # Check if this part is a function finding pattern
                            part_match = re.match(func_assignment_pattern, part)
                            is_func_finding = False
                            if part_match:
                                # Check if arguments contain numeric values (function finding) or are purely symbolic (function definition)
                                func_name_check = part_match.group(1)
                                args_match_check = re.search(
                                    rf"{re.escape(func_name_check)}\s*\(([^)]+)\)", part
                                )
                                if args_match_check:
                                    args_str_check = args_match_check.group(1).strip()
                                    # Check if any argument contains a digit or numeric operator
                                    if re.search(r"[-+]?\d", args_str_check):
                                        is_func_finding = True

                            if is_func_finding:
                                function_finding_parts.append(part)
                            else:
                                evaluation_parts.append(part)

                        # Process each function finding part
                        if function_finding_parts:
                            # Process the first function finding part (will be handled by main function finding block)
                            first_part = function_finding_parts[0]
                            func_name = re.match(
                                func_assignment_pattern, first_part
                            ).group(1)
                            args_match = re.search(
                                rf"{re.escape(func_name)}\s*\(([^)]+)\)", first_part
                            )
                            if args_match:
                                args_str = args_match.group(1)
                                arg_list = _split_commas(args_str)
                                param_names = []
                                param_chars = "xyzuvwrst"
                                for i, _ in enumerate(arg_list):
                                    if i < len(param_chars):
                                        param_names.append(param_chars[i])
                                    else:
                                        param_names.append(f"x{i+1}")

                                # Process first function finding
                                raw_with_find = f"{first_part}, find {func_name}({', '.join(param_names)})"
                                find_func_cmd = parse_find_function_command(
                                    raw_with_find
                                )
                                if find_func_cmd:
                                    raw = raw_with_find
                                    is_find_command = True

                                    # Store remaining parts for processing after function finding
                                    remaining_function_parts = function_finding_parts[
                                        1:
                                    ]
                                    remaining_evaluation_parts = evaluation_parts
                        else:
                            # No function finding patterns, continue to normal evaluation
                            pass

            if is_find_command:
                # Parse again if we haven't already parsed (when "find" keyword was present)
                if find_func_cmd is None:
                    find_func_cmd = parse_find_function_command(raw)
                if find_func_cmd is not None:
                    # Extract data points from the input
                    # Format: f(5,7.1) = 6.43182, f(0, 7.1) = 4.84091, f(10,10)= 10, find f(x,y)
                    # Or: f(2,1)= 15, find f(x,y)
                    func_name, param_names = find_func_cmd

                    # Remove "find f(x,y)" part to get data points
                    find_pattern = rf"find\s+{re.escape(func_name)}\s*\([^)]*\)"
                    data_str = re.sub(
                        find_pattern, "", raw, flags=re.IGNORECASE
                    ).strip()

                    # Remove trailing commas
                    data_str = data_str.rstrip(",").strip()

                    # First, process any function definitions in data_str (like main_entry path)
                    from .function_manager import parse_function_definition

                    parts_to_process = split_top_level_commas(data_str)
                    remaining_parts = []
                    for part in parts_to_process:
                        part = part.strip()
                        if not part:
                            continue
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
                            remaining_parts.append(part)
                    data_str = ", ".join(remaining_parts)

                    # Check if this is an inverse solving case (e.g., "find f(x,y) = 5")
                    # vs function finding case (e.g., "f(1)=1, f(2)=4, find f(x)")
                    is_inverse_solve = False
                    target_value = None

                    if data_str.strip().startswith("="):
                        is_inverse_solve = True
                        target_value = data_str.strip()[1:].strip()
                    elif not any(
                        f"{func_name}(" in part
                        for part in split_top_level_commas(data_str)
                    ):
                        # No function calls with this func_name, likely inverse solve
                        is_inverse_solve = True
                        if "=" in data_str:
                            target_value = data_str.split("=", 1)[1].strip()
                        else:
                            target_value = data_str.strip()

                    if is_inverse_solve and target_value:
                        # Check if function is defined for inverse solving
                        from .function_manager import _function_registry

                        if func_name in _function_registry:
                            from .solver import solve_inverse_function

                            result = solve_inverse_function(
                                func_name, target_value, param_names
                            )
                            _format_inverse_solutions(
                                result, func_name, param_names, target_value
                            )
                            continue  # Skip further processing
                        else:
                            print(
                                f"Error: Function '{func_name}' is not defined. Define it first or provide data points."
                            )
                            continue

                    # Parse data points: f(arg1,arg2,...) = value
                    data_points = []
                    # Split by commas, but be careful with commas inside function calls
                    parts = split_top_level_commas(data_str)

                    # Track solve requests separately (where arguments are parameter names)
                    solve_requests = []

                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        # Pattern: func_name(arg1,arg2,...) = value
                        # Allow spaces around =
                        pattern = rf"{re.escape(func_name)}\s*\(([^)]+)\)\s*=\s*(.+)"
                        match = re.match(pattern, part)
                        if match:
                            args_str = match.group(1)
                            value_str = match.group(2).strip()

                            # Check if arguments are parameter names (solve request) vs numeric values (data point)
                            arg_list = split_top_level_commas(args_str)
                            is_solve_request = False
                            if len(arg_list) == len(param_names):
                                # Check if all arguments match parameter names exactly
                                is_solve_request = True
                                for arg, param in zip(arg_list, param_names):
                                    if arg.strip() != param.strip():
                                        is_solve_request = False
                                        break

                            if is_solve_request:
                                # This is a solve request, not a data point
                                solve_requests.append((part, value_str))
                                continue

                            # Check if value_str is a simple numeric string (preserve for exact precision)
                            value_str_preserved = None
                            try:
                                # Try to parse as float to see if it's a simple number
                                float(value_str)
                                # Check if it's just a number (no operators, functions, etc.)
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
                                    # Preserve original string for exact conversion
                                    value_str_preserved = value_str
                            except (ValueError, TypeError):
                                pass

                            # Parse arguments (can be numeric or symbolic like pi, e)
                            args = []
                            arg_strings_preserved = []
                            for arg in split_top_level_commas(args_str):
                                try:
                                    arg_stripped = arg.strip()
                                    # Check if it's a simple numeric string
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

                                    # Try parsing as SymPy expression first (handles pi, e, etc.)
                                    arg_expr = _parse_preprocessed(arg_stripped)
                                    # Convert to numeric if possible, otherwise keep as symbolic
                                    try:
                                        arg_val = float(sp.N(arg_expr))
                                    except (ValueError, TypeError):
                                        # Keep as symbolic expression
                                        arg_val = arg_expr
                                    args.append(arg_val)
                                except Exception as e:
                                    print(
                                        f"Error: Invalid argument '{arg}' in data point: {e}"
                                    )
                                    break
                            else:
                                # Parse value (can be numeric or symbolic)
                                try:
                                    if value_str_preserved:
                                        # Use preserved string
                                        value = value_str_preserved
                                    else:
                                        value_expr = _parse_preprocessed(value_str)
                                        # Convert to numeric if possible, otherwise keep as symbolic
                                        try:
                                            value = float(sp.N(value_expr))
                                        except (ValueError, TypeError):
                                            # Keep as symbolic expression
                                            value = value_expr
                                    if len(args) == len(param_names):
                                        # Pass values directly to function finder, preserving strings for exact precision
                                        # The function finder will handle string-to-rational conversion
                                        final_args = []
                                        for i, arg_val in enumerate(args):
                                            # Use preserved string if available
                                            if (
                                                i < len(arg_strings_preserved)
                                                and arg_strings_preserved[i]
                                            ):
                                                final_args.append(
                                                    arg_strings_preserved[i]
                                                )
                                            elif isinstance(arg_val, str):
                                                # Keep as string - function finder will convert with exact precision
                                                final_args.append(arg_val)
                                            elif isinstance(arg_val, (int, float)):
                                                final_args.append(arg_val)
                                            else:
                                                # SymPy expression, convert to float or string
                                                try:
                                                    # Try to get string representation if it's a simple number
                                                    if isinstance(
                                                        arg_val,
                                                        (
                                                            sp.Rational,
                                                            sp.Integer,
                                                            sp.Float,
                                                        ),
                                                    ):
                                                        # Convert to string for exact preservation
                                                        final_args.append(str(arg_val))
                                                    else:
                                                        final_args.append(
                                                            float(sp.N(arg_val))
                                                        )
                                                except (ValueError, TypeError):
                                                    final_args.append(arg_val)

                                        # CRITICAL: Preserve value as string to maintain exact decimal precision
                                        # Use preserved original string if available (e.g., "70.833")
                                        if value_str_preserved:
                                            final_value = value_str_preserved
                                        elif isinstance(value, str):
                                            final_value = value
                                        elif isinstance(value, (int, float)):
                                            # Convert to string to preserve exact decimal representation
                                            # Prefer using original value_str if available
                                            if "value_str" in locals() and value_str:
                                                final_value = value_str
                                            else:
                                                # Fallback: convert float to string
                                                if isinstance(value, float):
                                                    # Format with enough precision to preserve exact decimal
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
                                                    value,
                                                    (sp.Rational, sp.Integer, sp.Float),
                                                ):
                                                    final_value = str(value)
                                                else:
                                                    # Try to get string representation
                                                    final_value = (
                                                        str(value)
                                                        if hasattr(value, "__str__")
                                                        else float(sp.N(value))
                                                    )
                                            except (ValueError, TypeError):
                                                final_value = value

                                        data_points.append((final_args, final_value))
                                    else:
                                        print(
                                            f"Error: Expected {len(param_names)} arguments, got {len(args)}"
                                        )
                                except Exception as e:
                                    print(
                                        f"Error: Invalid value '{value_str}' in data point: {e}"
                                    )

                    if data_points:
                        # Check if system is underdetermined (fewer points than needed for exact solution)
                        n_params = len(param_names)
                        n_coeffs = n_params + 1 if n_params > 1 else n_params
                        is_underdetermined = len(data_points) < n_coeffs

                        success, func_str, factored_form, error_msg = (
                            find_function_from_data(data_points, param_names)
                        )
                        if success:
                            params_str = ", ".join(param_names)
                            # Display the function string as-is (it's already in a readable format)
                            # Note: We used to remove * for display, but that made it unparseable
                            # The function string is already human-readable
                            print(f"{func_name}({params_str}) = {func_str}")
                            # if factored_form is not None:
                            #     print(
                            #         f"Equivalent: {func_name}({params_str}) = {factored_form}"
                            #     )

                            # Warn if this is a best-fit solution (underdetermined system)
                            if is_underdetermined:
                                print(
                                    f"Note: Best-fit solution (underdetermined system - {len(data_points)} point(s) for {n_coeffs} coefficient(s)). "
                                    f"Not all data points may be exactly satisfied."
                                )

                            # Compute and display L_func for canonical integerized form
                            try:
                                # Parse function body to extract coefficients
                                local_dict = {
                                    param: sp.Symbol(param) for param in param_names
                                }
                                func_body = sp.sympify(func_str, locals=local_dict)
                                if len(param_names) == 2:
                                    var1, var2 = sp.Symbol(param_names[0]), sp.Symbol(
                                        param_names[1]
                                    )
                                    equation_expanded = sp.expand(func_body)
                                    coeff_x_raw = equation_expanded.coeff(var1)
                                    coeff_y_raw = equation_expanded.coeff(var2)
                                    const_raw = equation_expanded.subs(
                                        [(var1, 0), (var2, 0)]
                                    )

                                    # Convert to rationals
                                    coeff_x = (
                                        sp.Rational(coeff_x_raw)
                                        if isinstance(coeff_x_raw, (sp.Float, float))
                                        else sp.simplify(coeff_x_raw)
                                    )
                                    coeff_y = (
                                        sp.Rational(coeff_y_raw)
                                        if isinstance(coeff_y_raw, (sp.Float, float))
                                        else sp.simplify(coeff_y_raw)
                                    )
                                    const = (
                                        sp.Rational(const_raw)
                                        if isinstance(const_raw, (sp.Float, float))
                                        else sp.simplify(const_raw)
                                    )

                                    if (
                                        isinstance(coeff_x, sp.Rational)
                                        and isinstance(coeff_y, sp.Rational)
                                        and isinstance(const, sp.Rational)
                                    ):

                                        def lcm(a: int, b: int) -> int:
                                            return (
                                                abs(a * b) // gcd(a, b)
                                                if a and b
                                                else 0
                                            )

                                        L_func = lcm(
                                            coeff_x.denominator, coeff_y.denominator
                                        )
                                        L_func = lcm(L_func, const.denominator)
                                        # Display canonical integerized form with L_func
                                        a_int = int(coeff_x * L_func)
                                        b_int = int(coeff_y * L_func)
                                        c_const = int(
                                            -const * L_func
                                        )  # Negate const to get positive constant on RHS
                                        # Template: A*x + B*y = L_func*F + C_const (for target F)
                                        # Example: 127*x + 50*y = 12*F + 4670
                                        print(
                                            f"Canonical integerized form (L_func = {L_func}): {a_int}*{param_names[0]} + {b_int}*{param_names[1]} = {L_func}*F + {c_const}"
                                        )
                            except Exception:
                                pass  # Skip if computation fails

                            # Also define the function for future use (keep * for SymPy parsing)
                            try:
                                define_function(func_name, param_names, func_str)
                                print(
                                    f"Function '{func_name}' is now available. You can call it like: {func_name}(values)"
                                )
                            except Exception as e:
                                print(
                                    f"Warning: Could not define function automatically: {e}"
                                )

                            # Process solve requests if any
                            if solve_requests:
                                from .function_manager import list_functions

                                print()  # Empty line for readability
                                for solve_part, target_value_str in solve_requests:
                                    try:
                                        # CRITICAL: Capture target_value_str immediately to avoid closure issues
                                        current_target_str = target_value_str

                                        # Parse target value with ambiguity detection
                                        # CRITICAL: Capture target_expr in a local variable to avoid modification
                                        (
                                            parsed_target_expr,
                                            literal_frac,
                                            was_simplified,
                                        ) = _parse_target_with_ambiguity_detection(
                                            current_target_str
                                        )

                                        # Use parsed_target_expr throughout this iteration
                                        target_expr = parsed_target_expr

                                        # Get the function body - parse from func_str since it's already defined
                                        # Create local dict with parameter symbols
                                        local_dict = {}
                                        for param in param_names:
                                            local_dict[param] = sp.Symbol(param)

                                        # Parse function body with parameter symbols
                                        func_body = sp.sympify(
                                            func_str, locals=local_dict
                                        )

                                        # Build equation: func_body - target = 0
                                        equation = func_body - target_expr

                                        # Solve the equation
                                        solutions = []
                                        param_symbols = [
                                            sp.Symbol(p) for p in param_names
                                        ]

                                        # Initialize integer solution state variables before any try/except that references them
                                        found_integers = []
                                        no_int_solution_msg = None
                                        int_solution_info = None

                                        if len(param_names) == 1:
                                            # Single variable - direct solve
                                            var_symbol = param_symbols[0]
                                            try:
                                                sols = sp.solve(equation, var_symbol)
                                                if sols:
                                                    if isinstance(sols, list):
                                                        for sol in sols:
                                                            sol_str = str(
                                                                sp.simplify(sol)
                                                            )
                                                            solutions.append(
                                                                {
                                                                    param_names[
                                                                        0
                                                                    ]: sol_str
                                                                }
                                                            )
                                                    else:
                                                        sol_str = str(sp.simplify(sols))
                                                        solutions.append(
                                                            {param_names[0]: sol_str}
                                                        )
                                            except Exception:
                                                pass
                                        elif len(param_names) == 2:
                                            # Two variables - solve for parametric solution
                                            # Prefer solving for the second variable (usually y) in terms of the first (usually x)
                                            var1, var2 = (
                                                param_symbols[0],
                                                param_symbols[1],
                                            )
                                            try:
                                                # Try to solve for var2 (y) in terms of var1 (x) first
                                                sol_var2 = sp.solve(equation, var2)
                                                if sol_var2:
                                                    if isinstance(sol_var2, list):
                                                        sol_var2 = sol_var2[0]
                                                    # Simplify the solution
                                                    sol_var2_simplified = sp.simplify(
                                                        sol_var2
                                                    )
                                                    sol_dict = {
                                                        param_names[0]: param_names[0],
                                                        param_names[1]: str(
                                                            sol_var2_simplified
                                                        ),
                                                    }
                                                    solutions.append(sol_dict)
                                            except Exception:
                                                pass

                                            # Try to find integer solutions if the equation is linear in both variables
                                            # This runs independently of sp.solve success
                                            try:
                                                sols = (
                                                    find_integer_solutions_for_linear(
                                                        equation, var1, var2
                                                    )
                                                )
                                                if sols:
                                                    # Handle special sentinel for "all integer pairs"
                                                    if (
                                                        isinstance(sols, list)
                                                        and sols
                                                        and isinstance(sols[0], dict)
                                                        and sols[0].get(
                                                            "all_integer_pairs"
                                                        )
                                                    ):
                                                        int_solution_info = {
                                                            "type": "all_integer_pairs",
                                                            "message": "Equation is tautologically 0=0 after scaling; every integer pair satisfies it.",
                                                        }
                                                    else:
                                                        # Convert diophantine solutions to a normalized list format
                                                        # Extract multiple positive integer solutions from diophantine parameterization
                                                        # diophantine returns parameterized solutions like (50*t_0 + 71942, -127*t_0 - 182622)
                                                        # We need to find multiple specific integer solutions
                                                        found_integers = []
                                                        try:
                                                            # Get the first solution tuple from diophantine
                                                            if sols and len(sols) > 0:
                                                                # diophantine returns parameterized solutions, get first one
                                                                first_sol = (
                                                                    list(sols[0])
                                                                    if hasattr(
                                                                        sols[0],
                                                                        "__iter__",
                                                                    )
                                                                    and not isinstance(
                                                                        sols[0], str
                                                                    )
                                                                    else sols[0]
                                                                )

                                                                # Parse the equation to get coefficients
                                                                poly = sp.Poly(
                                                                    equation, var1, var2
                                                                )
                                                                a = poly.coeff_monomial(
                                                                    var1
                                                                )
                                                                b = poly.coeff_monomial(
                                                                    var2
                                                                )
                                                                c = poly.coeff_monomial(
                                                                    1
                                                                )

                                                                # Build integerized equation
                                                                from math import (
                                                                    gcd as math_gcd,
                                                                )

                                                                def lcm(
                                                                    a: int, b: int
                                                                ) -> int:
                                                                    return (
                                                                        abs(a * b)
                                                                        // math_gcd(
                                                                            a, b
                                                                        )
                                                                        if a and b
                                                                        else 0
                                                                    )

                                                                def denom_of(
                                                                    sympy_number,
                                                                ):
                                                                    t = sp.together(
                                                                        sympy_number
                                                                    )
                                                                    if t.is_Rational:
                                                                        return int(t.q)
                                                                    if t.is_integer:
                                                                        return 1
                                                                    try:
                                                                        r = sp.nsimplify(
                                                                            t
                                                                        )
                                                                        if (
                                                                            r.is_Rational
                                                                        ):
                                                                            return int(
                                                                                r.q
                                                                            )
                                                                    except Exception:
                                                                        pass
                                                                    return 1

                                                                denoms = [
                                                                    denom_of(v)
                                                                    for v in (a, b, c)
                                                                ]
                                                                lcm_den = (
                                                                    math.lcm(*denoms)
                                                                    if denoms
                                                                    else 1
                                                                )
                                                                A = int(
                                                                    sp.Integer(
                                                                        sp.simplify(
                                                                            a * lcm_den
                                                                        )
                                                                    )
                                                                )
                                                                B = int(
                                                                    sp.Integer(
                                                                        sp.simplify(
                                                                            b * lcm_den
                                                                        )
                                                                    )
                                                                )
                                                                C = int(
                                                                    sp.Integer(
                                                                        sp.simplify(
                                                                            -c * lcm_den
                                                                        )
                                                                    )
                                                                )

                                                                # Find a particular solution using extended gcd
                                                                g = (
                                                                    math.gcd(A, B)
                                                                    if B != 0
                                                                    else (
                                                                        abs(A)
                                                                        if A != 0
                                                                        else 1
                                                                    )
                                                                )
                                                                if g > 0 and C % g == 0:
                                                                    (
                                                                        s,
                                                                        t_coeff,
                                                                        g_val,
                                                                    ) = _extended_gcd(
                                                                        abs(A), abs(B)
                                                                    )
                                                                    if A < 0:
                                                                        s = -s
                                                                    if B < 0:
                                                                        t_coeff = (
                                                                            -t_coeff
                                                                        )
                                                                    mult = C // g_val
                                                                    x0 = s * mult
                                                                    y0 = t_coeff * mult

                                                                    # Find the smallest positive solution (normalized)
                                                                    if B != 0:
                                                                        b_div_g = (
                                                                            abs(B)
                                                                            // g_val
                                                                        )
                                                                        mod = b_div_g
                                                                        x0_norm = (
                                                                            x0 % mod
                                                                        )
                                                                        if x0_norm < 0:
                                                                            x0_norm += (
                                                                                mod
                                                                            )
                                                                        y0_norm = (
                                                                            C
                                                                            - A
                                                                            * x0_norm
                                                                        ) // B
                                                                    else:
                                                                        x0_norm = x0
                                                                        y0_norm = y0

                                                                    # General solution: x = x0_norm + (B/g)*k, y = y0_norm - (A/g)*k
                                                                    # Find multiple positive solutions by trying different k values
                                                                    if B != 0:
                                                                        step_x = (
                                                                            abs(B)
                                                                            // g_val
                                                                        )
                                                                        A_div_g = (
                                                                            A // g_val
                                                                        )  # This handles the sign correctly

                                                                        # Try k values to find multiple positive solutions
                                                                        # Start from k=0 (gives normalized solution) and try both directions
                                                                        max_solutions = 20  # Limit to reasonable number
                                                                        k_range = 100  # Search k from -100 to 100
                                                                        for k in range(
                                                                            -k_range,
                                                                            k_range + 1,
                                                                        ):
                                                                            x_val = (
                                                                                x0_norm
                                                                                + step_x
                                                                                * k
                                                                            )
                                                                            # y = y0_norm - (A/g)*k (note the minus sign)
                                                                            y_val = (
                                                                                y0_norm
                                                                                - A_div_g
                                                                                * k
                                                                            )
                                                                            # Verify it's a solution and both are non-negative
                                                                            if (
                                                                                x_val
                                                                                >= 0
                                                                                and y_val
                                                                                >= 0
                                                                            ):
                                                                                # Verify it's actually a solution
                                                                                if (
                                                                                    A
                                                                                    * x_val
                                                                                    + B
                                                                                    * y_val
                                                                                    == C
                                                                                ):
                                                                                    found_integers.append(
                                                                                        (
                                                                                            x_val,
                                                                                            y_val,
                                                                                        )
                                                                                    )
                                                                                    if (
                                                                                        len(
                                                                                            found_integers
                                                                                        )
                                                                                        >= max_solutions
                                                                                    ):
                                                                                        break
                                                                        # Remove duplicates and sort
                                                                        found_integers = sorted(
                                                                            set(
                                                                                found_integers
                                                                            )
                                                                        )
                                                                    else:
                                                                        # B == 0 case: only one solution
                                                                        if (
                                                                            x0_norm >= 0
                                                                            and y0_norm
                                                                            >= 0
                                                                        ):
                                                                            found_integers.append(
                                                                                (
                                                                                    x0_norm,
                                                                                    y0_norm,
                                                                                )
                                                                            )
                                                        except Exception:
                                                            logger.exception(
                                                                "Error extracting specific solution from diophantine result"
                                                            )

                                                        # Build integerized equation string for display
                                                        int_eq_str = None
                                                        try:
                                                            poly = sp.Poly(
                                                                equation, var1, var2
                                                            )
                                                            a = poly.coeff_monomial(
                                                                var1
                                                            )
                                                            b = poly.coeff_monomial(
                                                                var2
                                                            )
                                                            c = poly.coeff_monomial(1)
                                                            from math import (  # noqa: F811
                                                                gcd as math_gcd,
                                                            )

                                                            def lcm(
                                                                a: int, b: int
                                                            ) -> int:
                                                                return (
                                                                    abs(a * b)
                                                                    // math_gcd(a, b)
                                                                    if a and b
                                                                    else 0
                                                                )

                                                            def denom_of(sympy_number):
                                                                t = sp.together(
                                                                    sympy_number
                                                                )
                                                                if t.is_Rational:
                                                                    return int(t.q)
                                                                if t.is_integer:
                                                                    return 1
                                                                try:
                                                                    r = sp.nsimplify(t)
                                                                    if r.is_Rational:
                                                                        return int(r.q)
                                                                except Exception:
                                                                    pass
                                                                return 1

                                                            denoms = [
                                                                denom_of(v)
                                                                for v in (a, b, c)
                                                            ]
                                                            lcm_den = (
                                                                math.lcm(*denoms)
                                                                if denoms
                                                                else 1
                                                            )
                                                            A = int(
                                                                sp.Integer(
                                                                    sp.simplify(
                                                                        a * lcm_den
                                                                    )
                                                                )
                                                            )
                                                            B = int(
                                                                sp.Integer(
                                                                    sp.simplify(
                                                                        b * lcm_den
                                                                    )
                                                                )
                                                            )
                                                            C = int(
                                                                sp.Integer(
                                                                    sp.simplify(
                                                                        -c * lcm_den
                                                                    )
                                                                )
                                                            )
                                                            int_eq_str = f"  Integerized equation (L_total = {lcm_den}): {A}*{param_names[0]} + {B}*{param_names[1]} = {C}"
                                                        except Exception:
                                                            pass

                                                        int_solution_info = {
                                                            "type": "diophantine",
                                                            "solutions": sols,
                                                            "integers": found_integers,
                                                            "equation": int_eq_str,
                                                            "no_solution": None,
                                                        }
                                                else:
                                                    no_int_solution_msg = "No integer solutions found (Diophantine check failed or none exist)."
                                                    int_solution_info = {
                                                        "type": "none",
                                                        "message": no_int_solution_msg,
                                                    }
                                            except ValueError as ve:
                                                # coefficient non-numeric or not linear
                                                logger.exception(
                                                    "Integer-solution validation failed"
                                                )
                                                no_int_solution_msg = f"Integer-solution validation failed: {ve}"
                                                int_solution_info = {
                                                    "type": "error",
                                                    "message": no_int_solution_msg,
                                                }
                                            except Exception as ex:
                                                # unexpected: log and keep a useful message for debugging
                                                logger.exception(
                                                    "Unexpected error while finding integer solutions"
                                                )
                                                no_int_solution_msg = f"Unexpected error while finding integer solutions: {ex}"
                                                int_solution_info = {
                                                    "type": "error",
                                                    "message": no_int_solution_msg,
                                                }

                                            # If sp.solve failed, try fallback
                                            if not solutions:
                                                try:
                                                    sol_var1 = sp.solve(equation, var1)
                                                    if sol_var1:
                                                        if isinstance(sol_var1, list):
                                                            sol_var1 = sol_var1[0]
                                                        sol_var1_simplified = (
                                                            sp.simplify(sol_var1)
                                                        )
                                                        sol_dict = {
                                                            param_names[0]: str(
                                                                sol_var1_simplified
                                                            ),
                                                            param_names[1]: param_names[
                                                                1
                                                            ],
                                                        }
                                                        solutions.append(sol_dict)
                                                except Exception:
                                                    pass

                                            # Ensure int_solution_info is set even if integer solution finding didn't run
                                            if int_solution_info is None:
                                                # Try to build integerized equation from function body for display
                                                try:
                                                    equation_expanded = sp.expand(
                                                        func_body
                                                    )
                                                    coeff_x_raw = (
                                                        equation_expanded.coeff(var1)
                                                    )
                                                    coeff_y_raw = (
                                                        equation_expanded.coeff(var2)
                                                    )
                                                    const_raw = equation_expanded.subs(
                                                        [(var1, 0), (var2, 0)]
                                                    )
                                                    target_for_int_solve, _, _ = (
                                                        _parse_target_with_ambiguity_detection(
                                                            current_target_str
                                                        )
                                                    )
                                                    if isinstance(
                                                        target_for_int_solve,
                                                        (sp.Rational, sp.Integer),
                                                    ):
                                                        coeff_x = (
                                                            sp.Rational(coeff_x_raw)
                                                            if isinstance(
                                                                coeff_x_raw,
                                                                (sp.Float, float),
                                                            )
                                                            else sp.simplify(
                                                                coeff_x_raw
                                                            )
                                                        )
                                                        coeff_y = (
                                                            sp.Rational(coeff_y_raw)
                                                            if isinstance(
                                                                coeff_y_raw,
                                                                (sp.Float, float),
                                                            )
                                                            else sp.simplify(
                                                                coeff_y_raw
                                                            )
                                                        )
                                                        const = (
                                                            sp.Rational(const_raw)
                                                            if isinstance(
                                                                const_raw,
                                                                (sp.Float, float),
                                                            )
                                                            else sp.simplify(const_raw)
                                                        )
                                                        if (
                                                            isinstance(
                                                                coeff_x, sp.Rational
                                                            )
                                                            and isinstance(
                                                                coeff_y, sp.Rational
                                                            )
                                                            and isinstance(
                                                                const, sp.Rational
                                                            )
                                                        ):
                                                            from math import (  # noqa: F811
                                                                gcd as math_gcd,
                                                            )

                                                            def lcm(
                                                                a: int, b: int
                                                            ) -> int:
                                                                return (
                                                                    abs(a * b)
                                                                    // math_gcd(a, b)
                                                                    if a and b
                                                                    else 0
                                                                )

                                                            denom_A = (
                                                                coeff_x.denominator
                                                            )
                                                            denom_B = (
                                                                coeff_y.denominator
                                                            )
                                                            denom_C = const.denominator
                                                            L_func = lcm(
                                                                denom_A, denom_B
                                                            )
                                                            L_func = lcm(
                                                                L_func, denom_C
                                                            )
                                                            coeffs = [
                                                                coeff_x,
                                                                coeff_y,
                                                                const,
                                                            ]
                                                            int_eq = _compute_integerized_equation(
                                                                coeffs,
                                                                target_for_int_solve,
                                                                L_func,
                                                            )
                                                            if int_eq is not None:
                                                                (
                                                                    a_total,
                                                                    b_total,
                                                                    RHS_total_int,
                                                                    L_total,
                                                                ) = int_eq
                                                                int_eq_str = f"  Integerized equation (L_total = {L_total}): {a_total}*{param_names[0]} + {b_total}*{param_names[1]} = {RHS_total_int}"
                                                                int_solution_info = {
                                                                    "type": "equation_only",
                                                                    "equation": int_eq_str,
                                                                    "integers": [],
                                                                    "no_solution": None,
                                                                }
                                                except Exception:
                                                    logger.exception(
                                                        "Error building integerized equation fallback"
                                                    )
                                                    pass

                                            if int_solution_info is None:
                                                int_solution_info = {
                                                    "type": "none",
                                                    "message": "Could not compute integer solution info",
                                                }
                                        else:
                                            # Multiple variables - solve for first variable in terms of others
                                            try:
                                                first_var = param_symbols[0]
                                                sol = sp.solve(equation, first_var)
                                                if sol:
                                                    if isinstance(sol, list):
                                                        sol = sol[0]
                                                    sol_dict = {
                                                        param_names[0]: str(
                                                            sp.simplify(sol)
                                                        )
                                                    }
                                                    # Other variables remain free
                                                    for i in range(1, len(param_names)):
                                                        sol_dict[param_names[i]] = (
                                                            param_names[i]
                                                        )
                                                    solutions.append(sol_dict)
                                            except Exception:
                                                logger.exception(
                                                    "Error solving for multiple variables"
                                                )
                                                pass

                                        if solutions:
                                            print(
                                                f"Solving {func_name}({', '.join(param_names)}) = {current_target_str}:"
                                            )

                                            # Display ambiguity information if applicable
                                            if (
                                                was_simplified
                                                and literal_frac is not None
                                            ):
                                                print(
                                                    f"  Parsed target '{current_target_str}' as {target_expr.numerator}/{target_expr.denominator} (literal {literal_frac.numerator}/{literal_frac.denominator})."
                                                )
                                                print(
                                                    f"  Using {target_expr.numerator}/{target_expr.denominator} for integer-solution search; to force literal use, input as {literal_frac.numerator}/{literal_frac.denominator} or supply override."
                                                )

                                            # Print integer solution info if available (computed earlier)
                                            if int_solution_info is not None:
                                                info_type = int_solution_info.get(
                                                    "type", "unknown"
                                                )
                                                if info_type == "all_integer_pairs":
                                                    print(
                                                        f"  {int_solution_info.get('message', 'All integer pairs are solutions')}"
                                                    )
                                                elif info_type == "diophantine":
                                                    # Display integerized equation if available
                                                    if "equation" in int_solution_info:
                                                        print(
                                                            int_solution_info[
                                                                "equation"
                                                            ]
                                                        )
                                                    elif (
                                                        "solutions" in int_solution_info
                                                        and int_solution_info[
                                                            "solutions"
                                                        ]
                                                    ):
                                                        # Build equation string from coefficients
                                                        try:
                                                            poly = sp.Poly(
                                                                equation, var1, var2
                                                            )
                                                            a = poly.coeff_monomial(
                                                                var1
                                                            )
                                                            b = poly.coeff_monomial(
                                                                var2
                                                            )
                                                            c = poly.coeff_monomial(1)
                                                            from math import (  # noqa: F811
                                                                gcd as math_gcd,
                                                            )

                                                            def lcm(
                                                                a: int, b: int
                                                            ) -> int:
                                                                return (
                                                                    abs(a * b)
                                                                    // math_gcd(a, b)
                                                                    if a and b
                                                                    else 0
                                                                )

                                                            def denom_of(sympy_number):
                                                                t = sp.together(
                                                                    sympy_number
                                                                )
                                                                if t.is_Rational:
                                                                    return int(t.q)
                                                                if t.is_integer:
                                                                    return 1
                                                                try:
                                                                    r = sp.nsimplify(t)
                                                                    if r.is_Rational:
                                                                        return int(r.q)
                                                                except Exception:
                                                                    pass
                                                                return 1

                                                            denoms = [
                                                                denom_of(v)
                                                                for v in (a, b, c)
                                                            ]
                                                            lcm_den = (
                                                                math.lcm(*denoms)
                                                                if denoms
                                                                else 1
                                                            )
                                                            A = int(
                                                                sp.Integer(
                                                                    sp.simplify(
                                                                        a * lcm_den
                                                                    )
                                                                )
                                                            )
                                                            B = int(
                                                                sp.Integer(
                                                                    sp.simplify(
                                                                        b * lcm_den
                                                                    )
                                                                )
                                                            )
                                                            C = int(
                                                                sp.Integer(
                                                                    sp.simplify(
                                                                        -c * lcm_den
                                                                    )
                                                                )
                                                            )
                                                            print(
                                                                f"  Integerized equation: {A}*{param_names[0]} + {B}*{param_names[1]} = {C}"
                                                            )
                                                        except Exception:
                                                            print(
                                                                "  Integer solutions exist (Diophantine parameterization found)"
                                                            )
                                                    # Display specific integer solutions if found
                                                    if int_solution_info.get(
                                                        "integers"
                                                    ):
                                                        integer_sols = (
                                                            int_solution_info[
                                                                "integers"
                                                            ]
                                                        )
                                                        if len(integer_sols) == 1:
                                                            print("  Integer solution:")
                                                            x_int, y_int = integer_sols[
                                                                0
                                                            ]
                                                            print(
                                                                f"    ({param_names[0]}, {param_names[1]}) = ({x_int}, {y_int})"
                                                            )
                                                        else:
                                                            print(
                                                                f"  Integer solutions (found {len(integer_sols)}):"
                                                            )
                                                            for (
                                                                x_int,
                                                                y_int,
                                                            ) in integer_sols:
                                                                print(
                                                                    f"    ({param_names[0]}, {param_names[1]}) = ({x_int}, {y_int})"
                                                                )
                                                            if len(integer_sols) >= 20:
                                                                print(
                                                                    "    ... (showing first 20 of potentially infinite solutions)"
                                                                )
                                                elif info_type == "none":
                                                    if "equation" in int_solution_info:
                                                        print(
                                                            int_solution_info[
                                                                "equation"
                                                            ]
                                                        )
                                                    if "message" in int_solution_info:
                                                        print(
                                                            f"  {int_solution_info['message']}"
                                                        )
                                                elif info_type == "error":
                                                    if "equation" in int_solution_info:
                                                        print(
                                                            int_solution_info[
                                                                "equation"
                                                            ]
                                                        )
                                                    if "message" in int_solution_info:
                                                        print(
                                                            f"  {int_solution_info['message']}"
                                                        )
                                                elif info_type == "equation_only":
                                                    if "equation" in int_solution_info:
                                                        print(
                                                            int_solution_info[
                                                                "equation"
                                                            ]
                                                        )
                                                else:
                                                    # Legacy format support
                                                    if "equation" in int_solution_info:
                                                        print(
                                                            int_solution_info[
                                                                "equation"
                                                            ]
                                                        )
                                                    if int_solution_info.get(
                                                        "no_solution"
                                                    ):
                                                        print(
                                                            int_solution_info[
                                                                "no_solution"
                                                            ]
                                                        )
                                                    elif int_solution_info.get(
                                                        "integers"
                                                    ):
                                                        print("  Integer solution:")
                                                        for (
                                                            x_int,
                                                            y_int,
                                                        ) in int_solution_info[
                                                            "integers"
                                                        ]:
                                                            print(
                                                                f"    ({param_names[0]}, {param_names[1]}) = ({x_int}, {y_int})"
                                                            )

                                            # Try to show the equation in simplified form first (for linear equations)
                                            try:
                                                if len(param_names) == 2:
                                                    # Rearrange to show: A*x + B*y = C
                                                    equation_rearranged = sp.simplify(
                                                        equation
                                                    )
                                                    # Try to get it in the form: coeff_x*x + coeff_y*y = const
                                                    var1, var2 = (
                                                        param_symbols[0],
                                                        param_symbols[1],
                                                    )
                                                    coeff_x = equation_rearranged.coeff(
                                                        var1
                                                    )
                                                    coeff_y = equation_rearranged.coeff(
                                                        var2
                                                    )
                                                    const = -equation_rearranged.subs(
                                                        [(var1, 0), (var2, 0)]
                                                    )

                                                    if coeff_x != 0 and coeff_y != 0:
                                                        # Convert to exact rationals if possible, format nicely
                                                        try:
                                                            coeff_x_rational = (
                                                                sp.Rational(coeff_x)
                                                                if isinstance(
                                                                    coeff_x,
                                                                    (sp.Float, float),
                                                                )
                                                                else sp.simplify(
                                                                    coeff_x
                                                                )
                                                            )
                                                            coeff_y_rational = (
                                                                sp.Rational(coeff_y)
                                                                if isinstance(
                                                                    coeff_y,
                                                                    (sp.Float, float),
                                                                )
                                                                else sp.simplify(
                                                                    coeff_y
                                                                )
                                                            )
                                                            const_rational = (
                                                                sp.Rational(const)
                                                                if isinstance(
                                                                    const,
                                                                    (sp.Float, float),
                                                                )
                                                                else sp.simplify(const)
                                                            )

                                                            # Try to simplify to integer form if possible
                                                            # Check if we can multiply through by common denominator to get integers
                                                            if (
                                                                isinstance(
                                                                    coeff_x_rational,
                                                                    sp.Rational,
                                                                )
                                                                and isinstance(
                                                                    coeff_y_rational,
                                                                    sp.Rational,
                                                                )
                                                                and isinstance(
                                                                    const_rational,
                                                                    sp.Rational,
                                                                )
                                                            ):
                                                                from math import gcd

                                                                def lcm(a, b):
                                                                    return (
                                                                        abs(a * b)
                                                                        // gcd(a, b)
                                                                        if a and b
                                                                        else 0
                                                                    )

                                                                denom_x = (
                                                                    coeff_x_rational.denominator
                                                                )
                                                                denom_y = (
                                                                    coeff_y_rational.denominator
                                                                )
                                                                denom_const = (
                                                                    const_rational.denominator
                                                                )
                                                                common_denom = lcm(
                                                                    lcm(
                                                                        denom_x, denom_y
                                                                    ),
                                                                    denom_const,
                                                                )

                                                                # Multiply through to get integer coefficients
                                                                a_int_display = int(
                                                                    coeff_x_rational
                                                                    * common_denom
                                                                )
                                                                b_int_display = int(
                                                                    coeff_y_rational
                                                                    * common_denom
                                                                )
                                                                c_int_display = int(
                                                                    const_rational
                                                                    * common_denom
                                                                )

                                                                # Simplify by dividing by gcd
                                                                g_all = gcd(
                                                                    gcd(
                                                                        abs(
                                                                            a_int_display
                                                                        ),
                                                                        abs(
                                                                            b_int_display
                                                                        ),
                                                                    ),
                                                                    abs(c_int_display),
                                                                )
                                                                if g_all > 1:
                                                                    a_int_display //= (
                                                                        g_all
                                                                    )
                                                                    b_int_display //= (
                                                                        g_all
                                                                    )
                                                                    c_int_display //= (
                                                                        g_all
                                                                    )

                                                                # Format: A*x + B*y = C (integer form)
                                                                eq_str = f"{a_int_display}*{param_names[0]} + {b_int_display}*{param_names[1]} = {c_int_display}"
                                                                print(
                                                                    f"  Equation: {eq_str}"
                                                                )
                                                            else:
                                                                # Fallback to rational form
                                                                eq_str = f"{coeff_x_rational}*{param_names[0]} + {coeff_y_rational}*{param_names[1]} = {const_rational}"
                                                                print(
                                                                    f"  Equation: {eq_str}"
                                                                )
                                                        except (TypeError, ValueError):
                                                            # Fallback to simplified form
                                                            eq_str = f"{sp.simplify(coeff_x)}*{param_names[0]} + {sp.simplify(coeff_y)}*{param_names[1]} = {sp.simplify(const)}"
                                                            print(
                                                                f"  Equation: {eq_str}"
                                                            )
                                            except Exception:
                                                pass

                                            for i, sol in enumerate(solutions, 1):
                                                sol_parts = [
                                                    f"{k} = {v}" for k, v in sol.items()
                                                ]
                                                print(
                                                    f"  Solution {i}: {', '.join(sol_parts)}"
                                                )
                                        else:
                                            print(
                                                f"Could not solve {func_name}({', '.join(param_names)}) = {current_target_str}"
                                            )
                                    except Exception as e:
                                        logger.exception(
                                            "Outer try block failed while processing equation"
                                        )
                                        print(f"Error solving {solve_part}: {e}")
                                        # Ensure int_solution_info exists (defensive)
                                        if int_solution_info is None:
                                            int_solution_info = {
                                                "type": "error",
                                                "message": "Processing failed; see logs.",
                                            }
                        else:
                            print(f"Error: {error_msg}")
                    else:
                        print(
                            "Error: No valid data points found. Format: f(arg1,arg2,...) = value, ..."
                        )

                    # Process remaining function finding parts and evaluation parts if any
                    if (
                        "remaining_function_parts" in locals()
                        and remaining_function_parts
                    ):
                        # Process each remaining function finding part
                        func_assignment_pattern_local = (
                            r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]+\)\s*=\s*[^,]+"
                        )
                        for part in remaining_function_parts:
                            part_match = re.match(func_assignment_pattern_local, part)
                            if part_match:
                                func_name = part_match.group(1)
                                args_match = re.search(
                                    rf"{re.escape(func_name)}\s*\(([^)]+)\)", part
                                )
                                if args_match:
                                    args_str = args_match.group(1)
                                    arg_list = _split_commas(args_str)
                                    param_names = []
                                    param_chars = "xyzuvwrst"
                                    for i, _ in enumerate(arg_list):
                                        if i < len(param_chars):
                                            param_names.append(param_chars[i])
                                        else:
                                            param_names.append(f"x{i+1}")

                                    # Process this function finding
                                    part_with_find = f"{part}, find {func_name}({', '.join(param_names)})"
                                    # Recursively process this part (simplified - just call the function finding logic)
                                    try:
                                        # Extract data point from this part
                                        value_match = re.search(r"=\s*(.+)", part)
                                        if value_match:
                                            value_str = value_match.group(1).strip()
                                            # Parse arguments
                                            final_args = []
                                            for arg in arg_list:
                                                try:
                                                    arg_val = float(arg.strip())
                                                    final_args.append(arg_val)
                                                except ValueError:
                                                    final_args.append(arg.strip())

                                            # Parse value
                                            try:
                                                value = float(value_str)
                                            except ValueError:
                                                value = value_str

                                            # Find function
                                            data_points = [
                                                (
                                                    (
                                                        [arg]
                                                        if len(param_names) == 1
                                                        else final_args
                                                    ),
                                                    value,
                                                )
                                            ]
                                            (
                                                success,
                                                func_str,
                                                factored_form,
                                                error_msg,
                                            ) = find_function_from_data(
                                                data_points, param_names
                                            )
                                            if success:
                                                params_str = ", ".join(param_names)
                                                print(
                                                    f"{func_name}({params_str}) = {func_str}"
                                                )
                                                if factored_form:
                                                    print(
                                                        f"Equivalent: {func_name}({params_str}) = {factored_form}"
                                                    )
                                                try:
                                                    define_function(
                                                        func_name, param_names, func_str
                                                    )
                                                    print(
                                                        f"Function '{func_name}' is now available."
                                                    )
                                                except Exception as e:
                                                    print(
                                                        f"Warning: Could not define function automatically: {e}"
                                                    )
                                    except Exception as e:
                                        print(f"Error processing {part}: {e}")

                    # Process evaluation parts
                    if (
                        "remaining_evaluation_parts" in locals()
                        and remaining_evaluation_parts
                    ):
                        for part in remaining_evaluation_parts:
                            part = part.strip()
                            if not part:
                                continue
                            try:
                                import time

                                start_time = time.perf_counter()
                                eva = _evaluate_safely(part)
                                elapsed = time.perf_counter() - start_time
                                if eva.get("ok"):
                                    res_str = eva.get("result")
                                    formatted_res = _format_special_values(res_str)
                                    print(
                                        f"{part} = {format_superscript(formatted_res)}"
                                    )
                                    if _timing_enabled:
                                        print(f"[Time: {elapsed:.4f}s]")
                                else:
                                    print(f"Error: {eva.get('error', 'Unknown error')}")
                            except Exception as e:
                                print(f"Error evaluating '{part}': {e}")

                    continue
        except ImportError:
            pass
        except Exception as e:
            # If "find" was detected but processing failed, don't silently continue
            # Give an error instead of trying to parse as normal expression
            if is_find_command:
                try:
                    logger.exception("Error processing function finding command")
                except Exception:
                    pass
                print(f"Error: Failed to process function finding command: {e}")
                print(
                    "  Make sure the format is: f(arg1,arg2,...) = value, ..., find f(x,y)"
                )
                continue
            # Otherwise, silently continue (function finding wasn't intended)
            # But check if it looks like function finding pattern to avoid wrong path
            func_finding_check = re.search(
                r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]+\)\s*=\s*[^,]+", raw
            )
            if func_finding_check:
                # Looks like function finding but detection failed - don't try function definition
                args_str = re.search(
                    rf"{re.escape(func_finding_check.group(1))}\s*\(([^)]+)\)", raw
                )
                if args_str and re.search(r"[-+]?\d", args_str.group(1)):
                    # Has numeric arguments - skip function definition parsing
                    pass
            pass

        # Check for pattern: f(x,y)=expression=value (solve for inputs that give output value)
        # Count number of '=' signs
        equals_count = raw.count("=")
        if equals_count == 2:
            # Pattern: func_name(params)=body=target_value
            # Split by '=' to get parts
            parts = raw.split("=")
            if len(parts) == 3:
                left_part = parts[0].strip()  # f(x,y)
                body_part = parts[1].strip()  # expression
                target_part = parts[2].strip()  # target value

                # Check if left_part matches function definition pattern
                func_def_pattern = r"^([a-zA-Z][a-zA-Z0-9]*)\s*\(([^)]*)\)$"
                match = re.match(func_def_pattern, left_part)
                if match:
                    func_name = match.group(1)
                    params_str = match.group(2).strip()
                    params = [p.strip() for p in params_str.split(",") if p.strip()]

                    if params and body_part and target_part:
                        # Define the function first
                        try:
                            from .function_manager import define_function

                            define_function(func_name, params, body_part)

                            # Now solve: func_name(params) = target_value
                            # Build equation: body_part - target_part = 0
                            # But we need to substitute the parameters as symbols
                            # Note: solve_system is already imported at module level
                            try:

                                # Parse the target value
                                target_expr = _parse_preprocessed(target_part)

                                # Build equation: f(x,y) - target = 0
                                # Create symbols for parameters
                                param_symbols = [sp.Symbol(p) for p in params]

                                # Parse body and substitute symbols
                                body_expr = _parse_preprocessed(body_part)

                                # Build equation: body_expr - target_expr = 0
                                equation = body_expr - target_expr

                                # Solve the equation
                                # For a single equation with multiple variables, we need to solve symbolically
                                # Convert to equation string format
                                equation_str = f"{body_part} - ({target_part})"

                                # Try to solve using SymPy
                                # For linear equations, we can solve for one variable in terms of others
                                solutions = []

                                if len(params) == 1:
                                    # Single variable - direct solve
                                    var_symbol = param_symbols[0]
                                    try:
                                        sols = sp.solve(equation, var_symbol)
                                        if sols:
                                            if isinstance(sols, list):
                                                for sol in sols:
                                                    sol_str = str(sp.simplify(sol))
                                                    solutions.append(
                                                        {params[0]: sol_str}
                                                    )
                                            else:
                                                sol_str = str(sp.simplify(sols))
                                                solutions.append({params[0]: sol_str})
                                    except Exception:
                                        pass
                                elif len(params) == 2:
                                    # Two variables - solve for parametric solution
                                    var1, var2 = param_symbols[0], param_symbols[1]
                                    try:
                                        # Try to solve for var1 in terms of var2
                                        sol_var1 = sp.solve(equation, var1)
                                        if sol_var1:
                                            if isinstance(sol_var1, list):
                                                sol_var1 = sol_var1[0]
                                            sol_dict = {
                                                params[0]: str(sp.simplify(sol_var1)),
                                                params[1]: params[1],
                                            }
                                            solutions.append(sol_dict)
                                        else:
                                            # Try solving for var2 in terms of var1
                                            sol_var2 = sp.solve(equation, var2)
                                            if sol_var2:
                                                if isinstance(sol_var2, list):
                                                    sol_var2 = sol_var2[0]
                                                sol_dict = {
                                                    params[0]: params[0],
                                                    params[1]: str(
                                                        sp.simplify(sol_var2)
                                                    ),
                                                }
                                                solutions.append(sol_dict)
                                    except Exception:
                                        pass
                                else:
                                    # Multiple variables - solve for first variable in terms of others
                                    try:
                                        first_var = param_symbols[0]
                                        sol = sp.solve(equation, first_var)
                                        if sol:
                                            if isinstance(sol, list):
                                                sol = sol[0]
                                            sol_dict = {
                                                params[0]: str(sp.simplify(sol))
                                            }
                                            # Other variables remain free
                                            for i in range(1, len(params)):
                                                sol_dict[params[i]] = params[i]
                                            solutions.append(sol_dict)
                                    except Exception:
                                        pass

                                if solutions:
                                    print(
                                        f"Solving {func_name}({', '.join(params)}) = {target_part}:"
                                    )
                                    for i, sol in enumerate(solutions, 1):
                                        sol_parts = [
                                            f"{k} = {v}" for k, v in sol.items()
                                        ]
                                        print(f"  Solution {i}: {', '.join(sol_parts)}")

                                    # Also show that the function is defined
                                    params_display = ", ".join(params)
                                    print(
                                        f"\nFunction '{func_name}({params_display})' is defined as: {body_part}"
                                    )
                                    continue
                                else:
                                    print(
                                        f"Could not solve {func_name}({', '.join(params)}) = {target_part}"
                                    )
                                    print(
                                        f"Function '{func_name}({', '.join(params)})' is defined as: {body_part}"
                                    )
                                    continue

                            except Exception as e:
                                # If solving fails, at least define the function
                                params_display = ", ".join(params)
                                print(
                                    f"Function '{func_name}({params_display})' is defined as: {body_part}"
                                )
                                print(
                                    f"Note: Could not solve equation automatically: {e}"
                                )
                                continue
                        except Exception as e:
                            print(f"Error: {e}")
                            pass

        # Check for function finding patterns BEFORE system/equation solving
        # This prevents f(-1)=3 from being treated as an equation
        func_finding_pattern_early = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]+\)\s*=\s*[^,]+"
        if re.search(func_finding_pattern_early, raw):
            matches_early = list(re.finditer(func_finding_pattern_early, raw))
            # Check if any match has numeric arguments (function finding) vs parameter names (function definition)
            has_numeric_args = False
            for m in matches_early:
                func_name_early = m.group(1)
                args_match_early = re.search(
                    rf"{re.escape(func_name_early)}\s*\(([^)]+)\)", m.group(0)
                )
                if args_match_early:
                    args_str_early = args_match_early.group(1).strip()
                    if re.search(r"[-+]?\d", args_str_early):
                        has_numeric_args = True
                        break

            if has_numeric_args:
                # This is a function finding pattern - it should have been handled earlier
                # But if we reach here, the detection failed, so provide helpful message
                print(
                    f"Error: '{raw}' looks like a function finding pattern (f(value)=result), not an equation."
                )
                if len(matches_early) == 1:
                    func_name_early = matches_early[0].group(1)
                    print(
                        f"  For single data point, use: {raw}, find {func_name_early}(x)"
                    )
                    print("  This will return the constant function f(x) = <value>")
                else:
                    func_name_early = matches_early[0].group(1)
                    print(
                        f"  For function finding, use: {raw}, find {func_name_early}(x)"
                    )
                continue

        # Check for system of equations FIRST (before trying to evaluate parts individually)
        # This prevents parse errors when input like "x+y=5, xy=25" is evaluated part-by-part
        parts_for_system = split_top_level_commas(raw)
        if len(parts_for_system) > 1:
            all_eq_check = all("=" in p for p in parts_for_system)
            # Check if these look like function calls (not equations)
            looks_like_function_calls = False
            try:
                from .function_manager import list_functions  # noqa: F811

                defined_funcs = list_functions()
                for part in parts_for_system:
                    func_match = re.match(
                        r"^(\w+)\s*\([^)]*\)\s*=\s*(.+)$", part.strip()
                    )
                    if func_match:
                        func_name = func_match.group(1)
                        if func_name in defined_funcs:
                            looks_like_function_calls = True
                            break
            except Exception:
                pass

            # Check for "find" command with multiple variables (e.g., "find x, y")
            find_pattern = re.search(
                r"\bfind\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)\b",
                raw,
                re.IGNORECASE,
            )
            find_vars = None
            find = None  # Initialize find variable
            if find_pattern:
                find_vars_str = find_pattern.group(1)
                # Extract all variable names from "find x, y" -> ["x", "y"]
                find_vars = [
                    v.strip()
                    for v in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", find_vars_str)
                ]
                raw_no_find = re.sub(
                    r"\bfind\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*\b",
                    "",
                    raw,
                    flags=re.IGNORECASE,
                ).strip()
                # Initialize find from find_vars
                find = find_vars[0] if find_vars else None
            else:
                # Try single variable pattern for backward compatibility
                find_tokens = re.findall(r"\bfind\s+(\w+)\b", raw, re.IGNORECASE)
                find_vars = find_tokens if find_tokens else None
                raw_no_find = re.sub(
                    r"\bfind\s+\w+\b", "", raw, flags=re.IGNORECASE
                ).strip()
                find = find_tokens[0] if find_tokens else None

            # If all parts contain "=" and don't look like function calls, treat as system
            if all_eq_check and not looks_like_function_calls:
                # Check if all assignments are to the same variable FIRST
                # This handles cases like "x = 1 % 2, x=3 % 6, x=3 % 7" where we want to evaluate each expression
                all_assign_same_var_check = all(
                    "=" in p and VAR_NAME_RE.match(p.split("=", 1)[0].strip())
                    for p in parts_for_system
                )
                if all_assign_same_var_check and len(parts_for_system) > 1:
                    assigned_vars_check = [
                        p.split("=", 1)[0].strip() for p in parts_for_system if "=" in p
                    ]
                    if (
                        len(assigned_vars_check) > 1
                        and len(set(assigned_vars_check)) == 1
                    ):
                        # All assignments are to the same variable
                        var = assigned_vars_check[0]
                        import time

                        start_time = time.perf_counter()
                        # Try to solve as system of congruences first
                        solved, exit_code = _solve_modulo_system_if_applicable(
                            parts_for_system, var, output_format
                        )
                        if solved:
                            if _timing_enabled:
                                print(
                                    f"[Time: {time.perf_counter() - start_time:.4f}s]"
                                )
                            continue

                        # If not all modulo or CRT solving failed, evaluate each expression separately
                        for p in parts_for_system:
                            if "=" in p:
                                left, right = p.split("=", 1)
                                var = left.strip()
                                rhs = right.strip() or "0"
                                # Evaluate the RHS expression (like "1 % 2")
                                res = _evaluate_safely(rhs)
                                if not res.get("ok"):
                                    print(
                                        f"Error evaluating '{var} = {rhs}': {res.get('error')}"
                                    )
                                    continue
                                try:
                                    # Format and print the result
                                    val_str = res.get("result", "")
                                    approx_str = res.get("approx", "")
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
                        elapsed = time.perf_counter() - start_time
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                        continue

                # This is a system of equations - handle it directly
                # If we have assignments like "xy=12" and find_vars like ["x", "y"], convert xy to x*y
                if find_vars and len(find_vars) > 0:
                    # Convert variable names like "xy" to "x*y" if x and y are in find_vars
                    parts_for_convert = split_top_level_commas(raw_no_find)
                    converted_parts = []
                    for part in parts_for_convert:
                        if "=" in part:
                            lhs, rhs = part.split("=", 1)
                            lhs = lhs.strip()
                            # Check if lhs is a concatenation of find_vars (e.g., "xy" when find_vars=["x", "y"])
                            # Try to split lhs into individual variables
                            var_pattern = "|".join(find_vars)
                            # Check if lhs matches a pattern like "xy" where x and y are in find_vars
                            # This is a heuristic: if lhs has no operators and all find_vars are single chars
                            if (
                                all(len(v) == 1 for v in find_vars)
                                and len(lhs) == len(find_vars)
                                and all(c in find_vars for c in lhs)
                            ):
                                # Convert "xy" to "x*y"
                                converted_lhs = "*".join(lhs)
                                converted_parts.append(f"{converted_lhs} = {rhs}")
                            else:
                                converted_parts.append(part)
                        else:
                            converted_parts.append(part)
                    raw_no_find = ", ".join(converted_parts)
                    # Use first find var for backward compatibility with solve_system
                    find = find_vars[0] if find_vars else None
                else:
                    find = find_vars[0] if find_vars and len(find_vars) > 0 else None

                try:
                    import time

                    start_time = time.perf_counter()
                    res = solve_system(raw_no_find, find)
                    elapsed = time.perf_counter() - start_time
                    print_result_pretty(res, output_format=output_format)
                    if _timing_enabled:
                        print(f"[Time: {elapsed:.4f}s]")
                    continue
                except Exception:
                    # If system solving fails, fall through to normal processing
                    logger.exception("Error solving system of equations")

        # Check for mixed function definition + function call (e.g., f(x)=2x, f(2))
        # Split by top-level commas and handle each part separately
        parts = split_top_level_commas(raw)
        if len(parts) > 1:
            # Skip if any part contains "find" - let the function finding handler process it
            # This prevents "find" from being parsed as implicit multiplication
            if any("find" in part.lower() for part in parts):
                # Don't process mixed parts if "find" is present
                # Let it fall through to normal processing which will catch it
                pass
            else:
                # Try to handle mixed definitions and calls
                handled_parts = []
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue

                    # Check if it's a function definition
                    try:
                        from .function_manager import (
                            define_function,
                            parse_function_definition,
                        )

                        func_def = parse_function_definition(part)
                        if func_def is not None:
                            func_name, params, body = func_def
                            try:
                                define_function(func_name, params, body)
                                params_str = ", ".join(params) if params else ""
                                print(
                                    f"Function '{func_name}({params_str})' defined as: {body}"
                                )
                                handled_parts.append(part)
                                continue
                            except ValidationError as e:
                                print(f"Error: {e.message}")
                                handled_parts.append(part)
                                continue
                    except Exception:
                        pass

                    # Check if it's a function call (f(1)=value or just f(1))
                    # Pattern: func_name(args) = value or func_name(args)
                    try:
                        from .function_manager import list_functions

                        defined_funcs = list_functions()

                        # Check if this looks like a function call
                        func_call_pattern = re.match(
                            r"^(\w+)\s*\([^)]*\)\s*(?:=\s*(.+))?$", part
                        )
                        if func_call_pattern:
                            func_name_in_call = func_call_pattern.group(1)
                            if func_name_in_call in defined_funcs:
                                # This is a function call
                                if "=" in part:
                                    # Format: f(1)=41 - evaluate the function and show result
                                    func_call_part = part.split("=", 1)[0].strip()
                                    expected_value_str = part.split("=", 1)[1].strip()

                                    # Evaluate the function call
                                    import time

                                    start_time = time.perf_counter()
                                    eva = _evaluate_safely(func_call_part)
                                    elapsed = time.perf_counter() - start_time

                                    if eva.get("ok"):
                                        res_str = eva.get("result")
                                        formatted_res = _format_special_values(res_str)
                                        print(
                                            f"{func_call_part} = {format_superscript(formatted_res)}"
                                        )
                                        # Check if it matches expected value
                                        try:
                                            expected_val = _parse_preprocessed(
                                                expected_value_str
                                            )
                                            computed_val = _parse_preprocessed(res_str)
                                            if (
                                                abs(
                                                    float(
                                                        sp.N(
                                                            expected_val - computed_val
                                                        )
                                                    )
                                                )
                                                < 1e-10
                                            ):
                                                print(
                                                    f"✓ Matches expected value: {expected_value_str}"
                                                )
                                            else:
                                                print(
                                                    f"  (Expected: {expected_value_str}, Got: {res_str})"
                                                )
                                        except Exception:
                                            pass
                                        if _timing_enabled:
                                            print(f"[Time: {elapsed:.4f}s]")
                                    else:
                                        print(
                                            f"Error evaluating {func_call_part}: {eva.get('error', 'Unknown error')}"
                                        )
                                else:
                                    # Format: f(1) - just evaluate
                                    import time

                                    start_time = time.perf_counter()
                                    eva = _evaluate_safely(part)
                                    elapsed = time.perf_counter() - start_time
                                    if eva.get("ok"):
                                        res_str = eva.get("result")
                                        formatted_res = _format_special_values(res_str)
                                        print(
                                            f"{part} = {format_superscript(formatted_res)}"
                                        )
                                        if _timing_enabled:
                                            print(f"[Time: {elapsed:.4f}s]")
                                    else:
                                        print(
                                            f"Error: {eva.get('error', 'Unknown error')}"
                                        )
                                handled_parts.append(part)
                                continue
                    except Exception:
                        pass

                    # Check if it's a regular expression (not a function call)
                    # If not handled yet, let it fall through to normal evaluation
                    if part not in handled_parts:
                        # Evaluate this part
                        try:
                            import time

                            start_time = time.perf_counter()
                            eva = _evaluate_safely(part)
                            elapsed = time.perf_counter() - start_time
                            if eva.get("ok"):
                                res_str = eva.get("result")
                                # Don't show "= None" for expressions that return None (like print())
                                # The side effect (e.g., printing) already happened, so showing "= None" is confusing
                                if res_str == "None" and "(" in part and ")" in part:
                                    # Looks like a function call that returned None - skip displaying the result
                                    # The side effect (like print output) was already displayed
                                    pass
                                else:
                                    print(f"{part} = {format_superscript(res_str)}")
                                if _timing_enabled:
                                    print(f"[Time: {elapsed:.4f}s]")
                            else:
                                error_msg = eva.get("error", "Unknown error")
                                error_code = eva.get("error_code", "")
                                # Provide helpful error message
                                if error_code == "PARSE_ERROR" and "f(" in part:
                                    print(
                                        f"Error: Could not parse '{part}'. Did you mean to call a function?"
                                    )
                                    print(
                                        f"  If '{part.split('(')[0] if '(' in part else part}' is a function, make sure it's defined first."
                                    )
                                    print(
                                        "  Use 'find f(x)' to find a function from data points, or 'f(x)=...' to define it."
                                    )
                                else:
                                    print(f"Error: {error_msg}")
                        except Exception as e:
                            print(f"Error evaluating '{part}': {e}")

                if handled_parts:
                    continue  # Already handled, don't process further

        # Check for single function definition (e.g., f(x)=2x)
        # BUT skip if it looks like a function finding pattern (f(-1)=3, not f(x)=3)
        # Function finding patterns have numeric arguments, function definitions have parameter names
        try:
            from .function_manager import define_function, parse_function_definition

            # Check if thisq(1)=1, sq(4)=2, sq(16)=4, find sq(x)s looks like a function finding pattern (numeric args) vs definition (parameter names)
            func_finding_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]+\)\s*=\s*[^,]+"
            if re.search(func_finding_pattern, raw):
                # Check if arguments are numeric (function finding) vs parameter names (function definition)
                matches = list(re.finditer(func_finding_pattern, raw))
                is_function_finding = False
                for m in matches:
                    func_name = m.group(1)
                    args_match = re.search(
                        rf"{re.escape(func_name)}\s*\(([^)]+)\)", m.group(0)
                    )
                    if args_match:
                        args_str = args_match.group(1).strip()
                        # If args contain numbers or negative signs, it's likely function finding
                        # If args are just letters, it's likely function definition
                        if (
                            re.search(r"[-+]?\d", args_str)
                            or args_str.startswith("-")
                            or args_str.startswith("+")
                        ):
                            is_function_finding = True
                            break

                if is_function_finding:
                    # Skip function definition parsing - this is a function finding pattern
                    func_def = None
                else:
                    # Might be a function definition, try parsing it
                    func_def = parse_function_definition(raw)
            else:
                func_def = parse_function_definition(raw)

            if func_def is not None:
                func_name, params, body = func_def
                try:
                    define_function(func_name, params, body)
                    params_str = ", ".join(params) if params else ""
                    print(f"Function '{func_name}({params_str})' defined as: {body}")
                except ValidationError as e:
                    print(f"Error: {e.message}")
                continue
        except ImportError:
            pass
        except Exception:
            pass

        try:
            import time

            start_time = time.perf_counter()
            if any(op in raw for op in ("<", ">", "<=", ">=")):
                res = solve_inequality(raw, None)
                elapsed = time.perf_counter() - start_time
                print_result_pretty(res, output_format=output_format)
                if _timing_enabled:
                    print(f"[Time: {elapsed:.4f}s]")
                continue

            # Check for "find f(x)" or "find f(x,y) = value" command

            if "find" in raw.lower():
                try:
                    from .function_manager import parse_find_function_command

                    find_func_cmd = parse_find_function_command(raw)

                    if find_func_cmd is not None:
                        # This is a function finding/inverse solving command
                        # Use the same logic as in --eval mode
                        try:
                            from .function_manager import (
                                define_function,
                                find_function_from_data,
                                parse_function_definition,
                            )

                            func_name, param_names = find_func_cmd
                            find_pattern = rf"find\s+{re.escape(func_name)}\s*\([^)]*\)"
                            data_str = re.sub(
                                find_pattern, "", raw, flags=re.IGNORECASE
                            ).strip()
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
                                func_def = parse_function_definition(part)
                                if func_def is not None:
                                    def_func_name, def_params, def_body = func_def
                                    try:
                                        define_function(
                                            def_func_name, def_params, def_body
                                        )
                                        params_str = (
                                            ", ".join(def_params) if def_params else ""
                                        )
                                        print(
                                            f"Function '{def_func_name}({params_str})' defined as: {def_body}"
                                        )
                                    except ValidationError as e:
                                        print(f"Error defining function: {e.message}")
                                else:
                                    # Not a function definition, keep for further processing
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
                                f"{func_name}(" in part
                                for part in split_top_level_commas(data_str)
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
                                from .function_manager import _function_registry

                                logger.debug(
                                    f"Checking registry for {func_name}: {func_name in _function_registry}, registry keys: {list(_function_registry.keys())}"
                                )
                                if func_name in _function_registry:
                                    from .solver import solve_inverse_function

                                    result = solve_inverse_function(
                                        func_name, target_value, param_names
                                    )
                                    _format_inverse_solutions(
                                        result, func_name, param_names, target_value
                                    )
                                    continue  # Skip further processing
                                else:
                                    print(
                                        f"Error: Function '{func_name}' is not defined. Define it first or provide data points."
                                    )
                                    continue

                            # If not inverse solve, fall through to function finding logic below (or implement it here)
                            # For now, let's implement the function finding logic here too to be complete

                            # Parse data points
                            data_points = []
                            parts = split_top_level_commas(data_str)
                            for part in parts:
                                part = part.strip()
                                if not part:
                                    continue
                                pattern = (
                                    rf"{re.escape(func_name)}\s*\(([^)]+)\)\s*=\s*(.+)"
                                )
                                match = re.match(pattern, part)
                                if match:
                                    args_str = match.group(1)
                                    value_str = match.group(2).strip()

                                    # Parse arguments
                                    args = []
                                    for arg in split_top_level_commas(args_str):
                                        try:
                                            arg_expr = _parse_preprocessed(arg.strip())
                                            try:
                                                arg_val = float(sp.N(arg_expr))
                                            except (ValueError, TypeError):
                                                arg_val = arg_expr
                                            args.append(arg_val)
                                        except Exception:
                                            break

                                    if len(args) == len(param_names):
                                        try:
                                            value_expr = _parse_preprocessed(value_str)
                                            try:
                                                value = float(sp.N(value_expr))
                                            except (ValueError, TypeError):
                                                value = value_expr
                                            data_points.append((args, value))
                                        except Exception:
                                            pass

                            if not data_points:
                                print(
                                    "Error: No valid data points found. Format: f(arg1,arg2,...) = value, ..."
                                )
                                continue

                            # Find function
                            success, func_str, factored_form, error_msg = (
                                find_function_from_data(data_points, param_names)
                            )
                            if success:
                                print(
                                    f"{func_name}({', '.join(param_names)}) = {func_str}"
                                )
                                if factored_form:
                                    print(
                                        f"Equivalent: {func_name}({', '.join(param_names)}) = {factored_form}"
                                    )
                                try:
                                    define_function(func_name, param_names, func_str)
                                    print(f"Function '{func_name}' is now available.")
                                except Exception as e:
                                    print(
                                        f"Warning: Could not define function automatically: {e}"
                                    )
                            else:
                                print(f"Error: {error_msg}")
                            continue

                        except Exception as e:
                            logger.exception("Error processing find function command")
                            print(f"Error: {e}")
                            continue
                except Exception:
                    pass

            # Check for "find" command with multiple variables (e.g., "find x, y")
            find_pattern = re.search(
                r"\bfind\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)\b",
                raw,
                re.IGNORECASE,
            )
            find_vars = None
            if find_pattern:
                find_vars_str = find_pattern.group(1)
                # Extract all variable names from "find x, y" -> ["x", "y"]
                find_vars = [
                    v.strip()
                    for v in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", find_vars_str)
                ]
                raw_no_find = re.sub(
                    r"\bfind\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*\b",
                    "",
                    raw,
                    flags=re.IGNORECASE,
                ).strip()
            else:
                # Try single variable pattern for backward compatibility
                find_tokens = re.findall(r"\bfind\s+(\w+)\b", raw, re.IGNORECASE)
                find_vars = find_tokens if find_tokens else None
                raw_no_find = re.sub(
                    r"\bfind\s+\w+\b", "", raw, flags=re.IGNORECASE
                ).strip()

            parts = split_top_level_commas(raw_no_find)
            if not parts:
                print("No valid parts parsed.")
                continue
            # Check for equations first (before trying to evaluate individual parts)
            all_assign = all(
                "=" in p and VAR_NAME_RE.match(p.split("=", 1)[0].strip())
                for p in parts
            )
            all_eq = all("=" in p for p in parts)
            if all_assign and len(parts) > 0:
                # If we have find_vars, convert assignments to equations and solve
                if find_vars and len(find_vars) > 0:
                    # Convert variable names like "xy" to "x*y" if x and y are in find_vars
                    converted_parts = []
                    for part in parts:
                        if "=" in part:
                            lhs, rhs = part.split("=", 1)
                            lhs = lhs.strip()
                            rhs = rhs.strip()
                            # Check if lhs is a concatenation of find_vars (e.g., "xy" when find_vars=["x", "y"])
                            # This is a heuristic: if lhs has no operators and all find_vars are single chars
                            if (
                                all(len(v) == 1 for v in find_vars)
                                and len(lhs) == len(find_vars)
                                and all(c in find_vars for c in lhs)
                            ):
                                # Convert "xy" to "x*y"
                                converted_lhs = "*".join(lhs)
                                converted_parts.append(f"{converted_lhs} = {rhs}")
                            else:
                                converted_parts.append(part)
                        else:
                            converted_parts.append(part)
                    raw_no_find = ", ".join(converted_parts)
                    # Use first find var for backward compatibility with solve_system
                    find = find_vars[0] if find_vars else None
                    res = solve_system(raw_no_find, find)
                    elapsed = time.perf_counter() - start_time
                    print_result_pretty(res, output_format=output_format)
                    if _timing_enabled:
                        print(f"[Time: {elapsed:.4f}s]")
                    continue

                # Check if all assignments are to the same variable
                # If so, evaluate each separately (like "x=1, x=2, x=3" -> evaluate each modulo/expression)
                assigned_vars = [p.split("=", 1)[0].strip() for p in parts if "=" in p]
                if len(assigned_vars) > 1 and len(set(assigned_vars)) == 1:
                    # All assignments are to the same variable - evaluate each expression separately
                    # This handles cases like "x = 1 % 2, x=3 % 6, x=3 % 7" where we want to evaluate each modulo
                    for p in parts:
                        if "=" in p:
                            left, right = p.split("=", 1)
                            var = left.strip()
                            rhs = right.strip() or "0"
                            # Evaluate the RHS expression (like "1 % 2")
                            res = _evaluate_safely(rhs)
                            if not res.get("ok"):
                                print(
                                    f"Error evaluating '{var} = {rhs}': {res.get('error')}"
                                )
                                if _timing_enabled:
                                    print(
                                        f"[Time: {time.perf_counter() - start_time:.4f}s]"
                                    )
                                continue
                            try:
                                # Format and print the result
                                val_str = res.get("result", "")
                                approx_str = res.get("approx", "")
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
                                if _timing_enabled:
                                    print(
                                        f"[Time: {time.perf_counter() - start_time:.4f}s]"
                                    )
                                continue
                    if _timing_enabled:
                        elapsed = time.perf_counter() - start_time
                        print(f"[Time: {elapsed:.4f}s]")
                    continue

                # Different variables - treat as chained assignments
                subs = {}
                for p in parts:
                    left, right = p.split("=", 1)
                    var = left.strip()
                    rhs = right.strip() or "0"

                    # Apply previous substitutions to the RHS expression BEFORE evaluation
                    if subs:
                        try:
                            rhs_expr = _parse_preprocessed(rhs)
                            rhs_expr = rhs_expr.subs(subs)
                            rhs = str(rhs_expr)
                        except Exception:
                            pass  # If parsing fails, use original rhs

                    res = _evaluate_safely(rhs)
                    if not res.get("ok"):
                        print("Error evaluating assignment RHS:", res.get("error"))
                        if _timing_enabled:
                            print(f"[Time: {time.perf_counter() - start_time:.4f}s]")
                        continue
                    try:
                        val = _parse_preprocessed(res["result"])
                    except Exception as e:
                        print("Error assembling assignment value:", e)
                        if _timing_enabled:
                            print(f"[Time: {time.perf_counter() - start_time:.4f}s]")
                        continue
                    # Use Symbol as key for proper SymPy substitution
                    subs[sp.Symbol(var)] = val
                    print(f"{var} = {format_solution(val)}")

                # Now handle find_vars if present
                if find_vars:
                    for find_var in find_vars:
                        find_sym = sp.Symbol(find_var)
                        if find_sym in subs:
                            print(
                                f"Result: {find_var} = {format_solution(subs[find_sym])}"
                            )
                        else:
                            print(f"Variable '{find_var}' not found in assignments.")

                if _timing_enabled:
                    elapsed = time.perf_counter() - start_time
                    print(f"[Time: {elapsed:.4f}s]")
                continue
            if len(parts) > 1 and all_eq:
                # Before solving as system, check if these look like function calls
                # Pattern: func_name(args) = value
                looks_like_function_calls = False
                try:
                    from .function_manager import list_functions

                    defined_funcs = list_functions()
                    for part in parts:
                        func_match = re.match(
                            r"^(\w+)\s*\([^)]*\)\s*=\s*(.+)$", part.strip()
                        )
                        if func_match:
                            func_name = func_match.group(1)
                            if func_name in defined_funcs:
                                looks_like_function_calls = True
                                break
                except Exception:
                    pass

                if looks_like_function_calls:
                    # These are function calls, not equations to solve
                    print(
                        "Error: These look like function calls, not equations to solve."
                    )
                    print("  To evaluate function calls, use: f(1), f(2), f(3)")
                    print("  To verify function values, use: f(1)=value, f(2)=value")
                    print(
                        "  To find a function from data, use: f(1)=value, f(2)=value, find f(x)"
                    )
                    if _timing_enabled:
                        print(f"[Time: {time.perf_counter() - start_time:.4f}s]")
                    continue

                res = solve_system(raw_no_find, find)
                elapsed = time.perf_counter() - start_time

                # Check if the system solving failed and provide better error message
                if not res.get("ok"):
                    error_msg = res.get("error", "Unknown error")
                    # Check if it looks like function calls were misinterpreted
                    if any(re.match(r"^\w+\s*\([^)]*\)", p.strip()) for p in parts):
                        print("Error: Could not solve this as a system of equations.")
                        print("  Did you mean to evaluate function calls?")
                        print("  Use: f(1), f(2), f(3) to evaluate functions")
                        print(
                            "  Or: f(1)=value, f(2)=value, f(3)=value to verify function values"
                        )
                    else:
                        print(f"Error: {error_msg}")
                else:
                    print_result_pretty(res, output_format=output_format)
                if _timing_enabled:
                    print(f"[Time: {elapsed:.4f}s]")
                continue
            elif len(parts) > 1:
                # Check if there's a "find" command that we missed
                find_pattern_in_raw = re.search(
                    r"\bfind\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)\b",
                    raw,
                    re.IGNORECASE,
                )
                if find_pattern_in_raw:
                    # Try to handle it as a system with find
                    find_vars_str = find_pattern_in_raw.group(1)
                    find_vars = [
                        v.strip()
                        for v in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", find_vars_str)
                    ]
                    raw_no_find = re.sub(
                        r"\bfind\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*\b",
                        "",
                        raw,
                        flags=re.IGNORECASE,
                    ).strip()
                    parts_for_find = split_top_level_commas(raw_no_find)
                    # Try to solve as system
                    try:
                        import time

                        start_time = time.perf_counter()
                        find = find_vars[0] if find_vars else None
                        res = solve_system(raw_no_find, find)
                        elapsed = time.perf_counter() - start_time
                        print_result_pretty(res, output_format=output_format)
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                        continue
                    except Exception as e:
                        logger.debug(f"Failed to solve system with find: {e}")
                        # Fall through to error message
                print(
                    "Error: Multiple expressions detected. Use commas to separate equations in a system, or use 'find' to specify which variables to solve for."
                )
                print("  Example: xy=12, find x, y")
                print("  Example: x+y=5, x-y=1, find x")
                continue
            else:
                part = parts[0]
                # Initialize find if not already set (check for "find" keyword in the input)
                if "=" in part:
                    # Check if this looks like a function finding pattern (f(-1)=3) vs equation
                    # This MUST happen before solve_single_equation to prevent π-style output
                    func_finding_pattern_check = re.match(
                        r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]+\)\s*=\s*[^,]+", part
                    )
                    if func_finding_pattern_check:
                        func_name_check = func_finding_pattern_check.group(1)
                        args_match_check = re.search(
                            rf"{re.escape(func_name_check)}\s*\(([^)]+)\)", part
                        )
                        if args_match_check:
                            args_str_check = args_match_check.group(1).strip()
                            # If args contain numbers, it's function finding, not equation solving
                            if re.search(r"[-+]?\d", args_str_check):
                                # Process as function finding directly
                                try:
                                    from .function_manager import (
                                        define_function,
                                        find_function_from_data,
                                    )

                                    # Extract value
                                    value_match = re.search(r"=\s*(.+)", part)
                                    if value_match:
                                        value_str = value_match.group(1).strip()
                                        # Parse arguments
                                        arg_list = []
                                        for arg in args_str_check.split(","):
                                            try:
                                                arg_list.append(float(arg.strip()))
                                            except ValueError:
                                                arg_list.append(arg.strip())

                                        # Parse value
                                        try:
                                            value = float(value_str)
                                        except ValueError:
                                            value = value_str

                                        # Find function (single data point = constant function)
                                        param_names_single = ["x"]
                                        data_points_single = [
                                            (
                                                (
                                                    [arg_list[0]]
                                                    if len(arg_list) == 1
                                                    else arg_list
                                                ),
                                                value,
                                            )
                                        ]
                                        success, func_str, factored_form, error_msg = (
                                            find_function_from_data(
                                                data_points_single, param_names_single
                                            )
                                        )
                                        if success:
                                            print(f"{func_name_check}(x) = {func_str}")
                                            if factored_form:
                                                print(
                                                    f"Equivalent: {func_name_check}(x) = {factored_form}"
                                                )
                                            try:
                                                define_function(
                                                    func_name_check,
                                                    param_names_single,
                                                    func_str,
                                                )
                                                print(
                                                    f"Function '{func_name_check}' is now available."
                                                )
                                            except Exception as e:
                                                print(
                                                    f"Warning: Could not define function automatically: {e}"
                                                )
                                            continue
                                        else:
                                            print(f"Error: {error_msg}")
                                            continue
                                except Exception as e:
                                    print(f"Error processing function finding: {e}")
                                    continue
                    # Initialize find if not already set (check for "find" keyword in the input)
                    find = None
                    if "find" in raw.lower():
                        find_tokens = re.findall(
                            r"\bfind\s+(\w+)\b", raw, re.IGNORECASE
                        )
                        find = find_tokens[0] if find_tokens else None
                    res = solve_single_equation(part, find)
                    elapsed = time.perf_counter() - start_time
                    # Show cache hits if enabled
                    if _cache_hits_enabled:
                        try:
                            # Get cache hits from result (solve_single_equation captures them)
                            cache_hits = res.get("cache_hits", [])
                            if cache_hits:
                                print("\n[Cache hits used:]")
                                for hit_expr, cache_type in cache_hits:
                                    cache_type_name = (
                                        "eval" if cache_type == "eval" else "subexpr"
                                    )
                                    print(f"  {hit_expr} ({cache_type_name})")
                        except Exception:
                            # Don't let cache hits display failure break the evaluation
                            # Silently continue - cache hits are optional
                            pass
                    print_result_pretty(res, output_format=output_format)
                    if _timing_enabled:
                        print(f"[Time: {elapsed:.4f}s]")
                else:
                    eva = _evaluate_safely(part)
                    elapsed = time.perf_counter() - start_time
                    # Show cache hits if enabled
                    if _cache_hits_enabled:
                        try:
                            # Get cache hits from result (evaluate_safely captures and attaches them)
                            cache_hits = eva.get("cache_hits", [])
                            if cache_hits:
                                print("\n[Cache hits used:]")
                                for hit_expr, cache_type in cache_hits:
                                    cache_type_name = (
                                        "eval" if cache_type == "eval" else "subexpr"
                                    )
                                    print(f"  {hit_expr} ({cache_type_name})")
                        except Exception:
                            # Don't let cache hits display failure break the evaluation
                            # Silently continue - cache hits are optional
                            pass
                    if not eva.get("ok"):
                        error_msg = eva.get("error", "Unknown error")
                        error_code = eva.get("error_code", "UNKNOWN_ERROR")
                        print(f"Error: {error_msg}")
                        if error_code == "COMMAND_IN_EXPRESSION":
                            print(
                                "Hint: Commands must be entered on separate lines. "
                                "Each command or expression should be on its own line."
                            )
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                    else:
                        res_str = eva.get("result")
                        # Don't show "= None" for expressions that return None (like print())
                        # The side effect (e.g., printing) already happened, so showing "= None" is confusing
                        if res_str == "None" and "(" in part and ")" in part:
                            # Looks like a function call that returned None - skip displaying the result
                            # The side effect (like print output) was already displayed
                            pass
                        else:
                            formatted_res = _format_special_values(res_str)
                            print(f"{part} = {format_superscript(formatted_res)}")
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                        try:
                            parsed = _parse_preprocessed(res_str)
                            expanded = sp.expand(parsed)
                            if str(expanded) != str(parsed):
                                print(f"Expanded: {format_solution(expanded)}")
                        except (
                            ParseError,
                            ValidationError,
                            ValueError,
                            TypeError,
                            AttributeError,
                        ):
                            # Expected errors for some expressions - silently skip expansion
                            pass
                        if eva.get("approx"):
                            approx_formatted = _format_number_no_trailing_zeros(
                                eva.get("approx")
                            )
                            # π-fraction conversion (Casio-style) disabled
                            print(
                                f"Decimal: {_format_special_values(approx_formatted)}"
                            )
        except Exception as e:
            try:
                from .logging_config import get_logger

                logger = get_logger("cli")
                logger.error(f"Unexpected error in REPL: {e}", exc_info=True)
            except ImportError:
                pass
            print("An error occurred. Please check your input and try again.")
            continue


def print_help_text() -> None:
    """Print help text for REPL commands."""
    from .config import VERSION

    help_text = f"""kalkulator-ai v{VERSION}

COMMANDS:
  help, ?              Show this command list
  quit, exit           Exit the calculator
  health               Run health check

CACHE:
  clearcache           Clear expression cache
  showcache [all]      Show cached expressions
  savecache [file]     Save cache to file
  loadcache [file]     Load cache from file

SETTINGS:
  timing [on|off]      Toggle calculation timing
  cachehits [on|off]   Toggle cache hit display

MATH:
  diff(expr, x)        Differentiate expression
  integrate(expr, x)   Integrate expression
  factor(expr)         Factor expression
  expand(expr)         Expand expression
  plot expr [opts]     Plot function

FUNCTION FINDING:
  f(1)=1, f(2)=4, find f(x)     Discover function from data
  evolve f(x) from x=[...], y=[...]   Genetic programming
  find ode from t=[...], x=[...]      ODE discovery (SINDy)
  benchmark N                          Run Feynman benchmarks

EXPORT:
  export <func> to <file.py>   Export function to Python

For full documentation: https://github.com/sizzlins/kalkulator-ai
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
        from .cache_manager import load_persistent_cache

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
        from .logging_config import setup_logging

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
        from .worker import worker_evaluate

        out = worker_evaluate(args.expr or "")
        print(json.dumps(out))
        return 0
    if args.worker_solve:
        from .worker import _worker_solve_dispatch

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
                from .function_manager import (
                    find_function_from_data,
                    parse_find_function_command,
                )

                find_func_cmd = parse_find_function_command(expr)
                if find_func_cmd is not None:
                    is_find_command = True
            except Exception:
                pass

        # If no explicit "find" keyword, check for multiple function assignments
        if not is_find_command:
            try:
                from .function_manager import (  # noqa: F811
                    find_function_from_data,
                    parse_find_function_command,
                )

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

                from .function_manager import find_function_from_data  # noqa: F811
                from .parser import parse_preprocessed as _parse_preprocessed

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
                        from .function_manager import (
                            ValidationError,
                            define_function,
                            parse_function_definition,
                        )

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
                    from .function_manager import _function_registry

                    if func_name in _function_registry:
                        from .solver import solve_inverse_function

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
                                arg_expr = _parse_preprocessed(arg_stripped)
                                try:
                                    arg_val = float(sp.N(arg_expr))
                                except (ValueError, TypeError):
                                    arg_val = arg_expr
                                args.append(arg_val)
                            except Exception:
                                break

                        if len(args) == len(param_names):
                            try:
                                if value_str_preserved:
                                    value = value_str_preserved
                                else:
                                    value_expr = _parse_preprocessed(value_str)
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
                logger.exception("Error processing function finding in --eval")
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
                                from .cache_manager import save_cache_to_disk

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
                            from .cache_manager import save_cache_to_disk

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
            from .cache_manager import save_cache_to_disk

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
            from .worker import _WORKER_MANAGER

            _WORKER_MANAGER.stop()
        except Exception:
            pass
        # Save persistent cache on exit
        try:
            from .cache_manager import save_cache_to_disk

            save_cache_to_disk()
        except ImportError:
            pass
    return 0


if __name__ == "__main__":
    """Allow running the CLI module directly with python -m kalkulator_pkg.cli"""
    import sys

    sys.exit(main_entry())
