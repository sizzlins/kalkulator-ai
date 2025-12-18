import json
import logging
import re
from math import gcd

import sympy as sp

logger = logging.getLogger(__name__)

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


def parse_target_with_ambiguity_detection(
    target_str: str, max_small_denominator: int = 12
) -> tuple[sp.Rational, sp.Rational | None, bool]:
    """Parse target string with ambiguity detection for simpler rationals."""
    from decimal import Decimal
    from fractions import Fraction

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
            from kalkulator_pkg.parser import parse_preprocessed

            target_expr = parse_preprocessed(target_str)
            if isinstance(target_expr, (sp.Float, float)):
                target_expr = sp.Rational(str(target_expr))
            return (target_expr, None, False)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot parse target '{target_str}'") from None


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm.
    Returns (s, t, g) such that s*a + t*b = g = gcd(a, b).
    """
    if b == 0:
        return (1, 0, a)
    else:
        s1, t1, g = extended_gcd(b, a % b)
        s = t1
        t = s1 - (a // b) * t1
        return (s, t, g)


def compute_integerized_equation(
    coeffs: list[sp.Rational], target: sp.Rational, L_func: int
) -> tuple[int, int, int, int] | None:
    """Compute integerized form of linear equation A*x + B*y = C for integer solution finding."""
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
            raise ValueError(f"Non-numeric coefficient found: {coef!r}")

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
        raise ValueError(f"Coefficient is not rational/integer: {sympy_number!r}")

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
    g = gcd(A, B)
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


def solve_modulo_system_if_applicable(
    parts: list[str], var: str, output_format: str = "human"
) -> tuple[bool, int]:
    """Check if parts form a system of congruences and solve it using CRT."""
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
            from kalkulator_pkg.solver import solve_system_of_congruences

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
