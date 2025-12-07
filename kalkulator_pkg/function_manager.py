"""Function definition, evaluation, and finding module.

This module provides comprehensive function management capabilities:

1. Function Definition:
   - Define functions with single or multiple variables (e.g., f(x)=2x, g(x,y,z)=x+y+z)
   - Support for nested function definitions
   - Validation of function names and parameters

2. Function Evaluation:
   - Evaluate functions with numeric or symbolic arguments (e.g., f(2), f(1,2,3))
   - Support for nested function calls (e.g., g(f(5)))
   - Automatic substitution of arguments into function body

3. Function Finding:
   - Polynomial interpolation using Lagrange method for exact fits
   - Linear regression for multi-variable functions
   - Newton divided differences as fallback method
   - Automatic simplification of rational coefficients
   - Support for symbolic values (e.g., pi) in data points

Examples:
    Define: f(x)=2*x, g(x,y)=x+y
    Evaluate: f(2) → 4, g(1,2) → 3
    Find: f(1)=1, f(2)=2, find f(x) → f(x) = x
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any

import sympy as sp
from sympy import parse_expr

from .config import ALLOWED_SYMPY_NAMES, TRANSFORMATIONS
from .types import ValidationError

# Built-in function names that should not be used as user-defined function names
BUILTIN_FUNCTION_NAMES = set(ALLOWED_SYMPY_NAMES.keys())


def _parse_to_exact_fraction(
    val: float | str | int | Fraction, max_denominator: int = 10000
) -> Fraction:
    """Parse input to exact Fraction for exact arithmetic.

    Accepts formats: integer, "a/b", decimal string.
    Uses fractions.Fraction for exact arithmetic.
    For decimal strings, tries to detect repeating patterns and suggest rational approximations.

    CRITICAL: For function finding, we need to detect when a decimal like "70.833" is actually
    a rounded version of a simpler rational like 425/6. This ensures exact coefficients.

    Args:
        val: Input value (int, float, str, or Fraction)
        max_denominator: Maximum denominator for detecting repeating patterns (default 10000)

    Returns:
        Fraction object with exact value
    """
    # If already a Fraction, return as is
    if isinstance(val, Fraction):
        return val

    # If it's an integer, convert directly
    if isinstance(val, int):
        return Fraction(val)

    # If it's a string, parse it
    if isinstance(val, str):
        val = val.strip()
        # Try to parse as fraction (e.g., "425/6")
        if "/" in val:
            try:
                parts = val.split("/")
                if len(parts) == 2:
                    num = int(parts[0].strip())
                    den = int(parts[1].strip())
                    return Fraction(num, den)
            except (ValueError, TypeError):
                # Not a simple a/b fraction - might be symbolic like "pi/2"
                pass

        # Try to parse as decimal string (e.g., "70.833")
        try:
            # First, parse as exact decimal representation
            # For "70.833", this gives 70833/1000
            exact_frac = Fraction(val)
            exact_val = float(exact_frac)

            # CRITICAL: For function finding, we need to detect if a decimal is a rounded
            # version of a simpler rational. For example:
            # - "70.833" might be rounded from 425/6 = 70.8333333...
            # - "15.333" might be rounded from 46/3 = 15.3333333...
            # - "122.833" might be rounded from 737/6 = 122.8333333...
            #
            # Strategy: Check if the decimal is very close to a simpler rational with
            # a small denominator. Use a tolerance that accounts for rounding to 3 decimals.

            # Determine decimal places in the input string
            decimal_places = 0
            if "." in val:
                decimal_part = val.split(".")[1]
                decimal_places = len(decimal_part)
                # Tolerance based on decimal places: for 3 decimals, use 0.0005 (half of 0.001)
                # This ensures we catch values rounded to 3 decimals
                if decimal_places <= 3:
                    tolerance = 0.0005  # For 1-3 decimal places, use 0.0005
                elif decimal_places <= 6:
                    tolerance = 0.0000005  # For 4-6 decimal places, use 0.0000005
                else:
                    tolerance = 10 ** (
                        -decimal_places - 1
                    )  # Half of last decimal place
            else:
                tolerance = 1e-6  # Default tolerance for integers

            # Try to find a simpler rational by checking progressively larger denominators
            # Start with small denominators (common in fractions)
            # Strategy: Find the fraction with the SMALLEST denominator that's within tolerance
            # This prioritizes simplicity (e.g., 425/6) over closeness (e.g., 35204/497)
            best_frac = exact_frac
            best_denom = exact_frac.denominator

            # Check denominators from small to larger
            # Stop as soon as we find a match - we want the smallest denominator
            for max_denom in [6, 8, 12, 16, 20, 24, 30, 50, 100, 200, 500, 1000, 2000]:
                if max_denom > max_denominator:
                    break

                limited_frac = exact_frac.limit_denominator(max_denom)

                # Check if the limited fraction is simpler and very close
                if limited_frac.denominator < exact_frac.denominator:
                    diff = abs(float(exact_frac - limited_frac))

                    # Use simpler rational if within tolerance
                    # This handles cases like:
                    # - 70.833 ≈ 425/6 (difference ~0.000333, within 0.0005)
                    # - 15.333 ≈ 46/3 (difference ~0.000333, within 0.0005)
                    # - 122.833 ≈ 737/6 (difference ~0.000333, within 0.0005)
                    if diff <= tolerance:
                        # Found a simpler rational within tolerance
                        # Prefer smaller denominator (simpler fraction)
                        if limited_frac.denominator < best_denom:
                            best_frac = limited_frac
                            best_denom = limited_frac.denominator
                            # Once we find a match with a small denominator, we can stop
                            # (or continue to check if an even smaller denominator exists)

            # Return the best (simplest) rational found, or exact if no simpler one found
            return best_frac
        except (ValueError, TypeError):
            # Not a simple decimal - try symbolic evaluation (e.g., "pi", "e", "pi/2")
            try:
                import sympy as sp_sym

                local_ns = {
                    "pi": sp_sym.pi,
                    "e": sp_sym.E,
                    "E": sp_sym.E,
                    "I": sp_sym.I,
                    "sqrt": sp_sym.sqrt,
                    "sin": sp_sym.sin,
                    "cos": sp_sym.cos,
                    "tan": sp_sym.tan,
                    "log": sp_sym.log,
                    "ln": sp_sym.log,
                    "exp": sp_sym.exp,
                }
                expr = sp_sym.sympify(val, locals=local_ns)
                result = expr.evalf()
                if hasattr(result, "free_symbols") and result.free_symbols:
                    raise ValueError(
                        f"Expression '{val}' contains free symbols: {result.free_symbols}"
                    )
                float_val = float(result)
                # Convert the float result to a Fraction
                val_str = format(float_val, ".15f").rstrip("0").rstrip(".")
                return Fraction(val_str)
            except Exception:
                # Try to parse as float first, then convert
                try:
                    float_val = float(val)
                    # Convert via string to preserve precision
                    val_str = format(float_val, ".15f").rstrip("0").rstrip(".")
                    return Fraction(val_str)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Cannot parse '{val}' as a number") from e

    # If it's a float, convert via string to preserve precision
    if isinstance(val, float):
        # Convert to string to avoid float rounding issues
        val_str = format(val, ".15f").rstrip("0").rstrip(".")
        try:
            exact_frac = Fraction(val_str)
            # Try to find simpler rational for floats too
            exact_val = float(exact_frac)
            tolerance = 1e-6  # Default tolerance for floats
            best_frac = exact_frac
            best_denom = exact_frac.denominator
            for max_denom in [6, 8, 12, 16, 20, 24, 30, 50, 100, 200, 500, 1000]:
                if max_denom > max_denominator:
                    break
                limited_frac = exact_frac.limit_denominator(max_denom)
                if limited_frac.denominator < exact_frac.denominator:
                    diff = abs(float(exact_frac - limited_frac))
                    if diff <= tolerance:
                        if limited_frac.denominator < best_denom:
                            best_frac = limited_frac
                            best_denom = limited_frac.denominator
            return best_frac
        except (ValueError, TypeError):
            return Fraction(val)

    raise ValueError(f"Cannot convert {val} to Fraction")


def _float_to_exact_rational(val: float | str) -> sp.Basic:
    """Convert a float or string to an exact SymPy Rational, preserving decimal precision.

    For floats or strings representing finite decimals (like 70.833), this preserves
    exact precision by using Fraction then converting to SymPy Rational.

    Args:
        val: Float value or string representation to convert

    Returns:
        SymPy Rational or Float expression with exact or high precision
    """
    try:
        # Use Fraction for exact parsing, then convert to SymPy Rational
        frac = _parse_to_exact_fraction(val)
        return sp.Rational(frac.numerator, frac.denominator)
    except (ValueError, TypeError):
        # Fallback to original behavior for symbolic expressions
        if isinstance(val, str):
            try:
                return sp.Rational(val)
            except (ValueError, TypeError):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    return sp.Float(val, 15)

        if isinstance(val, int):
            return sp.Rational(val)
        if isinstance(val, float) and val == int(val):
            return sp.Rational(int(val))

        # Try to use Decimal for exact decimal representation
        try:
            from decimal import Decimal

            dec_str = format(val, ".15f").rstrip("0").rstrip(".")
            dec_val = Decimal(dec_str)
            return sp.Rational(str(dec_val))
        except (ImportError, ValueError, TypeError):
            try:
                val_str = format(val, ".15f").rstrip("0").rstrip(".")
                if "e" not in val_str.lower() and "." in val_str:
                    return sp.Rational(val_str)
                else:
                    return sp.Float(val, 15)
            except (ValueError, TypeError):
                return sp.Float(val, 15)


# Global function registry: {function_name: (parameter_names, body_expression)}
_function_registry: dict[str, tuple[list[str], sp.Basic]] = {}


def clear_functions() -> None:
    """Clear all defined functions."""
    global _function_registry
    _function_registry.clear()


def list_functions() -> dict[str, tuple[list[str], str]]:
    """List all defined functions.

    Returns:
        Dictionary mapping function names to (parameters, body_string) tuples
    """
    result = {}
    for name, (params, body) in _function_registry.items():
        result[name] = (params, str(body))
    return result


def get_function_registry_hash() -> str:
    """Get a hash of the current function registry state.

    Returns:
        MD5 hash string of the registry content
    """
    import hashlib

    # Sort keys to ensure consistent order
    # Convert values to string representation for hashing
    items = []
    for name in sorted(_function_registry.keys()):
        params, body = _function_registry[name]
        items.append((name, tuple(params), str(body)))

    registry_str = str(items)
    return hashlib.md5(registry_str.encode()).hexdigest()


def define_function(name: str, params: list[str], body_expr: str) -> None:
    """Define a function.

    Args:
        name: Function name (e.g., "f", "g")
        params: List of parameter names (e.g., ["x", "y"])
        body_expr: Function body as string (e.g., "2*x", "x+y+z")

    Raises:
        ValidationError: If function name or parameters are invalid
    """
    # Validate function name
    if not name or not name.isalnum() or not name[0].isalpha():
        raise ValidationError(
            f"Invalid function name: {name}. Must start with a letter and contain only letters and numbers.",
            "INVALID_FUNCTION_NAME",
        )

    # Validate parameters
    for param in params:
        if not param or not param.isalnum() or not param[0].isalpha():
            raise ValidationError(
                f"Invalid parameter name: {param}. Must start with a letter and contain only letters and numbers.",
                "INVALID_PARAMETER_NAME",
            )

    # Parse body expression
    try:
        # Create a local dict with parameter symbols
        local_dict = {**ALLOWED_SYMPY_NAMES}
        for param in params:
            local_dict[param] = sp.Symbol(param)

        body = parse_expr(
            body_expr,
            local_dict=local_dict,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as e:
        raise ValidationError(
            f"Failed to parse function body '{body_expr}': {str(e)}",
            "FUNCTION_BODY_PARSE_ERROR",
        ) from e

    # Store function definition
    _function_registry[name] = (params, body)


def evaluate_function(name: str, args: list[Any]) -> sp.Basic:
    """Evaluate a function with given arguments.

    Args:
        name: Function name
        args: List of argument values (can be numbers or SymPy expressions)

    Returns:
        SymPy expression result

    Raises:
        ValidationError: If function not found or wrong number of arguments
    """
    if name not in _function_registry:
        raise ValidationError(
            f"Function '{name}' is not defined. Define it first with {name}(params)=body.",
            "FUNCTION_NOT_FOUND",
        )

    params, body = _function_registry[name]

    if len(args) != len(params):
        raise ValidationError(
            f"Function '{name}' expects {len(params)} argument(s) ({', '.join(params)}), "
            f"but got {len(args)} argument(s).",
            "WRONG_ARGUMENT_COUNT",
        )

    # Create substitution dictionary
    subs_dict = {}
    for param, arg in zip(params, args):
        # Convert arg to SymPy expression if needed
        if not isinstance(arg, sp.Basic):
            try:
                arg = sp.sympify(arg)
            except Exception:
                arg = sp.Symbol(str(arg))  # Fallback to symbol

        subs_dict[sp.Symbol(param)] = arg

    # Substitute and return
    try:
        result = body.subs(subs_dict)
        return result
    except Exception as e:
        raise ValidationError(
            f"Failed to evaluate function '{name}': {str(e)}",
            "FUNCTION_EVAL_ERROR",
        ) from e


def parse_function_definition(expr: str) -> tuple[str, list[str], str] | None:
    """Parse a function definition like 'f(x)=2x' or 'f(x,y,z)=x+y+z'.

    Args:
        expr: Expression string (e.g., "f(x)=2x", "g(x,y)=x+y")

    Returns:
        Tuple of (function_name, parameter_list, body_string) if valid, None otherwise
    """
    # Pattern: function_name(param1,param2,...) = body
    # Allow spaces: f(x, y) = 2*x
    # Must match the ENTIRE string (no trailing content after the body)
    # The body should not contain another function definition pattern
    expr_stripped = expr.strip()

    # First, check if there's a comma followed by another function pattern
    # This would indicate incomplete or malformed input like "f(x)=2x, f(x)="
    if "," in expr_stripped:
        # Check if there's another function definition pattern after comma
        comma_idx = expr_stripped.find(",")
        after_comma = expr_stripped[comma_idx + 1 :].strip()
        if re.match(r"^[a-zA-Z][a-zA-Z0-9]*\s*\([^)]*\)\s*=", after_comma):
            # Looks like incomplete definition, reject
            return None

    pattern = r"^([a-zA-Z][a-zA-Z0-9]*)\s*\(([^)]*)\)\s*=\s*(.+?)$"
    match = re.match(pattern, expr_stripped)
    if not match:
        return None

    func_name = match.group(1)
    params_str = match.group(2).strip()
    body = match.group(3).strip()

    # Validate: function name must not be a built-in function (e.g., sin, cos, log, etc.)
    # This prevents "sin(x)=cos(x)" from being treated as a function definition
    if func_name in BUILTIN_FUNCTION_NAMES:
        return None

    # Validate: body must not be empty
    if not body:
        return None

    # Validate: body should not contain another function definition pattern
    # (e.g., "f(x)=2x, f(x)=" should not match)
    if re.search(r"[a-zA-Z][a-zA-Z0-9]*\s*\([^)]*\)\s*=", body):
        return None

    # Parse parameters - must be valid identifiers (start with letter)
    if not params_str:
        params = []
    else:
        # Split by comma, handling spaces
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        # Validate all parameters are valid identifiers
        for param in params:
            if (
                not param
                or not param[0].isalpha()
                or not param.replace("_", "").isalnum()
            ):
                return None  # Invalid parameter name

            # CRITICAL FIX: Prevent using reserved names (like pi, e, sin) as parameters
            # This prevents "f(pi)=0" from being interpreted as a function definition
            if param in BUILTIN_FUNCTION_NAMES:
                return None

    return (func_name, params, body)


def parse_function_call(expr: str) -> tuple[str, list[str]] | None:
    """Parse a function call like 'f(2)' or 'f(1,2,3)'.

    Args:
        expr: Expression string (e.g., "f(2)", "g(1,2,3)", "f((pi)/(2))")

    Returns:
        Tuple of (function_name, argument_strings) if valid, None otherwise
    """
    expr = expr.strip()

    # Find function name at start: letters followed by alphanumerics
    if not expr or not expr[0].isalpha():
        return None

    i = 0
    while i < len(expr) and (expr[i].isalnum() or expr[i] == "_"):
        i += 1
    func_name = expr[:i]

    # Skip optional whitespace
    while i < len(expr) and expr[i].isspace():
        i += 1

    # Must have opening parenthesis
    if i >= len(expr) or expr[i] != "(":
        return None
    i += 1  # Skip opening paren

    # Find matching closing parenthesis (handle nested parens)
    args_start = i
    depth = 1
    while i < len(expr) and depth > 0:
        if expr[i] == "(":
            depth += 1
        elif expr[i] == ")":
            depth -= 1
        i += 1

    if depth != 0:
        return None  # Unbalanced parentheses

    # i now points just past the closing paren
    args_str = expr[args_start : i - 1].strip()  # -1 to exclude closing paren

    # Check no trailing content (after optional whitespace)
    while i < len(expr) and expr[i].isspace():
        i += 1
    if i != len(expr):
        return None  # Unexpected trailing content

    # Parse arguments by splitting at top-level commas
    if not args_str:
        args = []
    else:
        args = []
        current = []
        depth = 0
        for char in args_str:
            if char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            args.append("".join(current).strip())

    return (func_name, args)


def _monomial_exponents(
    n: int, degree: int = 2, include_inverse: bool = False
) -> list[tuple[int, ...]]:
    """Generate all monomial exponents up to total degree for n variables.

    Returns list of exponent-tuples with sum(|exps|) <= degree, sorted for stable order.
    For 4 variables and degree 2: [(0,0,0,0), (1,0,0,0), (0,1,0,0), ..., (2,0,0,0), (1,1,0,0), ...]

    If include_inverse=True, also includes negative exponents (inverse terms like 1/z, 1/z^2).
    For 3 variables with inverse: includes terms like (1,1,-2) for x*y/z^2.

    Args:
        n: Number of variables
        degree: Maximum total degree (sum of absolute values of exponents)
        include_inverse: If True, include negative exponents for inverse terms

    Returns:
        List of exponent tuples, ordered deterministically
    """
    exps = []

    def gen(prefix: list[int], vars_left: int, deg_left: int) -> None:
        if vars_left == 0:
            exps.append(tuple(prefix))
            return
        if include_inverse:
            # Include negative exponents: -deg_left to deg_left
            # But we need to ensure sum of absolute values <= degree
            # For each variable, try all exponents from -deg_left to deg_left
            for e in range(-deg_left, deg_left + 1):
                # Remaining degree after using |e| for this variable
                new_deg_left = deg_left - abs(e)
                if (
                    new_deg_left >= 0
                ):  # Only continue if we haven't exceeded the degree limit
                    gen(prefix + [e], vars_left - 1, new_deg_left)
        else:
            # Only non-negative exponents
            for e in range(deg_left + 1):
                gen(prefix + [e], vars_left - 1, deg_left - e)

    gen([], n, degree)
    return exps


def _eval_monomial(sample: list[Fraction], exps: tuple[int, ...]) -> Fraction:
    """Evaluate a monomial with given exponents on a sample.

    Supports both positive and negative exponents (for inverse terms).

    Args:
        sample: List of Fraction values for each variable
        exps: Tuple of exponents for each variable (can be negative)

    Returns:
        Fraction value of the monomial
    """
    val = Fraction(1)
    for x, e in zip(sample, exps):
        if e == 0:
            continue
        if e < 0:
            # Negative exponent: 1 / x^|e|
            if x == 0:
                raise ZeroDivisionError("Division by zero in monomial evaluation")
            val *= Fraction(1) / (Fraction(x) ** abs(e))
        else:
            val *= Fraction(x) ** e
    return val


def _build_polynomial_matrix(
    samples: list[tuple[list[Any], Any]], degree: int = 2, include_inverse: bool = False
) -> tuple[list[list[Fraction]], list[Fraction], list[tuple[int, ...]]]:
    """Build matrix and RHS for polynomial/rational basis expansion.

    Args:
        samples: List of (input_values, output_value) tuples
        degree: Maximum total degree of monomials (sum of absolute values)
        include_inverse: If True, include negative exponents for inverse terms

    Returns:
        Tuple of (matrix_rows, rhs_vector, exponent_list)
        All values are Fraction for exact arithmetic
    """
    if not samples:
        return [], [], []

    nvars = len(samples[0][0])
    exps_list = _monomial_exponents(nvars, degree, include_inverse=include_inverse)

    rows = []
    rhs = []

    for vars_tuple, out in samples:
        # Convert input values to Fractions
        sample_fracs = []
        for val in vars_tuple:
            if isinstance(val, Fraction):
                sample_fracs.append(val)
            elif isinstance(val, (int, float)):
                sample_fracs.append(_parse_to_exact_fraction(val))
            elif isinstance(val, str):
                sample_fracs.append(_parse_to_exact_fraction(val))
            else:
                # Try to convert to Fraction
                try:
                    sample_fracs.append(Fraction(str(val)))
                except (ValueError, TypeError):
                    sample_fracs.append(Fraction(float(val)))

        # Build row by evaluating each monomial
        row = [_eval_monomial(sample_fracs, exps) for exps in exps_list]
        rows.append(row)

        # Convert output to Fraction
        if isinstance(out, Fraction):
            rhs.append(out)
        elif isinstance(out, (int, float)):
            rhs.append(_parse_to_exact_fraction(out))
        elif isinstance(out, str):
            rhs.append(_parse_to_exact_fraction(out))
        else:
            try:
                rhs.append(Fraction(str(out)))
            except (ValueError, TypeError):
                rhs.append(Fraction(float(out)))

    return rows, rhs, exps_list


def _gaussian_elim_rational(
    A: list[list[Fraction]], b: list[Fraction]
) -> list[Fraction] | None:
    """Solve Ax = b using exact rational Gaussian elimination.

    Args:
        A: Matrix as list of rows (each row is list of Fractions), m x n
        b: Right-hand side vector (list of Fractions), length m

    Returns:
        Solution vector (list of Fractions) if consistent solution exists, None otherwise
        For underdetermined systems (m > n), checks if solution is consistent with all equations
        For overdetermined systems (m < n), returns None (use least squares separately)
    """
    if not A or not b:
        return None

    m = len(A)
    n = len(A[0]) if A else 0

    if m < n:
        return None  # Overdetermined in unknowns - use least squares

    # Work on augmented matrix copy
    M = [row[:] + [bval] for row, bval in zip(A, b)]

    r = 0
    piv_cols = []

    for c in range(n):
        # Find pivot in rows r..m-1
        piv = None
        for i in range(r, m):
            if M[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue  # Column is all zeros

        # Swap rows
        M[r], M[piv] = M[piv], M[r]
        piv_val = M[r][c]

        # Normalize row r
        M[r] = [val / piv_val for val in M[r]]

        # Eliminate other rows
        for i in range(m):
            if i == r:
                continue
            factor = M[i][c]
            if factor != 0:
                M[i] = [vi - factor * vr for vi, vr in zip(M[i], M[r])]

        piv_cols.append(c)
        r += 1
        if r == m:
            break

    # Extract solution for determined or consistent underdetermined systems
    if len(piv_cols) == n:  # Full rank in columns (all variables determined)
        sol = [Fraction(0)] * n
        for i, c in enumerate(piv_cols):
            if i < len(M):
                sol[c] = M[i][-1]

        # For underdetermined systems (m > n), verify all equations are satisfied
        if m > n:
            for i in range(m):
                computed = sum(A[i][j] * sol[j] for j in range(n))
                if computed != b[i]:
                    return None  # Not consistent

        return sol

    return None  # Singular or inconsistent


def _find_sparse_polynomial_solution(
    samples: list[tuple[list[Any], Any]], degree: int = 2, include_inverse: bool = False
) -> tuple[list[Fraction] | None, list[tuple[int, ...]]]:
    """Find sparse polynomial/rational solution using exact rational arithmetic.

    Implements polynomial/rational basis expansion (degree 2) with exact Fraction arithmetic.
    Algorithm:
    1. Generate all monomials up to total degree 2 (optionally including inverse terms)
    2. Build matrix M (samples x monomials) with exact Fraction values
    3. Solve M * coeffs = F using rational Gaussian elimination
    4. If underdetermined (m < n), search for sparse solutions:
       - Try all subsets of monomials of size m (or smaller for sparser solutions)
       - Verify exact match with all samples
       - Prioritize product terms, inverse terms, and simpler forms

    Args:
        samples: List of (input_values, output_value) tuples
        degree: Maximum total degree of monomials (default: 2)
        include_inverse: If True, include negative exponents for inverse terms

    Returns:
        Tuple of (solution_coefficients, exponent_list)
        solution_coefficients is None if no exact solution found
        All arithmetic uses Fraction for exact results
    """
    A, b, exps_list = _build_polynomial_matrix(
        samples, degree, include_inverse=include_inverse
    )

    if not A:
        return None, exps_list

    m = len(A)  # Number of samples
    n = len(exps_list)  # Number of monomials

    # Case 1: Square system (m == n) - try direct solve
    if m == n:
        sol = _gaussian_elim_rational(A, b)
        if sol is not None:
            # Verify exact match (residuals all zero)
            for row, bb in zip(A, b):
                s = sum(coef * val for coef, val in zip(row, sol))
                if s != bb:
                    sol = None
                    break
            if sol is not None:
                return sol, exps_list

    # Case 2: Overdetermined (m > n) - system has more equations than unknowns
    # For exact solution, we need the system to be consistent
    # Special case: if n == 1 (single term), check if all samples give the same coefficient
    if m > n:
        if n == 1:
            # Single term: check if all samples give consistent coefficient
            # For each sample, compute coefficient = output / monomial_value
            coeffs = []
            for row, bb in zip(A, b):
                monomial_val = row[0]
                if monomial_val == 0:
                    # Can't divide by zero
                    return None, exps_list
                coeff = bb / monomial_val
                coeffs.append(coeff)

            # Check if all coefficients are equal (exact match)
            if len(coeffs) > 0:
                first_coeff = coeffs[0]
                if all(c == first_coeff for c in coeffs):
                    # All samples give the same coefficient - exact solution!
                    sol = [first_coeff]
                    return sol, exps_list

        # For other overdetermined cases, could implement rational least squares
        # For now, reject (system would need to be consistent for exact solution)
        return None, exps_list

    # m < n: underdetermined - search for sparse solutions
    # Try smaller subsets first (sparsest solutions) up to size m
    indices = list(range(n))

    # Limit search size to avoid combinatorial explosion
    # For large basis (e.g., 129 monomials), be more aggressive with limits
    if n > 50:
        max_subset_size = min(
            m, 3
        )  # Very sparse solutions only for large basis (1-3 terms)
    elif n > 20:
        max_subset_size = min(m, 5)  # Moderate limit for medium basis
    else:
        max_subset_size = min(m, 10)  # Limit to 10 monomials max for performance

    if max_subset_size > 0:
        # For large basis, prioritize sparse solutions and early termination
        # Try smallest subsets first, and stop early if we find a good solution
        from itertools import combinations

        # Score function for subsets: prefer solutions with product terms, inverse terms, and fewer terms
        # Also prefer solutions that use variables present in the data (especially x)
        def subset_score(subset_indices):
            """Score a subset: lower is better. Prefer rational terms like x*y/z^2, product terms, and simpler forms."""
            total_degree = 0
            has_product = False
            has_quadratic = False
            has_inverse = False
            has_rational = False  # e.g., x*y/z^2 (product with inverse)
            uses_variables = set()

            # Check which variables are actually used in the data (non-zero inputs)
            data_vars = set()
            for vars_tuple, _ in samples:
                for i, val in enumerate(vars_tuple):
                    # Convert to Fraction to check if non-zero
                    try:
                        val_frac = (
                            _parse_to_exact_fraction(val)
                            if not isinstance(val, Fraction)
                            else val
                        )
                        if val_frac != 0:
                            data_vars.add(i)
                    except Exception:
                        pass

            for idx in subset_indices:
                exps = exps_list[idx]
                abs_degree = sum(abs(e) for e in exps)
                total_degree += abs_degree

                # Track which variables are used
                for i, e in enumerate(exps):
                    if e != 0:
                        uses_variables.add(i)
                    if e < 0:
                        has_inverse = True

                # Check if it's a rational term like x*y/z^2 (product with inverse)
                pos_count = sum(1 for e in exps if e > 0)
                neg_count = sum(1 for e in exps if e < 0)
                if pos_count >= 2 and neg_count >= 1:
                    # Has product of positive terms and at least one inverse
                    has_rational = True

                # Check if it's a product term (exactly two exponents are 1, rest are 0)
                non_zero_count = sum(1 for e in exps if e != 0)
                if non_zero_count == 2 and all(e in (0, 1) for e in exps):
                    has_product = True
                elif any(e > 1 for e in exps):
                    has_quadratic = True

            # Count how many data variables are used
            vars_used_count = len(uses_variables & data_vars)
            vars_not_used = len(data_vars - uses_variables)

            # Prefer subsets that:
            # 1. Have rational terms (x*y/z^2) - highest priority
            # 2. Use variables present in data
            # 3. Have product terms
            # 4. Have lower total degree
            # Score: (not has_rational, vars_not_used, has_quadratic, not has_product, total_degree)
            # Lower score = better
            return (
                not has_rational,
                vars_not_used,
                has_quadratic,
                not has_product,
                total_degree,
            )

        # Try subsets of increasing size, starting from the smallest
        # This finds the sparsest solution first
        for subset_size in range(1, max_subset_size + 1):
            # Generate all subsets of this size
            all_subsets = list(combinations(indices, subset_size))

            # For size 2, prioritize product term pairs (like x*y and z*w)
            if subset_size == 2:
                product_pairs = []

                def is_product_term(exps):
                    non_zero = sum(1 for e in exps if e > 0)
                    return non_zero == 2 and all(e in (0, 1) for e in exps)

                # Find all product term pairs
                for i, exps_i in enumerate(exps_list):
                    for j, exps_j in enumerate(exps_list):
                        if i < j:
                            if is_product_term(exps_i) and is_product_term(exps_j):
                                pair_tuple = (i, j)
                                if pair_tuple in all_subsets:
                                    product_pairs.append(pair_tuple)

                # Remove duplicates
                product_pairs = list(set(product_pairs))

                # Further prioritize product pairs that use variables from data
                # Especially prioritize pairs where both are pure product terms (like x*y and z*w)
                product_pairs_pure = []
                product_pairs_mixed = []

                for pair in product_pairs:
                    i, j = pair
                    exps_i = exps_list[i]
                    exps_j = exps_list[j]
                    # Check if both are pure product terms (exactly 2 variables each, degree 1)
                    if is_product_term(exps_i) and is_product_term(exps_j):
                        # Check if they use different variable sets (like x*y and z*w)
                        vars_i = {idx for idx, e in enumerate(exps_i) if e > 0}
                        vars_j = {idx for idx, e in enumerate(exps_j) if e > 0}
                        if vars_i != vars_j:  # Different variable sets
                            product_pairs_pure.append(pair)
                        else:
                            product_pairs_mixed.append(pair)
                    else:
                        product_pairs_mixed.append(pair)

                # Sort each group by score
                product_pairs_pure.sort(key=subset_score)
                product_pairs_mixed.sort(key=subset_score)

                # Sort remaining subsets by score
                remaining = [s for s in all_subsets if s not in product_pairs]
                remaining.sort(key=subset_score)

                # Try pure product pairs first (like x*y and z*w), then mixed, then others
                search_order = product_pairs_pure + product_pairs_mixed + remaining
            elif subset_size == 1:
                # For single-term subsets, prioritize rational terms (x*y/z^2) and product terms
                rational_terms = []
                product_terms = []
                other_terms = []

                for idx in indices:
                    exps = exps_list[idx]
                    pos_count = sum(1 for e in exps if e > 0)
                    neg_count = sum(1 for e in exps if e < 0)
                    non_zero_count = sum(1 for e in exps if e != 0)

                    # Rational term: product with inverse (e.g., x*y/z^2)
                    if pos_count >= 2 and neg_count >= 1:
                        rational_terms.append(idx)
                    # Product term: exactly two variables, both degree 1
                    elif non_zero_count == 2 and all(e in (0, 1) for e in exps):
                        product_terms.append(idx)
                    else:
                        other_terms.append(idx)

                # Sort each group by score
                rational_terms.sort(key=lambda idx: subset_score([idx]))
                product_terms.sort(key=lambda idx: subset_score([idx]))
                other_terms.sort(key=lambda idx: subset_score([idx]))

                # Try rational terms first (most likely for physics laws), then products, then others
                search_order = [
                    (idx,) for idx in rational_terms + product_terms + other_terms
                ]
            else:
                # Sort by score (better solutions first)
                all_subsets.sort(key=subset_score)
                search_order = all_subsets

            # Limit the number of subsets to try for performance (especially for large basis)
            if len(search_order) > 1000:
                # For very large search space, only try top-scored subsets
                search_order = search_order[:1000]

            for subset in search_order:
                A_sub = [[row[j] for j in subset] for row in A]

                # Handle overdetermined case (m > subset_size) for single terms
                if len(subset) == 1 and m > 1:
                    # Single term with multiple samples: check if all give same coefficient
                    coeffs = []
                    for row_sub, bb in zip(A_sub, b):
                        monomial_val = row_sub[0]
                        if monomial_val == 0:
                            # Can't divide by zero
                            continue
                        coeff = bb / monomial_val
                        coeffs.append(coeff)

                    # Check if all coefficients are equal (exact match)
                    if len(coeffs) == m and len(coeffs) > 0:
                        first_coeff = coeffs[0]
                        if all(c == first_coeff for c in coeffs):
                            # All samples give the same coefficient - exact solution!
                            sol_sub = [first_coeff]
                            # Lift solution to full length
                            sol_full = [Fraction(0)] * n
                            sol_full[subset[0]] = sol_sub[0]
                            return sol_full, exps_list
                    continue

                # For determined or underdetermined cases, use Gaussian elimination
                sol_sub = _gaussian_elim_rational(A_sub, b)

                if sol_sub is None:
                    continue

                # Lift solution to full length
                sol_full = [Fraction(0)] * n
                for idx, val in zip(subset, sol_sub):
                    sol_full[idx] = val

                # Verify exact match
                ok = True
                for row, bb in zip(A, b):
                    s = sum(coef * val for coef, val in zip(row, sol_full))
                    if s != bb:
                        ok = False
                        break

                if ok:
                    return sol_full, exps_list

    return None, exps_list


def _polynomial_solution_to_string(
    coeffs: list[Fraction], exps_list: list[tuple[int, ...]], param_names: list[str]
) -> str:
    """Convert polynomial solution (coefficients and exponents) to readable function string.

    Args:
        coeffs: List of Fraction coefficients for each monomial
        exps_list: List of exponent tuples corresponding to each monomial
        param_names: List of parameter names (e.g., ["x", "y", "z", "w"])

    Returns:
        Human-readable function string (e.g., "x*y - z*w")
    """
    terms = []

    for coeff, exps in zip(coeffs, exps_list):
        if coeff == 0:
            continue

        # Build monomial string for this term
        monomial_parts = []
        denominator_parts = []
        for i, exp in enumerate(exps):
            if exp == 0:
                continue
            param = param_names[i]
            if exp == 1:
                monomial_parts.append(param)
            elif exp > 1:
                # Format exponent as integer or decimal
                try:
                    exp_f = float(exp)
                    if abs(exp_f - round(exp_f)) < 1e-12:
                        exp_str = str(int(round(exp_f)))
                    else:
                        exp_str = f"{exp_f:.10g}".rstrip("0").rstrip(".")
                except Exception:
                    exp_str = str(exp)
                monomial_parts.append(f"{param}**{exp_str}")
            elif exp == -1:
                denominator_parts.append(param)
            else:  # exp < -1
                try:
                    exp_f = abs(float(exp))
                    if abs(exp_f - round(exp_f)) < 1e-12:
                        exp_str = str(int(round(exp_f)))
                    else:
                        exp_str = f"{exp_f:.10g}".rstrip("0").rstrip(".")
                except Exception:
                    exp_str = str(abs(exp))
                denominator_parts.append(f"{param}**{exp_str}")

        # Build the full term string (numerator/denominator)
        if not monomial_parts and not denominator_parts:
            # Constant term (all exponents are 0)
            if coeff == 1:
                terms.append(("1", coeff))
            elif coeff == -1:
                terms.append(("-1", coeff))
            else:
                # Format coefficient
                if coeff.denominator == 1:
                    terms.append((str(coeff.numerator), coeff))
                else:
                    terms.append((f"({coeff.numerator}/{coeff.denominator})", coeff))
        else:
            # Non-constant monomial - build numerator and denominator
            if not monomial_parts:
                numerator_str = "1"
            else:
                numerator_str = "*".join(monomial_parts)

            # Format coefficient
            if coeff == 1:
                if denominator_parts:
                    term_str = f"{numerator_str}/{'*'.join(denominator_parts)}"
                else:
                    term_str = numerator_str
                terms.append((term_str, coeff))
            elif coeff == -1:
                if denominator_parts:
                    term_str = f"-{numerator_str}/{'*'.join(denominator_parts)}"
                else:
                    term_str = f"-{numerator_str}"
                terms.append((term_str, coeff))
            else:
                # Format coefficient
                if coeff.denominator == 1:
                    coeff_str = str(coeff.numerator)
                else:
                    coeff_str = f"({coeff.numerator}/{coeff.denominator})"

                if denominator_parts:
                    term_str = (
                        f"{coeff_str}*{numerator_str}/{'*'.join(denominator_parts)}"
                    )
                else:
                    term_str = f"{coeff_str}*{numerator_str}"
                terms.append((term_str, coeff))

    if not terms:
        return "0"

    # Sort terms: constants last, then by monomial order
    def term_key(term):
        expr_str, coeff = term
        # Extract monomial complexity (number of multiplications)
        mult_count = expr_str.count("*")
        # Constants go last
        if mult_count == 0 and (
            expr_str == "1" or expr_str == "-1" or expr_str.lstrip("-").isdigit()
        ):
            return (2, expr_str)
        return (0, mult_count, expr_str)

    terms.sort(key=term_key)

    # Build final string - prefer positive terms first for readability
    # Sort terms again: positive coefficients first, then negative
    # Constants go last
    positive_terms = []
    negative_terms = []
    constants = []

    for expr_str, coeff in terms:
        # Check if it's a constant (no variables, just a number)
        is_constant = (
            expr_str.lstrip("-")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "")
            .replace("*", "")
            .replace(" ", "")
            .isdigit()
            or expr_str in ["1", "-1"]
            or (expr_str.startswith("(") and expr_str.endswith(")") and "/" in expr_str)
        )

        if is_constant:
            constants.append(expr_str)
        elif expr_str.startswith("-") or (isinstance(coeff, Fraction) and coeff < 0):
            negative_terms.append(expr_str.lstrip("-"))
        else:
            positive_terms.append(expr_str)

    # Build result: positive terms first, then negative terms, then constants
    result_parts = []

    # Add positive terms
    for term in positive_terms:
        if result_parts:
            result_parts.append(f"+ {term}")
        else:
            result_parts.append(term)

    # Add negative terms with minus signs
    for term in negative_terms:
        if result_parts:
            result_parts.append(f"- {term}")
        else:
            result_parts.append(f"-{term}")

    # Add constants last
    for const in constants:
        if result_parts:
            if const.startswith("-"):
                result_parts.append(f"- {const.lstrip('-')}")
            else:
                result_parts.append(f"+ {const}")
        else:
            result_parts.append(const)

    func_str = " ".join(result_parts).replace("+ -", "- ").replace("- -", "+ ")

    return func_str


def _symbolify_coefficient(val):
    """
    Attempt to convert a float coefficient to a symbolic form (e.g., pi/2).
    """
    try:
        if abs(val) < 1e-10:
            return None

        import sympy as sp

        # Method 1: Check PI multiples directly (more robust for noise)
        # Check val / pi
        ratio_pi = val / sp.pi.evalf()
        simplified_pi = sp.nsimplify(ratio_pi, tolerance=1e-5, rational=True)
        # If simplified ratio is simple (e.g. 1/2, 1, 2), then val = ratio * pi
        if simplified_pi != ratio_pi:  # If it actually simplified something
            # Check complexity of the multiplier
            den = sp.denom(simplified_pi)
            num = sp.numer(simplified_pi)
            if abs(den) < 1000 and abs(num) < 1000:
                return f"{simplified_pi}*pi".replace("1*pi", "pi")

        return None
    except Exception:
        return None


def find_function_from_data(
    data_points: list[tuple[Any, Any]], param_names: list[str] = ["x"]
) -> tuple[bool, str | None, dict[str, Any] | None, str | None]:
    """Find a function from data points using interpolation/regression.

    For single-parameter functions:
    - Uses polynomial interpolation for exact fits:
      * Method 1: Lagrange interpolation (P(x) = Σᵢ yᵢ * Lᵢ(x) where Lᵢ(x) = Πⱼ≠ᵢ (x-xⱼ)/(xᵢ-xⱼ))
        Provides a direct closed-form expression
      * Method 2: Newton divided differences (more numerically stable for high-degree polynomials)
      * Method 3: Vandermonde matrix (fallback if other methods fail)
    - For overdetermined systems, uses least squares regression

    For multi-parameter functions:
    - **PRIORITY 1**: Extended polynomial/rational basis with inverse terms (x*y, 1/z, 1/z^2, x*y/z^2, etc.)
      - Generates all monomials up to degree 2 including negative exponents (for inverse terms)
      - Can discover rational functions like x*y/z^2 (Newton's law), x*y/z, etc.
      - Builds exact Fraction matrix and solves using rational Gaussian elimination
      - For underdetermined systems, searches for sparse solutions using subset search
      - Prioritizes solutions with product terms, inverse terms, and simpler forms
    - **PRIORITY 2**: Polynomial basis expansion (degree 2, no inverse terms) - finds non-linear relationships
      like x*y - z*w, x^2 + y^2, etc. Uses exact rational arithmetic throughout.
    - **PRIORITY 3**: Linear regression (fallback) - finds f(x₁, x₂, ..., xₙ) = a₁x₁ + a₂x₂ + ... + aₙxₙ

    Args:
        data_points: List of (input_values, output_value) tuples
                    e.g., [([5, 7.1], 6.43182), ([0, 7.1], 4.84091), ([10, 10], 10.0)]
        param_names: List of parameter names (e.g., ["x", "y"])

    Returns:
        Tuple of (success, function_string, factored_form, error_message)
        e.g., (True, "(7/22)*x + (15/22)*y", None, None)
    """
    import numpy as np

    # Helper function to convert symbolic expressions to floats
    def eval_to_float(val):
        """Convert a value (string or number) to float, evaluating symbolic expressions."""
        import sympy as sp

        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                # Try direct conversion first
                return float(val)
            except ValueError:
                pass
            # Try evaluating as a symbolic expression
            try:
                local_ns = {
                    "pi": sp.pi,
                    "e": sp.E,
                    "E": sp.E,
                    "I": sp.I,
                    "sqrt": sp.sqrt,
                    "sin": sp.sin,
                    "cos": sp.cos,
                    "tan": sp.tan,
                    "log": sp.log,
                    "ln": sp.log,
                    "exp": sp.exp,
                }
                expr = sp.sympify(val, locals=local_ns)
                # Force numeric evaluation
                result = expr.evalf()
                # Check if result is still symbolic (contains free symbols)
                if hasattr(result, "free_symbols") and result.free_symbols:
                    raise ValueError(
                        f"Expression '{val}' contains free symbols: {result.free_symbols}"
                    )
                return float(result)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Could not convert '{val}' to a numeric value: {e}"
                ) from e
            except Exception as e:
                raise ValueError(f"Could not convert '{val}' to a numeric value") from e
        # For SymPy expressions
        try:
            result = val.evalf()
            if hasattr(result, "free_symbols") and result.free_symbols:
                raise ValueError(
                    f"Expression contains free symbols: {result.free_symbols}"
                )
            return float(result)
        except Exception as e:
            raise ValueError(f"Could not convert '{val}' to a numeric value") from e

    # 0. Check for conflicting data points (Impossible Data)
    input_map = {}
    for x_tuple, y_val in data_points:
        # Create a hashable key for the input
        # Round floats to avoid tiny precision differences causing false mismatches
        x_key = tuple(round(eval_to_float(x), 9) for x in x_tuple)

        y_float = eval_to_float(y_val)
        if x_key in input_map:
            prev_y = input_map[x_key]
            if not np.isclose(prev_y, y_float, atol=1e-7):
                return (
                    False,
                    None,
                    None,
                    f"Error: Conflicting data points found for input {x_tuple}. Got both {prev_y} and {y_val}.",
                )
        else:
            input_map[x_key] = y_float

    if not data_points:
        return (False, None, None, "No data points provided")

    if len(param_names) == 0:
        return (False, None, None, "No parameters specified")

    n_params = len(param_names)

    # Special case: if we have 1 data point and 1 parameter, we can find a constant function
    if len(data_points) == 1 and n_params == 1:
        value = data_points[0][1]
        return (True, str(value), None, None)

    import numpy as np
    from sklearn.linear_model import LinearRegression

    # Helper for consistent power formatting
    def _format_power(base_name: str, exponent: float) -> str:
        if abs(exponent - round(exponent)) < 1e-8:
            return f"{base_name}^{int(round(exponent))}"
        # prefer decimal representation for non-integer exponents
        s = f"{exponent:.10g}"
        # remove trailing zeros
        s = s.rstrip("0").rstrip(".")
        return f"{base_name}^{s}"

    # --- NEW: PRE-CHECK - Linear Fit (Occam's Razor for sparse data) ---
    # Moved to TOP PRIORITY to prevent log-linear checks from claiming imperfect exponential fits
    # For single-variable, always check for perfect linear fit first.
    if n_params == 1:
        try:
            X_vals = [
                (
                    eval_to_float(p[0][0])
                    if isinstance(p[0], (list, tuple))
                    else eval_to_float(p[0])
                )
                for p in data_points
            ]
            y_vals = [eval_to_float(p[1]) for p in data_points]

            # Try y = a*x + b

            X_arr = np.array(X_vals).reshape(-1, 1)
            y_arr = np.array(y_vals)

            lr = LinearRegression()
            lr.fit(X_arr, y_arr)

            # Check if it's a perfect fit
            y_pred = lr.predict(X_arr)
            mse = np.mean((y_arr - y_pred) ** 2)

            if mse < 1e-10:  # Perfect linear fit
                a = lr.coef_[0]
                b = lr.intercept_
                var_name = param_names[0]

                # Format the result
                parts = []
                # Format slope
                if abs(a) > 1e-10:
                    a_round = round(a)
                    if abs(a - a_round) < 1e-9 and a_round != 0:
                        if a_round == 1:
                            parts.append(var_name)
                        elif a_round == -1:
                            parts.append(f"-{var_name}")
                        else:
                            parts.append(f"{int(a_round)}*{var_name}")
                    else:
                        parts.append(f"{a:.10g}*{var_name}")

                # Format intercept
                if abs(b) > 1e-10:
                    b_round = round(b)
                    if abs(b - b_round) < 1e-9:
                        if b_round > 0:
                            parts.append(
                                f"+ {int(b_round)}" if parts else str(int(b_round))
                            )
                        else:
                            parts.append(
                                f"- {abs(int(b_round))}" if parts else str(int(b_round))
                            )
                    else:
                        if b > 0:
                            parts.append(f"+ {b:.10g}" if parts else f"{b:.10g}")
                        else:
                            parts.append(f"- {abs(b):.10g}" if parts else f"{b:.10g}")

                if parts:
                    func_str = " ".join(parts)
                    return (True, func_str, None, None)
        except Exception:
            pass  # Fall through to other methods

    # --- NEW: PRE-CHECK - Sin(x) + C (Simple Harmonic) ---
    # Fast detection for "sin(x) + 1", "2*sin(x)", etc. before polynomial fallback
    if n_params == 1:
        try:
            X_vals_sin = []
            y_vals_sin = []
            for p in data_points:
                x_val = (
                    eval_to_float(p[0][0])
                    if isinstance(p[0], (list, tuple))
                    else eval_to_float(p[0])
                )
                X_vals_sin.append(x_val)
                y_vals_sin.append(eval_to_float(p[1]))

            if len(X_vals_sin) >= 2:
                import numpy as np
                from sklearn.linear_model import LinearRegression

                X_arr_sin = np.array(X_vals_sin)
                y_arr_sin = np.array(y_vals_sin)

                # Check A*x + B*sin(x) + C (linear + sine combination)
                # This catches x + sin(x), 2*x + sin(x), etc.
                x_feat = X_arr_sin.reshape(-1, 1)
                sin_feat = np.sin(X_arr_sin).reshape(-1, 1)
                combined_feat = np.hstack([x_feat, sin_feat])

                lr_combined = LinearRegression()
                lr_combined.fit(combined_feat, y_arr_sin)

                y_pred_combined = lr_combined.predict(combined_feat)
                mse_combined = np.mean((y_arr_sin - y_pred_combined) ** 2)

                if mse_combined < 1e-5:  # Excellent fit (relaxed for FP noise)
                    A_lin = lr_combined.coef_[0]  # coefficient for x
                    B_sin = lr_combined.coef_[1]  # coefficient for sin(x)
                    C = lr_combined.intercept_

                    # Stricter validation: only accept if coefficients are close to small integers
                    # This avoids false positives on exp(x) and power law data
                    A_r = round(A_lin)
                    B_r = round(B_sin)
                    C_r = round(C)

                    # Require both A and B to be close to small integers (within 1e-3)
                    # and at least one of them should be non-zero
                    is_valid_A = abs(A_lin - A_r) < 1e-3 and abs(A_r) <= 10
                    is_valid_B = abs(B_sin - B_r) < 1e-3 and abs(B_r) <= 10
                    is_valid_C = abs(C - C_r) < 1e-2 and abs(C_r) <= 10

                    # Only proceed if coefficients look like simple sine+linear combination
                    if is_valid_A and is_valid_B and (abs(A_r) >= 1 or abs(B_r) >= 1):
                        p_name = param_names[0]
                        parts_sin = []

                        # Format A*x
                        if abs(A_lin) > 1e-10:
                            if abs(A_lin - A_r) < 1e-3 and A_r != 0:
                                if A_r == 1:
                                    parts_sin.append(p_name)
                                elif A_r == -1:
                                    parts_sin.append(f"-{p_name}")
                                else:
                                    parts_sin.append(f"{int(A_r)}*{p_name}")
                            else:
                                parts_sin.append(f"{A_lin:.6g}*{p_name}")

                        # Format B*sin(x)
                        if abs(B_sin) > 1e-10:
                            if abs(B_sin - B_r) < 1e-3 and B_r != 0:
                                if B_r == 1:
                                    term = f"sin({p_name})"
                                elif B_r == -1:
                                    term = f"-sin({p_name})"
                                else:
                                    term = f"{int(B_r)}*sin({p_name})"
                            else:
                                term = f"{B_sin:.6g}*sin({p_name})"

                            if parts_sin:
                                if term.startswith("-"):
                                    parts_sin.append(f"- {term[1:]}")
                                else:
                                    parts_sin.append(f"+ {term}")
                            else:
                                parts_sin.append(term)

                        # Format C (only if significant)
                        if abs(C) > 1e-2:
                            if abs(C - C_r) < 1e-2:
                                ci = int(C_r)
                                parts_sin.append(
                                    f"+ {ci}"
                                    if parts_sin and ci > 0
                                    else (f"- {abs(ci)}" if parts_sin else f"{ci}")
                                )
                            else:
                                if C > 0:
                                    parts_sin.append(
                                        f"+ {C:.6g}" if parts_sin else f"{C:.6g}"
                                    )
                                else:
                                    parts_sin.append(
                                        f"- {abs(C):.6g}" if parts_sin else f"{C:.6g}"
                                    )

                        if parts_sin:
                            func_str = " ".join(parts_sin)
                            return (True, func_str, None, None)
        except Exception:
            pass

    # --- NEW: PRE-CHECK - Quadratic Shift Fit (A*x^2 + B) ---
    # For sparse single-variable data, regression often prefers linear if boost is high.
    if n_params == 1:
        try:
            X_vals = [
                (
                    eval_to_float(p[0][0])
                    if isinstance(p[0], (list, tuple))
                    else eval_to_float(p[0])
                )
                for p in data_points
            ]
            y_vals = [eval_to_float(p[1]) for p in data_points]
            var_name = param_names[0]

            # Try y = A*x^2 + B
            import numpy as np
            from sklearn.linear_model import LinearRegression

            X_sq_arr = np.array(X_vals).reshape(-1, 1) ** 2
            y_arr = np.array(y_vals)

            lr_sq = LinearRegression()
            lr_sq.fit(X_sq_arr, y_arr)

            # Check if it's a perfect fit
            y_pred_sq = lr_sq.predict(X_sq_arr)
            mse_sq = np.mean((y_arr - y_pred_sq) ** 2)

            if mse_sq < 1e-10:
                a = lr_sq.coef_[0]
                b = lr_sq.intercept_

                parts = []
                # Format A*x^2
                if abs(a) > 1e-10:
                    a_round = round(a)
                    if abs(a - a_round) < 1e-9 and a_round != 0:
                        if a_round == 1:
                            parts.append(f"{var_name}^2")
                        elif a_round == -1:
                            parts.append(f"-{var_name}^2")
                        else:
                            parts.append(f"{int(a_round)}*{var_name}^2")
                    else:
                        parts.append(f"{a:.10g}*{var_name}^2")

                # Format B
                if abs(b) > 1e-10:
                    b_round = round(b)
                    if abs(b - b_round) < 1e-9:
                        if b_round > 0:
                            parts.append(
                                f"+ {int(b_round)}" if parts else str(int(b_round))
                            )
                        else:
                            parts.append(
                                f"- {abs(int(b_round))}" if parts else str(int(b_round))
                            )
                    else:
                        if b > 0:
                            parts.append(f"+ {b:.10g}" if parts else f"{b:.10g}")
                        else:
                            parts.append(f"- {abs(b):.10g}" if parts else f"{b:.10g}")

                if parts:
                    func_str = " ".join(parts)
                    return (True, func_str, None, None)
        except Exception:
            pass  # Fall through to other methods

    # --- NEW: PRE-CHECK - Power Law (y = a * x^b) ---
    # Detects x^2.5, x^0.5, etc. using log-log regression: log(y) = b*log(x) + log(a)
    if n_params == 1:
        try:
            X_vals = []
            y_vals = []
            param_name = param_names[0]

            valid_data = True
            for p in data_points:
                x_val = (
                    eval_to_float(p[0][0])
                    if isinstance(p[0], (list, tuple))
                    else eval_to_float(p[0])
                )
                y_val = eval_to_float(p[1])

                # proportional requires positive x, y (for log)
                if x_val <= 0 or y_val <= 0:
                    valid_data = False
                    break
                X_vals.append(math.log(x_val))
                y_vals.append(math.log(y_val))

            if valid_data and len(X_vals) >= 2:
                import numpy as np
                from sklearn.linear_model import LinearRegression

                X_log = np.array(X_vals).reshape(-1, 1)
                y_log = np.array(y_vals)

                lr_pow = LinearRegression()
                lr_pow.fit(X_log, y_log)

                # Check fit quality
                y_pred_log = lr_pow.predict(X_log)
                mse_log = np.mean((y_log - y_pred_log) ** 2)

                if mse_log < 1e-9:  # Good power law fit
                    b = lr_pow.coef_[0]  # exponent
                    log_a = lr_pow.intercept_
                    a = math.exp(log_a)  # coefficient

                    # Round b to nearest 0.5 or integer if close
                    b_round = round(b * 2) / 2
                    if abs(b - b_round) < 1e-4:
                        b = b_round

                    # Round a if close to integer
                    if abs(a - round(a)) < 1e-4:
                        a = round(a)

                    # Format
                    term = ""
                    if b == 1:
                        term = param_name
                    elif b == 0:
                        term = "1"
                    else:
                        term = _format_power(param_name, b)

                    if abs(a - 1.0) < 1e-4:
                        func_str = term
                    else:
                        func_str = f"{a}*{term}"

                    return (True, func_str, None, None)
        except Exception:
            pass

    # --- NEW: FAST PATH - Log-Linear Checks (Exponential & Power Laws) ---
    # Check for simple relationships like y = A*e^(Bx) or y = A*x^B
    if n_params == 1:
        try:
            from .function_finder_advanced import check_log_linear_transformations

            # Extract X and y data - evaluate symbolic constants
            # p[0] is a tuple/list of x values for this data point
            X_data = [
                (
                    eval_to_float(p[0][0])
                    if isinstance(p[0], (list, tuple))
                    else eval_to_float(p[0])
                )
                for p in data_points
            ]
            y_data = [eval_to_float(p[1]) for p in data_points]

            success, func_str = check_log_linear_transformations(
                X_data, y_data, param_names
            )
            if success and func_str:
                return (True, func_str, None, None)
        except (ImportError, Exception) as e:
            import sys

            sys.stderr.write(f"CRITICAL ERROR in advanced path: {e}\n")
            import traceback

            traceback.print_exc()
            pass  # Fall back to other methods

    # --- NEW: PRE-CHECK - Pure Trig Fit (cos(x), sin(x)) ---
    # For single-variable, try cos(x) and sin(x) directly
    if n_params == 1 and len(data_points) >= 3:
        try:
            import numpy as np

            X_vals = [
                (
                    eval_to_float(p[0][0])
                    if isinstance(p[0], (list, tuple))
                    else eval_to_float(p[0])
                )
                for p in data_points
            ]
            y_vals = [eval_to_float(p[1]) for p in data_points]
            var_name = param_names[0]

            # Test cos(x) and sin(x) with frequency sweeping
            # Try frequencies: 0.5, 1, 2, 3, 4, 5, pi
            pure_trig_freqs = [0.5, 1, 2, 3, 4, 5, np.pi]
            pure_trig_labels = ["0.5", "1", "2", "3", "4", "5", "pi"]

            for freq, freq_label in zip(pure_trig_freqs, pure_trig_labels):
                # Test cos(freq*x)
                cos_vals = [np.cos(freq * x) for x in X_vals]
                cos_errors = [abs(c - y) for c, y in zip(cos_vals, y_vals)]
                if max(cos_errors) < 1e-3:
                    if freq_label == "1":
                        return (True, f"cos({var_name})", None, None)
                    else:
                        return (True, f"cos({freq_label}*{var_name})", None, None)

                # Test sin(freq*x)
                sin_vals = [np.sin(freq * x) for x in X_vals]
                sin_errors = [abs(s - y) for s, y in zip(sin_vals, y_vals)]
                if max(sin_errors) < 1e-3:
                    if freq_label == "1":
                        return (True, f"sin({var_name})", None, None)
                    else:
                        return (True, f"sin({freq_label}*{var_name})", None, None)

            # --- DAMPED OSCILLATION FREQUENCY SWEEPING ---
            # Test exp(-x)*cos(n*x) and exp(-x)*sin(n*x) for various frequencies
            # Common frequencies: 1, 2, 3, 4, 5, 0.5, pi, 2*pi
            frequencies = [1, 2, 3, 4, 5, 0.5, np.pi, 2 * np.pi]
            freq_labels = ["1", "2", "3", "4", "5", "0.5", "pi", "2*pi"]

            for freq, freq_label in zip(frequencies, freq_labels):
                # Test exp(-x)*cos(freq*x)
                decay_cos_vals = [np.exp(-x) * np.cos(freq * x) for x in X_vals]
                decay_cos_errors = [abs(d - y) for d, y in zip(decay_cos_vals, y_vals)]
                if max(decay_cos_errors) < 1e-3:
                    if freq_label == "1":
                        return (True, f"exp(-{var_name})*cos({var_name})", None, None)
                    else:
                        return (
                            True,
                            f"exp(-{var_name})*cos({freq_label}*{var_name})",
                            None,
                            None,
                        )

                # Test exp(-x)*sin(freq*x)
                decay_sin_vals = [np.exp(-x) * np.sin(freq * x) for x in X_vals]
                decay_sin_errors = [abs(d - y) for d, y in zip(decay_sin_vals, y_vals)]
                if max(decay_sin_errors) < 1e-3:
                    if freq_label == "1":
                        return (True, f"exp(-{var_name})*sin({var_name})", None, None)
                    else:
                        return (
                            True,
                            f"exp(-{var_name})*sin({freq_label}*{var_name})",
                            None,
                            None,
                        )

                # Test A*exp(-x)*cos(freq*x) with coefficient fitting
                if not np.any(np.isclose(decay_cos_vals, 0, atol=1e-10)):
                    ratios = [
                        y / d for y, d in zip(y_vals, decay_cos_vals) if abs(d) > 1e-10
                    ]
                    if len(ratios) >= 2 and abs(max(ratios) - min(ratios)) < 1e-3:
                        coeff = sum(ratios) / len(ratios)
                        if abs(coeff - 1.0) < 1e-3:
                            if freq_label == "1":
                                return (
                                    True,
                                    f"exp(-{var_name})*cos({var_name})",
                                    None,
                                    None,
                                )
                            else:
                                return (
                                    True,
                                    f"exp(-{var_name})*cos({freq_label}*{var_name})",
                                    None,
                                    None,
                                )
                        elif abs(coeff - round(coeff)) < 1e-3:
                            if freq_label == "1":
                                return (
                                    True,
                                    f"{int(round(coeff))}*exp(-{var_name})*cos({var_name})",
                                    None,
                                    None,
                                )
                            else:
                                return (
                                    True,
                                    f"{int(round(coeff))}*exp(-{var_name})*cos({freq_label}*{var_name})",
                                    None,
                                    None,
                                )

            # Test exp(-x^2) - Gaussian / Bell Curve
            gaussian_vals = [np.exp(-(x**2)) for x in X_vals]
            gaussian_errors = [abs(g - y) for g, y in zip(gaussian_vals, y_vals)]
            if (
                max(gaussian_errors) < 1e-2
            ):  # Use slightly looser tolerance for Gaussian
                return (True, f"exp(-{var_name}^2)", None, None)

            # Test exp(x) - Exponential growth
            exp_vals = [np.exp(x) for x in X_vals]
            exp_errors = [abs(e - y) for e, y in zip(exp_vals, y_vals)]
            if max(exp_errors) < 1e-3:
                return (True, f"exp({var_name})", None, None)

            # Test exp(-x) - Exponential decay
            exp_neg_vals = [np.exp(-x) for x in X_vals]
            exp_neg_errors = [abs(e - y) for e, y in zip(exp_neg_vals, y_vals)]
            if max(exp_neg_errors) < 1e-3:
                return (True, f"exp(-{var_name})", None, None)

            # Test A*log(x) - Logarithmic growth
            # Only valid for positive x values
            if all(x > 0 for x in X_vals):
                log_vals = np.array([np.log(x) for x in X_vals])
                y_arr_local = np.array(y_vals)

                # Use linear regression to find y = A*log(x) (no intercept)
                # Check if log values have enough variance
                if np.std(log_vals) > 1e-6:
                    # Compute A = sum(y*log(x)) / sum(log(x)^2)
                    A = np.dot(y_arr_local, log_vals) / np.dot(log_vals, log_vals)
                    predicted = A * log_vals
                    errors = np.abs(predicted - y_arr_local)

                    if np.max(errors) < 1e-2:  # Good fit
                        if abs(A - round(A)) < 1e-3:
                            func_str = f"{int(round(A))}*log({var_name})"
                        else:
                            func_str = f"{A:.10g}*log({var_name})"
                        return (True, func_str, None, None)

            # Test cosh(x) - Hyperbolic cosine (catenary)
            cosh_vals = [np.cosh(x) for x in X_vals]
            cosh_errors = [abs(c - y) for c, y in zip(cosh_vals, y_vals)]
            if max(cosh_errors) < 1e-3:
                return (True, f"cosh({var_name})", None, None)

            # Test sinh(x) - Hyperbolic sine
            sinh_vals = [np.sinh(x) for x in X_vals]
            sinh_errors = [abs(s - y) for s, y in zip(sinh_vals, y_vals)]
            if max(sinh_errors) < 1e-3:
                return (True, f"sinh({var_name})", None, None)

            # Test sigmoid(x) = 1 / (1 + exp(-x)) - Logistic Growth
            sigmoid_vals = [1 / (1 + np.exp(-x)) for x in X_vals]
            sigmoid_errors = [abs(s - y) for s, y in zip(sigmoid_vals, y_vals)]
            if max(sigmoid_errors) < 1e-3:
                return (True, f"1/(1+exp(-{var_name}))", None, None)

            # Test abs(x) - Absolute Value
            abs_vals = [abs(x) for x in X_vals]
            abs_errors = [abs(a - y) for a, y in zip(abs_vals, y_vals)]
            if max(abs_errors) < 1e-3:
                return (True, f"abs({var_name})", None, None)

            # Test Saturation: 1 - exp(-x)
            # Charging capacitor curve
            sat_vals = [1 - np.exp(-x) for x in X_vals]
            sat_errors = [abs(s - y) for s, y in zip(sat_vals, y_vals)]
            if max(sat_errors) < 1e-3:
                return (True, f"1-exp(-{var_name})", None, None)

            # Test x^2 - Quadratic
            sq_vals = [x**2 for x in X_vals]
            sq_errors = [abs(s - y) for s, y in zip(sq_vals, y_vals)]
            if max(sq_errors) < 1e-6:
                return (True, f"{var_name}^2", None, None)

            # Test A*x^2 - Scaled quadratic
            if not np.any(np.isclose(sq_vals, 0, atol=1e-10)):
                ratios = [y / s for y, s in zip(y_vals, sq_vals) if abs(s) > 1e-10]
                if len(ratios) >= 2:
                    if abs(max(ratios) - min(ratios)) < 1e-6:
                        coeff = sum(ratios) / len(ratios)
                        if abs(coeff - 1.0) < 1e-9:
                            return (True, f"{var_name}^2", None, None)
                        elif abs(coeff - round(coeff)) < 1e-6:
                            return (
                                True,
                                f"{int(round(coeff))}*{var_name}^2",
                                None,
                                None,
                            )
                        else:
                            return (True, f"{coeff:.10g}*{var_name}^2", None, None)

            # --- SHIFTED INVERSE DETECTION ---
            # Test y = A + B/(x+c) for common shift values
            # This catches patterns like 1/(x+1) and 1 - 1/(x+1) (Enzyme Kinetics)
            shift_values = [1, -1, 2, -2, 0.5, -0.5]
            for shift in shift_values:
                try:
                    # Compute 1/(x+shift) for each x value
                    inv_vals = []
                    valid = True
                    for x in X_vals:
                        denom = x + shift
                        if abs(denom) < 1e-10:  # Avoid division by zero
                            valid = False
                            break
                        inv_vals.append(1 / denom)

                    if valid and len(inv_vals) == len(y_vals):
                        # Use Linear Regression to find A and B in y = A + B * (1/(x+c))
                        import numpy as np
                        from sklearn.linear_model import LinearRegression

                        X_inv_arr = np.array(inv_vals).reshape(-1, 1)
                        y_arr = np.array(y_vals)

                        lr_inv = LinearRegression()
                        lr_inv.fit(X_inv_arr, y_arr)

                        y_pred_inv = lr_inv.predict(X_inv_arr)
                        mse_inv = np.mean((y_arr - y_pred_inv) ** 2)

                        if mse_inv < 1e-10:
                            B = lr_inv.coef_[0]
                            A = lr_inv.intercept_

                            # Format shift string
                            if shift >= 0:
                                shift_str = f"({var_name}+{int(shift) if shift == int(shift) else shift})"
                            else:
                                shift_str = f"({var_name}-{int(abs(shift)) if abs(shift) == int(abs(shift)) else abs(shift)})"

                            parts = []
                            # Format A (Intercept)
                            if abs(A) > 1e-10:
                                A_round = round(A)
                                if abs(A - A_round) < 1e-9:
                                    parts.append(str(int(A_round)))
                                else:
                                    parts.append(f"{A:.10g}")

                            # Format B/(x+c)
                            term = ""
                            if abs(B) > 1e-10:
                                B_round = round(B)
                                if abs(B - B_round) < 1e-9:
                                    B_int = int(B_round)
                                    if B_int == 1:
                                        term = f"1/{shift_str}"
                                    elif B_int == -1:
                                        term = f"-1/{shift_str}"
                                    else:
                                        term = f"{B_int}/{shift_str}"
                                else:
                                    term = f"{B:.10g}/{shift_str}"

                                if term:
                                    if term.startswith("-"):
                                        parts.append(f"- {term[1:]}" if parts else term)
                                    else:
                                        parts.append(f"+ {term}" if parts else term)

                            if parts:
                                func_str = " ".join(parts)
                                # Fix spacing for simple negative term
                                if func_str.startswith("- "):
                                    func_str = "-" + func_str[2:]
                                return (True, func_str, None, None)
                except Exception:
                    pass

            # --- POTENTIAL FIELD DETECTION (1/sqrt(x^2+c)) ---
            pot_shifts = [1, -1, 2, -2, 0.5, -0.5]
            for shift in pot_shifts:
                try:
                    # Check if domain is consistent (all positive or all negative)
                    vals_raw = [x**2 + shift for x in X_vals]
                    signs = [np.sign(v) for v in vals_raw if abs(v) > 1e-10]

                    if not signs:
                        continue  # All zero?
                    sign_factor = signs[0]
                    if not all(s == sign_factor for s in signs):
                        continue  # Mixed signs (crossing zero), can't model with simple sqrt

                    inv_sqrt_vals = []
                    for v in vals_raw:
                        val = sign_factor * v
                        if val < 1e-10:
                            val = 1e-10
                        inv_sqrt_vals.append(1.0 / np.sqrt(val))

                    if len(inv_sqrt_vals) == len(y_vals):
                        X_pot_arr = np.array(inv_sqrt_vals).reshape(-1, 1)
                        y_arr = np.array(y_vals)

                        lr_pot = LinearRegression()
                        lr_pot.fit(X_pot_arr, y_arr)
                        y_pred = lr_pot.predict(X_pot_arr)
                        mse = np.mean((y_arr - y_pred) ** 2)

                        if mse < 1e-10:
                            B = lr_pot.coef_[0]
                            A = lr_pot.intercept_

                            # Format shift string
                            # If sign_factor is 1: sqrt(x^2 + shift)
                            # If sign_factor is -1: sqrt(-shift - x^2)

                            if sign_factor == 1:
                                if shift >= 0:
                                    shift_str = f"{var_name}^2+{int(shift) if shift==int(shift) else shift}"
                                else:
                                    shift_str = f"{var_name}^2-{int(abs(shift)) if abs(shift)==int(abs(shift)) else abs(shift)}"
                            else:  # sign_factor == -1
                                # -shift - x^2
                                neg_shift = -shift
                                if neg_shift >= 0:
                                    shift_str = f"{int(neg_shift) if neg_shift==int(neg_shift) else neg_shift}-{var_name}^2"
                                else:
                                    shift_str = f"-{var_name}^2-{int(abs(neg_shift)) if abs(neg_shift)==int(abs(neg_shift)) else abs(neg_shift)}"

                            parts = []
                            if abs(A) > 1e-5:
                                parts.append(f"{float(A):.10g}")

                            term = ""
                            if abs(B) > 1e-10:
                                if abs(B - 1.0) < 1e-4:
                                    term = f"1/sqrt({shift_str})"
                                elif abs(B + 1.0) < 1e-4:
                                    term = f"-1/sqrt({shift_str})"
                                else:
                                    term = f"{float(B):.10g}/sqrt({shift_str})"

                            if term:
                                if term.startswith("-"):
                                    parts.append(f"- {term[1:]}" if parts else term)
                                else:
                                    parts.append(f"+ {term}" if parts else term)

                            if parts:
                                func_str = " ".join(parts)
                                if func_str.startswith("- "):
                                    func_str = "-" + func_str[2:]
                                return (True, func_str, None, None)
                except Exception:
                    pass

        except Exception:
            pass

    # --- NEW: PRE-CHECK - Simple Product Fit (x*y, x*y*z) ---
    # For multi-variable sparse data, try simple products directly
    if n_params >= 2 and len(data_points) >= 2:
        try:
            import numpy as np

            # Extract X and y data
            X_data = []
            for p in data_points:
                if isinstance(p[0], (list, tuple)):
                    X_data.append([eval_to_float(x) for x in p[0]])
                else:
                    X_data.append([eval_to_float(p[0])])
            y_data = [eval_to_float(p[1]) for p in data_points]

            X_arr = np.array(X_data)
            y_arr = np.array(y_data)

            # Test 2-variable product: y = x0 * x1
            if n_params == 2:
                product = X_arr[:, 0] * X_arr[:, 1]
                if not np.any(np.isclose(product, 0, atol=1e-10)):
                    # Check if y / product is constant
                    ratios = y_arr / product
                    if np.max(np.abs(ratios - ratios[0])) < 1e-6:
                        coeff = ratios[0]
                        if abs(coeff - 1.0) < 1e-9:
                            func_str = f"{param_names[0]}*{param_names[1]}"
                        elif abs(coeff - round(coeff)) < 1e-9:
                            func_str = (
                                f"{int(round(coeff))}*{param_names[0]}*{param_names[1]}"
                            )
                        else:
                            func_str = f"{coeff:.10g}*{param_names[0]}*{param_names[1]}"
                        return (True, func_str, None, None)

            # Test Geometric Mean: sqrt(x * y)
            if n_params == 2:
                product = X_arr[:, 0] * X_arr[:, 1]
                if np.all(product >= 0) and not np.any(
                    np.isclose(product, 0, atol=1e-10)
                ):
                    geo_mean = np.sqrt(product)
                    # Check if y / geo_mean is constant
                    ratios = y_arr / geo_mean
                    if np.max(np.abs(ratios - ratios[0])) < 1e-6:
                        coeff = ratios[0]
                        if abs(coeff - 1.0) < 1e-9:
                            return (
                                True,
                                f"sqrt({param_names[0]}*{param_names[1]})",
                                None,
                                None,
                            )
                        elif abs(coeff - round(coeff)) < 1e-9:
                            return (
                                True,
                                f"{int(round(coeff))}*sqrt({param_names[0]}*{param_names[1]})",
                                None,
                                None,
                            )
                        else:
                            return (
                                True,
                                f"{coeff:.10g}*sqrt({param_names[0]}*{param_names[1]})",
                                None,
                                None,
                            )

            # Test Ratio: x0 / x1
            if n_params == 2:
                # Try x0 / x1
                denom = X_arr[:, 1]
                if not np.any(np.isclose(denom, 0, atol=1e-10)):
                    ratio = X_arr[:, 0] / denom
                    mask = np.abs(ratio) > 1e-10
                    if np.sum(mask) >= 1:
                        ratios = y_arr[mask] / ratio[mask]
                        if np.max(np.abs(ratios - ratios[0])) < 1e-6:
                            if np.all(np.abs(y_arr[~mask]) < 1e-6):
                                coeff = ratios[0]
                                if abs(coeff - 1.0) < 1e-9:
                                    return (
                                        True,
                                        f"{param_names[0]}/{param_names[1]}",
                                        None,
                                        None,
                                    )
                                elif abs(coeff - round(coeff)) < 1e-9:
                                    return (
                                        True,
                                        f"{int(round(coeff))}*{param_names[0]}/{param_names[1]}",
                                        None,
                                        None,
                                    )
                                else:
                                    return (
                                        True,
                                        f"{coeff:.10g}*{param_names[0]}/{param_names[1]}",
                                        None,
                                        None,
                                    )

                # Try x1 / x0
                denom = X_arr[:, 0]
                if not np.any(np.isclose(denom, 0, atol=1e-10)):
                    ratio = X_arr[:, 1] / denom
                    mask = np.abs(ratio) > 1e-10
                    if np.sum(mask) >= 1:
                        ratios = y_arr[mask] / ratio[mask]
                        if np.max(np.abs(ratios - ratios[0])) < 1e-6:
                            if np.all(np.abs(y_arr[~mask]) < 1e-6):
                                coeff = ratios[0]
                                if abs(coeff - 1.0) < 1e-9:
                                    return (
                                        True,
                                        f"{param_names[1]}/{param_names[0]}",
                                        None,
                                        None,
                                    )
                                elif abs(coeff - round(coeff)) < 1e-9:
                                    return (
                                        True,
                                        f"{int(round(coeff))}*{param_names[1]}/{param_names[0]}",
                                        None,
                                        None,
                                    )
                                else:
                                    return (
                                        True,
                                        f"{coeff:.10g}*{param_names[1]}/{param_names[0]}",
                                        None,
                                        None,
                                    )

            # Test 3-variable product: y = x0 * x1 * x2
            if n_params >= 3:
                product = X_arr[:, 0] * X_arr[:, 1] * X_arr[:, 2]
                if not np.any(np.isclose(product, 0, atol=1e-10)):
                    # Check if y / product is constant
                    ratios = y_arr / product
                    if np.max(np.abs(ratios - ratios[0])) < 1e-6:
                        coeff = ratios[0]
                        if abs(coeff - 1.0) < 1e-9:
                            func_str = (
                                f"{param_names[0]}*{param_names[1]}*{param_names[2]}"
                            )
                        elif abs(coeff - round(coeff)) < 1e-9:
                            func_str = f"{int(round(coeff))}*{param_names[0]}*{param_names[1]}*{param_names[2]}"
                        else:
                            func_str = f"{coeff:.10g}*{param_names[0]}*{param_names[1]}*{param_names[2]}"
                        return (True, func_str, None, None)

            # Test single-variable x^2 (ghost variable detection)
            # For multi-variable data, check if y = x_i^2 for any variable
            for i in range(n_params):
                sq_vals = X_arr[:, i] ** 2
                if not np.any(np.isclose(sq_vals, 0, atol=1e-10)):
                    ratios = y_arr / sq_vals
                    if np.max(np.abs(ratios - ratios[0])) < 1e-6:
                        coeff = ratios[0]
                        var_name = param_names[i]
                        if abs(coeff - 1.0) < 1e-9:
                            return (True, f"{var_name}^2", None, None)
                        elif abs(coeff - round(coeff)) < 1e-9:
                            return (
                                True,
                                f"{int(round(coeff))}*{var_name}^2",
                                None,
                                None,
                            )
                        else:
                            return (True, f"{coeff:.10g}*{var_name}^2", None, None)

            # Test x*y^2 interaction (Kinetic Energy pattern)
            # For 2-variable data, check if y = A * x0 * x1^2
            if n_params == 2:
                # Try x0 * x1^2
                interaction1 = X_arr[:, 0] * (X_arr[:, 1] ** 2)
                if not np.any(np.isclose(interaction1, 0, atol=1e-10)):
                    ratios = y_arr / interaction1
                    if np.max(np.abs(ratios - ratios[0])) < 1e-3:
                        coeff = ratios[0]
                        if abs(coeff - 0.5) < 1e-3:  # Special case for kinetic energy
                            return (
                                True,
                                f"1/2*{param_names[0]}*{param_names[1]}^2",
                                None,
                                None,
                            )
                        elif abs(coeff - 1.0) < 1e-3:
                            return (
                                True,
                                f"{param_names[0]}*{param_names[1]}^2",
                                None,
                                None,
                            )
                        elif abs(coeff - round(coeff)) < 1e-3:
                            return (
                                True,
                                f"{int(round(coeff))}*{param_names[0]}*{param_names[1]}^2",
                                None,
                                None,
                            )
                        else:
                            return (
                                True,
                                f"{coeff:.10g}*{param_names[0]}*{param_names[1]}^2",
                                None,
                                None,
                            )

                # Try x0^2 * x1
                interaction2 = (X_arr[:, 0] ** 2) * X_arr[:, 1]
                if not np.any(np.isclose(interaction2, 0, atol=1e-10)):
                    ratios = y_arr / interaction2
                    if np.max(np.abs(ratios - ratios[0])) < 1e-3:
                        coeff = ratios[0]
                        if abs(coeff - 0.5) < 1e-3:
                            return (
                                True,
                                f"1/2*{param_names[0]}^2*{param_names[1]}",
                                None,
                                None,
                            )
                        elif abs(coeff - 1.0) < 1e-3:
                            return (
                                True,
                                f"{param_names[0]}^2*{param_names[1]}",
                                None,
                                None,
                            )
                        elif abs(coeff - round(coeff)) < 1e-3:
                            return (
                                True,
                                f"{int(round(coeff))}*{param_names[0]}^2*{param_names[1]}",
                                None,
                                None,
                            )
                        else:
                            return (
                                True,
                                f"{coeff:.10g}*{param_names[0]}^2*{param_names[1]}",
                                None,
                                None,
                            )

            # Test 2-variable Physics Patterns (Field, Hyp, Diff, Res)
            if n_params == 2:
                x0 = X_arr[:, 0]
                x1 = X_arr[:, 1]

                # 1. Field Pattern: x/y^2 (Coulomb 2D)
                # Try x0/x1^2
                denom = x1**2
                if not np.any(np.isclose(denom, 0, atol=1e-10)):
                    term = x0 / denom
                    mask = np.abs(term) > 1e-10
                    if np.sum(mask) >= 1:
                        ratios = y_arr[mask] / term[mask]
                        if np.max(np.abs(ratios - ratios[0])) < 1e-3:
                            if np.all(np.abs(y_arr[~mask]) < 1e-3):
                                coeff = ratios[0]
                                if abs(coeff - 1.0) < 1e-6:
                                    return (
                                        True,
                                        f"{param_names[0]}/{param_names[1]}^2",
                                        None,
                                        None,
                                    )
                                else:
                                    return (
                                        True,
                                        f"{coeff:.10g}*{param_names[0]}/{param_names[1]}^2",
                                        None,
                                        None,
                                    )
                # Try x1/x0^2
                denom = x0**2
                if not np.any(np.isclose(denom, 0, atol=1e-10)):
                    term = x1 / denom
                    mask = np.abs(term) > 1e-10
                    if np.sum(mask) >= 1:
                        ratios = y_arr[mask] / term[mask]
                        if np.max(np.abs(ratios - ratios[0])) < 1e-3:
                            if np.all(np.abs(y_arr[~mask]) < 1e-3):
                                coeff = ratios[0]
                                if abs(coeff - 1.0) < 1e-6:
                                    return (
                                        True,
                                        f"{param_names[1]}/{param_names[0]}^2",
                                        None,
                                        None,
                                    )
                                else:
                                    return (
                                        True,
                                        f"{coeff:.10g}*{param_names[1]}/{param_names[0]}^2",
                                        None,
                                        None,
                                    )

                # 2. Euclidean Distance: sqrt(x^2 + y^2) (Pythagoras)
                term = np.sqrt(x0**2 + x1**2)
                mask = np.abs(term) > 1e-10
                if np.sum(mask) >= 1:
                    ratios = y_arr[mask] / term[mask]
                    if np.max(np.abs(ratios - ratios[0])) < 1e-3:
                        if np.all(np.abs(y_arr[~mask]) < 1e-3):
                            coeff = ratios[0]
                            if abs(coeff - 1.0) < 1e-6:
                                return (
                                    True,
                                    f"sqrt({param_names[0]}^2+{param_names[1]}^2)",
                                    None,
                                    None,
                                )
                            else:
                                return (
                                    True,
                                    f"{coeff:.10g}*sqrt({param_names[0]}^2+{param_names[1]}^2)",
                                    None,
                                    None,
                                )

                # 3. Difference of Squares: x^2 - y^2
                # (Already fixed in previous step, checking next...)
                # Try x0^2 - x1^2
                term = x0**2 - x1**2
                mask = np.abs(term) > 1e-10
                if np.sum(mask) >= 1:
                    ratios = y_arr[mask] / term[mask]
                    if np.max(np.abs(ratios - ratios[0])) < 1e-3:
                        if np.all(np.abs(y_arr[~mask]) < 1e-3):
                            coeff = ratios[0]
                            if abs(coeff - 1.0) < 1e-6:
                                return (
                                    True,
                                    f"{param_names[0]}^2-{param_names[1]}^2",
                                    None,
                                    None,
                                )
                            else:
                                return (
                                    True,
                                    f"{coeff:.10g}*({param_names[0]}^2-{param_names[1]}^2)",
                                    None,
                                    None,
                                )

                # Try x1^2 - x0^2
                term = x1**2 - x0**2
                mask = np.abs(term) > 1e-10
                if np.sum(mask) >= 1:
                    ratios = y_arr[mask] / term[mask]
                    if np.max(np.abs(ratios - ratios[0])) < 1e-3:
                        if np.all(np.abs(y_arr[~mask]) < 1e-3):
                            coeff = ratios[0]
                            if abs(coeff - 1.0) < 1e-6:
                                return (
                                    True,
                                    f"{param_names[1]}^2-{param_names[0]}^2",
                                    None,
                                    None,
                                )
                            else:
                                return (
                                    True,
                                    f"{coeff:.10g}*({param_names[1]}^2-{param_names[0]}^2)",
                                    None,
                                    None,
                                )

                # 4. Resistors / Harmonic: x*y/(x+y)
                sum_xy = x0 + x1
                if not np.any(np.isclose(sum_xy, 0, atol=1e-10)):
                    term = (x0 * x1) / sum_xy
                    mask = np.abs(term) > 1e-10
                    if np.sum(mask) >= 1:
                        ratios = y_arr[mask] / term[mask]
                        if np.max(np.abs(ratios - ratios[0])) < 1e-3:
                            if np.all(np.abs(y_arr[~mask]) < 1e-3):
                                coeff = ratios[0]
                                if abs(coeff - 1.0) < 1e-6:
                                    return (
                                        True,
                                        f"{param_names[0]}*{param_names[1]}/({param_names[0]}+{param_names[1]})",
                                        None,
                                        None,
                                    )
                                else:
                                    return (
                                        True,
                                        f"{coeff:.10g}*{param_names[0]}*{param_names[1]}/({param_names[0]}+{param_names[1]})",
                                        None,
                                        None,
                                    )

            # Test Gravity / Coulomb: x*y/z^2 and Ideal Gas: x*y/z
            if n_params >= 3:
                from itertools import permutations

                indices = list(range(n_params))

                for i, j, k in permutations(indices, 3):
                    # Avoid redundant symmetric checks (only check i < j for numerator)
                    if i > j:
                        continue

                    # Term 1: x_i * x_j / x_k^2 (Gravity)
                    denom_sq = X_arr[:, k] ** 2
                    if not np.any(np.isclose(denom_sq, 0, atol=1e-10)):
                        term_grav = (X_arr[:, i] * X_arr[:, j]) / denom_sq
                        mask = np.abs(term_grav) > 1e-10
                        if np.sum(mask) >= 1:
                            ratios = y_arr[mask] / term_grav[mask]
                            if np.max(np.abs(ratios - ratios[0])) < 1e-3:
                                if np.all(np.abs(y_arr[~mask]) < 1e-3):
                                    coeff = ratios[0]
                                    if abs(coeff - 1.0) < 1e-6:
                                        return (
                                            True,
                                            f"{param_names[i]}*{param_names[j]}/{param_names[k]}^2",
                                            None,
                                            None,
                                        )
                                    else:
                                        return (
                                            True,
                                            f"{coeff:.10g}*{param_names[i]}*{param_names[j]}/{param_names[k]}^2",
                                            None,
                                            None,
                                        )

                    # Term 2: x_i * x_j / x_k (Ideal Gas)
                    denom = X_arr[:, k]
                    if not np.any(np.isclose(denom, 0, atol=1e-10)):
                        term_gas = (X_arr[:, i] * X_arr[:, j]) / denom
                        mask = np.abs(term_gas) > 1e-10
                        if np.sum(mask) >= 1:
                            ratios = y_arr[mask] / term_gas[mask]
                            if np.max(np.abs(ratios - ratios[0])) < 1e-3:
                                if np.all(np.abs(y_arr[~mask]) < 1e-3):
                                    coeff = ratios[0]
                                    if abs(coeff - 1.0) < 1e-6:
                                        return (
                                            True,
                                            f"{param_names[i]}*{param_names[j]}/{param_names[k]}",
                                            None,
                                            None,
                                        )
                                    else:
                                        return (
                                            True,
                                            f"{coeff:.10g}*{param_names[i]}*{param_names[j]}/{param_names[k]}",
                                            None,
                                            None,
                                        )

        except Exception:
            pass  # Fall through to other methods

    # --- NEW: PRE-CHECK - Shifted Polynomial (x+c)^2 ---
    # For single-variable, test if y = (x + c)^2 for common shifts
    if n_params == 1 and len(data_points) >= 3:
        try:
            import numpy as np

            X_vals = [
                (
                    eval_to_float(p[0][0])
                    if isinstance(p[0], (list, tuple))
                    else eval_to_float(p[0])
                )
                for p in data_points
            ]
            y_vals = [eval_to_float(p[1]) for p in data_points]
            var_name = param_names[0]

            shift_values = [1, -1, 2, -2, 0.5, -0.5]
            for shift in shift_values:
                # Test (x + shift)^2
                shifted_sq_vals = [(x + shift) ** 2 for x in X_vals]
                errors = [abs(s - y) for s, y in zip(shifted_sq_vals, y_vals)]
                if max(errors) < 1e-3:
                    if shift >= 0:
                        return (
                            True,
                            f"({var_name}+{int(shift) if shift == int(shift) else shift})^2",
                            None,
                            None,
                        )
                    else:
                        return (
                            True,
                            f"({var_name}-{int(abs(shift)) if abs(shift) == int(abs(shift)) else abs(shift)})^2",
                            None,
                            None,
                        )

                # Test A*(x + shift)^2 with coefficient
                if not np.any(np.isclose(shifted_sq_vals, 0, atol=1e-10)):
                    ratios = [
                        y / s for y, s in zip(y_vals, shifted_sq_vals) if abs(s) > 1e-10
                    ]
                    if len(ratios) >= 2 and abs(max(ratios) - min(ratios)) < 1e-3:
                        coeff = sum(ratios) / len(ratios)
                        if shift >= 0:
                            shift_str = f"({var_name}+{int(shift) if shift == int(shift) else shift})"
                        else:
                            shift_str = f"({var_name}-{int(abs(shift)) if abs(shift) == int(abs(shift)) else abs(shift)})"

                        if abs(coeff - 1.0) < 1e-3:
                            return (True, f"{shift_str}^2", None, None)
                        elif abs(coeff - round(coeff)) < 1e-3:
                            return (
                                True,
                                f"{int(round(coeff))}*{shift_str}^2",
                                None,
                                None,
                            )
                        else:
                            return (True, f"{coeff:.10g}*{shift_str}^2", None, None)
        except Exception:
            pass  # Fall through to other methods

    # Uses OMP to find combinations of sin, cos, exp, log, 1/x, etc.
    try:
        from .regression_solver import solve_regression_stage

        # Extract X and y data - evaluate symbolic constants
        # p[0] is a tuple/list of x values for this data point
        X_data = []
        for p in data_points:
            if isinstance(p[0], (list, tuple)):
                X_data.append([eval_to_float(x) for x in p[0]])
            else:
                X_data.append([eval_to_float(p[0])])

        # Ensure y_data contains floats, not strings
        y_data = [eval_to_float(p[1]) for p in data_points]

        # Occam's Razor: Two-Stage Discovery
        # Stage 1: Simple Features Only (Poly + Rational). Avoids "Gaussian Trap".
        # Stage 2: Complex Features (Transcendentals). Only if Stage 1 fails.
        best_result = None
        stages = [False, True]  # Re-enabled Phase 2

        for include_transcendentals in stages:
            res = solve_regression_stage(
                X_data,
                y_data,
                data_points,
                param_names,
                include_transcendentals=include_transcendentals,
            )

            if res and res[0]:
                success, func_str, coeffs, mse = res

                # Calculate R² to determine if Phase 1 is "good enough"
                y_var = sum((y - sum(y_data) / len(y_data)) ** 2 for y in y_data) / len(
                    y_data
                )
                r_squared = 1 - (mse / y_var) if y_var > 1e-10 else 0.0

                # If Simple Stage finds a GOOD fit (R² > 0.99), STOP.
                # This prevents Phase 2 transcendentals from overfitting.
                if not include_transcendentals and r_squared > 0.99:
                    return (True, func_str, coeffs, None)

                # Otherwise, keep as candidate
                best_result = res

        if best_result:
            success, func_str, coeffs, mse = best_result
            return (success, func_str, coeffs, None)

        print("DEBUG: Advanced Solver failed to find a model.", file=sys.stderr)
        return (False, None, None, None)

    except (ImportError, Exception) as e:
        import sys

        sys.stderr.write(f"CRITICAL ERROR in advanced path: {e}\n")
        import traceback

        traceback.print_exc()
        pass  # Fall back to polynomial interpolation

    # For single-parameter functions, try polynomial basis expansion (degree 2) first
    # This can find patterns like π*x², x², etc. with exact rational or symbolic coefficients
    if n_params == 1:
        try:
            # Try polynomial basis expansion with degree 2 for single-parameter functions
            # This will find quadratic, linear, and constant terms
            sol, exps_list = _find_sparse_polynomial_solution(data_points, degree=2)

            if sol is not None:
                # Convert solution to readable function string
                func_str = _polynomial_solution_to_string(sol, exps_list, param_names)

                # Try to detect if coefficient is close to π and replace with symbolic π
                try:
                    # Ensure sp is available (imported at module level)
                    import sympy as sp_module

                    # Parse the function string to check for π-like coefficients
                    param_name = param_names[0]
                    local_dict = {
                        param_name: sp_module.Symbol(param_name),
                        "pi": sp_module.pi,
                        "Pi": sp_module.pi,
                    }
                    func_expr = sp_module.sympify(func_str, locals=local_dict)

                    # Check if it's a simple polynomial and try to detect π
                    # For f(x) = π*x², we'd have a coefficient close to π
                    # Expand any fractions first
                    expanded = sp_module.expand(func_expr)

                    param_symbol = sp_module.Symbol(param_names[0])
                    pi_val = float(sp_module.N(sp_module.pi))
                    pi_approx = 3.14159
                    pi_detected_early = False

                    # First, check if it's essentially π*x² (dominant x² term with coefficient ≈ π)
                    if isinstance(expanded, sp_module.Add):
                        # Check coefficient of x² term
                        x2_coeff = (
                            expanded.coeff(param_symbol**2)
                            if hasattr(expanded, "coeff")
                            else None
                        )
                        if x2_coeff is not None and x2_coeff != 0:
                            x2_coeff_val = float(sp_module.N(x2_coeff))
                            # Check if x² coefficient is close to π
                            if (
                                abs(x2_coeff_val - pi_val) / pi_val < 0.0001
                                or abs(x2_coeff_val - pi_approx) / pi_approx < 0.0001
                            ):
                                # Check if other terms are negligible
                                x_coeff = (
                                    expanded.coeff(param_symbol)
                                    if hasattr(expanded, "coeff")
                                    else 0
                                )
                                const_coeff = expanded.subs(param_symbol, 0)
                                x_coeff_val = (
                                    abs(float(sp_module.N(x_coeff)))
                                    if x_coeff != 0
                                    else 0
                                )
                                const_coeff_val = (
                                    abs(float(sp_module.N(const_coeff)))
                                    if const_coeff != 0
                                    else 0
                                )

                                # If other terms are very small (< 0.1% of x² term at largest x), simplify to π*x²
                                # Use largest x value from data points
                                max_x = max(
                                    abs(float(_parse_to_exact_fraction(p[0][0])))
                                    for p in data_points
                                )
                                x2_term_magnitude = abs(x2_coeff_val) * max_x * max_x
                                if (
                                    x_coeff_val * max_x < 0.001 * x2_term_magnitude
                                    and const_coeff_val < 0.001 * x2_term_magnitude
                                ):
                                    func_expr = sp_module.pi * param_symbol**2
                                    func_str = str(func_expr)
                                    func_str = func_str.replace("pi", "pi")
                                    # Mark that we've detected π*x² - skip further processing
                                    pi_detected_early = True
                                    # Verify this simplified expression matches data points
                                    all_match = True
                                    for point in data_points:
                                        input_vals = point[0]
                                        expected_output = point[1]

                                        subs_dict = {
                                            param_symbol: (
                                                _parse_to_exact_fraction(input_vals[0])
                                                if isinstance(
                                                    input_vals[0], (int, float, str)
                                                )
                                                else Fraction(input_vals[0])
                                            )
                                        }
                                        computed = func_expr.subs(subs_dict)
                                        expected_frac = (
                                            _parse_to_exact_fraction(expected_output)
                                            if isinstance(
                                                expected_output, (int, float, str)
                                            )
                                            else Fraction(expected_output)
                                        )

                                        # For π*x², allow tolerance for floating point approximations
                                        # Convert both to float for comparison
                                        computed_val = float(sp_module.N(computed))
                                        expected_val = float(expected_frac)
                                        diff = abs(computed_val - expected_val)
                                        # Allow tolerance for π approximations (e.g., 12.5664 vs 12.566370...)
                                        # Use relative tolerance: 0.01% of expected value, or absolute 1e-4
                                        tolerance = max(1e-4, abs(expected_val) * 1e-4)
                                        if diff > tolerance:
                                            all_match = False
                                            break

                                    if all_match:
                                        factored_form = None
                                        return (True, func_str, factored_form, None)
                                    # If verification fails, continue with polynomial solution

                        # Check each term for π-like coefficient (for more complex cases)
                        # Only if we haven't already detected π*x²
                        if not pi_detected_early and isinstance(
                            expanded, sp_module.Add
                        ):
                            new_terms = []
                            pi_detected = False
                            for term in expanded.args:
                                if isinstance(term, sp_module.Mul):
                                    # Check if any factor is a numeric coefficient close to π
                                    coeff_part = sp_module.Integer(1)
                                    var_part = sp_module.Integer(1)
                                    for factor in term.args:
                                        if isinstance(
                                            factor,
                                            (
                                                sp_module.Number,
                                                sp_module.Rational,
                                                sp_module.Float,
                                            ),
                                        ):
                                            coeff_part = coeff_part * factor
                                        else:
                                            var_part = var_part * factor

                                    # Check if coefficient is close to π (works for Rational, Float, etc.)
                                    if isinstance(
                                        coeff_part,
                                        (
                                            sp_module.Float,
                                            sp_module.Rational,
                                            sp_module.Integer,
                                        ),
                                    ):
                                        coeff_val = float(sp_module.N(coeff_part))
                                        # Check if coefficient is very close to π (within 0.01%)
                                        if (
                                            abs(coeff_val - pi_val) / pi_val < 0.0001
                                            or abs(coeff_val - pi_approx) / pi_approx
                                            < 0.0001
                                        ):
                                            # Replace with π
                                            new_terms.append(sp_module.pi * var_part)
                                            pi_detected = True
                                            continue

                                new_terms.append(term)

                            if pi_detected and not hasattr(
                                func_expr, "has_pi"
                            ):  # Avoid double detection
                                func_expr = sp_module.Add(*new_terms)
                                func_str = str(func_expr)
                                # Replace pi with π for display if possible, or keep as "pi"
                                func_str = func_str.replace("pi", "pi")
                        elif not pi_detected_early and isinstance(
                            expanded, sp_module.Mul
                        ):
                            # Check if it's a simple multiplication like π*x²
                            coeff_part = sp_module.Integer(1)
                            var_part = sp_module.Integer(1)
                            for factor in expanded.args:
                                if isinstance(
                                    factor,
                                    (
                                        sp_module.Number,
                                        sp_module.Rational,
                                        sp_module.Float,
                                    ),
                                ):
                                    coeff_part = coeff_part * factor
                                else:
                                    var_part = var_part * factor

                            # Check if coefficient is close to π
                            if isinstance(
                                coeff_part,
                                (
                                    sp_module.Float,
                                    sp_module.Rational,
                                    sp_module.Integer,
                                ),
                            ):
                                coeff_val = float(sp_module.N(coeff_part))
                                pi_val = float(sp_module.N(sp_module.pi))
                                pi_approx = 3.14159
                                # Check if coefficient is very close to π (within 0.01%)
                                if (
                                    abs(coeff_val - pi_val) / pi_val < 0.0001
                                    or abs(coeff_val - pi_approx) / pi_approx < 0.0001
                                ):
                                    func_expr = sp_module.pi * var_part
                                    func_str = str(func_expr)
                                    func_str = func_str.replace("pi", "pi")

                    # Verify the solution matches all data points
                    all_match = True
                    for point in data_points:
                        input_vals = point[0]
                        expected_output = point[1]

                        # Convert to Fractions for exact comparison
                        subs_dict = {
                            sp_module.Symbol(param_names[0]): (
                                _parse_to_exact_fraction(input_vals[0])
                                if isinstance(input_vals[0], (int, float, str))
                                else Fraction(input_vals[0])
                            )
                        }

                        computed = func_expr.subs(subs_dict)
                        expected_frac = (
                            _parse_to_exact_fraction(expected_output)
                            if isinstance(expected_output, (int, float, str))
                            else Fraction(expected_output)
                        )

                        # Check exact or close match (for π cases, we need tolerance)
                        if isinstance(computed, sp_module.Number):
                            computed_val = float(sp_module.N(computed))
                            expected_val = float(expected_frac)
                            diff = abs(computed_val - expected_val)
                            # Allow small tolerance for floating point or π approximations
                            if diff > max(1e-6, abs(expected_val) * 1e-6):
                                all_match = False
                                break
                        else:
                            # Symbolic - evaluate numerically
                            diff = abs(float(sp_module.N(computed - expected_frac)))
                            if diff > 1e-6:
                                all_match = False
                                break

                    if all_match:
                        factored_form = None
                        return (True, func_str, factored_form, None)
                except Exception:
                    # If π detection fails, just use the polynomial solution
                    # Verify it matches
                    try:
                        import sympy as sp_module

                        param_name = param_names[0]
                        local_dict = {param_name: sp_module.Symbol(param_name)}
                        func_expr = sp_module.sympify(func_str, locals=local_dict)

                        all_match = True
                        for point in data_points:
                            input_vals = point[0]
                            expected_output = point[1]

                            subs_dict = {
                                sp_module.Symbol(param_name): (
                                    _parse_to_exact_fraction(input_vals[0])
                                    if isinstance(input_vals[0], (int, float, str))
                                    else Fraction(input_vals[0])
                                )
                            }
                            computed = func_expr.subs(subs_dict)
                            expected_frac = (
                                _parse_to_exact_fraction(expected_output)
                                if isinstance(expected_output, (int, float, str))
                                else Fraction(expected_output)
                            )

                            if isinstance(computed, sp_module.Number):
                                computed_val = float(sp_module.N(computed))
                                expected_val = float(expected_frac)
                                if abs(computed_val - expected_val) > max(
                                    1e-6, abs(expected_val) * 1e-6
                                ):
                                    all_match = False
                                    break
                            else:
                                diff = abs(float(sp_module.N(computed - expected_frac)))
                                if diff > 1e-6:
                                    all_match = False
                                    break

                        if all_match:
                            return (True, func_str, None, None)
                    except Exception:
                        pass
        except Exception:
            # If polynomial basis expansion fails, fall through to polynomial interpolation
            pass

    # For multi-parameter functions, try extended basis expansion first (with inverse terms)
    # This can find rational functions like x*y/z^2 (Newton's law), x*y/z, etc.
    if n_params > 1:
        try:
            # First try with inverse terms included (for rational functions like x*y/z^2)
            # Use higher degree (4) to allow terms like x*y/z^2 (sum of abs values = 4)
            sol, exps_list = _find_sparse_polynomial_solution(
                data_points, degree=4, include_inverse=True
            )

            if sol is None:
                # Fallback: try without inverse terms (standard polynomial basis)
                sol, exps_list = _find_sparse_polynomial_solution(
                    data_points, degree=2, include_inverse=False
                )

            if sol is not None:
                # Convert solution to readable function string
                func_str = _polynomial_solution_to_string(sol, exps_list, param_names)

                # Verify the solution matches all data points exactly
                # (This should already be verified in _find_sparse_polynomial_solution, but double-check)
                try:
                    # Use module-level sp (imported at top of file)
                    local_dict = {param: sp.Symbol(param) for param in param_names}
                    func_expr = sp.sympify(func_str, locals=local_dict)

                    all_match = True
                    for point in data_points:
                        input_vals = point[0]
                        expected_output = point[1]

                        # Convert to Fractions for exact comparison
                        subs_dict = {}
                        for param, val in zip(param_names, input_vals):
                            if isinstance(val, (int, float, str)):
                                val_frac = _parse_to_exact_fraction(val)
                            else:
                                val_frac = (
                                    Fraction(val)
                                    if isinstance(val, Fraction)
                                    else Fraction(float(val))
                                )
                            subs_dict[sp.Symbol(param)] = val_frac

                        computed = func_expr.subs(subs_dict)
                        expected_frac = (
                            _parse_to_exact_fraction(expected_output)
                            if isinstance(expected_output, (int, float, str))
                            else Fraction(expected_output)
                        )

                        # Check exact match
                        if isinstance(computed, sp.Number):
                            computed_frac = (
                                Fraction(computed.numerator, computed.denominator)
                                if hasattr(computed, "numerator")
                                else Fraction(float(computed))
                            )
                            if computed_frac != expected_frac:
                                all_match = False
                                break
                        else:
                            # Symbolic - evaluate numerically
                            diff = abs(float(sp.N(computed - expected_frac)))
                            if diff > 1e-10:
                                all_match = False
                                break

                    if all_match:
                        # Generate factored form (for now, just use the same string)
                        # Could enhance this later to factor common terms
                        factored_form = None

                        return (True, func_str, factored_form, None)
                except Exception:
                    # If verification fails, fall through to linear regression
                    pass
        except Exception:
            # If polynomial basis expansion fails, fall through to linear regression
            pass

    # Fall back to linear regression (original algorithm)
    # For linear regression:
    # - Single-parameter: need at least 1 point (f(x) = a*x, or constant if 1 point)
    # - Multi-parameter: need at least n_params + 1 points (f(x,y) = A*x + B*y + C)
    if n_params > 1:
        n_coeffs = n_params + 1
    else:
        n_coeffs = n_params

    is_underdetermined = len(data_points) < n_coeffs

    if is_underdetermined:
        # For underdetermined systems, we'll use least squares to find the best-fit solution
        # Note: This will be a best-fit, not an exact fit
        pass  # Continue to solve using least squares below

    try:
        # Build system of linear equations: Ax = b
        # Each data point gives: coeff1*x1 + coeff2*x2 + ... = output
        # For f(x,y) = a*x + b*y, we need to find a and b

        # Extract inputs and outputs
        inputs = [point[0] for point in data_points]
        outputs = [point[1] for point in data_points]

        # Verify all inputs have same length
        for inp in inputs:
            if len(inp) != n_params:
                return (
                    False,
                    None,
                    f"All inputs must have {n_params} values, but got input with {len(inp)} values",
                )

        # Convert inputs and outputs to SymPy expressions if they're not already
        # This handles both numeric values and symbolic expressions like pi
        # For decimal values, convert to exact Rationals to preserve precision
        sympy_inputs = []
        for inp in inputs:
            sympy_row = []
            for val in inp:
                if isinstance(val, str):
                    # String - use helper function to convert to exact Rational
                    sympy_row.append(_float_to_exact_rational(val))
                elif isinstance(val, (int, float)):
                    # Use helper function to convert float to exact Rational
                    sympy_row.append(_float_to_exact_rational(val))
                elif isinstance(val, sp.Basic):
                    sympy_row.append(val)
                else:
                    # Try to convert to SymPy
                    try:
                        sympy_row.append(sp.sympify(val))
                    except (ValueError, TypeError):
                        return (
                            False,
                            None,
                            None,
                            f"Could not convert input value {val} to SymPy expression",
                        )
            sympy_inputs.append(sympy_row)

        sympy_outputs = []
        for out in outputs:
            if isinstance(out, str):
                # String - use helper function to convert to exact Rational
                sympy_outputs.append(_float_to_exact_rational(out))
            elif isinstance(out, (int, float)):
                # Use helper function to convert float to exact Rational
                sympy_outputs.append(_float_to_exact_rational(out))
            elif isinstance(out, sp.Basic):
                sympy_outputs.append(out)
            else:
                # Try to convert to SymPy
                try:
                    sympy_outputs.append(sp.sympify(out))
                except (ValueError, TypeError):
                    return (
                        False,
                        None,
                        None,
                        f"Could not convert output value {out} to SymPy expression",
                    )

        # Build coefficient matrix A and output vector b
        # For linear function: f(x1, x2, ..., xn) = a1*x1 + a2*x2 + ... + an*xn + C
        # Each row: [x1, x2, ..., xn, 1] -> output (the 1 accounts for the constant term C)
        # For multi-parameter functions, add a column of 1s for the constant term
        if n_params > 1:
            sympy_inputs_with_const = []
            for inp_row in sympy_inputs:
                sympy_inputs_with_const.append(inp_row + [sp.Integer(1)])
            A = sp.Matrix(sympy_inputs_with_const)
        else:
            # For single-parameter, use original inputs (no constant term by default)
            A = sp.Matrix(sympy_inputs)
        b = sp.Matrix(sympy_outputs)

        # Solve Ax = b for coefficients
        try:
            # For single-parameter functions, try polynomial fitting if we have enough points
            if n_params == 1 and len(data_points) > 1:
                # For n data points, we can fit a polynomial of degree (n-1) exactly
                # f(x) = a₀ + a₁x + a₂x² + ... + aₙ₋₁xⁿ⁻¹
                n_points = len(data_points)
                degree = n_points - 1

                # Check if we should try polynomial fitting (when we have more points than needed for linear)
                if n_points > 1:
                    # Method 1: Try Lagrange interpolation (closed-form, explicit formula)
                    # P(x) = Σᵢ yᵢ * Lᵢ(x) where Lᵢ(x) = Πⱼ≠ᵢ (x - xⱼ)/(xᵢ - xⱼ)
                    # This gives a direct closed-form expression
                    try:
                        # Sort points by x-value for numerical stability
                        sorted_points = sorted(
                            zip(sympy_inputs, sympy_outputs),
                            key=lambda p: (
                                float(sp.N(p[0][0]))
                                if isinstance(p[0][0], (sp.Basic, int, float))
                                else p[0][0]
                            ),
                        )
                        sorted_inputs = [p[0] for p in sorted_points]
                        sorted_outputs = [p[1] for p in sorted_points]

                        # Extract x and y values
                        x_vals = [inp[0] for inp in sorted_inputs]
                        y_vals = sorted_outputs
                        n = len(x_vals)

                        param_name = param_names[0]
                        param_symbol = sp.Symbol(param_name)

                        # Build Lagrange polynomial: P(x) = Σᵢ yᵢ * Lᵢ(x)
                        lagrange_poly = sp.Integer(0)

                        for i in range(n):
                            # Build Lagrange basis polynomial Lᵢ(x) = Πⱼ≠ᵢ (x - xⱼ)/(xᵢ - xⱼ)
                            L_i = sp.Integer(1)
                            x_i = x_vals[i]

                            for j in range(n):
                                if i != j:
                                    x_j = x_vals[j]
                                    # Add term (x - x_j) / (x_i - x_j)
                                    L_i = L_i * (param_symbol - x_j) / (x_i - x_j)

                            # Add y_i * L_i(x) to the polynomial
                            lagrange_poly = lagrange_poly + y_vals[i] * L_i

                        # Simplify and expand
                        lagrange_poly = sp.simplify(sp.expand(lagrange_poly))

                        # Verify the polynomial fits all points (sanity check)
                        all_match = True
                        for inp, out in zip(sorted_inputs, sorted_outputs):
                            try:
                                x_val = inp[0]
                                computed = lagrange_poly.subs(param_symbol, x_val)
                                expected = out
                                diff = abs(float(sp.N(computed - expected)))
                                if diff > 1e-10:
                                    all_match = False
                                    break
                            except (ValueError, TypeError):
                                all_match = False
                                break

                        if all_match:
                            # Convert to string format
                            func_str = str(lagrange_poly)

                            # Try to factor out common denominator
                            try:
                                parsed_func = lagrange_poly
                                if isinstance(parsed_func, sp.Add):
                                    denominators = []
                                    for term in parsed_func.args:
                                        if isinstance(term, sp.Mul):
                                            for factor in term.args:
                                                if (
                                                    isinstance(factor, sp.Rational)
                                                    and factor.denominator != 1
                                                ):
                                                    denominators.append(
                                                        factor.denominator
                                                    )
                                        elif (
                                            isinstance(term, sp.Rational)
                                            and term.denominator != 1
                                        ):
                                            denominators.append(term.denominator)

                                    if denominators:
                                        from math import gcd

                                        def lcm(a, b):
                                            return (
                                                abs(a * b) // gcd(a, b)
                                                if a and b
                                                else 0
                                            )

                                        common_denom = denominators[0]
                                        for d in denominators[1:]:
                                            common_denom = lcm(common_denom, d)

                                        multiplied = parsed_func * common_denom
                                        expanded = sp.expand(multiplied)

                                        if isinstance(expanded, sp.Add):
                                            all_integers = True
                                            for term in expanded.args:
                                                if isinstance(term, sp.Mul):
                                                    for factor in term.args:
                                                        if (
                                                            isinstance(
                                                                factor, sp.Rational
                                                            )
                                                            and factor.denominator != 1
                                                        ):
                                                            all_integers = False
                                                            break
                                                elif (
                                                    isinstance(term, sp.Rational)
                                                    and term.denominator != 1
                                                ):
                                                    all_integers = False

                                            if all_integers:
                                                numerator_str = str(expanded)
                                                numerator_str = numerator_str.replace(
                                                    " + -", " - "
                                                )
                                                func_str = (
                                                    f"({numerator_str})/{common_denom}"
                                                )
                            except (ValueError, TypeError, AttributeError):
                                pass

                            # Clean up the string representation
                            func_str = (
                                func_str.replace("**", "^")
                                if "^" not in func_str
                                else func_str
                            )
                            func_str = func_str.replace(" + -", " - ")

                            return (True, func_str, None, None)
                    except (
                        ValueError,
                        TypeError,
                        ZeroDivisionError,
                        AttributeError,
                        NotImplementedError,
                    ):
                        # If Lagrange method fails, try Newton method
                        pass

                    # Method 2: Try Newton divided differences (more numerically stable)
                    # This is better for high-degree polynomials and avoids Vandermonde conditioning issues
                    try:
                        # Sort points by x-value for numerical stability
                        sorted_points = sorted(
                            zip(sympy_inputs, sympy_outputs),
                            key=lambda p: (
                                float(sp.N(p[0][0]))
                                if isinstance(p[0][0], (sp.Basic, int, float))
                                else p[0][0]
                            ),
                        )
                        sorted_inputs = [p[0] for p in sorted_points]
                        sorted_outputs = [p[1] for p in sorted_points]

                        # Build Newton divided differences table
                        x_vals = [inp[0] for inp in sorted_inputs]
                        y_vals = sorted_outputs

                        # Compute divided differences
                        n = len(x_vals)
                        dd_table = [[None] * n for _ in range(n)]

                        # First column: y values
                        for i in range(n):
                            dd_table[i][0] = y_vals[i]

                        # Fill in divided differences
                        for j in range(1, n):
                            for i in range(n - j):
                                dd_table[i][j] = (
                                    dd_table[i + 1][j - 1] - dd_table[i][j - 1]
                                ) / (x_vals[i + j] - x_vals[i])

                        # Build Newton polynomial: P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...
                        param_name = param_names[0]
                        param_symbol = sp.Symbol(param_name)

                        # Build polynomial terms
                        poly_terms = [dd_table[0][0]]  # Constant term
                        for j in range(1, n):
                            if dd_table[0][j] == 0:
                                continue
                            # Build product (x-x₀)(x-x₁)...(x-x_{j-1})
                            product = 1
                            for k in range(j):
                                product = product * (param_symbol - x_vals[k])
                            poly_terms.append(dd_table[0][j] * product)

                        # Sum all terms to get the polynomial
                        newton_poly = sum(poly_terms)
                        newton_poly = sp.simplify(sp.expand(newton_poly))

                        # Convert to string format
                        func_str = str(newton_poly)

                        # Verify the polynomial fits all points (sanity check)
                        all_match = True
                        for inp, out in zip(sorted_inputs, sorted_outputs):
                            try:
                                x_val = inp[0]
                                computed = newton_poly.subs(param_symbol, x_val)
                                expected = out
                                diff = abs(float(sp.N(computed - expected)))
                                if diff > 1e-10:
                                    all_match = False
                                    break
                            except (ValueError, TypeError):
                                all_match = False
                                break

                        if all_match:
                            # Try to factor out common denominator
                            try:
                                parsed_func = newton_poly
                                if isinstance(parsed_func, sp.Add):
                                    denominators = []
                                    for term in parsed_func.args:
                                        if isinstance(term, sp.Mul):
                                            for factor in term.args:
                                                if (
                                                    isinstance(factor, sp.Rational)
                                                    and factor.denominator != 1
                                                ):
                                                    denominators.append(
                                                        factor.denominator
                                                    )
                                        elif (
                                            isinstance(term, sp.Rational)
                                            and term.denominator != 1
                                        ):
                                            denominators.append(term.denominator)

                                    if denominators:
                                        from math import gcd

                                        def lcm(a, b):
                                            return (
                                                abs(a * b) // gcd(a, b)
                                                if a and b
                                                else 0
                                            )

                                        common_denom = denominators[0]
                                        for d in denominators[1:]:
                                            common_denom = lcm(common_denom, d)

                                        multiplied = parsed_func * common_denom
                                        expanded = sp.expand(multiplied)

                                        if isinstance(expanded, sp.Add):
                                            all_integers = True
                                            for term in expanded.args:
                                                if isinstance(term, sp.Mul):
                                                    for factor in term.args:
                                                        if (
                                                            isinstance(
                                                                factor, sp.Rational
                                                            )
                                                            and factor.denominator != 1
                                                        ):
                                                            all_integers = False
                                                            break
                                                elif (
                                                    isinstance(term, sp.Rational)
                                                    and term.denominator != 1
                                                ):
                                                    all_integers = False

                                            if all_integers:
                                                numerator_str = str(expanded)
                                                numerator_str = numerator_str.replace(
                                                    " + -", " - "
                                                )
                                                func_str = (
                                                    f"({numerator_str})/{common_denom}"
                                                )
                            except (ValueError, TypeError, AttributeError):
                                pass

                            # Clean up the string representation
                            func_str = (
                                func_str.replace("**", "^")
                                if "^" not in func_str
                                else func_str
                            )
                            func_str = func_str.replace(" + -", " - ")

                            return (True, func_str, None, None)
                    except (
                        ValueError,
                        TypeError,
                        ZeroDivisionError,
                        AttributeError,
                        NotImplementedError,
                    ):
                        # If Newton method fails, fall back to Vandermonde
                        pass

                    # Method 2: Vandermonde matrix (fallback, but check condition number)
                    # Build Vandermonde matrix for polynomial fitting
                    # Each row: [1, x, x², x³, ..., xⁿ⁻¹]
                    vandermonde_rows = []
                    for inp in sympy_inputs:
                        x_val = inp[0]
                        row = [x_val**i for i in range(n_points)]
                        vandermonde_rows.append(row)

                    A_poly = sp.Matrix(vandermonde_rows)
                    b_poly = sp.Matrix(sympy_outputs)

                    # Check if the Vandermonde matrix is invertible
                    try:
                        det = A_poly.det()
                        if det == 0:
                            # If Vandermonde is singular, fall back to linear fit
                            pass
                        else:
                            # Check condition number for numerical stability warning
                            # For high-degree polynomials, Vandermonde can be ill-conditioned
                            try:
                                # Estimate condition number (simplified)
                                if n_points > 5:
                                    # Warn about potential numerical issues for high-degree polynomials
                                    # For now, proceed but this could be logged
                                    pass
                            except (ValueError, TypeError):
                                pass

                            # Solve for polynomial coefficients using Vandermonde matrix
                            coeffs_poly = A_poly.inv() * b_poly

                            # Build polynomial function string
                            # Order terms from highest degree to lowest (x^n, x^(n-1), ..., x, constant)
                            param_name = param_names[0]
                            terms = []

                            # Process terms in reverse order (highest degree first)
                            for i in range(len(coeffs_poly) - 1, -1, -1):
                                coeff = coeffs_poly[i]

                                # Skip zero coefficients
                                try:
                                    if coeff == 0 or (
                                        hasattr(coeff, "is_zero") and coeff.is_zero
                                    ):
                                        continue
                                    # Check numerically
                                    if abs(float(sp.N(coeff))) < 1e-10:
                                        continue
                                except (ValueError, TypeError):
                                    pass

                                if i == 0:
                                    # Constant term
                                    if isinstance(coeff, (sp.Rational, sp.Integer)):
                                        if (
                                            hasattr(coeff, "denominator")
                                            and coeff.denominator == 1
                                        ):
                                            terms.append(str(coeff.numerator))
                                        else:
                                            terms.append(
                                                f"({coeff.numerator}/{coeff.denominator})"
                                            )
                                    else:
                                        # Symbolic coefficient
                                        coeff_str = str(sp.simplify(coeff))
                                        terms.append(coeff_str)
                                else:
                                    # x^i terms
                                    if isinstance(coeff, (sp.Rational, sp.Integer)):
                                        if (
                                            hasattr(coeff, "denominator")
                                            and coeff.denominator == 1
                                        ):
                                            if coeff.numerator == 1:
                                                if i == 1:
                                                    terms.append(param_name)
                                                else:
                                                    terms.append(f"{param_name}**{i}")
                                            elif coeff.numerator == -1:
                                                if i == 1:
                                                    terms.append(f"-{param_name}")
                                                else:
                                                    terms.append(f"-{param_name}**{i}")
                                            else:
                                                if i == 1:
                                                    terms.append(
                                                        f"{coeff.numerator}*{param_name}"
                                                    )
                                                else:
                                                    terms.append(
                                                        f"{coeff.numerator}*{param_name}**{i}"
                                                    )
                                        else:
                                            if i == 1:
                                                terms.append(
                                                    f"({coeff.numerator}/{coeff.denominator})*{param_name}"
                                                )
                                            else:
                                                terms.append(
                                                    f"({coeff.numerator}/{coeff.denominator})*{param_name}**{i}"
                                                )
                                    else:
                                        # Symbolic coefficient
                                        coeff_str = str(sp.simplify(coeff))
                                        if i == 1:
                                            if any(
                                                op in coeff_str
                                                for op in ["+", "-", "*", "/"]
                                            ) and not (
                                                coeff_str.startswith("(")
                                                and coeff_str.endswith(")")
                                            ):
                                                terms.append(
                                                    f"({coeff_str})*{param_name}"
                                                )
                                            else:
                                                terms.append(
                                                    f"{coeff_str}*{param_name}"
                                                )
                                        else:
                                            if any(
                                                op in coeff_str
                                                for op in ["+", "-", "*", "/"]
                                            ) and not (
                                                coeff_str.startswith("(")
                                                and coeff_str.endswith(")")
                                            ):
                                                terms.append(
                                                    f"({coeff_str})*{param_name}**{i}"
                                                )
                                            else:
                                                terms.append(
                                                    f"{coeff_str}*{param_name}**{i}"
                                                )

                            if not terms:
                                func_str = "0"
                            else:
                                func_str = " + ".join(terms).replace(" + -", " - ")

                            # Try to factor out common denominator for cleaner output
                            try:
                                # Parse the function string and check if we can factor out a denominator
                                parsed_func = sp.sympify(func_str)
                                # Check if all coefficients are rational fractions
                                if isinstance(parsed_func, sp.Add):
                                    # Get all denominators
                                    denominators = []
                                    for term in parsed_func.args:
                                        if isinstance(term, sp.Mul):
                                            for factor in term.args:
                                                if (
                                                    isinstance(factor, sp.Rational)
                                                    and factor.denominator != 1
                                                ):
                                                    denominators.append(
                                                        factor.denominator
                                                    )
                                        elif (
                                            isinstance(term, sp.Rational)
                                            and term.denominator != 1
                                        ):
                                            denominators.append(term.denominator)

                                    if denominators:
                                        # Find LCM of all denominators
                                        from math import gcd

                                        def lcm(a, b):
                                            return (
                                                abs(a * b) // gcd(a, b)
                                                if a and b
                                                else 0
                                            )

                                        common_denom = denominators[0]
                                        for d in denominators[1:]:
                                            common_denom = lcm(common_denom, d)

                                        # Multiply the polynomial by common_denom and factor it out
                                        multiplied = parsed_func * common_denom
                                        # Check if the multiplied polynomial has integer coefficients
                                        try:
                                            # Try to convert to integer coefficients
                                            expanded = sp.expand(multiplied)
                                            # Check if all coefficients are integers
                                            if isinstance(expanded, sp.Add):
                                                all_integers = True
                                                for term in expanded.args:
                                                    if isinstance(term, sp.Mul):
                                                        for factor in term.args:
                                                            if (
                                                                isinstance(
                                                                    factor, sp.Rational
                                                                )
                                                                and factor.denominator
                                                                != 1
                                                            ):
                                                                all_integers = False
                                                                break
                                                    elif (
                                                        isinstance(term, sp.Rational)
                                                        and term.denominator != 1
                                                    ):
                                                        all_integers = False

                                                if all_integers:
                                                    # Format as (polynomial)/denominator
                                                    numerator_str = str(expanded)
                                                    # Clean up the numerator string
                                                    numerator_str = (
                                                        numerator_str.replace(
                                                            " + -", " - "
                                                        )
                                                    )
                                                    func_str = f"({numerator_str})/{common_denom}"
                                        except (ValueError, TypeError, AttributeError):
                                            pass
                            except (ValueError, TypeError, AttributeError, SyntaxError):
                                # If simplification fails, use original format
                                pass

                            return (True, func_str, None, None)
                    except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                        # If polynomial fitting fails, fall through to linear fit
                        pass

                # For single-parameter linear functions f(x) = a*x, check if all ratios are consistent
                # Check if all data points satisfy f(x) = a*x for the same a
                # Calculate ratio f(x)/x for each point
                ratios = []
                for inp, out in zip(sympy_inputs, sympy_outputs):
                    x_val = inp[0]
                    # Check if x_val is zero (would cause division by zero)
                    try:
                        if abs(float(sp.N(x_val))) < 1e-10:
                            continue
                    except (ValueError, TypeError):
                        pass
                    try:
                        ratio = sp.simplify(out / x_val)
                        ratios.append(ratio)
                    except (ZeroDivisionError, TypeError):
                        continue

                # Check if all ratios are equal (within tolerance for numeric values)
                if len(ratios) >= 2:
                    all_equal = True
                    first_ratio = ratios[0]
                    for ratio in ratios[1:]:
                        try:
                            # Check numeric equality
                            diff = abs(float(sp.N(first_ratio - ratio)))
                            if diff > 1e-10:
                                all_equal = False
                                break
                        except (ValueError, TypeError):
                            # Check symbolic equality
                            if not sp.simplify(first_ratio - ratio).equals(0):
                                all_equal = False
                                break

                    if all_equal:
                        # All points fit f(x) = a*x with the same a
                        coeffs = sp.Matrix([first_ratio])
                    else:
                        # Points don't fit a linear function exactly - check if we should warn
                        # Use least squares as fallback
                        ATA = A.T * A
                        if ATA.det() == 0:
                            return (
                                False,
                                None,
                                None,
                                "System is singular. The data points may not be linearly independent.",
                            )
                        coeffs = ATA.inv() * A.T * b

                        # Verify the fit - check if the solution actually satisfies the data points
                        # For single parameter, check if f(x) = a*x matches the data
                        try:
                            a_val = float(sp.N(coeffs[0]))
                            # Check each data point
                            max_error = 0
                            for inp, out in zip(sympy_inputs, sympy_outputs):
                                x_val = float(sp.N(inp[0]))
                                expected = float(sp.N(out))
                                predicted = a_val * x_val
                                error = abs(expected - predicted)
                                max_error = max(max_error, error)

                            # If error is significant, the data points don't fit a linear function
                            if max_error > 1e-6:
                                # Return error message indicating data doesn't fit
                                return (
                                    False,
                                    None,
                                    None,
                                    f"Data points do not fit a linear function f(x) = a*x. "
                                    f"Maximum error: {max_error:.6f}. "
                                    f"Please check your data points - they may not be linear.",
                                )
                        except (ValueError, TypeError, IndexError):
                            # If we can't verify, proceed with the solution
                            pass
                else:
                    # Not enough non-zero x values, use standard method
                    if A.det() == 0:
                        return (
                            False,
                            None,
                            None,
                            "System is singular. The data points may not be linearly independent.",
                        )
                    coeffs = A.inv() * b
            # For multi-parameter functions, account for constant term: n_coeffs = n_params + 1
            # For single-parameter functions, n_coeffs = n_params (no constant term by default)
            if n_params > 1:
                n_coeffs = n_params + 1
            else:
                n_coeffs = n_params

            if len(data_points) > n_coeffs:
                # Overdetermined system, use least squares
                # Check if A.T * A is invertible
                ATA = A.T * A
                if ATA.det() == 0:
                    return (
                        False,
                        None,
                        None,
                        "System is singular. The data points may not be linearly independent.",
                    )
                coeffs = ATA.inv() * A.T * b
            elif len(data_points) == n_coeffs:
                # Exactly determined system
                if A.det() == 0:
                    # System is singular, but check if it's consistent
                    # For consistent singular systems, we can find a particular solution
                    try:
                        # Check if system is consistent by comparing ranks
                        A_aug = A.col_insert(A.shape[1], b)
                        rank_A = A.rank()
                        rank_aug = A_aug.rank()

                        if rank_A == rank_aug:
                            # System is consistent - find a particular solution using Gauss-Jordan
                            try:
                                # Use gauss_jordan_solve to get a particular solution
                                # This returns (particular_solution, free_vars_matrix)
                                sol_result = A.gauss_jordan_solve(b)
                                if (
                                    isinstance(sol_result, tuple)
                                    and len(sol_result) >= 1
                                ):
                                    coeffs_param = sol_result[0]
                                    free_vars = (
                                        sol_result[1]
                                        if len(sol_result) > 1
                                        else sp.Matrix([])
                                    )

                                    # Extract a particular solution by setting free variables to 0
                                    # Replace free variables (like tau0, tau1, etc.) with 0
                                    coeffs = coeffs_param.subs(
                                        {var: 0 for var in coeffs_param.free_symbols}
                                    )

                                    # Convert to Matrix if needed
                                    if isinstance(coeffs, list):
                                        coeffs = sp.Matrix(coeffs)

                                    # Verify the solution
                                    if A * coeffs == b:
                                        # Solution is valid, proceed
                                        pass
                                    else:
                                        # Solution doesn't match, try alternative
                                        raise ValueError("Solution verification failed")
                                else:
                                    raise ValueError("Unexpected solution format")
                            except (
                                ValueError,
                                TypeError,
                                AttributeError,
                                NotImplementedError,
                            ):
                                # If gauss_jordan_solve fails, try to find a particular solution manually
                                # Use SVD-like approach or find any solution
                                # For now, try using the null space to find a particular solution
                                try:
                                    # Find one particular solution by setting free variables to 0
                                    # This is a simplified approach - may not always work
                                    # Remove linearly dependent rows and solve
                                    A_reduced = A.row_join(b)
                                    rref, pivots = A_reduced.rref()
                                    # Extract solution from RREF
                                    if len(pivots) > 0:
                                        # There's a solution, extract it
                                        sol_col = (
                                            A_reduced.shape[1] - 1
                                        )  # Last column is b
                                        coeffs_list = []
                                        for i in range(A.shape[1]):
                                            if i in pivots:
                                                # Find row with pivot in this column
                                                pivot_row = (
                                                    pivots.index(i)
                                                    if i in pivots
                                                    else None
                                                )
                                                if pivot_row is not None:
                                                    coeffs_list.append(
                                                        rref[pivot_row, sol_col]
                                                    )
                                                else:
                                                    coeffs_list.append(sp.Integer(0))
                                            else:
                                                # Free variable, set to 0 for particular solution
                                                coeffs_list.append(sp.Integer(0))
                                        coeffs = sp.Matrix(coeffs_list)
                                        # Verify
                                        if A * coeffs == b:
                                            pass  # Valid solution
                                        else:
                                            raise ValueError(
                                                "RREF solution verification failed"
                                            )
                                    else:
                                        raise ValueError("No solution found")
                                except Exception:
                                    # Last resort: return error
                                    return (
                                        False,
                                        None,
                                        None,
                                        "System is singular and no particular solution could be found. The data points may be collinear or not linearly independent.",
                                    )
                        else:
                            # System is inconsistent
                            return (
                                False,
                                None,
                                None,
                                "System is inconsistent. The data points do not satisfy any linear function f(x,y) = a*x + b*y + c.",
                            )
                    except Exception as e:
                        # If checking consistency fails, return error
                        return (
                            False,
                            None,
                            None,
                            f"System is singular. The data points may not be linearly independent. Error: {str(e)}",
                        )
                else:
                    # System is non-singular, solve normally
                    try:
                        coeffs = A.solve(b)
                        if isinstance(coeffs, list):
                            coeffs = sp.Matrix(coeffs)
                    except (ValueError, TypeError, AttributeError, NotImplementedError):
                        # Fallback to inverse
                        coeffs = A.inv() * b
            else:
                # Underdetermined system (fewer points than needed for exact solution)
                # Use least squares to find the best-fit solution
                # This finds the solution with minimum norm that minimizes ||Ax - b||²
                try:
                    # For underdetermined systems, use least squares: x = A^T * (A * A^T)^(-1) * b
                    # This gives the minimum-norm solution
                    AAT = A * A.T
                    if AAT.det() == 0:
                        # A * A^T is singular, try using pseudo-inverse or SVD-like approach
                        # For now, use least squares with A^T * (A * A^T)^(-1) * b
                        # But if A * A^T is singular, we need a different approach
                        # Try using gauss_jordan_solve or find any solution
                        try:
                            # Try to find a solution using gauss_jordan_solve
                            sol_result = A.T.gauss_jordan_solve(b)
                            if isinstance(sol_result, tuple) and len(sol_result) >= 1:
                                # This gives us coefficients for the underdetermined system
                                # The solution may not be unique, but we'll use this one
                                coeffs_temp = sol_result[0]
                                # Replace free variables with 0 to get a particular solution
                                coeffs = coeffs_temp.subs(
                                    {var: 0 for var in coeffs_temp.free_symbols}
                                )
                                if isinstance(coeffs, list):
                                    coeffs = sp.Matrix(coeffs)
                            else:
                                raise ValueError("Unexpected solution format")
                        except (
                            ValueError,
                            TypeError,
                            AttributeError,
                            NotImplementedError,
                        ):
                            # Fallback: try to solve using least squares with Moore-Penrose pseudo-inverse
                            # For underdetermined: x = A^T * (A * A^T)^(-1) * b
                            # But if A * A^T is singular, we can't use this
                            # Instead, set some coefficients to zero to make it determined
                            # For now, return a helpful error
                            return (
                                False,
                                None,
                                None,
                                f"Underdetermined system: Need at least {n_coeffs} data point(s) for {n_params} parameter(s) (including constant term). "
                                f"Got {len(data_points)} data point(s). "
                                f"The system has infinitely many solutions. Please provide more data points for a unique solution.",
                            )
                    else:
                        # Use least squares: x = A^T * (A * A^T)^(-1) * b
                        coeffs = A.T * AAT.inv() * b
                except (
                    ValueError,
                    TypeError,
                    AttributeError,
                    NotImplementedError,
                ) as e:
                    # If least squares fails, return helpful error
                    return (
                        False,
                        None,
                        None,
                        f"Underdetermined system: Need at least {n_coeffs} data point(s) for {n_params} parameter(s) (including constant term). "
                        f"Got {len(data_points)} data point(s). "
                        f"Could not find a best-fit solution: {str(e)}",
                    )

            # Convert coefficients to rational form or keep as symbolic
            # Handle both numeric and symbolic coefficients
            # IMPORTANT: Preserve exact Rationals from solve() - do not round them
            coeff_list = []
            for coeff in coeffs:
                # If coefficient is already a Rational, keep it as is (preserve exactness)
                if isinstance(coeff, sp.Rational):
                    coeff_list.append(coeff)
                # If coefficient is an Integer, keep it as is
                elif isinstance(coeff, sp.Integer):
                    coeff_list.append(coeff)
                # Check if coefficient is purely numeric
                else:
                    try:
                        # Check if it's already a Rational but not detected (e.g., from simplification)
                        if isinstance(coeff, sp.Basic):
                            # Try to simplify and see if it becomes a Rational
                            simplified = sp.simplify(coeff)
                            if isinstance(simplified, sp.Rational):
                                coeff_list.append(simplified)
                            elif isinstance(simplified, sp.Integer):
                                coeff_list.append(simplified)
                            else:
                                # Not a simple rational, check if it's close to a rational
                                num_val = float(sp.N(coeff))
                                if abs(num_val - round(num_val)) < 1e-10:
                                    coeff_list.append(sp.Rational(int(round(num_val))))
                                else:
                                    # Keep as simplified symbolic expression
                                    coeff_list.append(simplified)
                        else:
                            # Try to convert to Rational preserving exactness
                            num_val = float(sp.N(coeff))
                            if abs(num_val - round(num_val)) < 1e-10:
                                coeff_list.append(sp.Rational(int(round(num_val))))
                            else:
                                # For exact preservation, use the original coeff if it's already rational
                                coeff_list.append(sp.simplify(coeff))
                    except (ValueError, TypeError, OverflowError):
                        # Coefficient contains symbolic parts (e.g., pi), keep as is but simplify
                        coeff_list.append(sp.simplify(coeff))

            # Try to simplify coefficients by finding common factors
            # This helps convert fractions like (7/3, -1/3) to simpler forms when possible
            # Only do this for purely numeric coefficients
            try:
                # Check if all coefficients are numeric (Rational or Float)
                all_numeric = all(
                    isinstance(c, (sp.Rational, sp.Float, sp.Integer))
                    for c in coeff_list
                )

                if all_numeric:
                    from math import gcd

                    # Get all denominators
                    denominators = [
                        c.denominator
                        for c in coeff_list
                        if c != 0 and hasattr(c, "denominator")
                    ]
                    if denominators:
                        # Find LCM of all denominators
                        def lcm(a, b):
                            return abs(a * b) // gcd(a, b) if a and b else 0

                        common_denom = denominators[0]
                        for d in denominators[1:]:
                            common_denom = lcm(common_denom, d)

                        # Multiply all coefficients by common denominator to get integers
                        scaled_coeffs = [int(c * common_denom) for c in coeff_list]

                        # Find GCD of all non-zero coefficients (absolute values)
                        non_zero = [abs(c) for c in scaled_coeffs if c != 0]
                        if non_zero:
                            common_factor = non_zero[0]
                            for c in non_zero[1:]:
                                common_factor = gcd(common_factor, c)

                            # Also include common_denom in GCD calculation
                            # This helps simplify when all coefficients share a factor with the denominator
                            if common_denom > 1:
                                common_factor = gcd(common_factor, common_denom)

                            # Divide by common factor to simplify (if it divides all)
                            if common_factor > 1:
                                # Check if common_factor divides all coefficients and denominator
                                if (
                                    all(c % common_factor == 0 for c in scaled_coeffs)
                                    and common_denom % common_factor == 0
                                ):
                                    scaled_coeffs = [
                                        c // common_factor for c in scaled_coeffs
                                    ]
                                    common_denom = common_denom // common_factor

                            # Convert back to rational coefficients
                            if common_denom > 0:
                                coeff_list = [
                                    sp.Rational(c, common_denom) for c in scaled_coeffs
                                ]

                    # Additional simplification: Try to detect if coefficients are close to simpler rationals
                    # This handles cases where decimal inputs are approximations of simpler fractions
                    # Use limit_denominator to see if coefficients simplify to nicer forms
                    try:
                        from math import gcd

                        simplified_coeffs = []
                        for coeff in coeff_list:
                            if isinstance(coeff, sp.Rational):
                                # Try to simplify using limit_denominator
                                # Check if the coefficient is close to a simpler rational
                                frac = Fraction(
                                    int(coeff.numerator), int(coeff.denominator)
                                )
                                # Try with different max denominators to find a simpler form
                                # Use a higher limit to catch more cases (e.g., 12, 100, etc.)
                                simplified_frac = frac.limit_denominator(1000)
                                # Only use simplified form if it's very close (within 1e-10)
                                if abs(float(frac - simplified_frac)) < 1e-10:
                                    simplified_coeffs.append(
                                        sp.Rational(
                                            simplified_frac.numerator,
                                            simplified_frac.denominator,
                                        )
                                    )
                                else:
                                    simplified_coeffs.append(coeff)
                            else:
                                simplified_coeffs.append(coeff)

                        # If we simplified any coefficients, try to find a common denominator again
                        if simplified_coeffs != coeff_list:
                            # Re-check if all are still numeric
                            all_numeric_simplified = all(
                                isinstance(c, (sp.Rational, sp.Float, sp.Integer))
                                for c in simplified_coeffs
                            )
                            if all_numeric_simplified:
                                # Find common denominator again
                                denominators_simplified = [
                                    c.denominator
                                    for c in simplified_coeffs
                                    if c != 0 and hasattr(c, "denominator")
                                ]
                                if denominators_simplified:

                                    def lcm(a, b):
                                        return abs(a * b) // gcd(a, b) if a and b else 0

                                    common_denom_simplified = denominators_simplified[0]
                                    for d in denominators_simplified[1:]:
                                        common_denom_simplified = lcm(
                                            common_denom_simplified, d
                                        )

                                    # Scale to integers
                                    scaled_simplified = [
                                        int(c * common_denom_simplified)
                                        for c in simplified_coeffs
                                    ]
                                    # Find GCD
                                    non_zero_simplified = [
                                        abs(c) for c in scaled_simplified if c != 0
                                    ]
                                    if non_zero_simplified:
                                        common_factor_simplified = non_zero_simplified[
                                            0
                                        ]
                                        for c in non_zero_simplified[1:]:
                                            common_factor_simplified = gcd(
                                                common_factor_simplified, c
                                            )

                                        if (
                                            common_factor_simplified > 1
                                            and common_denom_simplified
                                            % common_factor_simplified
                                            == 0
                                        ):
                                            if all(
                                                c % common_factor_simplified == 0
                                                for c in scaled_simplified
                                            ):
                                                scaled_simplified = [
                                                    c // common_factor_simplified
                                                    for c in scaled_simplified
                                                ]
                                                common_denom_simplified = (
                                                    common_denom_simplified
                                                    // common_factor_simplified
                                                )

                                        # Convert back
                                        if common_denom_simplified > 0:
                                            coeff_list = [
                                                sp.Rational(c, common_denom_simplified)
                                                for c in scaled_simplified
                                            ]
                    except Exception:
                        # If simplification fails, use original coefficients
                        pass
            except Exception:
                # If simplification fails, use original coefficients
                pass

            # Build function string
            # For multi-parameter functions, the last coefficient is the constant term
            # Separate parameter coefficients from constant term
            if n_params > 1:
                param_coeffs = coeff_list[:-1]
                const_coeff = coeff_list[-1]
            else:
                param_coeffs = coeff_list
                const_coeff = None

            # Check if we have an underdetermined system (fewer data points than coefficients)
            # If so, we should show all parameters explicitly, even if some have zero coefficients
            is_underdetermined_local = len(data_points) < n_coeffs

            terms = []
            for i, (coeff, param) in enumerate(zip(param_coeffs, param_names)):
                # Check if coefficient is zero
                is_zero = False
                try:
                    if coeff == 0 or (hasattr(coeff, "is_zero") and coeff.is_zero):
                        is_zero = True
                except (TypeError, AttributeError):
                    # For symbolic expressions, check numerically
                    try:
                        if abs(float(sp.N(coeff))) < 1e-10:
                            is_zero = True
                    except (ValueError, TypeError):
                        pass  # Keep the term if we can't evaluate

                # For underdetermined systems, show zero coefficients explicitly
                if is_zero:
                    if is_underdetermined_local:
                        # Show as 0*param to indicate the parameter exists but has no effect
                        terms.append(f"0*{param}")
                    # Otherwise skip zero coefficients (normal behavior)
                    continue

                # Check for coefficient of 1 or -1
                try:
                    if coeff == 1 or (hasattr(coeff, "is_one") and coeff.is_one):
                        terms.append(param)
                        continue
                    elif coeff == -1:
                        terms.append(f"-{param}")
                        continue
                    elif hasattr(coeff, "__neg__"):
                        # Check if it's -1 symbolically
                        try:
                            if str(coeff) == "-1" or (
                                hasattr(sp, "Integer") and coeff == sp.Integer(-1)
                            ):
                                terms.append(f"-{param}")
                                continue
                        except (TypeError, AttributeError):
                            pass
                except (TypeError, AttributeError):
                    pass

                # Format coefficient
                if isinstance(coeff, (sp.Rational, sp.Integer)):
                    # Numeric rational coefficient
                    if hasattr(coeff, "denominator") and coeff.denominator == 1:
                        terms.append(f"{coeff.numerator}*{param}")
                    else:
                        terms.append(f"({coeff.numerator}/{coeff.denominator})*{param}")
                else:
                    # Symbolic coefficient (e.g., containing pi)
                    coeff_str = str(sp.simplify(coeff))
                    # Wrap in parentheses if it contains operators
                    if any(op in coeff_str for op in ["+", "-", "*", "/"]) and not (
                        coeff_str.startswith("(") and coeff_str.endswith(")")
                    ):
                        terms.append(f"({coeff_str})*{param}")
                    else:
                        terms.append(f"{coeff_str}*{param}")

                        # Add constant term if present (for multi-parameter functions)
            if const_coeff is not None:
                try:
                    # Check if constant is zero
                    is_zero = False
                    try:
                        if const_coeff == 0 or (
                            hasattr(const_coeff, "is_zero") and const_coeff.is_zero
                        ):
                            is_zero = True
                    except (TypeError, AttributeError):
                        try:
                            if abs(float(sp.N(const_coeff))) < 1e-10:
                                is_zero = True
                        except (ValueError, TypeError):
                            pass

                    # For underdetermined systems, show zero constant term explicitly
                    if is_zero and is_underdetermined_local:
                        terms.append("0")
                    elif not is_zero:
                        # Format constant term
                        if isinstance(const_coeff, (sp.Rational, sp.Integer)):
                            if (
                                hasattr(const_coeff, "denominator")
                                and const_coeff.denominator == 1
                            ):
                                terms.append(str(const_coeff.numerator))
                            else:
                                terms.append(
                                    f"({const_coeff.numerator}/{const_coeff.denominator})"
                                )
                        else:
                            # Symbolic constant
                            const_str = str(sp.simplify(const_coeff))
                            if any(
                                op in const_str for op in ["+", "-", "*", "/"]
                            ) and not (
                                const_str.startswith("(") and const_str.endswith(")")
                            ):
                                terms.append(f"({const_str})")
                            else:
                                terms.append(const_str)
                except (ValueError, TypeError, AttributeError):
                    pass  # Skip constant if formatting fails

            if not terms:
                # All coefficients are zero, function is constant 0
                func_str = "0"
            else:
                func_str = " + ".join(terms).replace(" + -", " - ")

            # Try to generate factored integer form if all coefficients are rational
            # Format: (A*x + B*y + C) / D
            factored_form = None
            try:
                if n_params > 1 and const_coeff is not None:
                    # Check if all coefficients are rational
                    all_rational = all(
                        isinstance(c, (sp.Rational, sp.Integer)) for c in coeff_list
                    )
                    if all_rational:
                        # Find LCM of all denominators
                        from math import gcd

                        def lcm(a, b):
                            return abs(a * b) // gcd(a, b) if a and b else 0

                        denominators = [
                            c.denominator
                            for c in coeff_list
                            if c != 0 and hasattr(c, "denominator")
                        ]
                        if denominators:
                            common_denom = denominators[0]
                            for d in denominators[1:]:
                                common_denom = lcm(common_denom, d)

                            # Scale all coefficients to integers
                            scaled_ints = [int(c * common_denom) for c in coeff_list]

                            # Build factored form: (A*x + B*y + C) / D
                            factored_terms = []
                            for i, (coeff_int, param) in enumerate(
                                zip(scaled_ints[:-1], param_names)
                            ):
                                if coeff_int != 0:
                                    if coeff_int == 1:
                                        factored_terms.append(param)
                                    elif coeff_int == -1:
                                        factored_terms.append(f"-{param}")
                                    else:
                                        factored_terms.append(f"{coeff_int}*{param}")
                                elif is_underdetermined_local:
                                    # For underdetermined systems, show zero coefficients explicitly
                                    factored_terms.append(f"0*{param}")

                            # Add constant term
                            const_int = scaled_ints[-1]
                            if const_int != 0:
                                if const_int > 0:
                                    factored_terms.append(str(const_int))
                                else:
                                    factored_terms.append(
                                        str(const_int)
                                    )  # Already has minus sign
                            elif is_underdetermined_local:
                                # For underdetermined systems, show zero constant term explicitly
                                factored_terms.append("0")

                            if factored_terms:
                                # Join terms, handling signs
                                factored_expr = " + ".join(factored_terms).replace(
                                    " + -", " - "
                                )
                                # Only include denominator if it's not 1
                                if common_denom == 1:
                                    factored_form = factored_expr
                                else:
                                    factored_form = (
                                        f"({factored_expr}) / {common_denom}"
                                    )
            except Exception:
                pass  # If factored form generation fails, just use regular form

            return (True, func_str, factored_form, None)

        except (ValueError, TypeError, AttributeError) as e:
            # Matrix might be singular or not invertible
            error_msg = str(e).lower()
            if (
                "singular" in error_msg
                or "determinant" in error_msg
                or "invert" in error_msg
            ):
                return (
                    False,
                    None,
                    None,
                    "Degenerate points — matrix singular. The data points may be collinear or not linearly independent. Please check your input data points.",
                )
            return (
                False,
                None,
                None,
                f"Cannot solve linear system. The data points may not be linearly independent. Error: {str(e)}",
            )

    except Exception as e:
        return (False, None, None, f"Error finding function: {str(e)}")


def fit_linear_from_two_points(
    x1: float | int | str | Fraction,
    y1: float | int | str | Fraction,
    x2: float | int | str | Fraction,
    y2: float | int | str | Fraction,
    param_name: str = "x",
) -> tuple[Fraction, Fraction, tuple[int, int, int], str, str, str]:
    """Find and simplify a linear function f(x) = a*x + b from two points using exact rational arithmetic.

    Uses exact Fraction arithmetic throughout to ensure precise results.
    Returns multiple equivalent forms of the function.

    Args:
        x1, y1: First point (x1, y1)
        x2, y2: Second point (x2, y2)
        param_name: Name of the parameter (default: "x")

    Returns:
        Tuple of:
        - a: Slope as Fraction
        - b: Intercept as Fraction
        - (A, B, L): Integer coefficients for form (A*x + B) / L
        - standard_form: String like "f(x) = a*x + b"
        - integer_form: String like "f(x) = (A*x + B) / L" or "f(x) = (B - |A|*x) / L"
        - alternative_form: String like "f(x) = b - |a|*x" (if a < 0)

    Raises:
        ValueError: If x1 == x2 (points must be distinct)
    """
    from math import gcd

    # Convert all inputs to exact Fractions
    x1_frac = _parse_to_exact_fraction(x1)
    y1_frac = _parse_to_exact_fraction(y1)
    x2_frac = _parse_to_exact_fraction(x2)
    y2_frac = _parse_to_exact_fraction(y2)

    # Check that points are distinct
    if x1_frac == x2_frac:
        raise ValueError("x1 must differ from x2")

    # Compute slope a exactly: a = (y2 - y1) / (x2 - x1)
    a = (y2_frac - y1_frac) / (x2_frac - x1_frac)

    # Compute intercept b exactly: b = y1 - a * x1
    b = y1_frac - a * x1_frac

    # Both a and b are now reduced Fractions (Fraction automatically reduces)

    # Compute integer form: find LCM of denominators
    a_denom = a.denominator
    b_denom = b.denominator
    L = abs(a_denom * b_denom) // gcd(a_denom, b_denom) if a_denom and b_denom else 1

    # Compute integer coefficients
    A = int(a * L)
    B = int(b * L)

    # Generate standard form: f(x) = a*x + b
    if a == 0:
        standard_form = f"f({param_name}) = {b}"
    elif a == 1:
        if b == 0:
            standard_form = f"f({param_name}) = {param_name}"
        elif b > 0:
            standard_form = f"f({param_name}) = {param_name} + {b}"
        else:
            standard_form = f"f({param_name}) = {param_name} - {abs(b)}"
    else:
        a_str = str(a) if a.denominator != 1 else str(a.numerator)
        if b == 0:
            standard_form = f"f({param_name}) = {a_str}*{param_name}"
        elif b > 0:
            standard_form = f"f({param_name}) = {a_str}*{param_name} + {b}"
        else:
            standard_form = f"f({param_name}) = {a_str}*{param_name} - {abs(b)}"

    # Generate integer form: f(x) = (A*x + B) / L or f(x) = (B - |A|*x) / L
    if L == 1:
        if A == 0:
            integer_form = f"f({param_name}) = {B}"
        elif A == 1:
            if B == 0:
                integer_form = f"f({param_name}) = {param_name}"
            elif B > 0:
                integer_form = f"f({param_name}) = {param_name} + {B}"
            else:
                integer_form = f"f({param_name}) = {param_name} - {abs(B)}"
        else:
            if B == 0:
                integer_form = f"f({param_name}) = {A}*{param_name}"
            elif B > 0:
                integer_form = f"f({param_name}) = {A}*{param_name} + {B}"
            else:
                integer_form = f"f({param_name}) = {A}*{param_name} - {abs(B)}"
    else:
        # Prefer form with positive numerator when A < 0: (B - |A|*x) / L
        if A < 0:
            integer_form = f"f({param_name}) = ({B} - {abs(A)}*{param_name}) / {L}"
        else:
            integer_form = f"f({param_name}) = ({A}*{param_name} + {B}) / {L}"

    # Generate alternative form: f(x) = b - |a|*x (when a < 0)
    if a < 0:
        a_abs_str = str(-a) if (-a).denominator != 1 else str(-a.numerator)
        if b == 0:
            alternative_form = f"f({param_name}) = -{a_abs_str}*{param_name}"
        elif b > 0:
            alternative_form = f"f({param_name}) = {b} - {a_abs_str}*{param_name}"
        else:
            alternative_form = f"f({param_name}) = -{abs(b)} - {a_abs_str}*{param_name}"
    else:
        alternative_form = standard_form

    # Verify the solution by substituting both points
    # Check point 1: f(x1) should equal y1
    computed_y1 = a * x1_frac + b
    if computed_y1 != y1_frac:
        raise ValueError(
            f"Verification failed: f({x1_frac}) = {computed_y1} != {y1_frac}"
        )

    # Check point 2: f(x2) should equal y2
    computed_y2 = a * x2_frac + b
    if computed_y2 != y2_frac:
        raise ValueError(
            f"Verification failed: f({x2_frac}) = {computed_y2} != {y2_frac}"
        )

    return (a, b, (A, B, L), standard_form, integer_form, alternative_form)


def parse_find_function_command(expr: str) -> tuple[str, list[str]] | None:
    """Parse a 'find function' command like 'find f(x,y)'.

    Args:
        expr: Expression string (e.g., "find f(x,y)", "f(2,1)=15, find f(x,y)")

    Returns:
        Tuple of (function_name, parameter_list) if valid, None otherwise
    """
    # Pattern: find function_name(param1,param2,...)
    # "find" can appear anywhere in the string, extract it
    pattern = r"find\s+([a-zA-Z][a-zA-Z0-9]*)\s*\(([^)]*)\)"
    match = re.search(pattern, expr.strip(), re.IGNORECASE)
    if not match:
        return None

    func_name = match.group(1)
    params_str = match.group(2).strip()

    # Parse parameters
    if not params_str:
        params = []
    else:
        params = [p.strip() for p in params_str.split(",") if p.strip()]

    return (func_name, params)


def get_function_registry_dump() -> dict[str, tuple[list[str], str]]:
    """Get a serializable dump of the function registry.

    Returns:
        Dict mapping function name to (params, body_str)
    """
    dump = {}
    for name, (params, body) in _function_registry.items():
        dump[name] = (params, str(body))
    return dump


def update_function_registry_from_dump(dump: dict[str, tuple[list[str], str]]) -> None:
    """Update the function registry from a dump.

    Args:
        dump: Dict mapping function name to (params, body_str)
    """
    global _function_registry

    # Clear existing registry to ensure we match the dump exactly
    _function_registry.clear()

    for name, (params, body_str) in dump.items():
        try:
            # Parse body string back to SymPy expression
            # We need to use parse_expr with local_dict to handle parameters correctly
            local_dict = {**ALLOWED_SYMPY_NAMES}
            for param in params:
                local_dict[param] = sp.Symbol(param)

            body = parse_expr(
                body_str,
                local_dict=local_dict,
                transformations=TRANSFORMATIONS,
                evaluate=True,
            )
            _function_registry[name] = (params, body)
        except Exception:
            # If parsing fails, skip this function
            pass


def export_function_to_file(func_name: str, filename: str) -> tuple[bool, str]:
    """Export a function from the registry to a Python file.

    Converts the mathematical formula to valid Python code and creates
    a standalone .py file that can be imported in other projects.

    Args:
        func_name: Name of the function to export (must be in registry)
        filename: Destination filename (e.g., "volume_lib.py")

    Returns:
        Tuple of (success, message)

    Example:
        >>> V(1,1)=3.14159, V(2,1)=12.56637, find V(r,h)
        >>> export V to volume_lib.py
        Creates:
            import math
            from math import sin, cos, exp, log, sqrt, pi

            def V(r, h):
                '''Auto-generated by Kalkulator AI
                Equation: 3.1416*r^2*h
                '''
                return 3.1416*r**2*h
    """
    # Check if function exists
    if func_name not in _function_registry:
        return (False, f"Error: Function '{func_name}' is not defined")

    params, body = _function_registry[func_name]

    # Convert SymPy expression to string
    body_str = str(body)

    # Convert mathematical syntax to Python syntax
    python_expr = _convert_to_python_syntax(body_str)

    # Determine which math imports are needed
    imports_needed = _detect_required_imports(python_expr)

    # Generate the Python code
    params_str = ", ".join(params) if params else ""

    # Build the import section
    import_lines = ["import math"]
    if imports_needed:
        import_lines.append(f"from math import {', '.join(sorted(imports_needed))}")

    # Build the function code
    code_lines = [
        "# Auto-generated by Kalkulator",
        "# Scientific Discovery Engine",
        "",
        *import_lines,
        "",
        "",
        f"def {func_name}({params_str}):",
        '    """Auto-generated by Kalkulator AI',
        "    ",
        f"    Equation: {body_str}",
        '    """',
        f"    return {python_expr}",
        "",
    ]

    code = "\n".join(code_lines)

    # Ensure filename has .py extension
    if not filename.endswith(".py"):
        filename += ".py"

    # Write to file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        return (True, f"Function '{func_name}' exported to {filename}")
    except Exception as e:
        return (False, f"Error writing file: {e}")


def _convert_to_python_syntax(expr_str: str) -> str:
    """Convert mathematical expression to valid Python syntax.

    Handles:
    - Power operator: x^2 -> x**2
    - Scientific notation: 1e-5 (already valid)
    - SymPy functions to math functions
    """
    import re

    result = expr_str

    # Replace ^ with ** for power (but not in strings)
    result = re.sub(r"\^", "**", result)

    # Replace SymPy exp with math.exp (if standalone, not already prefixed)
    # exp(-x^2) -> math.exp(-x**2)
    result = re.sub(r"\bexp\(", "math.exp(", result)

    # Replace SymPy sin, cos, tan with math versions
    result = re.sub(r"\bsin\(", "math.sin(", result)
    result = re.sub(r"\bcos\(", "math.cos(", result)
    result = re.sub(r"\btan\(", "math.tan(", result)

    # Replace log with math.log
    result = re.sub(r"\blog\(", "math.log(", result)

    # Replace sqrt with math.sqrt
    result = re.sub(r"\bsqrt\(", "math.sqrt(", result)

    # Replace sinh, cosh, tanh with math versions
    result = re.sub(r"\bsinh\(", "math.sinh(", result)
    result = re.sub(r"\bcosh\(", "math.cosh(", result)
    result = re.sub(r"\btanh\(", "math.tanh(", result)

    # Replace pi with math.pi
    result = re.sub(r"\bpi\b", "math.pi", result)

    # Replace E (Euler's number) with math.e
    result = re.sub(r"\bE\b", "math.e", result)

    return result


def _detect_required_imports(python_expr: str) -> set[str]:
    """Detect which math functions are used in the expression."""
    imports = set()

    if "math.sin(" in python_expr:
        imports.add("sin")
    if "math.cos(" in python_expr:
        imports.add("cos")
    if "math.tan(" in python_expr:
        imports.add("tan")
    if "math.exp(" in python_expr:
        imports.add("exp")
    if "math.log(" in python_expr:
        imports.add("log")
    if "math.sqrt(" in python_expr:
        imports.add("sqrt")
    if "math.sinh(" in python_expr:
        imports.add("sinh")
    if "math.cosh(" in python_expr:
        imports.add("cosh")
    if "math.tanh(" in python_expr:
        imports.add("tanh")
    if "math.pi" in python_expr:
        imports.add("pi")
    if "math.e" in python_expr:
        imports.add("e")

    return imports
