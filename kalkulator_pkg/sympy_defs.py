"""SymPy definitions and configurations (Lazy Loaded).

This module contains SymPy-dependent configurations that were previously in config.py.
It should only be imported when SymPy is strictly required, to avoid slow startup.
"""
import sympy as sp
from sympy.parsing.sympy_parser import convert_xor
from sympy.parsing.sympy_parser import implicit_multiplication_application
from sympy.parsing.sympy_parser import standard_transformations

from .utils.custom_functions import log2
from .utils.custom_functions import log10

# -----------------------------------------------------------------------------
# SymPy Custom Functions (Bitwise Evaluation)
# -----------------------------------------------------------------------------
class lshift(sp.Function):
    @classmethod
    def eval(cls, x, y):
        try: x, y = sp.sympify(x), sp.sympify(y)
        except: return None
        if x.is_number and y.is_number:
            try:
                if not (x.is_real and y.is_real): return sp.Integer(0)
                # Python's lshift raises ValueError for negative shift count
                return sp.Integer(int(x) << int(y))
            except (ValueError, TypeError, OverflowError):
                # Return 0 for invalid shifts (matches safe_lshift behavior)
                return sp.Integer(0)
        return None

class rshift(sp.Function):
    @classmethod
    def eval(cls, x, y):
        try: x, y = sp.sympify(x), sp.sympify(y)
        except: return None
        if x.is_number and y.is_number:
            try:
                if not (x.is_real and y.is_real): return sp.Integer(0)
                return sp.Integer(int(x) >> int(y))
            except (ValueError, TypeError, OverflowError):
                return sp.Integer(0)
        return None

class bitwise_xor(sp.Function):
    @classmethod
    def eval(cls, x, y):
        try: x, y = sp.sympify(x), sp.sympify(y)
        except: return None
        if x.is_number and y.is_number:
            try:
                if not (x.is_real and y.is_real): return sp.Integer(0)
                return sp.Integer(int(x) ^ int(y))
            except (ValueError, TypeError, OverflowError):
                return sp.Integer(0)
        return None

class bitwise_and(sp.Function):
    @classmethod
    def eval(cls, x, y):
        try: x, y = sp.sympify(x), sp.sympify(y)
        except: return None
        if x.is_number and y.is_number:
            try:
                if not (x.is_real and y.is_real): return sp.Integer(0)
                return sp.Integer(int(x) & int(y))
            except (ValueError, TypeError, OverflowError):
                return sp.Integer(0)
        return None

class bitwise_or(sp.Function):
    @classmethod
    def eval(cls, x, y):
        try: x, y = sp.sympify(x), sp.sympify(y)
        except: return None
        if x.is_number and y.is_number:
            try:
                if not (x.is_real and y.is_real): return sp.Integer(0)
                return sp.Integer(int(x) | int(y))
            except (ValueError, TypeError, OverflowError):
                return sp.Integer(0)
        return None

# -----------------------------------------------------------------------------
# Named Helper Functions (Pickle-Safe)
# -----------------------------------------------------------------------------
def _trunc_func(x):
    """Truncation toward zero."""
    return sp.sign(x) * sp.floor(sp.Abs(x))

def _frac_func(x):
    """Fractional part."""
    return x - sp.floor(x)

def _heaviside_func(x):
    """Heaviside step function (0.5 at x=0)."""
    return sp.Heaviside(x, sp.Rational(1, 2))

def _round_func(x):
    """Round to nearest integer."""
    return sp.floor(x + sp.Rational(1, 2))

def _neg_func(x):
    """Negation."""
    return -x

def _inv_func(x):
    """Inverse (1/x)."""
    return 1/x

class SafePrime(sp.Function):
    """Safe wrapper for prime(n) that handles symbolic arguments."""
    @classmethod
    def eval(cls, n):
        if n.is_Number:
            try:
                # Handle floats that are integers
                val = float(n)
                if val.is_integer() and val > 0:
                    return sp.prime(int(val))
            except:
                pass
        return None

ALLOWED_SYMPY_NAMES = {
    "pow": sp.Pow, # Explicitly allow pow(b, e) syntax
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "I": sp.I,
    # Special constants (must be recognized to prevent implicit mult corruption)
    "zoo": sp.zoo,  # Complex infinity
    "AccumBounds": sp.AccumBounds,
    "oo": sp.oo,    # Positive infinity
    "nan": sp.nan,  # Not a Number
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    # Traditional math notation aliases (arcsin = asin, etc.)
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "log": sp.log,
    "ln": sp.log,
    # Use custom classes to ensure proper parsing behavior (lambdas can cause TypeErrors with implicit multiplication)
    "log2": log2,
    "log10": log10,
    "exp": sp.exp,
    "Abs": sp.Abs,
    "abs": sp.Abs,  # lowercase alias for convenience
    # Hyperbolic functions
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "cot": sp.cot,
    # Modulo
    "Mod": sp.Mod,
    "mod": sp.Mod,  # lowercase alias for convenience
    # Calculus & algebra
    "diff": sp.diff,
    "integrate": sp.Integral,  # Lazy evaluation (prevents parse timeouts)
    "limit": sp.limit,  # For evaluating limits: limit(sin(x)/x, x, 0) -> 1
    "factor": sp.factor,
    "expand": sp.expand,
    "simplify": sp.simplify,
    # Number theory
    "factorint": sp.factorint,
    "divisors": sp.divisors,
    "isprime": sp.isprime,
    # Matrices (basic)
    "Matrix": sp.Matrix,
    "matrix": sp.Matrix,  # lowercase alias for convenience
    "det": sp.det,
    # Special functions
    "LambertW": sp.LambertW,
    "lambertw": sp.LambertW, # lowercase alias
    "min": sp.Min,
    "Min": sp.Min,  # uppercase alias
    "max": sp.Max,
    "Max": sp.Max,  # uppercase alias
    # Factorial and combinatorics
    "factorial": sp.factorial,
    "binomial": sp.binomial,
    # Rounding functions
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "ceil": sp.ceiling,  # alias
    "trunc": _trunc_func,
    # Number theory
    "gcd": sp.gcd,
    "lcm": sp.lcm,
    # Sign and gamma
    "sign": sp.sign,
    "gamma": sp.gamma,
    # Missing trig functions
    "sec": sp.sec,
    "csc": sp.csc,
    # Inverse hyperbolic functions
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    # Two-argument arctangent
    "atan2": sp.atan2,
    # Roots
    "root": sp.root,
    "cbrt": sp.cbrt,
    # Bessel functions
    "besselj": sp.besselj,  # Bessel function of first kind
    "primepi": sp.primepi,  # Prime-counting function
    "prime_pi": sp.primepi, # Alias
    "prime": SafePrime,      # N-th prime (Safe Wrapper)
    "ith_prime": SafePrime,  # Alias
    "SafePrime": SafePrime,  # Internal class name for re-parsing
    "lshift": lshift,
    "rshift": rshift,
    "bitwise_xor": bitwise_xor,
    "bitwise_and": bitwise_and,
    "bitwise_or": bitwise_or,
    "frac": _frac_func,
    # New operators
    "erf": sp.erf,
    "sinc": sp.sinc,
    "heaviside": _heaviside_func,
    "Heaviside": _heaviside_func,
    "round": _round_func,
    "neg": _neg_func,
    "inv": _inv_func,
    # Singularity Locking
    "locked": sp.Function("locked"),
    # Recurrence
    "fibonacci": sp.fibonacci,
    "lucas": sp.lucas,
    # Piecewise and conditional
    "Piecewise": sp.Piecewise,
    "Eq": sp.Eq,
    "Ne": sp.Ne,
    "Lt": sp.Lt,
    "Le": sp.Le,
    "Gt": sp.Gt,
    "Ge": sp.Ge,
}

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# -----------------------------------------------------------------------------
# SymPy Monkey-Patches for Bitwise Operators
# -----------------------------------------------------------------------------
def _lshift_patch(self, other):
    return sp.Function("lshift")(self, other)

def _rshift_patch(self, other):
    return sp.Function("rshift")(self, other)

if not hasattr(sp.Expr, "__lshift__"):
    sp.Expr.__lshift__ = _lshift_patch
    sp.Expr.__rrshift__ = _lshift_patch 

if not hasattr(sp.Expr, "__rshift__"):
    sp.Expr.__rshift__ = _rshift_patch
