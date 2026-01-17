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
        if x.is_number and y.is_number:
            return sp.Integer(int(x) << int(y))

class rshift(sp.Function):
    @classmethod
    def eval(cls, x, y):
        if x.is_number and y.is_number:
            return sp.Integer(int(x) >> int(y))

class bitwise_xor(sp.Function):
    @classmethod
    def eval(cls, x, y):
        if x.is_number and y.is_number:
            return sp.Integer(int(x) ^ int(y))

class bitwise_and(sp.Function):
    @classmethod
    def eval(cls, x, y):
        if x.is_number and y.is_number:
            return sp.Integer(int(x) & int(y))

class bitwise_or(sp.Function):
    @classmethod
    def eval(cls, x, y):
        if x.is_number and y.is_number:
            return sp.Integer(int(x) | int(y))

ALLOWED_SYMPY_NAMES = {
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
    "integrate": sp.integrate,
    "limit": sp.limit,  # For evaluating limits: limit(sin(x)/x, x, 0) -> 1
    "factor": sp.factor,
    "expand": sp.expand,
    "simplify": sp.simplify,
    # Matrices (basic)
    "Matrix": sp.Matrix,
    "matrix": sp.Matrix,  # lowercase alias for convenience
    "det": sp.det,
    # Special functions
    "LambertW": sp.LambertW,
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
    "lshift": lshift,
    "rshift": rshift,
    "bitwise_xor": bitwise_xor,
    "bitwise_and": bitwise_and,
    "bitwise_or": bitwise_or,
    # Floor, ceiling, fractional part
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "ceiling": sp.ceiling,
    "frac": lambda x: x - sp.floor(x),
    # New operators
    "erf": sp.erf,
    "sinc": sp.sinc,
    "heaviside": lambda x: sp.Heaviside(x, sp.Rational(1, 2)), # 0.5 at x=0
    "Heaviside": lambda x: sp.Heaviside(x, sp.Rational(1, 2)),
    "round": lambda x: sp.floor(x + sp.Rational(1, 2)),
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
