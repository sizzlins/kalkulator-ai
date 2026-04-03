
import sympy as sp
import sys

print("Checking SymPy version:", sp.__version__)

try:
    print("\nTesting sp.sin(sp.Symbol('x'))")
    arg = sp.Symbol('x')
    print(sp.sin(arg))
except Exception as e:
    print(f"Error 1: {type(e).__name__}: {e}")

try:
    print("\nTesting sp.sin(sp.core.expr.Expr)")
    # We pass the CLASS itself, not an instance
    arg = sp.core.expr.Expr
    print(sp.sin(arg))
except TypeError as e:
    print(f"Error 2 (Target Match?): {type(e).__name__}: {e}")
except Exception as e:
    print(f"Error 2 (Other): {type(e).__name__}: {e}")

# Check calculate.py or similar where we might be doing something dangerous
# Like passing type(x) instead of x?
