
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from kalkulator_pkg.worker import evaluate_safely
import sympy as sp

print("Checking SymPy version:", sp.__version__)

try:
    print("\nAttempting evaluate_safely('sin(x)')...")
    res = evaluate_safely("sin(x)")
    print("Result:", res)
except TypeError as e:
    print(f"Caught Expected TypeError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Caught Unexpected Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
