
import sympy as sp
from kalkulator_pkg.utils.formatting import simplify_exponential_bases

def verify_fix():
    print("Testing simplify_exponential_bases with complex exp...")
    # exp(i*pi) = -1. This evaluates to -1.0 + 1.2e-16j potentially.
    # The function should simplify this to -1 without crashing.
    
    x = sp.Symbol('x')
    # Expression that evaluates to complex: exp(x) where x is complex?
    # No, the function transforms exp(c*x) or exp(c).
    # Let's try exp(c) where c results in complex.
    # exp(i * pi)
    expr = sp.exp(sp.I * sp.pi)
    print(f"Expression: {expr}")
    try:
        res = simplify_exponential_bases(expr)
        print(f"Result (should be -1): {res}")
    except TypeError as e:
        print(f"FAIL: Crashed with {e}")
        exit(1)

    # Test exp(c*x) with complex c? 
    # The bug was likely simple exp(val) where val is complex.
    # User's case: f(x)=sin(1/(x-3)) -> singularity -> complex?
    # Actually, user saw "Composed Hypothesis: Generates ...".
    # This involves `alt` command generation.
    
    # Try an expression that yields complex128
    # exp(3.5j)
    expr2 = sp.exp(3.5j)
    print(f"Expression 2: {expr2}")
    try:
        val = expr2.evalf()
        print(f"Value check: {val} (type: {type(val)})")
        # In formatting.py, it does val.evalf() then complex(val).
        res2 = simplify_exponential_bases(expr2)
        print(f"Result 2 (should be unchanged or simple): {res2}")
    except Exception as e:
        print(f"FAIL: Crashed with {e}")
        exit(1)

    print("PASS: No crashes observed.")

if __name__ == "__main__":
    verify_fix()
