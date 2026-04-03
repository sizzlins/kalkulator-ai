
import sys
import os
import sympy as sp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from kalkulator_pkg.solver.numeric import _numeric_roots_for_single_var
from kalkulator_pkg.solver.dispatch import solve_single_equation

def main():
    x = sp.Symbol('x')
    expr = sp.sin(x) - sp.cos(x)
    
    print(f"Testing numeric roots for: {expr}")
    roots = _numeric_roots_for_single_var(expr, x, interval=(-12, 12))
    print("Roots found:", roots)
    
    # Check correctness
    for r in roots:
        val = expr.subs(x, r).evalf()
        print(f"x={r}: err={val}")

    print("\n--- Full Solve Test ---")
    res = solve_single_equation("sin(x)=cos(x)", "x")
    print("Full result:", res)

if __name__ == "__main__":
    main()
