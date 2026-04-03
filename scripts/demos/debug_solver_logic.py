
import sys
import os
import sympy as sp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from kalkulator_pkg.solver.dispatch import solve_single_equation
from kalkulator_pkg.config import NUMERIC_FALLBACK_ENABLED

def main():
    print(f"NUMERIC_FALLBACK_ENABLED: {NUMERIC_FALLBACK_ENABLED}")
    
    eq = "sin(x)=cos(x)"
    print(f"\nSolving: {eq}")
    
    # 1. Solve
    res = solve_single_equation(eq, "x")
    print(f"Result OK: {res.get('ok')}")
    print(f"Type: {res.get('type')}")
    print(f"Exact: {res.get('exact')}")
    print(f"Approx: {res.get('approx')}")
    
    # 2. Check if specific erroneous values are present
    if res.get('exact'):
        for val in res.get('exact'):
            if "2.0" in val or val == "2":
                print(f"Found suspicious value: {val}")

if __name__ == "__main__":
    main()
