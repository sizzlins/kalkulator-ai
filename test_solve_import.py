import sys
import os

# Mock imports
try:
    from kalkulator_pkg.solver import solve_single_equation
    print("[PASS] Global import successful")
except ImportError as e:
    print(f"[FAIL] Global import failed: {e}")
    sys.exit(1)

import sympy as sp
from kalkulator_pkg.parser import preprocess, parse_preprocessed
from kalkulator_pkg.utils.formatting import format_solution

def test_solve_logic():
    eq_str = "x^2 - 1 = 0"
    print(f"Testing logic for: {eq_str}")
    
    try:
        # Simulate the logic inside app.py solve handler
        # (With local imports removed as per my fix)
        preprocessed = preprocess(eq_str)
        parsed = parse_preprocessed(preprocessed)
        
        target_var = sp.Symbol('x')
        if isinstance(parsed, sp.Eq):
             result = solve_single_equation(parsed, target_var)
             print(f"Result: {format_solution(result)}")
        else:
             eq = sp.Eq(parsed, 0)
             result = solve_single_equation(eq, target_var)
             print(f"Result: {format_solution(result)}")
             
        print("[PASS] Logic execution successful")
    except NameError as e:
        print(f"[FAIL] NameError (UnboundLocal?): {e}")
    except Exception as e:
        print(f"[FAIL] Other Error: {e}")

if __name__ == "__main__":
    test_solve_logic()
