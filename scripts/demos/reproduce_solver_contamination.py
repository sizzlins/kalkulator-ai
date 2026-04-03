
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import sympy as sp
from kalkulator_pkg.solver.dispatch import solve_single_equation
from kalkulator_pkg.worker import evaluate_safely

def main():
    print("Test 1: Normal solve without global x")
    res = solve_single_equation("sin(x)=cos(x)", "x")
    print(f"Result 1: {res.get('exact')}")

    # Simulate contamination
    # In the real app, variables are stored in a context dictionary or passed to evaluate_safely via user_functions/variables?
    # solve_single_equation calls evaluate_safely(lhs).
    # If evaluate_safely sees 'x' and 'x' is in its scope, it substitutes it.
    
    # Let's see if we can poison variable 'x' in evaluate_safely
    
    # HYPOTHESIS: The REPL environment (app.py) replaces 'x' in the string BEFORE calling solve_single_equation
    # OR evaluate_safely picks it up.
    
    print("\nTest 2: Simulating REPL substitution")
    # Simulate: User sets x = pi * (n + 1/4)
    # The REPL might be doing text substitution?
    
    n = sp.Symbol('n')
    val = sp.pi * (n + sp.Rational(1, 4))
    
    # If REPL substitutes 'x' in the string:
    expr_str = f"sin({val}) = cos({val})"
    print(f"Solving substituted: {expr_str}")
    
    res2 = solve_single_equation(expr_str, "n") # Solving for n now? No, user typed 'solve sin(x)=cos(x)'
    # If the user typed 'sin(x)=cos(x)', and 'x' is defined, does it become an equation of 'n'?
    
    # If x is defined, 'sin(x)=cos(x)' evaluates to 'sin(...) = cos(...)'
    # This is NO LONGER an equation in 'x'. It is an equation in 'n'.
    # So solve_single_equation("sin(..)=cos(..)", "x") would fail to find x?
    # But if no find_var is provided, it might infer 'n'.
    
    res3 = solve_single_equation(expr_str) # No find_var
    print(f"Result 3 (Implicit var): {res3.get('exact')}")

if __name__ == "__main__":
    main()
