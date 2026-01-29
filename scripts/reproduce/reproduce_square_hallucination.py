
import numpy as np
import sys
import sympy as sp
from kalkulator_pkg.function_manager import find_function_from_data

def reproduce_square_hallucination():
    print("Reproduction: Testing Square Space Hallucination")
    print("Target Function: y = -sqrt(x) (Negative root)")
    
    # Generate data for y = -sqrt(x)
    # Using a range where sqrt is well-defined
    X = np.linspace(1, 10, 10)
    y = -np.sqrt(X)
    
    print(f"Data X: {X}")
    print(f"Data y: {y}")
    
    # We expect the solver to find 'y = -sqrt(x)'.
    # If it falls into the Square Space Hallucination, it will likely find 'y = sqrt(x)'
    # because (-sqrt(x))^2 = x, which fits perfectly in the square space (y^2 = x).
    
    data_points = list(zip(X, y))
    try:
        # time_limit allows enough time for simple heuristics but maybe not full evolution
        result = find_function_from_data(None, data_points, ['x'])
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Solver crashed: {e}")
        sys.exit(1)
    
    print(f"Found Function: {result}")
    
    if result is None:
        print("FAILURE: No function found.")
        sys.exit(1)
        
    # Evaluate the found function at x=4
    # Should be -2.
    # If hallucinated as sqrt(x), it will be 2.
    
    x_sym = sp.symbols('x')
    try:
        expr = sp.sympify(result)
        val = float(expr.subs(x_sym, 4))
        print(f"f(4) = {val}")
        
        target = -2.0
        if abs(val - target) < 1e-3:
            print("SUCCESS: Found correct function (negative preserved).")
        elif abs(val - 2.0) < 1e-3:
            print("FAILURE: Hallucinated positive function (Sign lost via Square Space).")
            print("The solver fit y^2 = x and returned sqrt(x), ignoring the sign of y.")
            sys.exit(1) # Fail for CI/Test checking
        else:
            print(f"FAILURE: Found incorrect function {result} -> {val} (Expected {target})")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error evaluating result: {e}")
        sys.exit(1)

if __name__ == "__main__":
    reproduce_square_hallucination()
