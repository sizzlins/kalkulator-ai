import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from kalkulator_pkg.heuristics import solve_rational_function_svd

def test_svd_scaling():
    print("Testing SVD Scaling with Large Offset (1e9)...")
    
    # Data: y = 1e9 + x^2
    X = np.linspace(0, 10, 20)
    y = 1e9 + X**2
    
    # Format for solver
    X_arr = X.reshape(-1, 1)
    y_arr = y
    
    print(f"Data range: {y.min()} to {y.max()}")
    
    success, func_str, mse = solve_rational_function_svd(
        X_arr, y_arr, ["x"], 
        max_numerator_degree=3, max_denominator_degree=2,
        verbose=True
    )
    
    expected_val = 1e9 + 25
    print(f"\nResult:")
    print(f"Success: {success}")
    print(f"Function: {func_str}")
    print(f"MSE: {mse}")
    
    # Check result
    try:
        import math
        # Helper to eval
        # Replace python operators
        eval_str = func_str.replace("^", "**")
        
        safe_dict = {"x": 5, "math": math}
        val_at_5 = eval(eval_str, safe_dict)
        print(f"f(5) = {val_at_5} (Expected: {expected_val})")
        
        if abs(val_at_5 - expected_val) < 1e-3 * expected_val: # Relative error check
             print("PASS: Accurate result.")
        else:
             print("FAIL: Inaccurate result.")
            
    except Exception as e:
        print(f"Eval failed: {e}")

if __name__ == "__main__":
    test_svd_scaling()
