
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.symbolic_regression.forensic_analysis import generate_pattern_seeds
from kalkulator_pkg.heuristics import check_power_peeling

class MockContext:
    pass

def verify_power_peeling():
    print("Verifying Enhanced Power Peeling...")
    ctx = MockContext()
    
    # generate_pattern_seeds internally calls check_power_peeling via _detect_tower_patterns
    
    # Case 1: y = x^x
    print("\n[Case 1] y = x^x")
    X = np.linspace(0.1, 5, 50).reshape(-1, 1)
    y = X[:, 0] ** X[:, 0]
    
    # Direct check
    success, expr, mse = check_power_peeling(X.tolist(), y.tolist(), ["x"], verbose=True)
    print(f"Direct check_power_peeling: Success={success}, Expr={expr}, MSE={mse}")
    if expr == "x^x" or expr == "x^(x)":
        print("PASS")
    else:
        print("FAIL")

    # Case 2: y = x^sqrt(x) (Requires include_roots=True in SVD)
    print("\n[Case 2] y = x^sqrt(x)")
    y_sqrt = X[:, 0] ** np.sqrt(X[:, 0])
    
    success, expr, mse = check_power_peeling(X.tolist(), y_sqrt.tolist(), ["x"], verbose=True)
    print(f"Direct check_power_peeling: Success={success}, Expr={expr}, MSE={mse}")
    
    # Expected: x^(sqrt(x)) or similar
    if "sqrt(x)" in str(expr) and "^" in str(expr):
        print("PASS")
    else:
        print("FAIL (Did not find sqrt in exponent)")
        
    # Case 3: Integration check via generate_pattern_seeds
    print("\n[Case 3] Integration via generate_pattern_seeds (y=x^x)")
    seeds, exact_match = generate_pattern_seeds(ctx, X, y, ["x"], verbose=True)
    print(f"Seeds found: {seeds}")
    if exact_match and ("x^x" in exact_match or "x^(x)" in exact_match):
        print("PASS")
    elif any("x^x" in s for s in seeds):
         print("PASS (Found in seeds)")
    else:
        print("FAIL")

if __name__ == "__main__":
    verify_power_peeling()
