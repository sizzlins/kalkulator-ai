
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.symbolic_regression.genetic_engine import discover_equation

def verify_self_power():
    print("Verifying x^x discovery with Robust MSE...")
    
    # Generate data
    x_vals = [
        1, 2, 3, 4,         # Integers
        -1, -2, -3,         # Negative Integers
        0.5,                # Fractional
        10,                 # Large-ish
         # 20 would be 1e26
        6 
    ]
    
    X = np.array(x_vals).reshape(-1, 1)
    y = []
    
    for x in x_vals:
        val = complex(x) ** complex(x)
        y.append(val)
        
    y = np.array(y)
    
    print(f"Data matches x^x range. Starting discovery...")
         
    # Run discovery
    best_expr, best_mse, pareto = discover_equation(
        X, y, variable_names=['x'], timeout=30, verbose=True
    )
    
    print(f"\nDiscovered: {best_expr}")
    print(f"MSE: {best_mse}")
    
    if best_expr in ["x**x", "pow(x, x)"]:
        print("SUCCESS: Clean x^x found!")
    else:
        print(f"WARNING: Found {best_expr}. Check logs to see if Inverse space cheated (if MSE is small).")

if __name__ == "__main__":
    verify_self_power()
