
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.symbolic_regression.genetic_engine import discover_equation

def verify_relu_fail():
    print("Verifying ReLU discovery failure...")
    
    # Data from user log
    # f(-5)=0 ... f(0)=0 ... f(5)=5
    x_vals = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    y_vals = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    
    X = np.array(x_vals).reshape(-1, 1)
    y = np.array(y_vals)
    
    print(f"X: {X.flatten()}")
    print(f"y: {y}")
    
    # Run discovery
    # We expect (x + abs(x))/2 or similar. 
    # If we get 'x', we must check the MSE.
    
    best_expr, best_mse, pareto = discover_equation(
        X, y, variable_names=['x'], timeout=30, verbose=True
    )
    
    print(f"\nDiscovered: {best_expr}")
    print(f"MSE: {best_mse}")
    
    # Check if result is 'x'
    if best_expr.strip() == "x":
        print("Reproduced: Found 'x' (Incorrect for negative values)")
        # Calculate manual MSE for 'x'
        pred = X.flatten()
        
        # Relative fitness logic replication
        denom = np.abs(y)
        denom[denom < 1e-10] = 1.0
        diff_rel = (pred - y) / denom
        manual_mse_rel = np.mean(diff_rel**2)
        
        print(f"Manual Relative MSE for 'x': {manual_mse_rel}")
        
        if best_mse < 1e-10 and manual_mse_rel > 1.0:
            print("CRITICAL: Engine reports ~0 MSE but Manual calculation is High!")
    else:
        print("Did not reproduce 'x'. Found something else.")

if __name__ == "__main__":
    verify_relu_fail()
