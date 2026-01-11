
import numpy as np
import sys
import os

# Ensure we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalkulator_pkg.function_manager import find_function_from_data
from kalkulator_pkg.function_finder_advanced import generate_candidate_features

def test_exp_interaction_discovery():
    print("Generating data for f(x, y) = exp(x + y) ...")
    
    # Generate data
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    
    # Target: exp(x + y) = exp(x) * exp(y)
    # The current system generates exp(x) and exp(y), but NOT their product.
    Z_flat = np.exp(X_flat + Y_flat)
    
    # Combine into (N, 2) matrix
    X_data = np.column_stack((X_flat, Y_flat))
    
    print("Attempting to find function (Linear Solver only)...")
    
    # We want to see if the feature matrix even contains the interaction
    features, names = generate_candidate_features(
        X_data, 
        variable_names=['x', 'y'],
        include_transcendentals=True
    )
    
    missing = True
    interaction_patterns = [
        "exp(x)*exp(y)",
        "exp(y)*exp(x)",
        "exp(x+y)" 
    ]
    
    print(f"Generated {len(names)} features.")
    
    found_interaction = None
    for name in names:
        # Check for direct interaction match
        if name in interaction_patterns:
            found_interaction = name
            break
            
    if found_interaction:
        print(f"SUCCESS: Found feature '{found_interaction}' in candidate list.")
        sys.exit(0) # Green
    else:
        print("FAILURE: Did not find 'exp(x)*exp(y)' or equivalent in candidate features.")
        print("Debug - Interaction features found:")
        for n in names:
            if '*' in n and 'exp' in n:
                print(f"  - {n}")
        sys.exit(1) # Red

if __name__ == "__main__":
    test_exp_interaction_discovery()
