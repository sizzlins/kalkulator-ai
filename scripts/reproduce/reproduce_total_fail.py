
import numpy as np
import sys
import os
import time

sys.path.append(os.getcwd())

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig

def reproduction():
    print("Running reproduction of total failure...")
    
    # Data from user prompt (subset)
    # f(x) = floor(x)^2 + frac(x)^2
    # Include 0 and negatives
    X = np.array([
        [4.5], [4.4], [3.9], [3.0], [2.9], 
        [0.0], [-1.0], [-2.5], [-20.0],
        [0.001], [-0.0005], [e]
    ])
    # Corresponding Y (approximate based on function)
    # f(4.5) = 4^2 + 0.5^2 = 16.25
    # f(0) = 0
    # f(-2.5) = floor(-2.5)^2 + (-2.5 - floor(-2.5))^2 = (-3)^2 + 0.5^2 = 9.25
    
    # Let's use the explicit points from user log to be exact
    data_points = [
        (4.5, 16.25), (4.4, 16.16), (3.9, 9.81), (3.0, 9), (0, 0),
        (-2.5, 9.25), (-20, 400), (0.001, 1e-06),
        # Points that might cause issues?
        (-0.0005, 1.99900025) # f(-0.0005): floor=-1, frac=0.9995. 1 + 0.999... approx 2.
    ]
    
    X = np.array([[p[0]] for p in data_points])
    y = np.array([p[1] for p in data_points])
    
    # User had: evolve --hybrid --verbose --super-verbose --debug --boost 3 --transform
    config = GeneticConfig(
        population_size=50, # Smaller for repro
        generations=5,
        n_islands=1,
        verbose=True,
        boosting_rounds=1,
        # mimic transform mode manually? 
        # The user command implies multi-space. 
        # But 'Best result from none space' means even direct space failed.
    )
    
    reg = GeneticSymbolicRegressor(config)
    print(f"Fitting on {len(y)} points...")
    
    # Try direct fit first (should work)
    start = time.time()
    reg.fit(X, y)
    print(f"Fit took {time.time() - start:.2f}s")
    
    if reg.best_tree:
        print(f"Success: {reg.get_expression()}")
    else:
        print("FAIL: No solution found.")
        
    # Also try fit_with_transformations
    print("\nTesting fit_with_transformations...")
    res = reg.fit_with_transformations(X, y, ["x"])
    print(f"Result: {res}")

if __name__ == "__main__":
    # Define e for data construction
    from math import e
    reproduction()
