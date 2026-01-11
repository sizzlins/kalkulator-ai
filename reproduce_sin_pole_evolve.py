
import numpy as np
import sys
import os
import sympy as sp

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

def debug_sin_pole_fitness():
    print("Debugging sin(1/(x-3)) fitness...")
    
    # 1. Setup Data (exclude x=3)
    # Use points from the user logs to be exact
    # f(4.0) = 0.841470984807897
    # f(3.1) = -0.544021110889362
    # f(2.9) = 0.544021110889362
    
    X_list = [4.0, 3.1, 2.9, 2.0, 5.0]
    y_list = [0.841470984807897, -0.544021110889362, 0.544021110889362, -0.841470984807897, 0.479425538604203]
    
    X = np.array(X_list).reshape(-1, 1)
    y = np.array(y_list)
    
    # 2. Check Seed String
    seed_str = "sin(1/(x-3.0))"
    print(f"Testing seed: {seed_str}")
    
    # 3. Parse to Tree
    try:
        expr = sp.sympify(seed_str)
        tree = ExpressionTree.from_sympy(expr, ['x'])
        print(f"Tree created: {tree}")
        print(f"Tree structure: {tree.to_pretty_string()}")
    except Exception as e:
        print(f"FATAL: Failed to parse seed: {e}")
        return

    # 4. Evaluate Tree
    pred = tree.evaluate(X)
    print(f"Predictions: {pred}")
    print(f"Targets:     {y}")
    
    diff = pred - y
    mse = np.mean(diff**2)
    print(f"MSE: {mse}")
    
    if mse > 1e-5:
        print("FAILURE: MSE is too high!")
    else:
        print("SUCCESS: MSE is low/perfect.")

    # 5. Check if it works in Regressor Context
    config = GeneticConfig(population_size=10, generations=2, verbose=True, seeds=[seed_str])
    reg = GeneticSymbolicRegressor(config)
    
    print("\nRunning fit() with explicit seed...")
    reg.fit(X, y, ['x'])
    
    best_sol = reg.pareto_front.get_best()
    if best_sol:
        print(f"\nBest solution found: {best_sol.expression}")
        print(f"Best MSE: {best_sol.mse}")
        
        if "sin" in best_sol.expression and "x" in best_sol.expression:
             print("SUCCESS: Found sin based solution.")
        else:
             print("FAILURE: Did not find sin solution.")
    else:
        print("FAILURE: No solution found.")

if __name__ == "__main__":
    debug_sin_pole_fitness()
