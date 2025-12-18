
import sys
import os
import numpy as np

# Set up path
sys.path.insert(0, os.getcwd())

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

def test_max_discovery():
    print("Testing max(x,y) discovery...")
    
    # User provided data points:
    # f(3,4) = 4
    # f(5,6) = 6
    # f(7,8) = 8
    # f(9, 10) = 10
    # f(11, 12) = 12
    # f(e, pi) = 3.14159... (max)
    # f(pi, e) = 3.14159... (max)
    # f(sin(pi), cos(pi)) = 0 (max(0, -1) = 0)
    # f(99,99) = 99
    # f(1,1) = 1
    # f(23, 22) = 23  <-- The critical differentiator where y fails
    
    X = np.array([
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [np.e, np.pi],
        [np.pi, np.e],
        [np.sin(np.pi), np.cos(np.pi)], # 0, -1
        [99, 99],
        [1, 1],
        [23, 22]
    ])
    
    # Expected y is max(x1, x2)
    y_target = np.max(X, axis=1)
    
    # Config with max/min enabled (which is now default, but let's be implicit to verify defaults work)
    # We use a robust config to ensure success
    config = GeneticConfig(
        population_size=1000, # robust population
        n_islands=4,
        generations=20,
        parsimony_coefficient=0.001, # low penalty to allow 'max'
        verbose=True
    )
    
    regressor = GeneticSymbolicRegressor(config=config)
    
    try:
        pareto = regressor.fit(X, y_target, ['x', 'y'])
        best = pareto.get_best()
        
        print("\n--- Results ---")
        if best:
            print(f"Best Expression: {best.expression}")
            print(f"MSE: {best.mse}")
            print(f"Complexity: {best.complexity}")
            
            # Verify correctness on the trick case
            # f(23, 22) should be 23
            # If it found 'y', result is 22 (error)
            
            # We can evaluate the tree directly or rely on MSE logic
            if best.mse < 1e-6:
                print(f"[SUCCESS] Found exact match (MSE < 1e-6)")
                if "max" in best.expression or ("x" in best.expression and "y" in best.expression):
                     print("Structure looks correct (contains interaction or max).")
                else:
                     print("Warning: Expression might be 'y' or 'x' only?")
            else:
                print(f"[FAIL] High MSE. Did not find max(x,y). Found: {best.expression}")
                
        else:
            print("[FAIL] No solution found.")
            
    except Exception as e:
        print(f"[CRASH] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_max_discovery()
