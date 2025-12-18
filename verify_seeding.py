
import sys
import os
import numpy as np

# Set up path
sys.path.insert(0, os.getcwd())

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

def test_seeding_discovery():
    print("Testing Seeding Strategy (injecting 'abs(x-y)' to find max)...")
    
    # f(23, 22) = 23 test case
    X = np.array([
        [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
        [np.e, np.pi], [np.pi, np.e],
        [0, -1], [99, 99], [1, 1], [23, 22],
        [100, 50], [50, 100], [-10, 10], [10, -10]
    ])
    
    y_target = np.max(X, axis=1)
    
    # Disable max/min to force algebraic solution
    # BUT inject seed "abs(x-y)"
    config = GeneticConfig(
        population_size=1000,
        n_islands=2,
        generations=20,
        parsimony_coefficient=0.0001,
        verbose=True,
        operators=[
            "add", "sub", "mul", "div", 
            "abs", 
            "sqrt", "square", 
            "neg"
        ],
        seeds=["(x + y + abs(x - y)) / 2"] # Seed with the exact answer to PROVE injection works
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
            
            if best.mse < 1e-6:
                print("[SUCCESS] Found exact match using seed!")
            else:
                print(f"[FAIL] High MSE. Seed did not help enough? Found: {best.expression}")
                
        else:
            print("[FAIL] No solution found.")
            
    except Exception as e:
        print(f"[CRASH] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_seeding_discovery()
