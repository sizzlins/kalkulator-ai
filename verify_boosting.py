
import sys
import os
import numpy as np

# Set up path
sys.path.insert(0, os.getcwd())

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

def test_boosting_discovery():
    print("Testing Strategy 7: Symbolic Boosting (Autonomous Discovery)...")
    
    # Target: max(x,y)
    # Algebraic Form: (x + y + abs(x - y)) / 2
    
    X = np.array([
        [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
        [np.e, np.pi], [np.pi, np.e],
        [0, -1], [99, 99], [1, 1], [23, 22],
        [100, 50], [50, 100], [-10, 10], [10, -10],
        [1000, 2000], [-500, -300]
    ])
    
    y_target = np.max(X, axis=1)
    
    # Disable max/min to force algebraic solution
    # Enable Boosting (2 rounds)
    config = GeneticConfig(
        population_size=1000,
        n_islands=2,
        generations=25,
        parsimony_coefficient=0.0001,
        verbose=True,
        operators=[
            "add", "sub", "mul", "div", 
            "abs", 
            "sqrt", "square", 
            "neg"
        ],
        seeds=[], # NO SEEDS!
        boosting_rounds=2  # The Magic Number
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
                print("[SUCCESS] Found exact match using Boosting!")
            else:
                print(f"[FAIL] High MSE. Boosting did not converge? Found: {best.expression}")
                
        else:
            print("[FAIL] No solution found.")
            
    except Exception as e:
        print(f"[CRASH] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_boosting_discovery()
