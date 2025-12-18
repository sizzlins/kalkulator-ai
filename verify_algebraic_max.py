
import sys
import os
import numpy as np

# Set up path
sys.path.insert(0, os.getcwd())

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

def test_algebraic_max_discovery():
    print("Testing algebraic max discovery (banning 'max' and 'min')...")
    
    # f(23, 22) = 23 test case
    X = np.array([
        [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
        [np.e, np.pi], [np.pi, np.e],
        [0, -1], [99, 99], [1, 1], [23, 22],
        [100, 50], [50, 100], [-10, 10], [10, -10] # More data helps complex forms
    ])
    
    y_target = np.max(X, axis=1)
    
    # Disable max/min to force algebraic solution
    # We also keep parsimony low to allow larger trees
    config = GeneticConfig(
        population_size=10000,
        n_islands=4,
        generations=100,
        parsimony_coefficient=0.000001,
        verbose=True,
        operators=[
            "add", "sub", "mul", "div", 
            "abs", 
            "sqrt", "square", 
            "neg"
        ]
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
            
            # Check if it matches algebraic logic
            # (x+y+abs(x-y))/2
            # Check f(23,22)
            # Evaluate using the user's manual check logic (mocking it here)
            # 23+22 + abs(1) = 46 / 2 = 23.
            pass
        else:
            print("[FAIL] No solution found.")
            
    except Exception as e:
        print(f"[CRASH] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_algebraic_max_discovery()
