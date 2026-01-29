import numpy as np
import time
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig

def main():
    print("Generating data for f(x, y) = sin(x) * y")
    np.random.seed(42)
    n_samples = 1000
    X = np.random.uniform(-10, 10, size=(n_samples, 2))
    # y = sin(x) * y
    y_target = np.sin(X[:, 0]) * X[:, 1]
    
    print(f"Data shape: {X.shape}, {y_target.shape}")
    
    config = GeneticConfig(
        generations=10, # Keep short for verification
        population_size=100,
        verbose=True,
        timeout=30 # Short timeout
    )
    
    regressor = GeneticSymbolicRegressor(config)
    
    start_time = time.time()
    print("\nStarting fit...")
    regressor.fit(X, y_target, variable_names=['x', 'y'])
    end_time = time.time()
    
    print(f"\nFit completed in {end_time - start_time:.2f}s")
    
    best_tree = regressor.best_tree
    if best_tree:
        expr = best_tree.to_pretty_string()
        print(f"Best Expression: {expr}")
        
        # Verify correctness
        y_pred = best_tree.evaluate_fast(X)
        mse = np.mean((y_target - y_pred)**2)
        print(f"Final MSE: {mse}")
        
        if mse < 1e-6:
            print("SUCCESS: Function discovered explicitly.")
            if "sin(x)" in expr and "y" in expr:
                print("SUCCESS: Structure matches.")
            else:
                print("WARNING: Structure mismatch (might be equivalent).")
        else:
            print("FAILURE: MSE too high.")
    else:
        print("FAILURE: No solution found.")

if __name__ == "__main__":
    main()
