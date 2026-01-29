import numpy as np
import time
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig

def main():
    print("Generating random data for f(x, y) = sin((x+y)/2)")
    np.random.seed(42)
    # Replicating user's 1000 points or 214 points. Let's use 1000 for robust statistics.
    n_samples = 1000
    X = np.random.uniform(-10, 10, size=(n_samples, 2))
    # y = sin((x+y)/2)
    y_target = np.sin((X[:, 0] + X[:, 1]) / 2.0)
    
    print(f"Data shape: {X.shape}, {y_target.shape}")
    
    config = GeneticConfig(
        generations=15, 
        population_size=500, # Keeping it standard
        verbose=True,
        timeout=60
    )
    
    regressor = GeneticSymbolicRegressor(config)
    
    start_time = time.time()
    print("\nStarting fit...")
    regressor.fit(X, y_target, variable_names=['x', 'y'])
    end_time = time.time()
    
    print(f"\nFit completed in {end_time - start_time:.2f}s")
    
    if regressor.best_tree:
        expr = regressor.best_tree.to_pretty_string()
        print(f"Best Expression: {expr}")
        
        # Verify correctness
        y_pred = regressor.best_tree.evaluate_fast(X)
        mse = np.mean((y_target - y_pred)**2)
        print(f"Final MSE: {mse}")
        
        if mse < 1e-6:
            print("SUCCESS: Function discovered.")
        else:
            print("FAILURE: MSE too high (Expected, confirming theory).")
            # Check if it found sin(y)
            if "sin(y)" in expr:
                print("Observed Expected Failure Mode: Found sin(y).")

if __name__ == "__main__":
    main()
