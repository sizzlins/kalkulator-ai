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
    
    # Simulating Boost 3
    config = GeneticConfig(
        generations=5, 
        population_size=3100, 
        verbose=True,
        timeout=60
    )
    
    regressor = GeneticSymbolicRegressor(config)
    
    start_time = time.time()
    print("\nStarting fit (Boost 3 Simulation)...")
    regressor.fit(X, y_target, variable_names=['x', 'y'])
    end_time = time.time()
    
    print(f"\nFit completed in {end_time - start_time:.2f}s")
    
    if regressor.best_tree:
        print(f"Best: {regressor.best_tree.to_pretty_string()}")

if __name__ == "__main__":
    main()
