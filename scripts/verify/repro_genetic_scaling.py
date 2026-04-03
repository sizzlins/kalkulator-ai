
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.solver import genetic_solver_adapter

def test_genetic_scaling():
    print("Testing Genetic Solver Scaling Bug...")
    
    # 1. Generate data for y = 1000 * x
    # Perfect linear relationship, but large scale.
    # MSE should be small relative to variance, but > 1.0 if not normalized?
    # Actually if perfect, MSE=0.
    # So we add noise.
    
    np.random.seed(42)
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    # y = 1000*x + noise(std=5)
    noise = np.random.normal(0, 5, 20)
    y = 1000 * X.flatten() + noise
    
    data_points = []
    for x_val, y_val in zip(X, y):
        data_points.append(((x_val[0],), y_val))
        
    print(f"Data Range: y in [{min(y):.2f}, {max(y):.2f}]")
    print(f"Noise Std: {np.std(noise):.2f}")
    
    # Expected MSE is approx var(noise) = 25.
    # This is > 1.0.
    # So genetic adapter should REJECT it if the bug exists.
    
    # We turn OFF regression helper in strategy, but here we call adapter DIRECTLY.
    # So we don't need to worry about regression solving it first.
    
    print("Running genetic_solver_adapter...")
    result = genetic_solver_adapter.solve(
        data_points,
        param_names=['x'],
        verbose=True,
        timeout=10.0,
        generations=10,
        population_size=50
    )
    
    success, expr, details, error = result
    
    if success:
        print(f"SUCCESS: Genetic solver returned: {expr}")
        print("Bug NOT reproduced (or MSE was < 1.0 by chance?)")
    else:
        print(f"FAILURE: Genetic solver failed: {error}")
        if "poor fit" in str(error) and "MSE=" in str(error):
            print("Bug CONFIRMED: Genetic solver rejected valid model due to scale.")
        else:
            print("Failure reason unexpected.")

if __name__ == "__main__":
    test_genetic_scaling()
