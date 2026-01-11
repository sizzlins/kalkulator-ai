
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.symbolic_regression import GeneticSymbolicRegressor, GeneticConfig

def run():
    # Data: sqrt(x^2 - 16) with wings
    # User's wings: [-20, -4] and [4, 20]
    X_neg = np.linspace(-20, -4, 10).reshape(-1, 1)
    X_pos = np.linspace(4, 20, 10).reshape(-1, 1)
    X = np.vstack([X_neg, X_pos])
    
    # Calculate target y
    y = np.sqrt(X**2 - 16).flatten()
    
    # Simulate current behavior
    base_pop = 100
    base_gen = 30
    rounds = 3
    
    config = GeneticConfig(
        population_size=base_pop * rounds,
        generations=base_gen * rounds,
        n_islands=2,
        mutation_rate=0.4,
        crossover_rate=0.4,
        verbose=True
    )
    
    print(f"Running regression on sqrt(x^2-16) wings...")
    
    reg = GeneticSymbolicRegressor(config)
    front = reg.fit(X, y, variable_names=['x'])
    model = front.get_best()
    
    if not model:
        print("FAIL: No model found.")
        sys.exit(1)
            
    print(f"\nFinal Result: {model.expression}")
    print(f"MSE: {model.mse}")
    
    # Check for integer 16 vs float 15.99...
    expr_str = model.expression
    if "16" in expr_str and "15." not in expr_str:
        print("SUCCESS: Found exact integer 16.")
    elif "15.99" in expr_str or "16.00" in expr_str:
        print("PARTIAL: Found approximate constant (Float artifact).")
        # We want to change this to SUCCESS by fixing valid floats
    else:
        print("FAIL: Found neither 16 nor close approximation.")

if __name__ == "__main__":
    run()
