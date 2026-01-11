
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.symbolic_regression import GeneticSymbolicRegressor, GeneticConfig
from kalkulator_pkg.utils.formatting import format_solution

def run():
    # Data from user report: sin(x^2)
    # f(0)=0, f(1)=0.8415, f(1.5)=0.7781, f(2)=-0.7568, f(2.5)=-0.0332, f(3)=0.4121, f(3.5)=-0.4282
    X = np.array([[0], [1], [1.5], [2], [2.5], [3], [3.5]])
    y = np.array([0, 0.8415, 0.7781, -0.7568, -0.0332, 0.4121, -0.4282])
    
    # Simulate --boost 3 configuration
    # repl_commands.py scales parameters:
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
    
    print(f"Running regression with pop={config.population_size}, gen={config.generations}")
    
    reg = GeneticSymbolicRegressor(config)
    front = reg.fit(X, y)
    model = front.get_best()
    
    if not model:
        print("FAIL: No model found.")
        sys.exit(1)
    
    print(f"\nFinal Result: {model.expression}")
    print(f"MSE: {model.mse}")
    
    # Check if result is close to sin(x^2)
    # We can check MSE against user reported "bad" MSE (0.198) vs "good" MSE (0.002)
    if model.mse > 0.1:
        print("FAIL: MSE too high. Regression reproduced!")
        sys.exit(1)
    else:
        print("SUCCESS: Low MSE found.")

if __name__ == "__main__":
    run()
