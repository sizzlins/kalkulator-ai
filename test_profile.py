"""Profile test to find bottleneck."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

import numpy as np
import time
from kalkulator_pkg.symbolic_regression import GeneticConfig, GeneticSymbolicRegressor

# Small dataset
X = np.array([[1.0], [2.0]])
y = np.array([2.0, 1.732])

print("Profiling test - single generation only")
print()

config = GeneticConfig(
    population_size=100,
    generations=5,  # Just 5 generations
    verbose=True,
    parsimony_coefficient=0.001,
)

start = time.time()
regressor = GeneticSymbolicRegressor(config)
pareto = regressor.fit(X, y, ['x'])
elapsed = time.time() - start

print()
print(f"Total time: {elapsed:.2f}s")
print(f"Time per generation: {elapsed/5:.2f}s")

if elapsed > 30:
    print("WARNING: Extremely slow! Should be <10s for 5 generations")
