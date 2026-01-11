"""Debug: Check if anchor seeds are being used."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

import numpy as np
from kalkulator_pkg.symbolic_regression import GeneticConfig, GeneticSymbolicRegressor

X = np.array([[2.0]])
y = np.array([1.732])

print("DEBUG: Checking if (x+1)**(1/x) seed is being used")
print()

config = GeneticConfig(
    population_size=50,
    generations=1,  # Just 1 generation
    verbose=True,  # See anchor detection
    parsimony_coefficient=0.001,
)

print("Creating regressor and fitting...")
regressor = GeneticSymbolicRegressor(config)
pareto = regressor.fit(X, y, ['x'])

print()
print("Results after 1 generation:")
for i, sol in enumerate(pareto.solutions[:5]):
    print(f"  {i+1}. {sol.expression} (MSE: {sol.mse:.6f})")
