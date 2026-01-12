"""Verify Bessel J0 discovery capability."""
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())

from kalkulator_pkg.symbolic_regression import GeneticSymbolicRegressor, GeneticConfig

# User's Bessel J0 data
data = [
    (0.0, 1.0),
    (0.5, 0.93847),
    (1.0, 0.7652),
    (1.5, 0.51183),
    (2.0, 0.22389),
    (2.5, -0.04838),
    (3.0, -0.26005),
    (3.5, -0.38013),
    (4.0, -0.39715),
    (4.5, -0.32054),
    (5.0, -0.1776),
    (5.5, -0.00684),
    (6.0, 0.15065),
    (6.5, 0.26009),
    (7.0, 0.30008),
]

X = np.array([d[0] for d in data]).reshape(-1, 1)
y = np.array([d[1] for d in data])

print("Testing Bessel J0 discovery...")
print(f"Data points: {len(data)}")

# Configure with bessel operators enabled
config = GeneticConfig(
    population_size=200,
    generations=50,
    timeout=30,
    verbose=True,
    seeds=["bessel_j0(x)"],  # Seed the correct answer to verify operator works
)

regressor = GeneticSymbolicRegressor(config)
pareto = regressor.fit(X, y, variable_names=["x"])

best = pareto.get_best()
knee = pareto.get_knee_point()

print(f"\nBest MSE: {best.expression if best else 'None'} (MSE: {best.mse if best else 'N/A'})")
print(f"Knee Point: {knee.expression if knee else 'None'} (MSE: {knee.mse if knee else 'N/A'})")

# Check if J0 was found
result = best.expression if best else ""
if "bessel" in result.lower() or "j0" in result.lower():
    print("\nSUCCESS: Bessel J0 found!")
else:
    print(f"\nDid not find bessel_j0. Found: {result}")
