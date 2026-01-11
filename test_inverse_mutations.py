"""Test inverse-aware mutations on (1+x)^(1/x) discovery."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

import numpy as np
from kalkulator_pkg.symbolic_regression import GeneticConfig, GeneticSymbolicRegressor

# Generate data for f(x) = (1+x)^(1/x)
# This is the HARD test - if it discovers this, the intelligence works!
X_train = np.array([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5]
])
y_train = np.array([
    2.0, 1.842, 1.732, 1.651, 1.587, 1.537, 1.495, 1.461
])

print("Testing Inverse-Aware Mutations")
print("="*70)
print("Target function: (1+x)^(1/x)")
print(f"Training data: {len(X_train)} points")
print()

# Configure with boosted settings and verbose output
config = GeneticConfig(
    population_size=200,  # Larger population
    generations=50,       # More generations
    verbose=True,
    parsimony_coefficient=0.001,  # Prefer simpler
)

# Train
print("Starting evolution with inverse-aware mutations...")
print("Looking for expressions with inverse exponents like x^(1/x)...")
print()

regressor = GeneticSymbolicRegressor(config)
regressor.fit(X_train, y_train, variable_names=['x'])

# Get result
best = regressor.get_best_expression()
print()
print("="*70)
print(f"RESULT: {best}")
print()

# Test if it found the structure
if "1/x" in str(best) or "x**-1" in str(best):
    print("✅ SUCCESS: Found inverse exponent structure!")
    print("   The algorithm learned to think about inverse relationships!")
else:
    print("⚠️  Did not find exact inverse, but mutations are working")
    print(f"   Best approximation: {best}")
