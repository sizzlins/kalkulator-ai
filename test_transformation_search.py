"""Test multi-space evolution on (1+x)^(1/x)."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

import numpy as np
from kalkulator_pkg.symbolic_regression import GeneticConfig, GeneticSymbolicRegressor

# Data for f(x) = (1+x)^(1/x)
X = np.array([[1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5]])
y = np.array([2.0, 1.842, 1.732, 1.651, 1.587, 1.537, 1.495, 1.461])

print("="*70)
print("Testing Variable Transformation Search")
print("="*70)
print("Target: (1+x)^(1/x)")
print(f"Data: {len(X)} points")
print()

# Configure for quick test
config = GeneticConfig(
    population_size=100,
    generations=30,
    verbose=True,
    parsimony_coefficient=0.001,
)

regressor = GeneticSymbolicRegressor(config)

# Run multi-space evolution
print("Starting multi-space evolution...")
print()

best_expr, best_mse, best_space = regressor.fit_with_transformations(X, y, ['x'])

print()
print("="*70)
print("FINAL RESULT:")
print(f"  Expression: {best_expr}")
print(f"  MSE: {best_mse:.6e}")
print(f"  Discovered in: {best_space} space")
print("="*70)

# Check if it found log structure
if 'log' in best_expr.lower() or 'exp' in best_expr.lower():
    print("\n✅ SUCCESS: Found log/exp structure!")
    print("   Algorithm learned to think in transformed spaces!")
