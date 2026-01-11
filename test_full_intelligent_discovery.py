"""Full test: Constant anchors + Multi-space + Inverse mutations on (1+x)^(1/x)."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

import numpy as np
from kalkulator_pkg.symbolic_regression import GeneticConfig, GeneticSymbolicRegressor

# User's exact data for f(x) = (1+x)^(1/x)
X = np.array([[1.0], [2.0], [3.0], [4.0], [4.5]])
y = np.array([2.0, 1.73205080756888, 1.5874010519682, 1.49534878122122, 1.46057896631733])

print("="*70)
print("FULL TEST: All 3 Phases Combined")
print("="*70)
print("Target: (1+x)^(1/x)")
print(f"Data: {len(X)} points")
print()
print("Phases active:")
print("  ✅ Phase 1: Inverse-aware mutations")
print("  ✅ Phase 2: Multi-space transformation")
print("  ✅ Phase 3: Constant anchor detection")
print()

# Configure for moderately-sized test
config = GeneticConfig(
    population_size=200,
    generations=50,
    verbose=True,
    parsimony_coefficient=0.001,
)

print("Running fit_with_transformations() with anchor detection...")
print()

regressor = GeneticSymbolicRegressor(config)
best_expr, best_mse, best_space = regressor.fit_with_transformations(X, y, ['x'])

print()
print("="*70)
print("FINAL RESULT:")
print(f"  Expression: {best_expr}")
print(f"  MSE: {best_mse:.8f}")
print(f"  Found in: {best_space} space")
print("="*70)
print()

# Check if it's the exact function
if '(x+1)**(1/x)' in best_expr or '(1+x)**(1/x)' in best_expr:
    print("🎉 PERFECT! Discovered exact target function!")
elif 'x+1' in best_expr and '**(1/x)' in best_expr:
    print("✅ VERY CLOSE! Has key structure (x+1)^(1/x)")
elif best_mse < 1e-6:
    print("✅ EXCELLENT! Near-perfect numerical fit")
elif best_mse < 1e-3:
    print("✅ GOOD! Strong approximation")
else:
    print(f"⚠️  Approximation with MSE {best_mse:.6f}")
