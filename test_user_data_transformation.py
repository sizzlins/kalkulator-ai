"""Test transformation search with user's actual (1+x)^(1/x) data."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

import numpy as np
from kalkulator_pkg.symbolic_regression import GeneticConfig, GeneticSymbolicRegressor

# User's exact data for f(x) = (1+x)^(1/x)
X = np.array([
    [4.5], [4.4], [4.3], [4.2], [4.1], [4.0], [3.9], [3.8], [3.7], [3.6],
    [3.5], [3.4], [3.3], [3.2], [3.1], [3.0], [2.9], [2.8], [2.7], [2.6],
    [2.5], [2.4], [2.3], [2.2], [2.1], [2.0], [1.9], [1.8], [1.7], [1.6],
    [1.5], [1.4], [1.3], [1.2], [1.1], [1.0]
])

y = np.array([
    1.46057896631733, 1.46707773881062, 1.47379218317397, 1.48073378857426,
    1.48791489625403, 1.49534878122122, 1.50304974361412, 1.51103321111493,
    1.51931585401351, 1.52791571479526, 1.53685235445298, 1.54614701811358,
    1.55582282304186, 1.56590497265434, 1.57642100086962, 1.5874010519682,
    1.59887820217331, 1.61088883044436, 1.62347304755987, 1.63667519454208,
    1.65054442394899, 1.66513538067647, 1.68050900286556, 1.69673346855418,
    1.71388532019653, 1.73205080756888, 1.75132750053796, 1.77182623758817,
    1.79367349515146, 1.81701428845016, 1.84201574932019, 1.86887157406314,
    1.89780760028134, 1.92908886409376, 1.96302862058022, 2.0
])

print("="*70)
print("Testing Multi-Space Evolution on User's Data")
print("="*70)
print(f"Target: (1+x)^(1/x)")
print(f"Data points: {len(X)}")
print()

# Match user's config: boost mode 3x
config = GeneticConfig(
    population_size=300,  # 3x boost
    generations=90,       # 3x boost
    n_islands=2,
    verbose=True,
    parsimony_coefficient=0.001,
)

print("Running fit_with_transformations()...")
print("This will try DIRECT, LOG, and INVERSE spaces")
print()

regressor = GeneticSymbolicRegressor(config)
best_expr, best_mse, best_space = regressor.fit_with_transformations(X, y, ['x'])

print()
print("="*70)
print("FINAL RESULT:")
print(f"  Best expression: {best_expr}")
print(f"  MSE: {best_mse:.8f}")
print(f"  Found in: {best_space} space")
print("="*70)
print()

# Compare to user's REPL result
repl_mse = 0.00689045
if best_mse < repl_mse:
    improvement = (repl_mse / best_mse)
    print(f"✅ {improvement:.1f}x BETTER than direct-space REPL result!")
else:
    print(f"⚠️  Similar to REPL result (MSE {repl_mse:.6f})")
