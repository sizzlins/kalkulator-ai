"""Simple test: Direct space only with anchor detection."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

import numpy as np
from kalkulator_pkg.symbolic_regression import GeneticConfig, GeneticSymbolicRegressor

# User's exact data for f(x) = (1+x)^(1/x)
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 1.73205080756888, 1.5874010519682, 1.49534878122122])

print("="*70)
print("Testing Constant Anchor Detection in Direct Space")
print("="*70)
print("Target: (1+x)^(1/x)")
print(f"Data: {len(X)} points")
print()

config = GeneticConfig(
    population_size=200,
    generations=50,
    verbose=True,
    parsimony_coefficient=0.001,
)

regressor = GeneticSymbolicRegressor(config)
pareto = regressor.fit(X, y, ['x'])

best = pareto.get_best()
if best:
    print()
    print("="*70)
    print("RESULT:")
    print(f"  Expression: {best.expression}")
    print(f"  MSE: {best.mse:.8f}")
    print("="*70)
    
    # Check if exact
    expr_str = best.expression.replace(" ", "")
    if '(x+1)**(1/x)' in expr_str or '(1+x)**(1/x)' in expr_str:
        print("\nPERFECT - Found exact function!")
    elif '(x+1)' in expr_str and '**(1/x)' in expr_str:
        print("\nVERY CLOSE - Has (x+1)^(1/x) structure!")
    elif best.mse < 1e-6:
        print("\nEXCELLENT - Near perfect fit!")
