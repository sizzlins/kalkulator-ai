"""Final test: Did we discover (1+x)^(1/x)?"""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

import numpy as np
from kalkulator_pkg.symbolic_regression import GeneticConfig, GeneticSymbolicRegressor

X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 1.73205080756888, 1.5874010519682, 1.49534878122122])

print("FINAL TEST: Did constant anchors help discover (1+x)^(1/x)?")
print()

config = GeneticConfig(
    population_size=100,
    generations=30,
    verbose=False,  # Quiet mode to see result clearly
    parsimony_coefficient=0.001,
)

regressor = GeneticSymbolicRegressor(config)
pareto = regressor.fit(X, y, ['x'])

best = pareto.get_best()
if best:
    print("="*70)
    print("DISCOVERED EXPRESSION:")
    print(best.expression)
    print(f"MSE: {best.mse:.10f}")
    print("="*70)
    print()
    
    # Check if exact match
    expr_clean = best.expression.replace(" ", "").lower()
    
    if '(x+1)**(1/x)' in expr_clean or '(1+x)**(1/x)' in expr_clean:
        print("SUCCESS! Found EXACT target function: (1+x)^(1/x)")
    elif '(x+1)' in expr_clean and '1/x' in expr_clean and '**' in expr_clean:
        print("VERY CLOSE! Has structure with (x+1) and exponent 1/x")
        print("Likely equivalent to target!")
    else:
        print("Different expression found (approximation)")
        print(f"Target was: (1+x)^(1/x)")
