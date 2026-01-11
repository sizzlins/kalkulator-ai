
import numpy as np
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
import sympy as sp
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

# Data from user request
# f(4.5) = 4.25
# f(4.0) = 4.0
X = np.array([4.5, 4.4, 4.0, 3.0, 2.0, 1.0, 0.0, -0.2, -1.2, -2.5]).reshape(-1, 1)
y = np.array([4.25, 4.16, 4.0, 3.0, 2.0, 1.0, 0.0, -0.36, -1.36, -2.75])

print("Target Y:", y)

# Construct seed tree manually to emulate parsing
seed_str = "floor(x) + frac(x)**2"
print(f"\nEvaluating seed: {seed_str}")

try:
    # 1. Parse with SymPy (as done in genetic_engine.py)
    local_dict = {'x': sp.Symbol('x')}
    # Add custom functions to local dict as config.py does not expose them directly to sympify without context?
    # Actually genetic_engine lines 483-484: local_dict = {v: sp.Symbol(v) for v in variables}
    # It relies on sp.sympify finding 'floor' and 'frac' in global or standard sympy namespace?
    # sp.floor exists. sp.frac exists.
    
    expr = sp.sympify(seed_str, locals=local_dict)
    print(f"SymPy parsed: {expr}")
    
    # 2. Convert to ExpressionTree
    tree = ExpressionTree.from_sympy(expr, ['x'])
    print(f"Tree string: {tree.to_string()}")
    
    # 3. Evaluate
    y_pred = tree.evaluate(X)
    print(f"Predictions: {y_pred}")
    
    # 4. MSE
    mse = np.mean((y - y_pred)**2)
    print(f"MSE: {mse}")
    
    if mse < 1e-6:
        print("SEED WORKS PERFECTLY")
    else:
        print("SEED FAILS")
        print(f"Diff: {y - y_pred}")

except Exception as e:
    import traceback
    traceback.print_exc()

# 5. Check if 'frac' is available in sp.sympify default namespace
try:
    s = sp.sympify("frac(1.5)")
    print(f"\nsympify('frac(1.5)') -> {s} (type: {type(s)})")
except Exception as e:
    print(f"\nsympify('frac') failed: {e}")

