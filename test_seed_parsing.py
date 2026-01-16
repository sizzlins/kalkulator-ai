"""Test if atan seeds parse correctly into ExpressionTree."""
import numpy as np
import sympy as sp
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree

# Sample seed from bipolar detector
test_seeds = [
    "atan(y/(x+2))",
    "atan((x-2)/y)",
    "cos(16*(atan(y/(x+2))+atan((x-2)/y)))",
    "cos(8*atan(y/(x+2)))",
]

# Variable symbols
var_names = ["x", "y"]
symbols = {v: sp.Symbol(v) for v in var_names}

print("Testing seed parsing:")
for seed in test_seeds:
    try:
        # Parse seed string to SymPy expression
        local_dict = {"atan": sp.atan, "cos": sp.cos, "sin": sp.sin}
        local_dict.update(symbols)
        expr = sp.sympify(seed, locals=local_dict)
        print(f"  ✓ SymPy parsed: {seed} -> {expr}")
        
        # Convert to ExpressionTree
        tree = ExpressionTree.from_sympy(expr, var_names)
        print(f"    ✓ ExpressionTree: {tree.to_string()}")
        
        # Evaluate at test point
        X_test = np.array([[0, 2]])  # f(0, 2) should = 1
        result = tree.evaluate(X_test)
        print(f"    ✓ Evaluated at (0, 2): {result[0]}")
    except Exception as e:
        print(f"  ✗ FAILED: {seed}")
        print(f"    Error: {type(e).__name__}: {e}")
