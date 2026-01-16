"""Test atan2-based seed evaluation on y=0 cases."""
import numpy as np
import sympy as sp
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree

# Original atan-based seed (fails at y=0)
seed_atan = "cos(16*(atan(y/(x+2))+atan((x-2)/y)))"

# New atan2-based seed (handles y=0)
seed_atan2 = "cos(16*(atan2(y,x+2)+atan2(x-2,y)))"

# Test points including y=0 cases
test_points = [
    ((0, 2), 1.0),
    ((2, 0), 1.0),  # y=0 case!
    ((-5, -5), 0.961372022292441),
]

print("Comparing atan vs atan2 seeds:")
print("=" * 60)

for seed_name, seed_str in [("atan", seed_atan), ("atan2", seed_atan2)]:
    print(f"\n{seed_name}: {seed_str}")
    
    local_dict = {
        'x': sp.Symbol('x'), 'y': sp.Symbol('y'),
        'atan': sp.atan, 'atan2': sp.atan2, 'cos': sp.cos,
    }
    try:
        expr = sp.sympify(seed_str, locals=local_dict)
        tree = ExpressionTree.from_sympy(expr, ['x', 'y'])
        print(f"  Parsed: {tree.to_string()[:60]}...")
        
        for (x, y), expected in test_points:
            X = np.array([[x, y]], dtype=float)
            result = tree.evaluate(X)[0]
            status = "✓" if np.isfinite(result) else "✗ NaN"
            print(f"  ({x:2}, {y:2}): {result:8.4f} (expected {expected:.4f}) {status}")
    except Exception as e:
        print(f"  ERROR: {e}")
