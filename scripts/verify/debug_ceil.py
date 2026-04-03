
import numpy as np
import sympy as sp
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
from kalkulator_pkg.symbolic_regression.population import Population
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig
from kalkulator_pkg.parser import safe_sympy_parse
from kalkulator_pkg.sympy_defs import ALLOWED_SYMPY_NAMES

def test_ceil_behavior():
    print("Testing ceil behavior...")
    
    # 1. Check ALLOWED_SYMPY_NAMES
    if "ceil" in ALLOWED_SYMPY_NAMES:
        print("PASS: 'ceil' is in ALLOWED_SYMPY_NAMES")
    else:
        print("FAIL: 'ceil' is NOT in ALLOWED_SYMPY_NAMES")
        
    # 2. Check Parsing
    expr_str = "ceil(x/2)"
    local_dict = {'x': sp.Symbol('x')}
    full_dict = {**ALLOWED_SYMPY_NAMES, **local_dict}
    
    try:
        expr = safe_sympy_parse(expr_str, local_dict=full_dict)
        print(f"PASS: parsed '{expr_str}' -> {expr} ({type(expr)})")
    except Exception as e:
        print(f"FAIL: parsing '{expr_str}' failed: {e}")
        return

    # 3. Tree Conversion
    try:
        tree = ExpressionTree.from_sympy(expr, ['x'])
        print(f"PASS: Converted to tree: {tree}")
    except Exception as e:
        print(f"FAIL: Tree conversion failed: {e}")
        return

    # 4. Evaluation
    X = np.array([1, 2, 3, 4], dtype=float)
    y_expected = np.array([1, 1, 2, 2], dtype=float) # ceil(0.5)=1, ceil(1)=1, ceil(1.5)=2, ceil(2)=2
    
    try:
        # ExpressionTree.evaluate expects X as (n_samples, n_features) array
        # We have 1 variable 'x', so we need shape (4, 1)
        X_matrix = X.reshape(-1, 1)
        y_pred = tree.evaluate(X_matrix)
        print(f"Pred: {y_pred}")
        print(f"Expect: {y_expected}")
        
        mse = np.mean((y_pred - y_expected)**2)
        print(f"MSE: {mse}")
        
        if mse < 1e-9:
            print("PASS: Evaluation match")
        else:
            print("FAIL: Evaluation mismatch")
            
    except Exception as e:
        print(f"FAIL: Evaluation crashed: {e}")

    # 5. Check 'ceiling' vs 'ceil' mismatch handling
    print("\nChecking 'ceiling' alias...")
    expr_ceil = sp.ceiling(sp.Symbol('x')/2)
    # expression_tree.from_sympy maps sp.ceiling -> "ceiling" operator name?
    # Let's check internal node value
    print(f"Tree Root Value: '{tree.root.value}'")
    # expression_tree.UNARY_OPERATORS must have this key
    from kalkulator_pkg.symbolic_regression.expression_tree import UNARY_OPERATORS, BINARY_OPERATORS
    if tree.root.value in UNARY_OPERATORS:
        print(f"PASS: '{tree.root.value}' is in UNARY_OPERATORS")
    else:
        print(f"FAIL: '{tree.root.value}' is NOT in UNARY_OPERATORS")

if __name__ == "__main__":
    test_ceil_behavior()
