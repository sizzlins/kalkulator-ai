
import numpy as np
import ast
import sympy as sp
from kalkulator_pkg.parser import SafeSymPyVisitor
from kalkulator_pkg.symbolic_regression.forensic_analysis import _detect_general_staircase

def test_parser_floordiv():
    print("Testing Parser // Support...")
    visitor = SafeSymPyVisitor()
    expr_str = "(x + 1) // 2"
    tree = ast.parse(expr_str) # Default mode='exec' wraps in Module
    sympy_expr = visitor.visit(tree)
    print(f"Parsed '{expr_str}' -> {sympy_expr}")
    
    # Verify evaluation
    x = sp.Symbol('x')
    f = sp.lambdify(x, sympy_expr)
    assert f(1) == 1
    assert f(2) == 1
    assert f(3) == 2
    print("Parser Test Passed!")

def test_staircase_detection():
    print("\nTesting Staircase Detection...")
    # User Data
    # 1, 2 -> 1
    # 3, 4 -> 2
    # 5, 6 -> 3
    # 15, 16 -> 8
    # 17, 18 -> 9
    X = np.array([1, 2, 3, 4, 5, 6, 15, 16, 17, 18])
    # f(x) = (x+1)//2
    # 1->1, 2->1
    # 3->2, 4->2
    # 5->3, 6->3
    # 15->8, 16->8
    # 17->9, 18->9
    y = np.array([1, 1, 2, 2, 3, 3, 8, 8, 9, 9])
    
    seeds = _detect_general_staircase(X, y, variable_names=['x'], verbose=True)
    print(f"Detected Seeds: {seeds}")
    
    expected_seed = "floor(0.5*x + 0.5)" # equivalent to (x+1)//2
    # or ceil(0.5*x) ?
    # 0.5*1 = 0.5 -> ceil -> 1. Correct.
    # 0.5*2 = 1.0 -> ceil -> 1. Correct.
    # 0.5*3 = 1.5 -> ceil -> 2. Correct.
    
    found = any("floor" in s or "ceil" in s for s in seeds)
    if found:
        print("Staircase Detection Test Passed!")
    else:
        print("Staircase Detection Test Failed: No floor/ceil seeds found.")

if __name__ == "__main__":
    try:
        test_parser_floordiv()
        test_staircase_detection()
    except Exception as e:
        print(f"FAILED: {e}")
