
import numpy as np
import sympy as sp
import traceback
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, ExpressionNode, NodeType
from kalkulator_pkg.parser import safe_sympy_parse

def test_gamma_debug():
    print("=== DEBUG GAMMA START ===")
    
    # Test 1: Manual Tree construction for gamma(4)
    try:
        print("\n[Test 1] ExpressionTree('gamma', 4.0)")
        node = ExpressionNode(NodeType.UNARY_OP, "gamma", [ExpressionNode(NodeType.CONSTANT, 4.0)])
        tree = ExpressionTree(node, ["x"])
        
        # Evaluate
        X = np.array([[1.0]])
        val = tree.evaluate(X)[0]
        print(f"Evaluated: {val}")
        assert abs(val - 6.0) < 1e-9, "Evaluation failed"
        
        # To SymPy
        sym = tree.to_sympy()
        print(f"To SymPy: {sym}")
        
    except Exception:
        traceback.print_exc()

    # Test 2: safe_sympy_parse("factorial(x)")
    try:
        print("\n[Test 2] safe_sympy_parse('factorial(x)')")
        expr = safe_sympy_parse("factorial(x)")
        print(f"Parsed: {expr}")
        
        if expr is None:
             print("Parsed as None")
    except Exception:
         traceback.print_exc()
         
    # Test 3: safe_sympy_parse("gamma(x+1)")
    try:
        print("\n[Test 3] safe_sympy_parse('gamma(x+1)')")
        expr = safe_sympy_parse("gamma(x+1)")
        print(f"Parsed: {expr}")
    except Exception:
         traceback.print_exc()

    print("=== DEBUG GAMMA END ===")

if __name__ == "__main__":
    test_gamma_debug()
