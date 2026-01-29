
import sympy as sp
import numpy as np
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, ExpressionNode, NodeType, _get_sympy_ops
from kalkulator_pkg.sympy_defs import lshift

def test_residual_errors():
    print("Testing Residual Errors...")
    
    # 1. Complex to Int Conversion (SymPy Integer Error)
    # "Argument of Integer should be of numeric type, got 89547297578722.98 + 49577540924655.27*I"
    huge_complex = sp.Float(89547297578722.98) + sp.I * sp.Float(49577540924655.27)
    print(f"\n[Test 1] huge_complex: {huge_complex}")
    
    # Test direct lshift behavior
    try:
        res = lshift.eval(huge_complex, 2)
        print(f"lshift.eval(huge_complex, 2) -> {res}")
        assert res == 0 or res == sp.Integer(0)
    except Exception as e:
        print(f"FAIL: lshift crashed: {e}")

    # Test via ExpressionTree to_sympy
    # Tree: lshift(CONST(huge_complex), CONST(2))
    # We need to ensure ExpressionNode handles complex constant correctly?
    # ExpressionNode(CONSTANT, huge_complex) -> to_sympy
    
    # Mocking a tree node
    # But wait, ExpressionNode.to_sympy calls binary_ops['lshift'](left, right)
    
    unary, binary = _get_sympy_ops()
    op_lshift = binary['lshift']
    
    try:
        res = op_lshift(huge_complex, sp.Integer(2))
        print(f"op_lshift(huge_complex, 2) -> {res}")
    except Exception as e:
        print(f"FAIL: op_lshift crashed: {e}")

    # 2. Gamma Pole
    print("\n[Test 2] Gamma Pole via expression_tree ops")
    op_gamma = unary['gamma']
    try:
        res = op_gamma(sp.Integer(-5))
        print(f"op_gamma(-5) -> {res}")
        assert res == sp.zoo
    except Exception as e:
        print(f"FAIL: op_gamma crashed: {e}")

if __name__ == "__main__":
    test_residual_errors()
