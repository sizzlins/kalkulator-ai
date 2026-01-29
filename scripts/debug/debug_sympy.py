
import sys
import os
import numpy as np

# Mocking path
sys.path.append(os.getcwd())

from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, ExpressionNode, NodeType

def test_sympy_conversion():
    print("Testing SymPy conversion...")
    
    # 1. Simple CONST
    tree = ExpressionTree(ExpressionNode(NodeType.CONSTANT, 3.14))
    print(f"CONST: {tree.to_sympy()}")
    
    # 2. Variable
    tree = ExpressionTree(ExpressionNode(NodeType.VARIABLE, "x"))
    print(f"VAR: {tree.to_sympy()}")
    
    # 3. Unary Sin
    node = ExpressionNode(NodeType.UNARY_OP, "sin", [
        ExpressionNode(NodeType.VARIABLE, "x")
    ])
    tree = ExpressionTree(node)
    print(f"SIN: {tree.to_sympy()}")
    
    # 4. Binary Add
    node = ExpressionNode(NodeType.BINARY_OP, "add", [
        ExpressionNode(NodeType.VARIABLE, "x"),
        ExpressionNode(NodeType.CONSTANT, 1.0)
    ])
    tree = ExpressionTree(node)
    print(f"ADD: {tree.to_sympy()}")
    
    # 5. Protected Log (plog)
    node = ExpressionNode(NodeType.UNARY_OP, "plog", [
        ExpressionNode(NodeType.VARIABLE, "x")
    ])
    tree = ExpressionTree(node)
    try:
        print(f"PLOG: {tree.to_sympy()}")
    except Exception as e:
        print(f"PLOG Failed: {e}")

if __name__ == "__main__":
    try:
        test_sympy_conversion()
        print("Success!")
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
