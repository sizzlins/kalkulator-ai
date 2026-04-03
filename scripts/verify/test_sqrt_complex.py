
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, ExpressionNode, NodeType, UNARY_OPERATORS, safe_sqrt, psqrt, safe_pow

def test_complex_sqrt():
    print("--- Verifying safe_sqrt ---")
    x = np.array([-0.12])
    res = safe_sqrt(x)
    print(f"safe_sqrt(-0.12) = {res} (dtype: {res.dtype})")
    
    print("\n--- Verifying Operator Mapping ---")
    op = UNARY_OPERATORS.get("sqrt")
    print(f"UNARY_OPERATORS['sqrt'] is safe_sqrt? {op is safe_sqrt}")
    print(f"UNARY_OPERATORS['sqrt'] is psqrt? {op is psqrt}")
    print(f"UNARY_OPERATORS['sqrt'] name: {op.__name__}")

    print("\n--- Verifying ExpressionTree Evaluation ---")
    # Tree 1: sqrt(x)
    root1 = ExpressionNode(NodeType.UNARY_OP, "sqrt", [
        ExpressionNode(NodeType.VARIABLE, "x")
    ])
    tree1 = ExpressionTree(root1, variables=["x"])
    
    X = np.array([-0.12]).reshape(-1, 1)
    # Force float/complex input
    X_complex = X.astype(complex)
    
    print(f"Evaluating sqrt(x) on {X_complex}...")
    res1 = tree1.evaluate(X_complex)
    print(f"Result: {res1} (dtype: {res1.dtype})")
    
    # Tree 2: sqrt(x) + 7
    root2 = ExpressionNode(NodeType.BINARY_OP, "add", [
        ExpressionNode(NodeType.UNARY_OP, "sqrt", [
            ExpressionNode(NodeType.VARIABLE, "x")
        ]),
        ExpressionNode(NodeType.CONSTANT, 7.0)
    ])
    tree2 = ExpressionTree(root2, variables=["x"])
    
    print(f"Evaluating sqrt(x) + 7 on {X_complex}...")
    res2 = tree2.evaluate(X_complex)
    print(f"Result: {res2} (dtype: {res2.dtype})")

    print("\n--- Verifying pow(x, 0.5) ---")
    # Tree 3: pow(x, 0.5) + 7
    root3 = ExpressionNode(NodeType.BINARY_OP, "add", [
        ExpressionNode(NodeType.BINARY_OP, "pow", [
            ExpressionNode(NodeType.VARIABLE, "x"),
            ExpressionNode(NodeType.CONSTANT, 0.5)
        ]),
        ExpressionNode(NodeType.CONSTANT, 7.0)
    ])
    tree3 = ExpressionTree(root3, variables=["x"])
    
    print(f"Evaluating pow(x, 0.5) + 7 on {X_complex}...")
    res3 = tree3.evaluate(X_complex)
    print(f"Result: {res3} (dtype: {res3.dtype})")
    
    # Check naive pow
    print(f"safe_pow(-0.12, 0.5) = {safe_pow(-0.12, 0.5)}")

    print("\n--- Verifying nan_to_num ---")
    c_arr = np.array([1+1j, np.nan, np.inf])
    print(f"Original: {c_arr}")
    res_nan = np.nan_to_num(c_arr, nan=0.0, posinf=1e10, neginf=-1e10)
    print(f"nan_to_num(nan=0.0): {res_nan} (dtype: {res_nan.dtype})")
    
    res_nan2 = np.nan_to_num(c_arr, nan=1e9)
    print(f"nan_to_num(nan=1e9): {res_nan2} (dtype: {res_nan2.dtype})")

    print("\n--- Verifying Warning Trigger (psqrt) ---")
    # Tree 4: psqrt(x) (Expect Warning)
    root4 = ExpressionNode(NodeType.UNARY_OP, "psqrt", [
         ExpressionNode(NodeType.VARIABLE, "x")
    ])
    tree4 = ExpressionTree(root4, variables=["x"])
    print(f"Evaluating psqrt(x) on {X_complex}...")
    res4 = tree4.evaluate(X_complex)
    print(f"Result: {res4} (dtype: {res4.dtype})")

if __name__ == "__main__":
    test_complex_sqrt()
