
import numpy as np
import sys
import os

# Add project root to path
sys.path.append("C:\\Users\\LOQ\\PycharmProjects\\kalkulator-ai")

from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, ExpressionNode, NodeType

def reproduce():
    print("--- Reproducing Sqrt(x) + 7 Crash ---")

    # Manually construct x^0.5 + 7
    # Tree: (+) -> [pow(x, 0.5), 7]
    x_node = ExpressionNode(NodeType.VARIABLE, "x")
    point_five = ExpressionNode(NodeType.CONSTANT, 0.5)
    pow_node = ExpressionNode(NodeType.BINARY_OP, "pow", children=[x_node, point_five])
    const_node = ExpressionNode(NodeType.CONSTANT, 7.0)
    root = ExpressionNode(NodeType.BINARY_OP, "add", children=[pow_node, const_node])

    tree = ExpressionTree(root, variables=["x"])

    # Create data with negative values (causing complex sqrt)
    # Mix of positive and negative
    X = np.array([[-4.0], [-1.0], [4.0], [9.0]])
    print(f"Testing X: {X.flatten()}")

    try:
        preds = tree.evaluate_fast(X)
        print("\nPredictions Raw:", preds)
        
        # Check if they are 0 (bad) or complex (good)
        count_zeros = np.sum(preds == 0)
        print(f"Count of Zeros: {count_zeros}/{len(preds)}")
        
        if np.all(preds == 0):
            print("FAIL: Predictions are all 0.0 (Crash detected)")
        elif np.any(preds == 0) and not np.all(X == 0):
             # Some might be valid 0, but here sqrt(x)+7 shouldn't be 0
             print("WARNING: Some zeros detected.")

        if np.iscomplexobj(preds):
            print("PASS: Result is complex object.")
            print("Values:", preds)
        else:
            print("FAIL: Result is REAL (imaginary part lost/crashed).")

    except Exception as e:
        print(f"CRASHED EXTERNAL: {e}")

if __name__ == "__main__":
    reproduce()
