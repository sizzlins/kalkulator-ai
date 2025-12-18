
import numpy as np
import sympy as sp
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, NodeType, ExpressionNode

def test_eval():
    print("Testing Evaluation Logic...")
    
    # Construct tree: e * x
    # This matches the structure of 1084483*x/398959 approximately
    # e approx 2.718281828
    
    # Constant Node (e)
    node_c = ExpressionNode(NodeType.CONSTANT, 2.718281828)
    # Variable Node (x)
    node_x = ExpressionNode(NodeType.VARIABLE, 'x')
    # Mul Node
    root = ExpressionNode(NodeType.BINARY_OP, 'mul', children=[node_c, node_x])
    node_c.parent = root
    node_x.parent = root
    
    tree = ExpressionTree(root)
    
    # Input Data
    X = np.array([[1], [2], [3]], dtype=float) # Shape (3, 1)
    y_target = np.array([2, 4, 6], dtype=float) # Target is 2x
    
    # Evaluate
    # ExpressionTree.evaluate takes X directly
    y_pred = tree.evaluate(X)
    
    print(f"Input x: {X}")
    print(f"Target 2x: {y_target}")
    print(f"Prediction e*x: {y_pred}")
    
    # Calculate MSE
    diff = y_pred - y_target
    mse = np.mean(diff**2)
    print(f"MSE: {mse}")
    
    if mse < 1e-9:
        print("CRITICAL: e*x fits 2*x with MSE 0. This is impossible unless math is broken.")
    else:
        print("PASS: e*x does NOT fit 2*x (MSE > 0).")

    # Test Rationalization
    print("\nTesting Rationalization...")
    try:
        sympy_expr = tree.to_sympy({'x': sp.Symbol('x')})
        print(f"SymPy Expression: {sympy_expr}")
    except Exception as e:
        print(f"SymPy Error: {e}")

if __name__ == "__main__":
    test_eval()
