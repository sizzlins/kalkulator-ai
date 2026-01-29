
import numpy as np
import sys
import os

# Adjust path to find the package
sys.path.append(os.getcwd())

from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionNode, ExpressionTree, NodeType

def test_locked_evaluation():
    # 1. Manually construct sin(1/(x - locked(3.0)))
    
    # locked(3.0)
    # Note: ExpressionNode constructor: (node_type, value, children, parent, locked)
    locked_3 = ExpressionNode(NodeType.CONSTANT, 3.0, locked=True)
    
    # x
    var_x = ExpressionNode(NodeType.VARIABLE, "x")
    
    # x - 3.0
    # Note: Binary op children order [left, right]
    sub_node = ExpressionNode(NodeType.BINARY_OP, "sub", [var_x, locked_3])
    var_x.parent = sub_node
    locked_3.parent = sub_node
    
    # 1 (for 1/...)
    const_1 = ExpressionNode(NodeType.CONSTANT, 1.0)
    
    # 1 / (x-3)
    div_node = ExpressionNode(NodeType.BINARY_OP, "div", [const_1, sub_node])
    const_1.parent = div_node
    sub_node.parent = div_node
    
    # sin(...)
    sin_node = ExpressionNode(NodeType.UNARY_OP, "sin", [div_node])
    div_node.parent = sin_node
    
    tree = ExpressionTree(sin_node, ["x"])
    
    # 2. Test values
    # x = 4.0 -> 1/(4-3) = 1 -> sin(1) ≈ 0.84147
    x_val = np.array([4.0])
    y_pred = tree.evaluate(x_val)
    print(f"Test x=4.0: Pred={y_pred}, Expect={np.sin(1.0)}")
    
    # x = 3.1 -> 1/(3.1-3) = 10 -> sin(10) ≈ -0.544
    x_val2 = np.array([3.1])
    y_pred2 = tree.evaluate(x_val2)
    print(f"Test x=3.1: Pred={y_pred2}, Expect={np.sin(10.0)}")
    
    # 3. Test from_sympy parsing
    import sympy as sp
    # Create locked function
    # Note: The code patches in sympy_defs.py should be active if we import it, 
    # but we will just simulate the sympify string if needed.
    # Actually, let's try to verify if from_sympy creates the same structure.
    
    sp_x = sp.Symbol('x')
    sp_locked = sp.Function('locked')
    expr = sp.sin(1/(sp_x - sp_locked(3.0)))
    
    try:
        tree_parsed = ExpressionTree.from_sympy(expr, ["x"])
        print("\nParsed Tree Structure:")
        print(tree_parsed)
        
        # Check if 3.0 is locked
        # Traverse to find 3.0
        nodes = tree_parsed.get_all_nodes()
        locked_nodes = [n for n in nodes if getattr(n, 'locked', False)]
        print(f"Found {len(locked_nodes)} locked nodes.")
        for n in locked_nodes:
            print(f"Locked node value: {n.value}")
            
        y_parsed = tree_parsed.evaluate(x_val)
        print(f"Parsed x=4.0: Pred={y_parsed}")
        
    except Exception as e:
        print(f"Parsing failed: {e}")

if __name__ == "__main__":
    test_locked_evaluation()
