
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionNode, NodeType

def test_manual():
    print("Creating nodes...")
    x = ExpressionNode(NodeType.VARIABLE, "x")
    y = ExpressionNode(NodeType.VARIABLE, "y")
    add = ExpressionNode(NodeType.BINARY_OP, "add", children=[x, y])
    
    print(f"Node: {add}")
    print(f"Children len: {len(add.children)}")
    print(f"Children types: {[type(c) for c in add.children]}")
    
    count = add.count_nodes()
    print(f"Count: {count}")
    
    if count != 3:
        print("FAIL: Count should be 3")
        sys.exit(1)
    else:
        print("PASS: Count is 3")

if __name__ == "__main__":
    test_manual()
