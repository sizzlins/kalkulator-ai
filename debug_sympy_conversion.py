from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, ExpressionNode, NodeType
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticConfig
import sympy as sp

# Create SymPy expression using custom or generic function
# Note: In config.py bitwise_xor is defined, but here we simulate what happens 
# if we just use a SymPy Function with that name (which is what usually happens in roundtrips)
f_xor = sp.Function("bitwise_xor")
x = sp.Symbol("x")
expr = f_xor(x, 2)

# Convert to Tree
try:
    tree = ExpressionTree.from_sympy(expr, ["x"])
    print(f"Root Type: {tree.root.node_type}")
    print(f"Root Value: {tree.root.value}")
    if tree.root.node_type == NodeType.BINARY_OP:
        print("Success: Node is BINARY")
        print(f"Children count: {len(tree.root.children)}")
        print(f"Child 1: {tree.root.children[0].value}")
        print(f"Child 2: {tree.root.children[1].value}")
    else:
        print("Failure: Node is NOT binary")

    # Check complexity with config
    config = GeneticConfig()
    comp = tree.complexity(config.operator_weights)
    print(f"Weighted Complexity: {comp}")

except Exception as e:
    print(f"Error: {e}")
