from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticConfig
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, ExpressionNode, NodeType
import numpy as np

config = GeneticConfig()
w_xor = config.operator_weights.get("bitwise_xor")
print(f"XOR Weight in Config: {w_xor}")

# Create simple tree: bitwise_xor(x, 5)
tree = ExpressionTree(
    ExpressionNode(
        NodeType.BINARY_OP, 
        "bitwise_xor", 
        [
            ExpressionNode(NodeType.VARIABLE, "x"), 
            ExpressionNode(NodeType.CONSTANT, 5.0)
        ]
    ), 
    ["x"]
)

complexity = tree.complexity(config.operator_weights)
print(f"Calculated Complexity: {complexity}")

# Create nested tree: bitwise_xor(bitwise_xor(x, 5), 2)
tree2 = ExpressionTree(
    ExpressionNode(
        NodeType.BINARY_OP, 
        "bitwise_xor", 
        [
            tree.root.copy_subtree(),
            ExpressionNode(NodeType.CONSTANT, 2.0)
        ]
    ), 
    ["x"]
)
complexity2 = tree2.complexity(config.operator_weights)
print(f"Nested Complexity: {complexity2}")
