import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, ExpressionNode, NodeType
from kalkulator_pkg.symbolic_regression.strategies import EvolutionStrategy
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig

config = GeneticConfig()
strategy = EvolutionStrategy(config)

# Build exp(sqrt(x))
tree = ExpressionTree(
    ExpressionNode(NodeType.UNARY_OP, "exp", children=[
        ExpressionNode(NodeType.UNARY_OP, "sqrt", children=[
            ExpressionNode(NodeType.VARIABLE, "x")
        ])
    ])
)

X = np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 4.0]).reshape(-1, 1)
y = np.exp(np.sqrt(X.flatten().astype(complex)))
print(f"y dtype: {y.dtype}")
print(f"y: {y}")

# Evaluate tree
pred = tree.evaluate(X)
print(f"pred: {pred}")
print(f"pred dtype: {pred.dtype}")

# Calculate fitness
fitness = strategy.calculate_fitness(tree, X, y)
print(f"fitness: {fitness}")
is_inf = fitness == float("inf")
print(f"Is inf: {is_inf}")

if is_inf:
    print("FAILURE: fitness is inf -- complex handling still broken")
elif fitness < 1.0:
    print(f"SUCCESS: fitness={fitness:.6f} (should be near 0 for perfect match)")
else:
    print(f"WARNING: fitness={fitness:.6f} (expected near 0)")
