
import unittest
import numpy as np
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, NodeType, ExpressionNode
from kalkulator_pkg.symbolic_regression.strategies import EvolutionStrategy
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig

class TestDiscreteFixes(unittest.TestCase):
    def setUp(self):
        self.config = GeneticConfig()
        self.strategy = EvolutionStrategy(self.config)
        self.X = np.arange(-10, 11).reshape(-1, 1) # -10 to 10
        self.x_vals = self.X.flatten()
        # Target: x >> 1 (integer division by 2 approx)
        # Note: python bitwise >> is floor behavior for negative numbers usually
        self.y_target = np.array([int(x) >> 1 for x in self.x_vals])
        
    def test_integer_penalty(self):
        """Verify that 0.49*x receives a penalty vs x >> 1."""
        # 1. Cleaner: x >> 1
        # Construct tree: rshift(x, 1)
        root_clean = ExpressionNode(NodeType.BINARY_OP, 'rshift', [
            ExpressionNode(NodeType.VARIABLE, 'x'),
            ExpressionNode(NodeType.CONSTANT, 1.0)
        ])
        tree_clean = ExpressionTree(root_clean, variables=['x'])
        
        loss_clean = self.strategy.calculate_fitness(tree_clean, self.X, self.y_target)
        # Should be effectively 0 (plus complexity penalty)
        mse_clean = loss_clean - (self.config.parsimony_coefficient * tree_clean.complexity())
        self.assertLess(mse_clean, 1e-6, "Correct model should have ~0 MSE")

        # 2. Dirtier: 0.49 * x
        # 0.49 * 10 = 4.9. Target 5. Diff 0.1. MSE ~0.01.
        # BUT with penalty, should be +10.0.
        root_dirty = ExpressionNode(NodeType.BINARY_OP, 'mul', [
            ExpressionNode(NodeType.CONSTANT, 0.49),
            ExpressionNode(NodeType.VARIABLE, 'x')
        ])
        tree_dirty = ExpressionTree(root_dirty, variables=['x'])
        
        loss_dirty = self.strategy.calculate_fitness(tree_dirty, self.X, self.y_target)
        basic_complexity = tree_dirty.complexity()
        
        # Check if penalty applied
        # Raw MSE without penalty would be small (~0.05)
        # With penalty, should be > 10.0
        print(f"Dirty Loss: {loss_dirty}")
        self.assertGreater(loss_dirty, 9.0, "Integer penalty should trigger for 0.49*x")

    def test_discrete_snapper(self):
        """Verify rshift(x, 1.648) snaps to rshift(x, 1) if MSE allows."""
        # 1. Construct Funny Tree: rshift(x, 1.648)
        # 1.648 is roughly sqrt(e)
        funny_val = 1.648
        rshift_node = ExpressionNode(NodeType.BINARY_OP, 'rshift', [
            ExpressionNode(NodeType.VARIABLE, 'x'),
            ExpressionNode(NodeType.CONSTANT, funny_val)
        ])
        tree = ExpressionTree(rshift_node, variables=['x'])
        
        # Verify initial state
        const_node = tree.root.children[1]
        self.assertAlmostEqual(const_node.value, funny_val)
        
        # 2. Polish
        # x >> 1.648 (floored to 1) is SAME as x >> 1
        # So MSE should be identical/zero.
        tree.polish_discrete_constants(self.X, self.y_target)
        
        # 3. Verify Snap
        new_val = tree.root.children[1].value
        print(f"Polished Value: {new_val}")
        self.assertAlmostEqual(new_val, 1.0, places=5, msg="Should snap to 1.0")
        
    def test_discrete_snapper_revert(self):
        """Verify rshift(x, 1.9) snaps to 1.0 (truncation/floor wins over round/ceil)."""
        # x >> 1.9 (floor 1) == x >> 1.
        # old logic (round): 1.9 -> 2.0. Rejected. Reverted to 1.9.
        # new logic (floor/ceil): tries 2.0 (Reject), tries 1.0 (Accept).
        
        bad_val = 1.9
        rshift_node = ExpressionNode(NodeType.BINARY_OP, 'rshift', [
            ExpressionNode(NodeType.VARIABLE, 'x'),
            ExpressionNode(NodeType.CONSTANT, bad_val)
        ])
        tree = ExpressionTree(rshift_node, variables=['x'])
        
        # Polish
        tree.polish_discrete_constants(self.X, self.y_target)
        
        # Verify Snap to 1.0 (Better Behavior)
        new_val = tree.root.children[1].value
        print(f"Improved Snapping Value: {new_val}")
        self.assertAlmostEqual(new_val, 1.0, places=5, msg="Should snap to 1.0 (floor) even if 1.9 is closer to 2.0")

if __name__ == "__main__":
    unittest.main()
