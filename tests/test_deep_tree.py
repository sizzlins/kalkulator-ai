
import sys
import unittest
import inspect
import numpy as np
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, ExpressionNode, NodeType

print(f"DEBUG: ExpressionNode loaded from: {inspect.getfile(ExpressionNode)}")

class TestDeepTree(unittest.TestCase):
    def setUp(self):
        # Increase recursion limit slightly just to ensure we hit it naturally if algorithmic
        # but we want to prove it works even with default limit for reasonable depths
        self.recursion_limit = sys.getrecursionlimit()
        # Create a deep tree: x + (x + (x + ...))
        # Depth 2000 should blow default stack (1000)
        self.depth = 2000
        
        # Build iteratively to avoid crash during construction!
        self.root = ExpressionNode(NodeType.VARIABLE, "x")
        curr = self.root
        for _ in range(self.depth):
            # new_node = x + curr
            # We wrap current in a ADD node
            new_parent = ExpressionNode(NodeType.BINARY_OP, "add", children=[
                ExpressionNode(NodeType.VARIABLE, "x"),
                curr
            ])
            curr.parent = new_parent
            curr = new_parent
            
        self.tree = ExpressionTree(curr)
        print(f"DEBUG: Root: {self.tree.root}, Type: {self.tree.root.node_type}")
        print(f"DEBUG: Root children count: {len(self.tree.root.children)}")
        if self.tree.root.children:
             print(f"DEBUG: First Check: {self.tree.root.children[0]}")

    def test_contains_variables(self):
        """Should not crash."""
        print(f"Testing contains_variables with depth {self.depth}...")
        self.assertTrue(self.tree.contains_variables())

    def test_count_nodes(self):
        """Should not crash."""
        print("Testing count_nodes...")
        # Use tree.root, not self.root (which is the leaf)
        count = self.tree.root.count_nodes()
        print(f"DEBUG: count_nodes result: {count}")
        self.assertGreater(count, self.depth)

    def test_to_rpn(self):
        """Should not crash."""
        print("Testing to_rpn...")
        rpn = self.tree.root.to_rpn()
        self.assertGreater(len(rpn), self.depth)

    def test_str(self):
        """Should not crash (Iterative - Should PASS)."""
        print("Testing __str__...")
        try:
            s = str(self.tree)
            print("String generation successful.")
            self.assertTrue(len(s) > self.depth)
        except RecursionError:
            self.fail("RecursionError raised in __str__")

    def test_fold_constants(self):
        """Should not crash (Iterative - Should PASS)."""
        print("Testing fold_constants...")
        # Add some constants to fold: 1 + (1 + (...))
        # Rebuild a constant-heavy tree
        root_const = ExpressionNode(NodeType.CONSTANT, 1.0)
        curr = root_const
        for _ in range(self.depth):
            new_parent = ExpressionNode(NodeType.BINARY_OP, "add", children=[
                ExpressionNode(NodeType.CONSTANT, 1.0),
                curr
            ])
            curr.parent = new_parent
            curr = new_parent
        
        tree = ExpressionTree(curr)
        
        try:
            tree.fold_constants()
            print("Fold successful.")
        except RecursionError:
            self.fail("RecursionError raised in fold_constants")
            
    def test_mutation_depth_limit(self):
        """Test that mutations respect max_depth."""
        print("Testing mutation depth limits...")
        from kalkulator_pkg.symbolic_regression.operators import insert_mutation, subtree_mutation
        
        # Create a linear unary chain (depth 5)
        # sin(sin(sin(sin(x))))
        root = ExpressionNode(NodeType.VARIABLE, "x")
        curr = root
        for _ in range(4):
            # Wrap current in SIN
            new_parent = ExpressionNode(NodeType.UNARY_OP, "sin", children=[curr])
            curr.parent = new_parent
            curr = new_parent
        tree = ExpressionTree(curr)
        initial_depth = tree.depth()
        # Expect depth 5 (root(add) -> add -> add -> add -> x)
        # Actually count:
        # 1(add) -> 2(add) -> 3(add) -> 4(add) -> 5(x)
        
        # 1. Insert mutation with max_depth=initial_depth. Should FAIL (return copy).
        # insert adds 1 to depth.
        mutated = insert_mutation(tree, operators=['sin'], max_depth=initial_depth)
        self.assertEqual(mutated.depth(), initial_depth)
        # Verify it's a copy but identical structure (for deterministic failure behavior in this specific operator)
        # Actually insert_mutation returns tree.copy() if limit hit.
        
        # 2. Insert mutation with max_depth=initial_depth+1. Should SUCCEED.
        mutated_success = insert_mutation(tree, operators=['sin'], max_depth=initial_depth + 1)
        # Note: insert_mutation is random, it might pick a node where wrapping doesn't increase total depth?
        # No, it wraps a node. If it wraps root, depth +1. If it wraps leaf, depth IS maintained?
        # No, if it wraps leaf (depth D at level L), new node is level L, leaf is L+1.
        # So depth ALWAYS increases by 1 if we define depth as longest path.
        # Yes.
        self.assertEqual(mutated_success.depth(), initial_depth + 1)

if __name__ == '__main__':
    unittest.main()
