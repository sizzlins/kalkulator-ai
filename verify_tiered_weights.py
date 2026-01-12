
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticConfig
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, ExpressionNode, NodeType

def test_tiered_economics():
    print("Testing Tiered Economic Reform...")
    
    config = GeneticConfig()
    weights = config.operator_weights
    
    # Check key weights
    assert weights["abs"] == 4.0, f"Abs should be 4.0, got {weights.get('abs')}"
    assert weights["max"] == 5.0, f"Max should be 5.0, got {weights.get('max')}"
    assert weights["sqrt"] == 1.0, f"Sqrt should be 1.0, got {weights.get('sqrt')}"
    assert weights["pow"] == 1.0, f"Pow should be 1.0, got {weights.get('pow')}"
    
    print("PASS: Weight Config")
    
    # 1. The Gateway Drug: Abs(x)
    # Cost: 4 (abs) + 1 (x) = 5.0
    abs_node = ExpressionNode(NodeType.UNARY_OP, "abs", [ExpressionNode(NodeType.VARIABLE, "x")])
    abs_tree = ExpressionTree(abs_node, variables=["x"])
    abs_cost = abs_tree.complexity(weights=weights)
    print(f"Cost of Abs(x): {abs_cost}")
    
    # 2. The Physics Solution: sqrt(x^2 + 1)
    # Cost: 1(sqrt) + 1(add) + 1(pow) + 1(x) + 1(const) + 1(const 2) = 6.0?
    # Let's build it:
    x = ExpressionNode(NodeType.VARIABLE, "x")
    two = ExpressionNode(NodeType.CONSTANT, 2.0)
    one = ExpressionNode(NodeType.CONSTANT, 1.0)
    pow_node = ExpressionNode(NodeType.BINARY_OP, "pow", [x, two])
    add_node = ExpressionNode(NodeType.BINARY_OP, "add", [pow_node, one])
    sqrt_node = ExpressionNode(NodeType.UNARY_OP, "sqrt", [add_node])
    
    phys_tree = ExpressionTree(sqrt_node, variables=["x"])
    phys_cost = phys_tree.complexity(weights=weights)
    print(f"Cost of sqrt(x^2 + 1): {phys_cost}")
    
    # 3. The Cheat: max(0.5, Abs(x))
    # Cost: 5(max) + 1(const) + 5(Abs(x) tree) = 11.0
    max_node = ExpressionNode(NodeType.BINARY_OP, "max", [
        ExpressionNode(NodeType.CONSTANT, 0.5),
        abs_node # Reuse abs(x)
    ])
    cheat_tree = ExpressionTree(max_node, variables=["x"])
    cheat_cost = cheat_tree.complexity(weights=weights)
    print(f"Cost of max(0.5, Abs(x)): {cheat_cost}")
    
    # Verdict
    # Physics (6.0) vs Abs (5.0) -> Gap is small (1.0).
    # Physics (6.0) vs Cheat (11.0) -> Gap is HUGE.
    
    # Conclusion: 
    # The engine might start with Abs(x) (5.0) because it's slightly cheaper than Physics (6.0).
    # BUT, to improve fit, it cannot go to Cheat (11.0).
    # It MUST go to Physics (6.0) because 6.0 << 11.0.
    
    assert phys_cost < cheat_cost, "Physics MUST be cheaper than Cheat"
    # It's okay if Abs is cheaper than Physics, closer is better.
    # Ideally Abs(x) should be MORE expensive than sqrt(x^2)?
    # sqrt(x^2) = 1(sqrt) + 1(pow) + 1(x) + 1(2) = 4.0.
    # Abs(x) = 5.0. 
    # YES! sqrt(x^2) is now 4.0, while Abs(x) is 5.0.
    # The engine should prefer sqrt(x^2) over Abs(x)!
    
    # Let's verify sqrt(x^2)
    pow_x_2 = ExpressionNode(NodeType.BINARY_OP, "pow", [x, two])
    sqrt_pow = ExpressionNode(NodeType.UNARY_OP, "sqrt", [pow_x_2])
    sqrt_pow_tree = ExpressionTree(sqrt_pow, variables=["x"])
    sqrt_pow_cost = sqrt_pow_tree.complexity(weights=weights)
    print(f"Cost of sqrt(x^2): {sqrt_pow_cost}")
    
    assert sqrt_pow_cost < abs_cost, f"sqrt(x^2) ({sqrt_pow_cost}) should be cheaper than abs(x) ({abs_cost})"
    print("PASS: Physics Bridge Established")

if __name__ == "__main__":
    test_tiered_economics()
