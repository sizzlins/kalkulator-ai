"""Final sanity check: Verify ** is used for power, not ^ (XOR)."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.symbolic_regression.expression_tree import (
    ExpressionTree, ExpressionNode, NodeType
)

print("=== Power Operator Sanity Check ===\n")

# 1. Build tree manually: x ** 2
x_node = ExpressionNode(NodeType.VARIABLE, 'x')
two_node = ExpressionNode(NodeType.CONSTANT, 2)
pow_node = ExpressionNode(NodeType.BINARY_OP, 'pow', [x_node, two_node])
x_node.parent = pow_node
two_node.parent = pow_node
tree = ExpressionTree(root=pow_node, variables=['x'])

# 2. Check to_string() output (used by compile_secure)
code_str = tree.to_string()
print(f"Code String: {code_str}")
assert "**" in code_str or "pow" in code_str, "CRITICAL: Compiler is using XOR (^)"
print("[PASS] Test A: Power operator uses ** (not ^)\n")

# 3. Compile and eval round-trip
code_obj = tree.compile_secure()
func = eval(code_obj)
result = func(5)
print(f"eval(5) = {result}")
assert result == 25, f"Expected 25, got {result}"
print("[PASS] Test B: compile_secure produces correct result (5**2 = 25)\n")

# 4. Parse round-trip: ExpressionTree.from_string('x**2')
tree2 = ExpressionTree.from_string('x**2')
code_str2 = tree2.to_string()
print(f"Round-trip string: {code_str2}")
code_obj2 = tree2.compile_secure()
func2 = eval(code_obj2)
assert func2(5) == 25, "Round-trip failed"
print("[PASS] Test C: from_string('x**2') round-trips correctly\n")

# 5. Bonus: Verify bitwise_xor still uses ^
xor_node = ExpressionNode(NodeType.BINARY_OP, 'bitwise_xor', [
    ExpressionNode(NodeType.VARIABLE, 'x'),
    ExpressionNode(NodeType.CONSTANT, 3)
])
xor_tree = ExpressionTree(root=xor_node, variables=['x'])
xor_str = xor_tree.to_string()
print(f"XOR String: {xor_str}")
assert "^" in xor_str, "bitwise_xor should use ^"
assert "**" not in xor_str, "bitwise_xor must NOT use **"
print("[PASS] Test D: bitwise_xor correctly uses ^ (distinct from pow)\n")

print("=" * 40)
print("All checks passed. Power operator is safe.")
print("  pow  -> **  (exponentiation)")
print("  xor  -> ^   (bitwise XOR)")
print("No ambiguity. Compiler is physics-safe.")
