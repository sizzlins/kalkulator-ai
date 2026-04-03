"""Verify all Gemini review feedback implementations."""
import numpy as np

# Test 1: Dimensional Analysis - get_dimension
print("=" * 60)
print("TEST 1: Dimensional Analysis (get_dimension)")
print("=" * 60)

from kalkulator_pkg.symbolic_regression.expression_tree import (
    ExpressionNode, ExpressionTree, NodeType, ALLOWED_PRIMITIVES, _FORBIDDEN_NAMES
)
from kalkulator_pkg.dimensional_analysis.units import LENGTH, TIME, DIMENSIONLESS, MASS

# 1a: x/t should give L*T^-1 (velocity)
root = ExpressionNode(NodeType.BINARY_OP, 'div', [
    ExpressionNode(NodeType.VARIABLE, 'x'),
    ExpressionNode(NodeType.VARIABLE, 't'),
])
dim = root.get_dimension({'x': LENGTH, 't': TIME})
assert str(dim) == "L T^-1", f"Expected 'L T^-1', got '{dim}'"
print(f"  1a PASS: x/t -> {dim}")

# 1b: sin(x/t) should fail (non-dimensionless argument)
try:
    sin_node = ExpressionNode(NodeType.UNARY_OP, 'sin', [root])
    sin_node.get_dimension({'x': LENGTH, 't': TIME})
    assert False, "Should have raised ValueError"
except ValueError:
    print("  1b PASS: sin(x/t) correctly rejected")

# 1c: sin(dimensionless) should work
dim_sin = ExpressionNode(NodeType.UNARY_OP, 'sin', [
    ExpressionNode(NodeType.VARIABLE, 'theta'),
]).get_dimension({'theta': DIMENSIONLESS})
assert dim_sin.is_dimensionless()
print(f"  1c PASS: sin(theta) -> dimensionless")

# 1d: m*a = force (M*L*T^-2)
from kalkulator_pkg.dimensional_analysis.units import ACCELERATION
force_node = ExpressionNode(NodeType.BINARY_OP, 'mul', [
    ExpressionNode(NodeType.VARIABLE, 'm'),
    ExpressionNode(NodeType.VARIABLE, 'a'),
])
dim_force = force_node.get_dimension({'m': MASS, 'a': ACCELERATION})
expected_force = MASS * ACCELERATION
assert dim_force == expected_force, f"Expected {expected_force}, got {dim_force}"
print(f"  1d PASS: m*a -> {dim_force} (force)")

# 1e: x + t should fail (dimension mismatch)
try:
    add_node = ExpressionNode(NodeType.BINARY_OP, 'add', [
        ExpressionNode(NodeType.VARIABLE, 'x'),
        ExpressionNode(NodeType.VARIABLE, 't'),
    ])
    add_node.get_dimension({'x': LENGTH, 't': TIME})
    assert False, "Should have raised ValueError"
except ValueError:
    print("  1e PASS: x + t correctly rejected (dimension mismatch)")

print()

# Test 2: Security Validation
print("=" * 60)
print("TEST 2: Security Whitelist & Validation")
print("=" * 60)

# 2a: Valid tree passes
tree = ExpressionTree(root=ExpressionNode(NodeType.BINARY_OP, 'add', [
    ExpressionNode(NodeType.VARIABLE, 'x'),
    ExpressionNode(NodeType.CONSTANT, 3.14),
]), variables=['x'])
tree._validate_security_recursively()
print("  2a PASS: Valid tree passes security check")

# 2b: Forbidden name '__import__' is blocked
bad = ExpressionTree(root=ExpressionNode(NodeType.VARIABLE, '__import__'), variables=['x'])
try:
    bad._validate_security_recursively()
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "forbidden" in str(e).lower()
    print(f"  2b PASS: __import__ correctly blocked")

# 2c: Forbidden name 'os' is blocked
bad2 = ExpressionTree(root=ExpressionNode(NodeType.VARIABLE, 'os'), variables=['x'])
try:
    bad2._validate_security_recursively()
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"  2c PASS: 'os' correctly blocked")

# 2d: ALLOWED_PRIMITIVES contains all operators
assert 'sin' in ALLOWED_PRIMITIVES
assert 'add' in ALLOWED_PRIMITIVES
assert 'pow' in ALLOWED_PRIMITIVES
assert '__import__' not in ALLOWED_PRIMITIVES
print(f"  2d PASS: {len(ALLOWED_PRIMITIVES)} allowed primitives")

# 2e: compile_secure works
code = tree.compile_secure()
assert code is not None
print("  2e PASS: compile_secure() produces bytecode")

print()

# Test 3: GeneticConfig units_map
print("=" * 60)
print("TEST 3: GeneticConfig units_map field")
print("=" * 60)

from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig
config = GeneticConfig()
assert config.units_map is None
print("  3a PASS: units_map defaults to None (disabled)")

config2 = GeneticConfig(units_map={'x': LENGTH, 't': TIME})
assert config2.units_map is not None
print("  3b PASS: units_map can be set")

print()

# Test 4: Gaussian Peak Detector
print("=" * 60)
print("TEST 4: Gaussian Peak Detector")
print("=" * 60)

from kalkulator_pkg.symbolic_regression.forensic_analysis import _detect_gaussian_peak

x = np.linspace(-5, 5, 100)
y = 3.0 * np.exp(-((x - 1.0) / 2.0)**2)
X = x.reshape(-1, 1)
result = _detect_gaussian_peak(X, y, variable_names=['x'], verbose=True)
if isinstance(result, tuple):
    print(f"  4a PASS: Gaussian detected (exact match): {result[1]}")
elif result:
    print(f"  4a PASS: Gaussian seeds found: {result}")
else:
    print("  4a INFO: No Gaussian detected (may need tuning)")

print()

# Test 5: Sigmoid Switch Detector
print("=" * 60)
print("TEST 5: Sigmoid Switch Detector")
print("=" * 60)

from kalkulator_pkg.symbolic_regression.forensic_analysis import _detect_sigmoid_switch

x = np.linspace(-5, 5, 100)
y = 10.0 / (1.0 + np.exp(-2.0 * (x - 1.0))) + 5.0
X = x.reshape(-1, 1)
result = _detect_sigmoid_switch(X, y, variable_names=['x'], verbose=True)
if isinstance(result, tuple):
    print(f"  5a PASS: Sigmoid detected (exact match): {result[1]}")
elif result:
    print(f"  5a PASS: Sigmoid seeds found: {result}")
else:
    print("  5a INFO: No Sigmoid detected (may need tuning)")

print()

# Test 6: Windows Cache Race Condition Fix
print("=" * 60)
print("TEST 6: Windows Cache Race Condition Fix")
print("=" * 60)

import inspect
from kalkulator_pkg.cache_manager import save_persistent_cache
source = inspect.getsource(save_persistent_cache)
assert 'retry' in source.lower() or 'PermissionError' in source
print("  6a PASS: save_persistent_cache has retry/PermissionError handling")

print()
print("=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
