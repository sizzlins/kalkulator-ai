"""Test step function pattern detection"""

import numpy as np
import sys
sys.path.insert(0, "C:/Users/LOQ/PycharmProjects/kalkulator-ai")

from kalkulator_pkg.cli.repl_commands import _detect_step_patterns

# Test floor(x) detection
print("Testing floor(x) detection:")
x_vals = [4.5, 4.4, 3.9, 3.1, 2.5, 1.9, 1.1, 0.5, -0.5, -1.5]
X = np.array([[x] for x in x_vals])
y = np.array([np.floor(x) for x in x_vals])
result = _detect_step_patterns(X, y)
print(f"  X: {x_vals[:5]}...")
print(f"  Y: {y[:5]}...")
print(f"  Result: {result}")
assert "floor(x)" in result, f"Expected floor(x), got {result}"
print("  ✅ floor(x) detection passed!")

# Test ceil(x) detection  
print("\nTesting ceil(x) detection:")
y_ceil = np.array([np.ceil(x) for x in x_vals])
result = _detect_step_patterns(X, y_ceil)
print(f"  Result: {result}")
assert "ceil(x)" in result, f"Expected ceil(x), got {result}"
print("  ✅ ceil(x) detection passed!")

# Test round(x) detection
print("\nTesting round(x) detection:")
y_round = np.array([round(x) for x in x_vals])
result = _detect_step_patterns(X, y_round)
print(f"  Result: {result}")
assert "round(x)" in result, f"Expected round(x), got {result}"
print("  ✅ round(x) detection passed!")

# Test non-step function (should return empty)
print("\nTesting non-step function (should return empty):")
y_linear = np.array([2*x + 1 for x in x_vals])
result = _detect_step_patterns(X, y_linear)
print(f"  Result: {result}")
assert result == [], f"Expected empty, got {result}"
print("  ✅ Non-step detection passed!")

print("\n" + "="*60)
print("ALL STEP FUNCTION TESTS PASSED!")
print("="*60)
