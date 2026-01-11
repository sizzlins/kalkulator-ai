"""Test integration of generate_pattern_seeds tuple return"""

import numpy as np
import sys
import warnings
sys.path.insert(0, "C:/Users/LOQ/PycharmProjects/kalkulator-ai")

# Mock the _detect_step_patterns if needed, or rely on real one
from kalkulator_pkg.cli.repl_commands import generate_pattern_seeds

# 1. Test Normal Case (No step function)
print("Testing Normal Case (linear function)...")
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
result = generate_pattern_seeds(X, y, ['x'], verbose=True)

print(f"Result type: {type(result)}")
if isinstance(result, tuple):
    seeds, exact = result
    print(f"Seeds: {seeds}")
    print(f"Exact match: {exact}")
    assert exact is None, "Should not have exact match for linear function"
    assert isinstance(seeds, list), "First element should be list of seeds"
else:
    print("FAILED: Did not return tuple")
    exit(1)

print("\n" + "="*30 + "\n")

# 2. Test Step Function Case
print("Testing Step Function Case (floor)...")
x_vals = [4.5, 4.4, 3.9, 3.1]
X_step = np.array([[x] for x in x_vals])
y_step = np.array([np.floor(x) for x in x_vals])

result_step = generate_pattern_seeds(X_step, y_step, ['x'], verbose=True)
if isinstance(result_step, tuple):
    seeds, exact = result_step
    print(f"Seeds: {seeds}")
    print(f"Exact match: {exact}")
    assert exact == "floor(x)", f"Expected 'floor(x)', got {exact}"
else:
    print("FAILED: Did not return tuple")
    exit(1)

print("\nPASSED: Integration test successful!")
