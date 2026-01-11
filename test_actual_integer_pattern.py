"""Test the actual _detect_integer_patterns function"""

import numpy as np
import sys
sys.path.insert(0, "C:/Users/LOQ/PycharmProjects/kalkulator-ai")

from kalkulator_pkg.cli.repl_commands import _detect_integer_patterns

# Simulate the user's data - key integer points
X = np.array([[2.0], [3.0], [4.0], [5.0], [-1.0]])
y = np.array([1.28571428571429, 1.07692307692308, 1.03174603174603, 1.01612903225806, 0.0])

print("Testing _detect_integer_patterns directly:")
print(f"X shape: {X.shape}")
result = _detect_integer_patterns(X, y)
print(f"Result: {result}")

if result:
    print("✅ Pattern detection works!")
else:
    print("❌ Pattern detection returned empty!")
