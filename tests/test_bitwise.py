"""Test bitwise XOR pattern detection."""
import numpy as np

# Test data: f(x) = int(x) ^ 5
test_points = [
    # Integers
    (0, 5), (1, 4), (2, 7), (3, 6), (4, 1), (5, 0), (6, 3), (7, 2),
    (-1, -6), (-2, -5), (-3, -8), (-4, -7), (-5, -2), (-6, -1),
    # Decimals (should use truncation)
    (4.5, 1),   # int(4.5) = 4, 4 ^ 5 = 1
    (-4.5, -7), # int(-4.5) = -4, -4 ^ 5 = -7 (truncation, not floor!)
    (0.5, 5),   # int(0.5) = 0, 0 ^ 5 = 5
    (-0.5, 5),  # int(-0.5) = 0, 0 ^ 5 = 5
]

X = np.array([p[0] for p in test_points]).reshape(-1, 1)
y = np.array([p[1] for p in test_points])

print(f"Test data: {len(test_points)} points")
print(f"Sample: {test_points[:5]} ...")

# Test the detector directly
from kalkulator_pkg.symbolic_regression.forensic_analysis import _detect_bitwise_patterns

result = _detect_bitwise_patterns(X, y, variable_names=["x"], verbose=True)
print(f"\nResult: {result}")

if result:
    if isinstance(result, tuple):
        print(f"\n✓ SUCCESS: Found exact match: {result[1]}")
    else:
        print(f"\n✓ SUCCESS: Found seeds: {result}")
else:
    print(f"\n✗ FAILURE: No pattern detected")
