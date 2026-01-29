"""Test that scalloped staircase functions are correctly identified.

The bug was: floor(x) + frac(x)^2 was being misidentified as 'x - 0.1' because
the linear check returned early with high R² without checking piecewise patterns.
"""
import math
import numpy as np

def floor_frac_squared(x):
    """f(x) = floor(x) + frac(x)^2"""
    floor_x = math.floor(x)
    frac_x = x - floor_x
    return floor_x + frac_x ** 2

# Generate test data points (same pattern as user's data)
test_points = []
for x in [4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.3, 0.1]:
    y = floor_frac_squared(x)
    test_points.append(((x,), y))

# Add some negative values
for x in [-0.5, -1.0, -1.5, -2.0, -2.5]:
    y = floor_frac_squared(x)
    test_points.append(((x,), y))

# Add integers
for x in range(-5, 6):
    y = floor_frac_squared(x)
    test_points.append(((x,), y))

print("Test data points (first 10):")
for p in test_points[:10]:
    print(f"  f({p[0][0]}) = {p[1]}")

print(f"\nTotal points: {len(test_points)}")

# Run the function finder
from kalkulator_pkg.function_manager import find_function_from_data

class MockContext:
    data = {}
    _named_decimals = {}

ctx = MockContext()

success, func_str, factored, note = find_function_from_data(ctx, test_points, param_names=["x"])

print(f"\n{'='*50}")
print(f"Result:")
print(f"  Success: {success}")
print(f"  Function: {func_str}")
print(f"  Note: {note}")
print(f"{'='*50}")

# Verify the result contains floor/frac
expected_keywords = ["floor", "frac"]
found_keywords = [kw for kw in expected_keywords if kw in str(func_str).lower()]

if found_keywords:
    print(f"\n✓ SUCCESS: Found piecewise pattern with keywords: {found_keywords}")
else:
    if "x" in str(func_str) and "-" in str(func_str) and "floor" not in str(func_str).lower():
        print(f"\n✗ FAILURE: Still returning linear approximation instead of floor/frac pattern")
    else:
        print(f"\n? UNCLEAR: Result doesn't match expected pattern, manual check needed")
