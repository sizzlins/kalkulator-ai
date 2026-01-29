"""Test Fibonacci pattern detection."""
import numpy as np

# Test data: fibonacci(x)
phi = 1.618033988749895
sqrt5 = 2.23606797749979

def fib(x):
    """Analytic continuation of Fibonacci."""
    return (phi**x - np.cos(np.pi * x) * phi**(-x)) / sqrt5

# Generate test data
test_points = []
# Integers
for n in range(0, 11):
    test_points.append((n, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55][n]))
# Some non-integers
for x in [0.5, 1.5, 2.5, 3.5, 4.5]:
    test_points.append((x, fib(x)))

X = np.array([p[0] for p in test_points]).reshape(-1, 1)
y = np.array([p[1] for p in test_points])

print(f"Test data: {len(test_points)} points")
print(f"Integers: 0→{int(y[0])}, 1→{int(y[1])}, 5→{int(y[5])}, 10→{int(y[10])}")

# Test the detector
from kalkulator_pkg.symbolic_regression.forensic_analysis import _detect_fibonacci_patterns

result = _detect_fibonacci_patterns(X, y, variable_names=["x"], verbose=True)
print(f"\nResult: {result}")

if result:
    if isinstance(result, tuple):
        print(f"\n✓ SUCCESS: Found exact match: {result[1]}")
    else:
        print(f"\n✓ SUCCESS: Found seeds: {result}")
else:
    print(f"\n✗ FAILURE: No pattern detected")
