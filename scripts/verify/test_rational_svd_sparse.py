import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.heuristics import solve_rational_function_svd

print("--- Verifying Rational SVD Threshold Relaxation ---")

# Test 1: Triangular numbers (Quadratic, 3 points)
# f(1)=1, f(2)=3, f(3)=6 -> f(x) = x(x+1)/2 = 0.5x^2 + 0.5x
X = [[1], [2], [3]]
y = [1, 3, 6]
success, expr, mse = solve_rational_function_svd(X, y, ['x'], max_numerator_degree=2, max_denominator_degree=0)
print(f"Test 1 (Quadratic, 3 pts): Success={success}")
if success:
    print(f"  Expr: {expr}")
    print(f"  MSE: {mse}")
else:
    print("  FAILED")

# Test 2: Simple Rational (4 points)
# f(x) = x/(x+1) -> f(1)=0.5, f(2)=2/3, f(3)=0.75, f(4)=0.8
X = [[1], [2], [3], [4]]
y = [0.5, 2.0/3.0, 0.75, 0.8]
success, expr, mse = solve_rational_function_svd(X, y, ['x'], max_numerator_degree=1, max_denominator_degree=1)
print(f"Test 2 (Rational 1/1, 4 pts): Success={success}")
if success:
    print(f"  Expr: {expr}")
    print(f"  MSE: {mse}")
else:
    print("  FAILED")

# Test 3: Linear (2 points)
# f(x) = 2x + 1 -> f(1)=3, f(2)=5
X = [[1], [2]]
y = [3, 5]
success, expr, mse = solve_rational_function_svd(X, y, ['x'], max_numerator_degree=1, max_denominator_degree=0)
print(f"Test 3 (Linear, 2 pts): Success={success}")
if success:
    print(f"  Expr: {expr}")
    print(f"  MSE: {mse}")
else:
    print("  FAILED")
