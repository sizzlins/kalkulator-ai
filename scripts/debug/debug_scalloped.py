"""Debug why _detect_scalloped_staircase returns [] - add more tracing."""
import math
import numpy as np

def floor_frac_squared(x):
    """f(x) = floor(x) + frac(x)^2"""
    floor_x = math.floor(x)
    frac_x = x - floor_x
    return floor_x + frac_x ** 2

# Generate test data
X = []
y = []
for x in [4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.3, 0.1]:
    X.append(x)
    y.append(floor_frac_squared(x))
for x in [-0.5, -1.0, -1.5, -2.0, -2.5]:
    X.append(x)
    y.append(floor_frac_squared(x))
for x in range(-5, 6):
    X.append(x)
    y.append(floor_frac_squared(x))

X_arr = np.array(X)
y_arr = np.array(y)

print(f"Data: {len(X_arr)} points")

# Manual trace of the algorithm
X_flat = X_arr.flatten()
y_flat = y_arr.flatten()

print(f"\n1. X_flat shape: {X_flat.shape}, y_flat shape: {y_flat.shape}")
print(f"   len(X_flat) >= 5? {len(X_flat) >= 5}")

# Step 1: Integer Anchor Analysis
integer_mask = np.abs(X_flat - np.round(X_flat)) < 1e-9
print(f"\n2. Integer mask count: {np.sum(integer_mask)} (need >= 3)")

integer_x = X_flat[integer_mask]
integer_y = y_flat[integer_mask]
print(f"   Integer anchors (first 5): x={integer_x[:5]}, y={integer_y[:5]}")

# Try to find 'a' such that f(n) = n^a
pos_int_mask = integer_x > 0.5
print(f"\n3. Positive integer mask count: {np.sum(pos_int_mask)} (need >= 3)")

if np.sum(pos_int_mask) >= 3:
    pos_int_x = integer_x[pos_int_mask]
    pos_int_y = integer_y[pos_int_mask]
    print(f"   Positive integers: x={pos_int_x}, y={pos_int_y}")
    print(f"   All positive y? {np.all(pos_int_y > 0)}")
    
    if np.all(pos_int_y > 0):
        log_x = np.log(pos_int_x)
        log_y = np.log(pos_int_y)
        a_estimates = log_y / log_x
        print(f"   a_estimates: {a_estimates}")
        a_median = np.median(a_estimates)
        print(f"   a_median: {a_median}")
        
        a_candidates = [0.5, 1, 1.5, 2, 3, 4]
        best_a = min(a_candidates, key=lambda a: abs(a - a_median))
        print(f"   best_a: {best_a}")
    else:
        print("   -> Checking if f(n) = n (a=1)")
        anchor_errors = np.abs(integer_y - integer_x)
        print(f"   anchor_errors: {anchor_errors}")
        print(f"   max anchor error: {np.max(anchor_errors)}")
else:
    print("   -> Not enough positive integers, checking f(n) = n")
    anchor_errors = np.abs(integer_y - integer_x)
    print(f"   anchor_errors: {anchor_errors}")
    print(f"   max anchor error: {np.max(anchor_errors)}")

# The key bug: for floor(x) + frac(x)^2, f(n) = n for integers
# So anchor_errors should be 0
# But the pos_int_y check at line 176 expects y > 0
# For negative integers: f(-1) = -1, so y = -1 is NOT > 0
# This path leads to log-log regression on positive integers only

# For f(n) = n, log(y)/log(x) = log(n)/log(n) = 1
# So best_a should be 1
