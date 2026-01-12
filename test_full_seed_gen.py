"""Test the full generate_pattern_seeds with realistic user data"""

import numpy as np
import sys
sys.path.insert(0, "C:/Users/LOQ/PycharmProjects/kalkulator-ai")

from kalkulator_pkg.cli.repl_commands import generate_pattern_seeds

# Create realistic data like the user's input
# (x³+1)/(x³-1) for various x values
def f(x):
    return (x**3 + 1) / (x**3 - 1)

# Include integer points 2, 3, 4, 5 plus many decimal points
x_vals = [2.0, 3.0, 4.0, 5.0, -20, -19, -18, -17, -16, -15, -1, 0, -2]
X = np.array([[x] for x in x_vals])
y = np.array([f(x) if x != 1 else np.inf for x in x_vals])

print("Testing generate_pattern_seeds with verbose=True:")
print(f"X shape: {X.shape}")
print(f"Integer points in X: {[x for x in X.flatten() if abs(x - round(x)) < 0.01 and 1 < abs(x) < 10]}")

seeds = generate_pattern_seeds(X, y, ["x"], verbose=True)

print(f"\n{'='*60}")
print(f"TOTAL SEEDS: {len(seeds)}")
print(f"{'='*60}")

# Check for the cubic pattern
cubic_pattern = [s for s in seeds if "x^3" in s or "x**3" in s]
print(f"\nSeeds containing x^3: {cubic_pattern}")

if any("(x^3 + 1) / (x^3 - 1)" in s or "(x**3 + 1)/(x**3 - 1)" in s for s in seeds):
    print("✅ SUCCESS: Found the exact pattern!")
else:
    print("❌ FAILURE: Exact pattern not found")
    print(f"All seeds: {seeds[:20]}...")
