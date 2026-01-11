"""Simulate the EXACT data the REPL would produce from user input"""

import numpy as np
import sympy as sp
import sys
sys.path.insert(0, "C:/Users/LOQ/PycharmProjects/kalkulator-ai")

from kalkulator_pkg.cli.repl_commands import _detect_integer_patterns

# These are some of the EXACT x-values from user input, as they might be parsed
# Note: The REPL parses inputs like f(2) -> evaluates "2" via sympy
# and inputs like f(e) -> evaluates "e" via sympy to get 2.71828...

# Simulate the values as they would appear after sympy evaluation
x_values = [
    4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0,
    2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0,  # <-- 2.0 is here!
    1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0,
    -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # <-- Integers 2,3,4,5,... are here!
    float(sp.E),  # e = 2.71828...
    float(sp.pi),  # pi = 3.14159...
    complex(0, 1),  # i is complex!
    float(sp.sin(1)),
    float(sp.sin(sp.pi)),
    4.1, -2.5, 0.001, -0.99, 12.345, -19.9, 15.5, 3.333,
    float(sp.sqrt(2)),
    float(sp.sqrt(5)),
]

# Corresponding y values for (x³+1)/(x³-1), handle singularity
def f(x):
    if x == 1:
        return np.inf
    if isinstance(x, complex):
        return complex(x)  # Keep it complex
    try:
        return (x**3 + 1) / (x**3 - 1)
    except:
        return np.nan

y_values = [f(x) for x in x_values]

# Convert to numpy, mimicking what the REPL does
X = np.array([[x] for x in x_values], dtype=object)  # dtype=object to handle complex
y = np.array(y_values, dtype=object)

print(f"X shape: {X.shape}")
print(f"X dtype: {X.dtype}")
print(f"y dtype: {y.dtype}")

# Check the types of specific values
print(f"\nType of X[25] (should be 2.0): {type(X[25,0])} = {X[25,0]}")
print(f"Type of X[62] (integer 2): {type(X[62,0])} = {X[62,0]}")

# Try to detect integer patterns
print("\nRunning _detect_integer_patterns...")
result = _detect_integer_patterns(X, y)
print(f"Result: {result}")

# Debug the filtering logic
print("\n--- Manual filter check ---")
x_flat = X.flatten()
for i, x_val in enumerate(x_flat):
    # Skip complex
    if np.iscomplex(x_val) or (hasattr(x_val, 'imag') and abs(x_val.imag) > 1e-9):
        continue
    try:
        real_val = float(x_val.real if hasattr(x_val, 'real') else x_val)
        if abs(real_val - round(real_val)) < 1e-9 and abs(real_val) > 1 and abs(real_val) < 10:
            print(f"  INTEGER FOUND: x[{i}] = {real_val}, y = {y[i]}")
    except Exception as e:
        print(f"  Exception at i={i}: {e}, x_val type={type(x_val)}")
