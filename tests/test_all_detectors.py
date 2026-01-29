"""Test all regenerated pattern detectors."""
import numpy as np

from kalkulator_pkg.symbolic_regression.forensic_analysis import (
    _detect_self_power,
    _detect_modulo_patterns,
    _detect_signum_patterns,
    _detect_relu_patterns,
    _detect_clamp_patterns,
    _detect_pulse_patterns,
    _detect_prime_counting_patterns,
    _detect_anchor_patterns,
)

print("=" * 60)
print("TESTING ALL REGENERATED PATTERN DETECTORS")
print("=" * 60)

# 1. Self-Power: f(x) = x^x
print("\n1. Self-Power: x^x")
x_vals = np.array([1, 2, 3, 4, 5, 1.5, 2.5]).reshape(-1, 1)
y_vals = x_vals.flatten() ** x_vals.flatten()
result = _detect_self_power(x_vals, y_vals, variable_names=["x"], verbose=True)
print(f"   Result: {result}")
assert result, "Self-power detection failed!"
print("   ✓ PASSED")

# 2. Modulo: f(x) = x % 3
print("\n2. Modulo: x % 3")
x_vals = np.linspace(0, 12, 50).reshape(-1, 1)
y_vals = x_vals.flatten() % 3
result = _detect_modulo_patterns(x_vals, y_vals, variable_names=["x"], verbose=True)
print(f"   Result: {result}")
assert result, "Modulo detection failed!"
print("   ✓ PASSED")

# 3. Signum: f(x) = sign(x)
print("\n3. Signum: sign(x)")
x_vals = np.array([-10, -5, -3, -1, -0.5, 0, 0.5, 1, 3, 5, 10]).reshape(-1, 1)
y_vals = np.sign(x_vals.flatten())
result = _detect_signum_patterns(x_vals, y_vals, variable_names=["x"], verbose=True)
print(f"   Result: {result}")
assert result, "Signum detection failed!"
print("   ✓ PASSED")

# 4. ReLU: f(x) = max(0, x)
print("\n4. ReLU: max(0, x)")
x_vals = np.linspace(-5, 5, 21).reshape(-1, 1)
y_vals = np.maximum(0, x_vals.flatten())
result = _detect_relu_patterns(x_vals, y_vals, variable_names=["x"], verbose=True)
print(f"   Result: {result}")
assert result, "ReLU detection failed!"
print("   ✓ PASSED")

# 5. Clamp: f(x) = min(x, 5)
print("\n5. Clamp: min(x, 5)")
x_vals = np.linspace(0, 10, 21).reshape(-1, 1)
y_vals = np.minimum(x_vals.flatten(), 5)
result = _detect_clamp_patterns(x_vals, y_vals, variable_names=["x"], verbose=True)
print(f"   Result: {result}")
assert result, "Clamp detection failed!"
print("   ✓ PASSED")

# 6. Pulse: f(x) = Heaviside(x-3) - Heaviside(x-7)
print("\n6. Pulse: Heaviside(x-3) - Heaviside(x-7)")
x_vals = np.array([0, 1, 2, 2.5, 3.5, 4, 5, 6, 6.5, 7.5, 8, 9, 10]).reshape(-1, 1)
y_vals = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float)  # Clean 0/1 pulse
result = _detect_pulse_patterns(x_vals, y_vals, variable_names=["x"], verbose=True)
print(f"   Result: {result}")
assert result, "Pulse detection failed!"
print("   ✓ PASSED")

# 7. Prime Counting: f(x) = π(x)
print("\n7. Prime Counting: π(x)")
prime_counts = [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 8]  # π(0) to π(20)
x_vals = np.arange(len(prime_counts)).reshape(-1, 1)
y_vals = np.array(prime_counts, dtype=float)
result = _detect_prime_counting_patterns(x_vals, y_vals, variable_names=["x"], verbose=True)
print(f"   Result: {result}")
assert result, "Prime counting detection failed!"
print("   ✓ PASSED")

# 8. Anchor: f(x) = (x+1)^(1/x)
print("\n8. Anchor: (x+1)^(1/x)")
x_vals = np.linspace(1, 100, 50).reshape(-1, 1)
y_vals = (x_vals.flatten() + 1) ** (1.0 / x_vals.flatten())
result = _detect_anchor_patterns(x_vals, y_vals, variable_names=["x"], verbose=True)
print(f"   Result: {result}")
assert result, "Anchor detection failed!"
print("   ✓ PASSED")


print("\n" + "=" * 60)
print("ALL 8 DETECTORS PASSED!")
print("=" * 60)
