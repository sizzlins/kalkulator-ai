"""End-to-end test for Bessel J0 discovery with pattern detection."""
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())

# User's Bessel J0 data
data = [
    (0.0, 1.0),
    (0.5, 0.93847),
    (1.0, 0.7652),
    (1.5, 0.51183),
    (2.0, 0.22389),
    (2.5, -0.04838),
    (3.0, -0.26005),
    (3.5, -0.38013),
    (4.0, -0.39715),
    (4.5, -0.32054),
    (5.0, -0.1776),
    (5.5, -0.00684),
    (6.0, 0.15065),
    (6.5, 0.26009),
    (7.0, 0.30008),
]

X = np.array([d[0] for d in data]).reshape(-1, 1)
y = np.array([d[1] for d in data])

print("=" * 60)
print("BESSEL J0 DISCOVERY TEST")
print("=" * 60)

# Test 1: Pattern Detection
print("\n[TEST 1] Pattern Detection")
from kalkulator_pkg.cli.repl_commands import _detect_bessel_patterns

detected = _detect_bessel_patterns(X, y)
print(f"Detected patterns: {detected}")

if "bessel_j0(x)" in detected:
    print("✓ PASS: Bessel J0 pattern detected!")
else:
    print("✗ FAIL: Bessel J0 pattern NOT detected")

# Test 2: Genetic Engine Discovery (with seed)
print("\n[TEST 2] Genetic Engine Discovery")
from kalkulator_pkg.symbolic_regression import GeneticSymbolicRegressor, GeneticConfig

config = GeneticConfig(
    population_size=100,
    generations=20,
    timeout=15,
    verbose=False,
    seeds=detected if detected else [],
)

regressor = GeneticSymbolicRegressor(config)
pareto = regressor.fit(X, y, variable_names=["x"])

best = pareto.get_best()
print(f"Best result: {best.expression if best else 'None'}")
print(f"MSE: {best.mse if best else 'N/A'}")

if best and ("bessel" in best.expression.lower() or "j0" in best.expression.lower()):
    print("✓ PASS: Bessel J0 discovered!")
else:
    print("✗ Check result manually")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
