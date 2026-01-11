 """
Reproduce the solver bug where valid equations return "Contradiction (numeric)"
"""
import sympy as sp

# Test case 1: 1326784*x = 35.4607291932873
print("=" * 60)
print("Test 1: 1326784*x = 35.4607291932873")
print("=" * 60)

x = sp.Symbol('x')
eq = sp.Eq(1326784 * x, 35.46072919328738)
print(f"Equation: {eq}")

# Try sp.solve
try:
    sols = sp.solve(eq, x)
    print(f"sp.solve result: {sols}")
    print(f"Type: {type(sols)}")
    if sols:
        for sol in sols:
            print(f"  Solution: {sol}")
            print(f"  Numerical: {float(sol)}")
except Exception as e:
    print(f"sp.solve FAILED: {e}")

print()

# Manual calculation
expected = 35.4607291932873 / 1326784
print(f"Manual calculation: x = {expected}")

# Verify
print(f"Verification: 1326784 * {expected} = {1326784 * expected}")

print("\n" + "=" * 60)
print("Test 2: 1326784/x = x")
print("=" * 60)

eq2 = sp.Eq(1326784 / x, x)
print(f"Equation: {eq2}")

try:
    sols2 = sp.solve(eq2, x)
    print(f"sp.solve result: {sols2}")
    if sols2:
        for sol in sols2:
            print(f"  Solution: {sol}")
            try:
                print(f"  Numerical: {float(sp.N(sol))}")
            except:
                print(f"  Numerical: {sp.N(sol)}")
except Exception as e:
    print(f"sp.solve FAILED: {e}")

print()
# Manual calculation
expected2 = sp.sqrt(1326784)
print(f"Manual calculation: x = ±√1326784 = ±{float(expected2)}")

# Verify
print(f"Verification: 1326784/{float(expected2)} = {1326784/float(expected2)}")
