"""Test finding (1+x)^(1/x) from user's data."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

# Test data for f(x) = (1+x)^(1/x)
# This approaches e≈2.718 as x→0
test_data = [
    (4.5, 1.46057896631733),
    (4.0, 1.49534878122122),
    (3.5, 1.53685235445298),
    (3.0, 1.5874010519682),
    (2.5, 1.65054442394899),
    (2.0, 1.73205080756888),
    (1.5, 1.84201574932019),
    (1.0, 2.0),
]

print("Testing function discovery for f(x) = (1+x)^(1/x)")
print("="*70)
print("\nData sample (8 strategic points):")
for x, y in test_data:
    print(f"  f({x}) = {y}")

print("\n" + "="*70)
print("Expected: f(x) = (1+x)^(1/x) or (x+1)**(1/x)")
print("\nThis is a HARD problem - transcendental function with fractional exponent")
print("Will need: evolve with transcendental features enabled")
print("\nRecommended command:")
print("  all " + ", ".join([f"f({x})={y}" for x, y in test_data]))
