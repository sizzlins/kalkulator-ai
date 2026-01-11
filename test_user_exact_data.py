"""Test with the EXACT data from user's log."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

from kalkulator_pkg.function_manager import find_function_from_data

# User's exact function: f(x)=x^2+4x-221
# With 66 points including complex and irrational values
data_points = []

# Generate all integer points from -20 to 20
for x in range(-20, 21):
    y = x**2 + 4*x - 221
    data_points.append(([x], y))

# Add some irrational/transcendental points
import math
extra_points = [
    ([math.e], math.e**2 + 4*math.e - 221),
    ([math.pi], math.pi**2 + 4*math.pi - 221),
    ([0.1], 0.1**2 + 4*0.1 - 221),
    ([0.2], 0.2**2 + 4*0.2 - 221),
    ([0.3], 0.3**2 + 4*0.3 - 221),
    ([1.1], 1.1**2 + 4*1.1 - 221),
    ([1.2], 1.2**2 + 4*1.2 - 221),
    ([1.3], 1.3**2 + 4*1.3 - 221),
]
data_points.extend(extra_points)

print(f"Testing with {len(data_points)} points...")
print("Function: x^2 + 4x - 221")
print("="*70)

success, func_str, factored, confidence_note = find_function_from_data(
    data_points, ['x']
)

print(f"\nResult:")
print(f"  Success: {success}")
print(f"  Function: {func_str}")
print(f"  Factored: {repr(factored)}")
print(f"  Confidence: {repr(confidence_note)}")
print()

if confidence_note and "R²=" in str(confidence_note):
    print("✅ SUCCESS: R² is now included in confidence note!")
    print(f"   Confidence note: {confidence_note}")
    
    # Extract R²
    import re
    r2_match = re.search(r"R²=([\d.]+)", str(confidence_note))
    if r2_match:
        r_squared = float(r2_match.group(1))
        print(f"   Extracted R²: {r_squared}")
        
        if r_squared > 0.7:
            print(f"   ✅ R² > 0.7: Perfect seed would be ACCEPTED by hybrid mode!")
else:
    print("❌ FAIL: R² not found in confidence note")
    print(f"   Confidence note: {repr(confidence_note)}")
