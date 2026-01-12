"""Test that polyfit results now include R² in confidence note."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

from kalkulator_pkg.function_manager import find_function_from_data

# Test case: x^2 + 4x - 221
data_points = [
    ([1], -216),
    ([2], -209),
    ([3], -200),
    ([4], -189),
    ([5], -176),
    ([6], -161),
]

print("Testing polyfit R² inclusion...")
print("="*70)

success, func_str, factored, confidence_note = find_function_from_data(
    data_points, ['x']
)

print(f"\nResult:")
print(f"  Success: {success}")
print(f"  Function: {func_str}")
print(f"  Factored: {factored}")
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
            print(f"   ✅ R² > 0.7: Seed would be ACCEPTED by hybrid mode!")
        else:
            print(f"   ❌ R² <= 0.7: Seed would be rejected")
else:
    print("❌ FAIL: R² not found in confidence note")
    print(f"   Confidence note: {repr(confidence_note)}")
