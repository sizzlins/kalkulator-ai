"""Test bipolar detector with realistic bounded output data."""
import numpy as np
from kalkulator_pkg.symbolic_regression.forensic_analysis import _detect_bipolar_poles

# Create data with bounded output (simulating cos output in [-1, 1])
X = np.array([
    [-5, -5], [-5, -4], [-5, 0],  # y=0 pole
    [-2, -5], [-2, 0],  # x=-2 pole
    [0, 2], [2, 0],  # Special points
    [3, 3], [-3, -3],
])
# Bounded output in [-1, 1] range (simulating cos function)
y = np.array([
    0.96, 0.97, np.nan,  # y=0 is nan
    np.nan, np.nan,  # x=-2 is nan
    1.0, 1.0,  # Special values
    -0.5, -0.8,  # More bounded values
])

print("Testing _detect_bipolar_poles with bounded output:")
print(f"  y range: [{np.nanmin(y)}, {np.nanmax(y)}]")
seeds = _detect_bipolar_poles(X, y, variable_names=['x', 'y'], verbose=True)
print(f"\nGenerated {len(seeds)} seeds")

# Find cos seeds
cos_seeds = [s for s in seeds if 'cos' in s]
print(f"\ncos-wrapped seeds ({len(cos_seeds)}):")
for s in cos_seeds[:10]:
    print(f"  {s}")

# Check for target pattern
target = "cos(16*(atan(y/(x+2))+atan((x-2)/y)))"
if target in seeds:
    print(f"\n✓ EXACT TARGET FOUND: {target}")
else:
    # Check for similar patterns
    similar = [s for s in seeds if 'cos(16' in s and 'atan' in s]
    print(f"\n✗ Exact target not found")
    print(f"Similar patterns with cos(16*...atan...):")
    for s in similar[:5]:
        print(f"  {s}")
