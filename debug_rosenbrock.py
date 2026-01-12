"""Debug why Rosenbrock detection isn't firing."""
import numpy as np

# Sample of the Rosenbrock data (low values)
data_points = [
    # (x, y, z)
    (1, 1, 0),      # Minimum!
    (2, 4, 1),      # y = x²
    (0, 0, 1),      # y = x²
    (-1, 1, 4),     # y = x²
    (-2, 4, 9),     # y = x²
    (3, 9, 4),      # y = x²
    (-3, 9, 16),    # y = x²
    (0, 1, 101),
    (1, 2, 100),
    (1, 0, 100),
    (0, -1, 101),
]

# Build data_map
data_map = {}
for x_val, y_val, z_val in data_points:
    data_map[(round(x_val, 4), round(y_val, 4))] = z_val

print(f"Data map has {len(data_map)} entries")
print(f"Values: {sorted(data_map.values())}")

# Calculate low_threshold
low_threshold = sorted(data_map.values())[min(10, len(data_map) - 1)]
print(f"Low threshold (10th smallest): {low_threshold}")

# Get low points
low_points = [(k, v) for k, v in data_map.items() if v <= low_threshold]
print(f"Low points ({len(low_points)}):")
for (x_val, y_val), z_val in low_points:
    print(f"  f({x_val}, {y_val}) = {z_val}, x² = {x_val**2}, y-x² = {y_val - x_val**2}")

# Check valley pattern
valley_matches = 0
for (x_val, y_val), z_val in low_points:
    diff = abs(y_val - x_val**2)
    threshold = 0.1 * max(1, abs(y_val))
    match = diff < threshold
    print(f"  Check y≈x²: |{y_val} - {x_val**2}| = {diff} < {threshold}? {match}")
    if match:
        valley_matches += 1

valley_ratio = valley_matches / len(low_points) if low_points else 0
print(f"\nValley matches: {valley_matches}/{len(low_points)} = {valley_ratio:.2%}")
print(f"Detection threshold: 50%")
print(f"Would detect: {valley_ratio >= 0.5}")
