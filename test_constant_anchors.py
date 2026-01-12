"""Test constant anchor detection on user's (1+x)^(1/x) data."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

import numpy as np
from kalkulator_pkg.symbolic_regression.constant_anchors import detect_anchors, generate_hypotheses

# User's data for f(x) = (1+x)^(1/x)
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 1.73205080756888, 1.5874010519682, 1.49534878122122])

print("="*70)
print("Testing Constant Anchor Detection")
print("="*70)
print("Target function: (1+x)^(1/x)")
print()

# Detect anchors
anchors = detect_anchors(X, y, tolerance=1e-3)

print(f"Detected {len(anchors)} constant anchors:")
for x_int, name, value in anchors:
    print(f"  f({x_int}) = {name} ≈ {value:.6f}")
print()

# Generate hypotheses
if anchors:
    hypotheses = generate_hypotheses(anchors, 'x')
    print(f"Generated {len(hypotheses)} hypothesis expressions:")
    for h in hypotheses:
        print(f"  {h}")
    print()
    
    # Check if target is in hypotheses
    target = "(x+1)**(1/x)"
    if target in hypotheses or "(x+1)**(1/x)" in [h.replace(" ", "") for h in hypotheses]:
        print("✅ SUCCESS: Target expression found in hypotheses!")
    else:
        print("⚠️  Target not found, but similar structures generated")
        print(f"    Looking for: {target}")
else:
    print("❌ No anchors detected")
