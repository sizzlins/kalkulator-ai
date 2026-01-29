
import numpy as np
from kalkulator_pkg.symbolic_regression.forensic_analysis import generate_pattern_seeds

# Test case: sin(x^2)
X = np.array([[1.77245385], [2.50662827], [3.06998012], [3.5449077]])
y = np.array([0.0, 0.0, 0.0, 0.0]) # Perfect zeros at these points

class MockContext:
    pass

print("Testing Chirp Analysis...")
# Pass explicit variable name to match expected output
seeds = generate_pattern_seeds(MockContext(), X, y, variable_names=["x"], verbose=True)
print("Seeds found:", seeds)

if "sin(x**2)" in seeds:
    print("SUCCESS: Detected sin(x^2)")
else:
    print("FAILURE: Did not detect sin(x^2)")
