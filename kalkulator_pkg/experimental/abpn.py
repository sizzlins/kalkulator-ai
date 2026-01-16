# Empty
from kalkulator_pkg.symbolic_regression.constant_anchors import detect_anchors, generate_hypotheses
import numpy as np
X = np.array([[2.0]])
y = np.array([1.732])
anchors = detect_anchors(X, y)
print(f"Anchors: {anchors}")
if anchors:
    hyp = generate_hypotheses(anchors)
    print(f"Hypotheses: {hyp}")