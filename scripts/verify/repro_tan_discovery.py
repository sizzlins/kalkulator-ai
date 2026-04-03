
import numpy as np
import sys
import os

# Ensure package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.symbolic_regression.forensic_analysis import generate_pattern_seeds

def test_tan_seeds():
    print("Testing generate_pattern_seeds for sin(cos(tan(x)))...")
    
    # Generate data: f(x) = sin(cos(tan(x)))
    # Range [-10, 10] like user
    X = np.linspace(-10, 10, 1000).reshape(-1, 1)
    # Be careful with singularities
    try:
        y = np.sin(np.cos(np.tan(X[:, 0])))
    except:
        y = np.zeros_like(X[:, 0])
        
    print(f"Data range Y: [{np.min(y):.4f}, {np.max(y):.4f}]")
    
    seeds = generate_pattern_seeds(None, X, y, variable_names=['x'], verbose=True)
    
    print("\nSeeds found:")
    found_target = False
    for s in seeds:
        print(f"  - {s}")
        if "sin(cos(tan(x)))" in s:
            found_target = True
            
    if found_target:
        print("\nSUCCESS: Target seed found!")
    else:
        print("\nFAILURE: Target seed NOT found.")
        sys.exit(1)

if __name__ == "__main__":
    test_tan_seeds()
