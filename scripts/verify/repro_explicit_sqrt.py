
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from kalkulator_pkg.heuristics import generate_candidate_features

def test_explicit_sqrt_generation():
    print("Testing explicit sqrt(x) generation...")
    X = np.linspace(1, 10, 10).reshape(-1, 1)
    # y doesn't matter for feature generation
    
    features, names = generate_candidate_features(X, ['x'], include_transcendentals=True)
    
    print(f"Generated features: {names}")
    
    if "sqrt(x)" in names:
        print("PASS: Generated 'sqrt(x)'")
    else:
        print("FAIL: 'sqrt(x)' not found in features")
        
    if "x^1.5" in names:
        print("PASS: Generated 'x^1.5'") 
    else:
        print("FAIL: 'x^1.5' not found in features")

def test_explicit_sqrt_negative_rejection():
    print("\nTesting negative input rejection...")
    X = np.linspace(-10, -1, 10).reshape(-1, 1)
    
    features, names = generate_candidate_features(X, ['x'], include_transcendentals=True)
    
    if "sqrt(x)" in names:
        print("FAIL: Generated 'sqrt(x)' for negative input")
    else:
        print("PASS: Rejected 'sqrt(x)' for negative input")

if __name__ == "__main__":
    test_explicit_sqrt_generation()
    test_explicit_sqrt_negative_rejection()
