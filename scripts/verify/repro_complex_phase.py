
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from kalkulator_pkg.symbolic_regression.forensic_analysis import detect_complex_phase_patterns
from kalkulator_pkg.symbolic_regression.forensic_analysis import generate_pattern_seeds as find_symbolic_structure

def test_complex_phase_logic():
    print("Testing detect_complex_phase_patterns logic...")
    # Generate data for exp(sqrt(x))
    # For x < 0: sqrt(x) = i*sqrt(|x|)
    # exp(i*sqrt(|x|)) has phase sqrt(|x|)
    
    X_neg = np.linspace(-10, -1, 20).reshape(-1, 1)
    # y = exp(sqrt(x))
    # numpy sqrt(-x) returns nan, need complex type
    X_complex = X_neg.astype(complex)
    y_complex = np.exp(np.sqrt(X_complex))
    
    seeds = detect_complex_phase_patterns(X_neg, y_complex, ['x'])
    print(f"Seeds found: {seeds}")
    
    if "exp(sqrt(x))" in seeds:
        print("PASS: Logic detected 'exp(sqrt(x))'")
    else:
        print("FAIL: Logic failed to detect 'exp(sqrt(x))'")

def test_integration():
    print("\nTesting integration in find_symbolic_structure...")
    # Provide mixed data (positive and negative) to see if detector triggers
    X = np.linspace(-5, 5, 20).reshape(-1, 1)
    X_complex = X.astype(complex)
    y = np.exp(np.sqrt(X_complex))
    
    print(f"y in repro shape: {y.shape}, dtype: {y.dtype}")
    print(f"y in repro max imag: {np.max(np.abs(np.imag(y)))}")
    # print(f"y content: {y}")
    
    # find_symbolic_structure expects (ctx, X, y, variable_names, ...)
    
    try:
        seeds = find_symbolic_structure(None, X, y, ['x'], verbose=True)
        # It returns tuple (seeds, best_match)? Or list?
        # Definition: return seeds (list) or tuple.
        
        extracted_seeds = []
        if isinstance(seeds, tuple):
             extracted_seeds = seeds[0]
        else:
             extracted_seeds = seeds
             
        print(f"All seeds from structure finding: {extracted_seeds}")
        
        if any("exp(sqrt(x))" in s for s in extracted_seeds):
            print("PASS: Integrated detector found 'exp(sqrt(x))'")
        else:
            print("FAIL: Integrated detector missed 'exp(sqrt(x))'")
            
    except Exception as e:
        print(f"ERROR during integration test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # test_complex_phase_logic()
    test_integration()
