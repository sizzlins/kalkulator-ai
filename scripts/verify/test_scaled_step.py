
import numpy as np
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.symbolic_regression.forensic_analysis import generate_pattern_seeds

def test_scaled_steps():
    print("Testing scaled step functions...")
    
    # Test 1: ceil(x/2)
    # x: 0, 1, 2, 3, 4, 5, 6
    # y: 0, 1, 1, 2, 2, 3, 3 ... wait ceil(0)=0, ceil(0.5)=1, ceil(1)=1
    # Let's generate data typical of what user provided.
    # User had f(1)=1, f(2)=1... maybe distinct values?
    # User input: f(1)=1, f(2)=1, f(3)=2, f(4)=2, f(5)=3, f(6)=3...
    # This is consistent with ceil(x/2).
    # x=1 -> 0.5 -> ceil -> 1
    # x=2 -> 1.0 -> ceil -> 1
    # x=3 -> 1.5 -> ceil -> 2
    
    x_vals = np.arange(1, 22)
    y_vals = np.ceil(x_vals / 2.0)
    
    X = x_vals.reshape(-1, 1)
    
    print("\n--- Test Case 1: ceil(x/2) ---")
    seeds = generate_pattern_seeds(None, X, y_vals, variable_names=['x'], verbose=True)
    print(f"Seeds: {seeds}")
    
    # Check types because sometimes we get lists of lists?
    flat_seeds = []
    for s in seeds:
        if isinstance(s, list):
            flat_seeds.extend(map(str, s))
        else:
            flat_seeds.append(str(s))
            
    print(f"Flat Seeds: {flat_seeds}")
    
    if any("ceil(x/2" in s.replace(" ","") for s in flat_seeds):
        print("PASS: Found ceil(x/2) pattern")
    else:
        print("FAIL: Did not find ceil(x/2)")
        
    # Test 2: floor(x/3)
    x_vals2 = np.arange(0, 30)
    y_vals2 = np.floor(x_vals2 / 3.0)
    X2 = x_vals2.reshape(-1, 1)
    
    print("\n--- Test Case 2: floor(x/3) ---")
    seeds2 = generate_pattern_seeds(None, X2, y_vals2, variable_names=['t'], verbose=True)
    print(f"Seeds: {seeds2}")
    
    flat_seeds2 = []
    for s in seeds2:
        if isinstance(s, list):
            flat_seeds2.extend(map(str, s))
        else:
            flat_seeds2.append(str(s))
            
    print(f"Flat Seeds: {flat_seeds2}")
    
    if any("floor(t/3)" in s.replace(" ","") for s in flat_seeds2):
        print("PASS: Found floor(t/3) pattern")
    else:
        print("FAIL: Did not find floor(t/3)")

if __name__ == "__main__":
    test_scaled_steps()
