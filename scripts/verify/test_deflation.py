
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.symbolic_regression.forensic_analysis import detect_forensic_deflation

def test_deflation():
    print("Testing Forensic Deflation...")
    
    # Generate data for f(x) = exp(sqrt(x)) * log(x^2 + 1)
    # Range includes negative numbers for complex behavior
    x = np.linspace(-10, 10, 200)
    # y = exp(sqrt(x)) * log(x^2 + 1)
    
    # Handle complex sqrt
    sqrt_x = np.sqrt(x.astype(complex))
    y = np.exp(sqrt_x) * np.log(x**2 + 1)
    
    # Simulation: We detected 'exp(sqrt(x))' from phase analysis
    seeds = ["exp(sqrt(x))"]
    var_names = ["x"]
    
    # Prepare X (2D array)
    X = x.reshape(-1, 1)
    
    # Run Deflation
    print(f"Input Seed: {seeds}")
    deflated_seeds = detect_forensic_deflation(X, y, seeds, var_names, verbose=True)
    
    print(f"Deflated Seeds: {deflated_seeds}")
    
    expected = "exp(sqrt(x)) * log(1.0*x^2 + 1.0)"
    # Note: formatting might vary "1.0*x^2 + 1.0" or similar
    
    found = False
    for s in deflated_seeds:
        if "log" in s and "x^2" in s and "exp(sqrt(x))" in s:
            found = True
            break
            
    if found:
        print("SUCCESS: Found composite seed!")
    else:
        print("FAILURE: Did not find composite seed.")

if __name__ == "__main__":
    test_deflation()
