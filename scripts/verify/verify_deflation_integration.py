
import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.symbolic_regression.forensic_analysis import generate_pattern_seeds

class MockContext:
    def __init__(self):
        self.banned_operators = []

def verify_deflation_integration():
    print("Verifying Deflation Integration...")
    
    # 1. Generate Composite Data: f(x) = exp(sqrt(x)) * log(x^2 + 1)
    # Range should include negative numbers (complex domain) and positive (real domain)
    # The heuristic needs negative inputs to trigger Phase Analysis.
    x = np.linspace(-10, 10, 200)
    X = x.reshape(-1, 1)
    
    # y = exp(sqrt(x)) * log(x^2 + 1)
    # Handle complex computation safely
    sqrt_x = np.sqrt(x.astype(complex))
    y = np.exp(sqrt_x) * np.log(x**2 + 1)
    
    ctx = MockContext()
    var_names = ["x"]
    
    # 2. Run Forensic Analysis
    t0 = time.time()
    seeds = generate_pattern_seeds(ctx, X, y, variable_names=var_names, verbose=True)
    dt = time.time() - t0
    
    print(f"\nAnalysis time: {dt:.4f}s")
    print(f"Discovered Seeds: {seeds}")
    
    # 3. Check for Composite Seed
    # We expect something like 'exp(sqrt(x)) * log(x^2 + 1.0)'
    # Or simplified variants.
    
    found_phase = False
    found_composite = False
    
    for s in seeds:
        if "exp(sqrt(x))" in s:
            found_phase = True
            if "log" in s and "x^2" in s:
                found_composite = True
                print(f"MATCH: {s}")
                
    if found_composite:
        print("[PASS] Successfully found composite function via deflation!")
    elif found_phase:
        print("[PARTIAL] Found phase component but MISSED composite.")
        # Debug why? 
        # Check logs if captured (verbose=True above prints to stdout)
        sys.exit(1)
    else:
        print("[FAIL] Completely missed the structure.")
        sys.exit(1)

if __name__ == "__main__":
    verify_deflation_integration()
