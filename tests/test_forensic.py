
import numpy as np
from kalkulator_pkg.symbolic_regression.forensic_analysis import _detect_singularity_periodogram

def test_detector():
    print("Testing _detect_singularity_periodogram...")
    
    # Create data for sin(x/(x-3))
    # Dense grid matches reproduction script
    X_vals = np.linspace(-20, 20, 257)
    mask = np.abs(X_vals - 3.0) > 1e-9
    X_vals = X_vals[mask]
    X = X_vals.reshape(-1, 1)
    
    # y = sin(x / (x - 3))
    y = np.sin(X_vals / (X_vals - 3.0))
    
    print(f"Data shape: {X.shape}")
    
    # Run detector
    seeds = _detect_singularity_periodogram(X, y, variable_names=["x"], verbose=True)
    
    print(f"\nSeeds found: {seeds}")
    
    if len(seeds) > 0:
        print("SUCCESS: Seeds found.")
    else:
        print("FAILURE: No seeds found.")

if __name__ == "__main__":
    test_detector()
