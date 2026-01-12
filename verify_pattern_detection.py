
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalkulator_pkg.cli.repl_commands import generate_pattern_seeds

def test_sin_pole():
    print("Testing sin(1/(x-3))...")
    
    # Generate data for f(x) = sin(1/(x-3))
    # Include the pole at x=3
    X = np.linspace(0, 6, 100)
    y = np.sin(1/(X - 3))
    
    # Manually insert pole/nan at x=3 (or close to it)
    # The actual function at x=3 is undefined.
    # Let's simulate what likely happened in the user's data: explicit complex or nan
    # The REPL command handling normally filters bad values but passed them to us in y
    # Here we simulate the raw values that might trigger the pole detection.
    
    # Actually, let's match the user's data point explicitly mentioned: f(3.0) = nan
    X_pole = np.array([3.0])
    y_pole = np.array([np.nan])
    
    X_combined = np.concatenate([X, X_pole])
    y_combined = np.concatenate([y, y_pole])
    
    # Shuffle to mix it up
    idx = np.random.permutation(len(X_combined))
    X_shuffled = X_combined[idx]
    y_shuffled = y_combined[idx]
    
    seeds = generate_pattern_seeds(X_shuffled, y_shuffled, ['x'])
    
    print(f"Generated {len(seeds)} seeds.")
    found_target = False
    for s in seeds:
        print(f"  - {s}")
        if 'sin(1/(x-(3.0' in s.replace(" ", ""): # Flexible matching for float formatting
            found_target = True
        if 'sin(1/(x-3' in s.replace(" ", ""):
             found_target = True

    if found_target:
        print("\nSUCCESS: Found composed seed for sin(1/(x-3))!")
    else:
        print("\nFAILURE: Did not find composed seed.")

    return found_target

if __name__ == "__main__":
    test_sin_pole()
