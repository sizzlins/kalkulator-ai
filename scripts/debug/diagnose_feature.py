"""Diagnostic script to understand why exp(x)*exp(y) is not being selected."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalkulator_pkg.function_finder_advanced import generate_candidate_features

def diagnose():
    print("Diagnosing exp(x)*exp(y) feature selection...")
    
    # Generate same data as test
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    
    X_data = []
    y_data = []
    for i in range(len(x)):
        for j in range(len(y)):
            val_x = x[i]
            val_y = y[j]
            val_z = np.exp(val_x + val_y)
            X_data.append([val_x, val_y])
            y_data.append(val_z)
    
    X_arr = np.array(X_data)
    y_arr = np.array(y_data)
    
    print(f"Target y range: [{min(y_arr):.4f}, {max(y_arr):.4f}]")
    print(f"y_arr: {y_arr}")
    
    # Generate features
    X_matrix, feature_names = generate_candidate_features(
        X_arr,
        ['x', 'y'],
        include_transcendentals=True,
        y_data=y_arr
    )
    
    print(f"\nGenerated {len(feature_names)} features")
    
    # Find exp(x)*exp(y) index
    target_idx = None
    for i, name in enumerate(feature_names):
        if name == "exp(x)*exp(y)" or name == "exp(y)*exp(x)":
            target_idx = i
            print(f"\nFound target feature at index {i}: '{name}'")
            break
    
    if target_idx is None:
        print("\nERROR: exp(x)*exp(y) NOT FOUND in features!")
        # Print all exp-related features
        print("Exp-related features:")
        for i, name in enumerate(feature_names):
            if "exp" in name:
                print(f"  [{i}] {name}")
        sys.exit(1)
    
    # Check the feature values
    target_feature = X_matrix[:, target_idx]
    print(f"\nTarget feature values: {target_feature}")
    print(f"Expected (y_arr): {y_arr}")
    
    # Check if they match
    error = np.abs(target_feature - y_arr)
    print(f"\nAbsolute error: {error}")
    print(f"Max error: {np.max(error):.2e}")
    
    # Check correlation
    correlation = np.corrcoef(target_feature, y_arr)[0, 1]
    print(f"\nCorrelation with target: {correlation:.6f}")
    
    # Check if feature is constant or near-zero
    print(f"\nFeature stats:")
    print(f"  Mean: {np.mean(target_feature):.6f}")
    print(f"  Std: {np.std(target_feature):.6f}")
    print(f"  Min: {np.min(target_feature):.6f}")
    print(f"  Max: {np.max(target_feature):.6f}")
    
    if np.allclose(target_feature, y_arr, rtol=1e-6):
        print("\nSUCCESS: Feature matches target exactly!")
        sys.exit(0)
    else:
        print("\nWARNING: Feature does NOT match target exactly.")
        sys.exit(1)

if __name__ == "__main__":
    diagnose()
