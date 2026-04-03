
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.heuristics import generate_candidate_features, detect_poles_from_data

def test_multivariate_pole_blindness():
    print("Testing Multivariate Pole Blindness...")
    
    # 1. Generate data for f(x, y) = 1/(y - 2)
    # x is irrelevant, y has a pole at 2
    
    # Generate 50 random points
    np.random.seed(42)
    N = 100
    x_rand = np.random.uniform(0, 10, N)
    
    # Generate y values in range [0, 4], avoiding exactly 2.0 initially
    y_rand = np.random.uniform(0.1, 3.9, N)
    
    # Enforce some points CLOSE to 2.0 to give signal
    y_rand[:10] = np.linspace(2.01, 2.1, 10)
    y_rand[10:20] = np.linspace(1.9, 1.99, 10)
    
    # Calculate Z for random points
    # z = 1/(y-2)
    z_rand = 1.0 / (y_rand - 2.0)
    
    # Add pole points (Huge/Inf)
    y_near_pole = np.array([2.0 + 1e-10, 2.0 - 1e-10, 2.0 + 1e-12, 2.0]) # 2.0 will give inf
    x_dummy = np.ones_like(y_near_pole)
    
    with np.errstate(divide='ignore'):
         z_pole = 1.0 / (y_near_pole - 2.0)
         
    # Combine
    X_normal = np.column_stack([x_rand, y_rand])
    Z_normal = z_rand
    
    X_test = np.vstack([X_normal, np.column_stack([x_dummy, y_near_pole])])
    y_test = np.concatenate([Z_normal, z_pole]) # Output vector
    
    print(f"Test Data Shape: {X_test.shape}")
    print(f"Max y_test value: {np.max(np.abs(y_test[np.isfinite(y_test)]))}")
    print(f"Has Inf/NaN: {np.any(~np.isfinite(y_test))}")
    
    # Debug: Check detect_poles directly
    print("Running detect_poles_from_data directly...")
    poles = detect_poles_from_data(X_test, y_test)
    print(f"Detected Poles: {poles}")
    
    # 2. Run Feature Generation
    try:
        features, feature_names = generate_candidate_features(
            X_test, 
            ['x', 'y'], 
            include_transcendentals=True,
            y_data=y_test,
            X_original=X_test, # Pass for pole detection
            y_original=y_test
        )
        
        print(f"Generated {len(feature_names)} features.")
        
        # Check for expected feature
        pole_found = False
        for name in feature_names:
            # Look for 1/(y-2) or similar
            if "1/(y-2" in name or "1/(y - 2" in name or "1/(y-2.0" in name:
                pole_found = True
                print(f"SUCCESS: Found pole feature '{name}'")
                break
            
            # Also check if it incorrectly found 1/(x-...)
            if "1/(x-" in name:
                print(f"WARNING: Found x-pole '{name}' (likely false positive)")
                
        if not pole_found:
            print("FAILURE: Did not find '1/(y-2)' feature.")
            print("Bug confirmed: Multivariate Pole Blindness.")
        else:
            print("VERIFIED: Fix is working.")
            
    except Exception as e:
        print(f"Crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multivariate_pole_blindness()
