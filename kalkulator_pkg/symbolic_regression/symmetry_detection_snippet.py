import numpy as np

def _detect_symmetry_pole(X, y, variable_names=None, verbose=False):
    """Detect points of symmetry which often indicate poles or centers."""
    if X.ndim > 1 and X.shape[1] > 1: return []
    
    seeds = []
    
    # Sort data
    xy = sorted(zip(X.flatten(), y), key=lambda p: p[0])
    xs = np.array([p[0] for p in xy])
    ys = np.array([p[1] for p in xy])
    
    n_points = len(xs)
    if n_points < 4: return []
    
    # Check each integer and half-integer in range as candidate center
    x_min, x_max = np.min(xs), np.max(xs)
    candidates = []
    for c in np.arange(np.ceil(x_min), np.floor(x_max) + 0.1, 0.5):
        candidates.append(c)
        
    var = variable_names[0] if variable_names else "x"
    
    for c in candidates:
        # Check odd symmetry around c: f(c+h) = -f(c-h)
        # Find pairs (x1, x2) such that (x1+x2)/2 ≈ c
        
        # Simple grid check if points are regularly spaced?
        # Robust check: Interpolate or find closest pairs
        
        errors = []
        count = 0
        
        for i in range(n_points):
            x1 = xs[i]
            if x1 >= c: break # Only check left side
            
            # Find mirror point x2 ≈ 2c - x1
            target_x2 = 2*c - x1
            
            # Find closest actual point
            idx2 = np.argmin(np.abs(xs - target_x2))
            x2 = xs[idx2]
            
            if abs(x2 - target_x2) < 1e-4:
                y1, y2 = ys[i], ys[idx2]
                
                # Odd symmetry: y1 + y2 ≈ 0
                errors.append(abs(y1 + y2))
                count += 1
        
        if count > 2:
            mean_sym_err = np.mean(errors)
            # If symmetry error is small relative to magnitude
            y_mag = np.mean(np.abs(ys))
            if mean_sym_err < 0.05 * y_mag:
                if verbose: print(f"   Symmetry Analysis: Found ODD symmetry check at {c} (err={mean_sym_err:.4f})")
                
                # Check if it looks like a pole (values divergent near c?)
                # Actually, for sin(1/(x-3)), it effectively has odd symmetry if 1/(x-3) maps to odd
                # sin(1/(-h)) = sin(-1/h) = -sin(1/h). Yes!
                
                c_str = str(float(c))
                seeds.append(f"1/({var}-locked({c_str}))")
                seeds.append(f"1/({var}-locked({c_str}))**2")
    
    return seeds
