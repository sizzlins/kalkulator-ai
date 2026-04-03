
import numpy as np
import sys

def check_numpy():
    print(f"Numpy Version: {np.__version__}")
    
    # Create complex array with NaNs/Infs
    c = np.array([1+1j, np.nan+0j, np.inf+1j, 1-1j])
    print("Original:", c)
    
    try:
        clean = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        print("Cleaned:", clean)
        print("PASS: np.nan_to_num supports complex.")
    except Exception as e:
        print(f"FAIL: np.nan_to_num crashed: {e}")
        
    # Check if 'nan' arg is supported (added in 1.17)
    try:
        clean = np.nan_to_num(c, nan=1e9)
    except TypeError:
        print("FAIL: np.nan_to_num does not support 'nan' keyword (old numpy?)")

if __name__ == "__main__":
    check_numpy()
