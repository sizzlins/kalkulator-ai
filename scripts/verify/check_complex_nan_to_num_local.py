
import numpy as np
import warnings

def test_nan_to_num():
    print(f"Numpy version: {np.__version__}")
    
    # Test 1: Complex numbers with NaNs
    x = np.array([1+1j, np.nan, 2-2j])
    print(f"\nEncoding: {x.dtype}")
    y = np.nan_to_num(x, nan=1e9)
    print(f"Result: {y}")
    print(f"Is complex: {np.iscomplexobj(y)}")
    
    # Test 2: Complex numbers without NaNs
    x2 = np.array([1+1j, -4+0j])
    print(f"\nEncoding: {x2.dtype}")
    y2 = np.nan_to_num(x2, nan=1e9)
    print(f"Result: {y2}")
    print(f"Is complex: {np.iscomplexobj(y2)}")
    
    # Test 3: What if we pass float array with negatives to safe_sqrt manually?
    x3 = np.array([-1.0, 4.0])
    res = np.lib.scimath.sqrt(x3)
    print(f"\nSqrt(-1, 4): {res}")
    print(f"Is complex: {np.iscomplexobj(res)}")
    
    y3 = np.nan_to_num(res, nan=1e9)
    print(f"Result: {y3}")
    print(f"Is complex: {np.iscomplexobj(y3)}")

if __name__ == "__main__":
    test_nan_to_num()
