
import numpy as np
import sys

def reproduce_crash():
    print("--- Reproducing nan_to_num crash ---")
    
    # Simulate the data from logs
    # [DEBUG Evaluate] X shape: (15, 1), dtype: complex128
    # [DEBUG Evaluate] Post-Broadcast: [ 6.52387639e+00+0.j ...]
    
    arr = np.array([
        6.52387639e+00+0.j, -4.89290729e-02+0.j, -3.80559456e+00+0.j,
        3.35571892e+01+0.j, -6.79570457e+00+0.j, -4.07742274e+00+0.j,
        2.99011001e+01+0.j, -9.24215822e+00+0.j,  4.21333683e+01+0.j,
        -4.62107911e+00+0.j, -2.71828183e-04+0.j, -7.38905610e+00+0.j,
        2.71828183e+00+0.j,  9.06003333e+00+0.j, -2.03871137e+00+0.j
    ]).reshape(15, 1)
    
    print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"Is complex obj? {np.iscomplexobj(arr)}")
    
    try:
        if np.iscomplexobj(arr):
            print("Attempting nan_to_num with complex replacements...")
            res = np.nan_to_num(arr, nan=1e9+0j, posinf=1e9+0j, neginf=1e9+0j)
            print("Success!")
            print(res)
        else:
            print("Array is not complex?!")
            
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()

    # Another case: Mixed nan
    arr_nan = arr.copy()
    arr_nan[0] = np.nan
    try:
        print("\nAttempting nan_to_num with complex replacements (on NaN array)...")
        res = np.nan_to_num(arr_nan, nan=1e9+0j, posinf=1e9+0j, neginf=1e9+0j)
        print("Success!")
    except Exception as e:
        print(f"CRASHED: {e}")

if __name__ == "__main__":
    reproduce_crash()
