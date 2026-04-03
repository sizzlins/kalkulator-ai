
import numpy as np

def check_clip():
    val = 0.5
    clipped = np.clip(val, -100, 100)
    print(f"Input: {val} ({type(val)})")
    print(f"Clipped: {clipped} ({type(clipped)})")
    
    try:
        c = clipped.astype(complex)
        print("Success: .astype(complex) works.")
    except AttributeError:
        print("FAIL: .astype(complex) failed (AttributeError).")

if __name__ == "__main__":
    check_clip()
