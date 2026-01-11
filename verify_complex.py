
import numpy as np

def verify_complex_relu():
    x = 1j # i
    # Formula: (x + |x|) / 2
    res = (x + np.abs(x)) / 2
    
    print(f"x = {x}")
    print(f"|x| = {np.abs(x)}")
    print(f"(x + |x|) / 2 = {res}")
    
    expected = 0.5 + 0.5j
    print(f"Expected: {expected}")
    
    if np.isclose(res, expected):
        print("MATCH")
    else:
        print("NO MATCH")
        
if __name__ == "__main__":
    verify_complex_relu()
