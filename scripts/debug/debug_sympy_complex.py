
import sympy as sp
import numpy as np
import sys

def test_complex_handling():
    print("Testing SymPy Complex Handling...")
    
    c1 = 1.23 + 4.56j
    c2 = np.complex128(1.23 + 4.56j)
    
    print(f"Original: {c1}")
    
    try:
        print("Attempting sp.Float(c1)...")
        res = sp.Float(c1)
        print(f"Success sp.Float: {res}")
    except Exception as e:
        print(f"Failed sp.Float: {e}")

    try:
        print("Attempting sp.nsimplify(c1)...")
        # Timeout logic? No, just run it.
        res = sp.nsimplify(c1)
        print(f"Success sp.nsimplify: {res}")
    except Exception as e:
        print(f"Failed sp.nsimplify: {e}")

    # Correct way to handle complex in sympy?
    try:
        print("Attempting manual complex construction...")
        res = sp.Float(c1.real) + sp.I * sp.Float(c1.imag)
        print(f"Success manual: {res}")
    except Exception as e:
        print(f"Failed manual: {e}")

if __name__ == "__main__":
    test_complex_handling()
