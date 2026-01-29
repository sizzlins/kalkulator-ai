
import sympy as sp
import numpy as np

def test_sympy_funcs():
    print("Testing SymPy Functions with unusual inputs...")
    
    # Test cases that might appear in GP
    inputs = [
        10.5,           # Float
        -5.0,           # Negative float
        1e6 + 0.5,      # Large float
        1.2 + 3.4j,     # Complex (Python)
        sp.Float(1.2) + sp.I * sp.Float(3.4) # SymPy Complex
    ]
    
    funcs = {
        "primepi": sp.primepi,
        "prime": sp.prime,
        "fibonacci": sp.fibonacci,
        "lucas": sp.lucas,
        "erf": sp.erf,
        "gamma": sp.gamma
    }

    for name, func in funcs.items():
        print(f"\n--- Testing {name} ---")
        for x in inputs:
            try:
                print(f"Input: {x}")
                # Some functions might hang or crash here
                res = func(x)
                print(f"Result: {res}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    test_sympy_funcs()
