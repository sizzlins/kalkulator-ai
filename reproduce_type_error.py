
import sys
import os
import sympy as sp

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.function_manager import find_function_from_data

def test_type_error():
    print("Testing TypeError crash (Symbolic Object passed to float())...")
    
    # Simulate data points where x input is ALREADY a SymPy object (not string)
    # This matches the stack trace indicating float() failed on an expression
    x = sp.Symbol('x')
    
    data_points = [
        ([x], '0'), 
        ([x/2], '1')
    ]
    param_names = ['x']
    
    print(f"Data points input types: {[type(p[0][0]) for p in data_points]}")
    
    try:
        find_function_from_data(data_points, param_names)
        print("Success: No crash")
    except TypeError as e:
        print(f"Caught expected TypeError: {e}")
    except ValueError as e:
        print(f"Caught ValueError instead: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_type_error()
