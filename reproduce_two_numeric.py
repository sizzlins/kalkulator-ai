
import sys
import os
import math

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.function_manager import find_function_from_data

def test_two_numeric_points():
    print("Testing with 2 numeric points (no f(0)=0)...")
    
    # User's failing case: 2 numeric points + symbolic
    data_points = [
        # Numeric (only 2!)
        (['1'], str(math.sin(1))),
        (['-1'], str(math.sin(-1))),
        
        # Symbolic
        (['x'], '0'),
        (['x/2'], '1'),
        (['x/3'], '0.866025403784439'),
        (['x/4'], '0.707106781186548'),
        (['x/6'], '0.5')
    ]
    param_names = ['val']
    
    print(f"Feeding {len(data_points)} points (2 numeric, 5 symbolic)...")
    
    result = find_function_from_data(data_points, param_names)
    
    print("\nResult:")
    print(f"Function: {result[1]}")
    print(f"Message: {result[3]}")
    
    func_str = str(result[1])
    
    if "sin" in func_str and "0.84" not in func_str:
        print("PASS: Found sine function.")
    else:
        print("FAIL: Did not find sine function or got linear trap.")

if __name__ == "__main__":
    test_two_numeric_points()
