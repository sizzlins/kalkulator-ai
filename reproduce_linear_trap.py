
import sys
import os
import math

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.function_manager import find_function_from_data

def test_linear_trap():
    print("Testing Linear Trap (Sparse Numeric sin(x) vs Symbolic Constraints)...")
    
    # User's EXACT sparse data:
    data_points = [
        # Numeric
        (['0'], '0'),
        (['1'], str(math.sin(1))),  # 0.84147...
        (['-1'], str(math.sin(-1))), # -0.84147...
        
        # Symbolic
        (['x'], '0'),
        (['x/2'], '1'),
        (['x/3'], '0.866025403784439'),
        (['x/4'], '0.707106781186548'),
        (['x/6'], '0.5')
    ]
    param_names = ['val']
    
    print(f"Feeding {len(data_points)} mixed points (3 numeric, 5 symbolic)...")
    
    result = find_function_from_data(data_points, param_names)
    
    print("\nResult:")
    print(f"Function: {result[1]}")
    print(f"Message: {result[3]}")
    
    func_str = str(result[1])
    
    # We expect rejection of linear fit 0.84*x
    if "sin" in func_str:
        print("PASS: Found sine function.")
    elif "x" in func_str and "sin" not in func_str:
        print("FAIL: Stuck in Linear Trap (y = ax).")
    else:
        print("FAIL: Undetermined.")

if __name__ == "__main__":
    test_linear_trap()
