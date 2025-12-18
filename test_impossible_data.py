
import os
import sys

import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.function_manager import find_function_from_data


def test_impossible_data():
    print("Testing Mission Impossible: h(1)=5, h(1)=10")
    # Conflicting data: x=1 has two different y values
    data_h = [
        ([1], 5),
        ([1], 10),
        ([2], 20)
    ]
    param_names_h = ["x"]
    
    try:
        result_h = find_function_from_data(data_h, param_names_h)
        print(f"Result h: {result_h}")
        
        success, func_str, _, error_msg = result_h
        if not success and "Conflicting data points" in (error_msg or ""):
             print("SUCCESS: Gracefully handled conflicting data")
        else:
             print(f"FAILURE: Unexpected result: {result_h}")
             
    except Exception as e:
        print(f"FAILURE: Crashed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_impossible_data()
