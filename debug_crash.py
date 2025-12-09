
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.function_manager import find_function_from_data
# from kalkulator_pkg.parser import parse_input (Removed)

def reproduce_crash():
    print("Reproducing crash with symbolic args in find_function_from_data...")
    
    # Simulating the user input: f(x)=0, f(x/2)=1, f(x/4)=0.707...
    # The parser in app.py would extract data points.
    
    # Based on app.py logic, find_function_from_data expects:
    # data_points = [ ([arg1, arg2], value), ... ]
    # param_names = ['x']
    
    data_points = [
        (['x'], '0'), 
        (['x/2'], '1'), 
        (['x/4'], '0.707106781186548')
    ]
    param_names = ['x']
    
    # print(f"Data points: {data_points}")
    
    try:
        find_function_from_data(data_points, param_names)
        print("Success: No crash!?")
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_crash()
