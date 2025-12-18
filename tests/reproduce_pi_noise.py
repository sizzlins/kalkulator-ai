import math

from kalkulator_pkg.function_manager import find_function_from_data


def test_pi_noise():
    # f(x, y) = pi/2 * (1 + x^2 + y^2)
    data = [
        ([0, 0], 1.5707963268),
        ([1, 0], 3.1415926536),
        ([0, 1], 3.1415926536),
        ([1, 1], 4.7123889804),
        ([2, 2], 14.1371669412),
        ([3, 4], 40.8407044967),
        ([-2, 2], 14.1371669412),
        ([10, 10], 315.7300629358),
        ([0.5, 0.5], 2.3561944902),
        ([1.5, 2], 11.3882733693)
    ]
    
    print("Running function finder...")
    success, func_str, _, error = find_function_from_data(data, ["x", "y"])
    
    print(f"Success: {success}")
    print(f"Function: {func_str}")
    
    if "pi" in func_str:
        print("PASS: Found pi symbolic")
    else:
        print("FAIL: No 'pi' symbol found")
        
    # Check for noise
    if "cos" in func_str or "sin" in func_str or "exp" in func_str:
        print("FAIL: Ghost noise found (cos/sin/exp)")
    else:
        print("PASS: No ghost noise")

if __name__ == "__main__":
    test_pi_noise()
