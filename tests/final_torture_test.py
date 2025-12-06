
import sys
import os
import math
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from kalkulator_pkg.function_manager import find_function_from_data

def test_case(name, data_points, param_names, expected_pattern):
    print(f"\n--- Testing {name} ---")
    try:
        success, func_str, _, mse = find_function_from_data(data_points, param_names)
        if success:
            print(f"Result: {func_str}")
            # Normalize strings for basic check (remove spaces)
            res_clean = func_str.replace(" ", "")
            pass_check = True
            
            # Key features check
            if expected_pattern == "rho*u*L/mu":
                if "rho*u*L/mu" not in res_clean and "rho*L*u/mu" not in res_clean: pass_check = False
            elif expected_pattern == "9000*q1*q2/r^2":
                if not ("9000" in res_clean and "/r^2" in res_clean): pass_check = False
            elif expected_pattern == "exp(-t)*cos(2*t)":
                if "exp(-t)*cos(2*t)" not in res_clean: pass_check = False
            elif expected_pattern == "-5*x*log(x)":
                # Loose check for -4.999
                if not ("x*log(x)" in res_clean and ("-5" in res_clean or "-4.999" in res_clean)): pass_check = False
            elif "r^2*h" in expected_pattern: # Cone
                if "r^2*h" not in res_clean and "h*r^2" not in res_clean: pass_check = False
            elif "-t^2+x^2+y^2+z^2" in expected_pattern: # Spacetime
                if "-t^2" not in res_clean or "x^2" not in res_clean or "z^2" not in res_clean: pass_check = False
                if "z^3" in res_clean or "exp" in res_clean: pass_check = False
            elif "2*x^3" in expected_pattern:
                if "2*x^3" not in res_clean: pass_check = False

            if pass_check:
                print("STATUS: PASS")
            else:
                print(f"STATUS: FAIL (Expected pattern '{expected_pattern}' not found or forbidden terms present)")
        else:
            print("STATUS: FAIL (No function found)")
    except Exception as e:
        print(f"STATUS: CRASH ({e})")
        import traceback
        traceback.print_exc()

def run_tests():
    # 1. Reynolds (R = rho u L / mu)
    data_R = [
        ([1,1,1,1], 1), ([10,2,5,2], 50), ([5,100,0.1,0.5], 100), ([2,50,2,10], 20), ([8,10,2,4], 40)
    ]
    test_case("Reynolds", data_R, ["rho", "u", "L", "mu"], "rho*u*L/mu")

    # 2. Coulomb (F = 9000 q1 q2 / r^2)
    data_F = [
        ([1,1,1], 9000), ([1,1,3], 1000), ([2,2,2], 9000), ([0,5,10], 0), ([5,4,10], 1800)
    ]
    test_case("Coulomb", data_F, ["q1", "q2", "r"], "9000*q1*q2/r^2")

    # 3. Damped Oscillation (x = exp(-t) cos(2t))
    data_x = [
        ([0], 1), ([1.570796], -0.20788), ([3.14159], 0.043214), ([0.785398], 0), ([2], 0.0886)
    ]
    test_case("Damped", data_x, ["t"], "exp(-t)*cos(2*t)")

    # 4. Entropy (S = -5 x log(x))
    data_S = [
        ([1], 0), ([2.71828], -13.5914), ([2], -6.93147), ([10], -115.129), ([0.5], 1.73286)
    ]
    test_case("Entropy", data_S, ["x"], "-5*x*log(x)")

    # 5. Cone Volume (V = 1/3 pi r^2 h)
    data_V = [
        ([3,10], 94.2477796), ([1,3], 3.14159265), ([0,10], 0), ([6,2], 75.3982237), ([2,6], 25.1327412)
    ]
    test_case("Cone", data_V, ["r", "h"], "pi*r^2*h")

    # 6. Spacetime (s = -t^2 + x^2 + y^2 + z^2)
    # Corrected data point -88 based on formula assumption
    data_s = [
        ([10,2,2,2], -88), ([0,3,4,0], 25), ([1,1,1,1], 2), ([5,0,0,0], -25), ([2,5,5,5], 71)
    ]
    test_case("Spacetime", data_s, ["t", "x", "y", "z"], "-t^2+x^2+y^2+z^2")

    # 7. Cubic (y = 2 x^3)
    data_y = [
        ([1], 2), ([2], 16), ([3], 54), ([4], 128), ([5], 250)
    ]
    test_case("Cubic", data_y, ["x"], "2*x^3")

if __name__ == "__main__":
    run_tests()
