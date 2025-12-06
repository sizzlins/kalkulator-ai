
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from kalkulator_pkg.function_manager import find_function_from_data

# Test 1: Damped with 8 points (was crashing)
print("--- Test 1: Damped (8 points) ---")
data_x = [
    ([0], 1), ([0.5], 0.32768), ([1.570796], -0.20788), ([3.14159], 0.043214),
    ([0.785398], 0), ([2], -0.08846), ([1], -0.15309), ([0.25], 0.68414)
]
try:
    success, func_str, _, mse = find_function_from_data(data_x, ["t"])
    print(f"Result: {func_str}")
    if func_str and "exp(-t)" in func_str and "cos(2*t)" in func_str:
        print("STATUS: PASS")
    else:
        print("STATUS: PARTIAL (Found complex transient)")
except Exception as e:
    print(f"STATUS: CRASH - {e}")

# Test 2: Entropy with 7 points (was showing -4.999)
print("\n--- Test 2: Entropy (7 points) ---")
data_S = [
    ([1], 0), ([2.71828], -13.5914), ([2], -6.93147), ([10], -115.129),
    ([0.5], 1.73287), ([4], -27.72589), ([3], -16.47918)
]
try:
    success, func_str, _, mse = find_function_from_data(data_S, ["x"])
    print(f"Result: {func_str}")
    if func_str and "-5*x*log(x)" in func_str.replace(" ", ""):
        print("STATUS: PASS")
    else:
        print("STATUS: PARTIAL")
except Exception as e:
    print(f"STATUS: CRASH - {e}")

# Test 3: Cone with 7 points (was showing 0*pi)
print("\n--- Test 3: Cone (7 points) ---")
data_V = [
    ([3,10], 94.24778), ([1,3], 3.14159), ([0,10], 0), ([6,2], 75.39822),
    ([2,6], 25.13274), ([2,9], 37.69911), ([4,3], 50.26548)
]
try:
    success, func_str, _, mse = find_function_from_data(data_V, ["r", "h"])
    print(f"Result: {func_str}")
    if func_str and "1/3*pi*r^2*h" in func_str.replace(" ", "") and "0*pi" not in func_str:
        print("STATUS: PASS")
    elif "0*pi" in func_str:
        print("STATUS: PARTIAL (0*pi still present)")
    else:
        print("STATUS: PARTIAL")
except Exception as e:
    print(f"STATUS: CRASH - {e}")

# Test 4: Spacetime with 7 points
print("\n--- Test 4: Spacetime (7 points) ---")
data_s = [
    ([10,2,2,2], -88), ([0,3,4,0], 25), ([1,1,1,1], 2), ([5,0,0,0], -25),
    ([2,5,5,5], 71), ([4,4,4,4], 32), ([3,0,0,0], -9)
]
try:
    success, func_str, _, mse = find_function_from_data(data_s, ["t", "x", "y", "z"])
    print(f"Result: {func_str}")
    if func_str and "-t^2" in func_str and "x^2" in func_str and "y^2" in func_str and "z^2" in func_str:
        print("STATUS: PASS")
    else:
        print("STATUS: PARTIAL")
except Exception as e:
    print(f"STATUS: CRASH - {e}")

print("\nDone!")
