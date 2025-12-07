import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from kalkulator_pkg.function_manager import find_function_from_data

# Entropy (S = -5 x log(x))
data_S = [
    ([1], 0),
    ([2.71828], -13.5914),
    ([2], -6.93147),
    ([10], -115.129),
    ([0.5], 1.73286),
]
success, func_str, coeffs, mse = find_function_from_data(data_S, ["x"])
print(f"Entropy Result: {func_str}")
print("Expected: -5*x*log(x)")
if func_str and "x*log(x)" in func_str.replace(" ", ""):
    print("STATUS: PASS")
else:
    print("STATUS: FAIL")
