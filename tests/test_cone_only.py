import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from kalkulator_pkg.function_manager import find_function_from_data

# Cone Volume (V = 1/3 pi r^2 h)
data_V = [
    ([3, 10], 94.2477796),
    ([1, 3], 3.14159265),
    ([0, 10], 0),
    ([6, 2], 75.3982237),
    ([2, 6], 25.1327412),
]
success, func_str, coeffs, mse = find_function_from_data(data_V, ["r", "h"])
print(f"Cone Result: {func_str}")
print(f"Expected: pi*r^2*h (or 1/3*pi*r^2*h)")
if func_str and (
    "r^2*h" in func_str.replace(" ", "") or "h*r^2" in func_str.replace(" ", "")
):
    print("STATUS: PASS")
else:
    print("STATUS: FAIL")
