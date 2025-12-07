import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from kalkulator_pkg.function_manager import find_function_from_data  # noqa: E402

# Reynolds (R = rho u L / mu)
data_R = [
    ([1, 1, 1, 1], 1),
    ([10, 2, 5, 2], 50),
    ([5, 100, 0.1, 0.5], 100),
    ([2, 50, 2, 10], 20),
    ([8, 10, 2, 4], 40),
]
success, func_str, coeffs, mse = find_function_from_data(
    data_R, ["rho", "u", "L", "mu"]
)
print(f"Reynolds Result: {func_str}")
print("Expected: rho*u*L/mu")
if func_str and "rho*u*L/mu" in func_str.replace(" ", ""):
    print("STATUS: PASS")
else:
    print("STATUS: FAIL")
