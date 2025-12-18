import math

from kalkulator_pkg.function_manager import find_function_from_data


def test_composite():
    # f(x) = sin(x) + 1
    # f(0)=1, f(pi/2)=2, f(pi)=1
    data = [([0], 1), ([math.pi / 2], 2), ([math.pi], 1)]
    print("Running composite test...")
    success, func_str, _, error = find_function_from_data(data, ["x"])
    print(f"Success: {success}")
    print(f"Function: {func_str}")
    
    if "sin(x)" in func_str and "1" in func_str:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    test_composite()
