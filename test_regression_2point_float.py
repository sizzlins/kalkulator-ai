from kalkulator_pkg.function_manager import find_function_from_data
import math

def test():
    # Simulate data after app.py converts pi to float
    x1 = -3.0
    y1 = 0.0
    
    x2 = math.pi # 3.14159...
    y2 = 3.0 + math.pi # 6.14159...
    
    # Expected function: f(x) = x + 3
    
    data = [
        ([x1], y1),
        ([x2], y2)
    ]
    
    print(f"Data: {data}")
    
    success, func, factored, msg = find_function_from_data(data, ["x"])
    
    print(f"Success: {success}")
    print(f"Func: {func}")
    # print(f"Msg: {msg}")

    if success and ("x" in func and "3" in func):
         print("[PASS] Found linear function")
    else:
         print("[FAIL] Did not find linear function")

if __name__ == "__main__":
    test()
