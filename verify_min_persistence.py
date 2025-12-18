
from kalkulator_pkg.function_manager import define_function, evaluate_function
import sympy as sp

def test_min_persistence():
    print("Testing persistence of min/max functions...")
    
    # 1. Define a function using min
    # piecewise: min(x, 0) is effectively -ReLU(-x) or 0 for x>0, x for x<0
    func_name = "g"
    inputs = ["x"]
    expr_str = "min(x, 0) + max(x, 1)"
    
    try:
        print(f"Defining {func_name}(x) = {expr_str}")
        define_function(func_name, inputs, expr_str)
        print("Success: Function defined.")
    except Exception as e:
        print(f"FAILED to define function: {e}")
        return

    # 2. Evaluate it
    # g(0.5) = min(0.5, 0) + max(0.5, 1) = 0 + 1 = 1
    # g(-2) = min(-2, 0) + max(-2, 1) = -2 + 1 = -1
    val1 = evaluate_function("g", [0.5])
    print(f"g(0.5) = {val1} (Expected 1)")
    
    val2 = evaluate_function("g", [-2])
    print(f"g(-2) = {val2} (Expected -1)")
    
    if abs(val1 - 1) < 1e-9 and abs(val2 - (-1)) < 1e-9:
        print("VERIFICATION PASSED: min/max behave correctly.")
    else:
        print("VERIFICATION FAILED: Incorrect evaluation results.")

if __name__ == "__main__":
    test_min_persistence()
