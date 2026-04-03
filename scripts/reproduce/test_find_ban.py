
import sys
import os
import math
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from kalkulator_pkg.function_manager import find_function_from_data

class MockContext:
    def __init__(self, banned=None):
        self.banned_operators = banned or []
        self.function_registry = {}

def test_find_ban():
    print("Testing find_function_from_data Ban Enforcement...")
    
    # Data for sqrt(x)
    X = [(x,) for x in [1, 4, 9, 16, 25]]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    data = list(zip(X, y))
    
    # 1. Test WITHOUT ban
    print("\n1. Testing without bans...")
    ctx = MockContext()
    success, func_str, _, _ = find_function_from_data(ctx, data, ["x"], verbose=True)
    print(f"Result: {func_str}")
    
    if not success or not ("sqrt" in func_str or "0.5" in func_str):
        print("WARNING: Baseline failed to find sqrt(x). SVD might be flaky on small data.")
        # Try more data points if needed, but 5 exact points should trigger SVD or heuristics
        if not success:
             print("Baseline failed.")
             # We can't test ban if baseline doesn't work.
             # SVD usually needs more points? Or maybe "Hybrid Mode" handles it.
             # Let's try to add more points.
             X = [(x,) for x in np.linspace(1, 100, 20)]
             y = [math.sqrt(x[0]) for x in X]
             data = list(zip(X, y))
             success, func_str, _, _ = find_function_from_data(ctx, data, ["x"], verbose=True)
             print(f"Retry Result: {func_str}")

    if success and ("sqrt" in func_str or "0.5" in func_str):
        print("PASS: Baseline found sqrt(x).")
    else:
        print("FAIL: Baseline could not find sqrt(x). Cannot proceed with ban test.")
        return

    # 2. Test WITH ban
    print("\n2. Testing with 'ban pow' AND 'ban sqrt'...")
    ctx_banned = MockContext(banned=["pow", "^", "**", "sqrt"])
    
    try:
        success, func_str, _, _ = find_function_from_data(ctx_banned, data, ["x"], verbose=True)
        print(f"Result: {func_str}")
        
        if success and ("^" in func_str or "**" in func_str or "pow" in func_str):
            print("FAIL: find_function_from_data returned power function despite ban!")
            # We want this to Fail initially
        elif success:
             print(f"PASS? Returned {func_str} (maybe linear approximation?)")
        else:
             print("PASS: Could not find function (as expected).")
             
    except Exception as e:
        print(f"Error during execution: {e}")
        pass

if __name__ == "__main__":
    test_find_ban()
