
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.calculus import differentiate
from kalkulator_pkg.function_manager import define_function, evaluate_function


def test_calculus_bug():
    print("Testing Calculus Expansion Bug")
    
    # 1. Define f(x) = x^2 + 100
    print("Defining f(x) = x^2 + 100")
    try:
        define_function("f", ["x"], "x^2 + 100")
    except Exception as e:
        print(f"Failed to define function: {e}")
        return

    # 2. Try to differentiate f(x)
    # The REPL likely calls differentiate("diff(f(x), x)") or similar?
    # Actually, the user typed `diff(f(x), x)`.
    # Based on cli.py logic (which I need to check), it probably passes "f(x)" to differentiate.
    
    # 2. Try to differentiate f(x) using evaluate_safely("diff(f(x), x)")
    # This simulates exactly what cli.py does.
    
    print("Differentiating 'f(x)' via evaluate_safely('diff(f(x), x)')...")
    from kalkulator_pkg.worker import evaluate_safely

    try:
        # In the REPL, the input is 'diff(f(x), x)'
        expr = "diff(f(x), x)"
        result = evaluate_safely(expr)
        print(f"Result: {result}")
        
        if result['ok']:
            # Expected: 2*x
            # Bug: f
            res_str = result['result']
            if res_str == "f":
                print("FAILURE: Reproduced bug! Result is 'f' (symbolic) instead of '2*x' (evaluated).")
            elif "2*x" in res_str:
                print("SUCCESS: Result contains 2*x")
            elif "Derivative" in res_str:
                print(f"FAILURE: Result is unevaluated derivative: {res_str}")
            else:
                 print(f"UNKNOWN: Result is {res_str}")
        else:
             print(f"ERROR: {result['error']}")

    except Exception as e:
        print(f"FAILURE: Crashed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_calculus_bug()
