"""Reproduce the f(pi/2) = pi*f/2 bug."""
import sys

sys.path.insert(0, ".")

if __name__ == "__main__":
    from kalkulator_pkg.function_manager import define_function
    from kalkulator_pkg.worker import evaluate_safely

    # Define f(x) = x*sin(x)
    print("Defining f(x) = x*sin(x)")
    define_function("f", ["x"], "x*sin(x)")

    # Test f(pi/2) - should be (pi/2) * sin(pi/2) = (pi/2) * 1 = pi/2
    print("\nTesting f(pi/2)...")
    result = evaluate_safely("f(pi/2)")
    print(f"Result: {result}")

    expected = "pi/2"
    if result['ok']:
        res_str = result['result']
        if "f" in res_str:
            print(f"FAILURE: Result contains 'f' as a symbol: {res_str}")
        elif "pi" in res_str.lower():
            print(f"Result involves pi: {res_str}")
        else:
            print(f"Result: {res_str}")
    else:
        print(f"ERROR: {result['error']}")
