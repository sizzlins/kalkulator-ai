import sys
import os
sys.path.insert(0, os.getcwd())

if __name__ == "__main__":
    from kalkulator_pkg.function_manager import define_function, _function_registry
    from kalkulator_pkg.worker import evaluate_safely
    
    # Define f(x) = i*pi
    define_function("f", ["x"], "I*pi")
    
    # Check what f is stored as
    f_info = _function_registry.get("f")
    print(f"Stored function f: {f_info}")
    
    # Now evaluate e^f(x)
    # The parser should substitute f(x) -> I*pi, then evaluate e^(I*pi)
    res = evaluate_safely("e^f(x)")
    print(f"e^f(x) result: {res}")
    
    # Also try direct evaluation
    res2 = evaluate_safely("e^(I*pi)")
    print(f"e^(I*pi) direct: {res2}")
    
    # Try with exp notation
    res3 = evaluate_safely("exp(I*pi)")
    print(f"exp(I*pi): {res3}")
