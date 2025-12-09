import sys
import os
sys.path.insert(0, os.getcwd())

if __name__ == "__main__":
    from kalkulator_pkg.function_manager import define_function, _function_registry
    from kalkulator_pkg.parser import preprocess, parse_preprocessed
    
    # Define f(x) = i*pi
    define_function("f", ["x"], "I*pi")
    print(f"Defined f: {_function_registry.get('f')}")
    
    # Try preprocessing without f defined in registry
    # First, clear and test raw preprocessing
    from kalkulator_pkg.function_manager import clear_functions
    clear_functions()
    
    preprocessed_no_f = preprocess("e^f(x)")
    print(f"Preprocessed (no f defined): {preprocessed_no_f}")
    
    # Now define f again
    define_function("f", ["x"], "I*pi")
    
    preprocessed_with_f = preprocess("e^f(x)")
    print(f"Preprocessed (f defined): {preprocessed_with_f}")
    
    # Check - are they different?
    print(f"Are they different? {preprocessed_no_f != preprocessed_with_f}")
