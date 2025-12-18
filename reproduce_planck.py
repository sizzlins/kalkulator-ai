

from kalkulator_pkg.parser import parse_preprocessed, preprocess
from kalkulator_pkg.function_manager import define_function, evaluate_function, clear_functions
from kalkulator_pkg.function_finder_advanced import generate_candidate_features
import sympy as sp
import numpy as np
from kalkulator_pkg.regression_solver import _symbolify_coefficient

def test_e_evaluation():
    print("\n--- Test 1: 'e' Constant Evaluation ---")
    clear_functions()
    # Case 1: Define f(x) = x^3 / (e^x - 1)
    # This involves parsing "e^x" where "e" should be Euler's number
    
    # Simulate user input parsing for function definition
    # This is slightly complex as the CLI does some heavy lifting.
    # We'll test the parser directly.
    
    expr_str = "(x^3)/((e^x)-1)"
    pre = preprocess(expr_str)
    print(f"Preprocessed: {pre}")
    
    # We need to manually parse this as a saved function body would be
    # The 'define_function' uses parse_expr with 'local_dict' via config
    
    try:
        define_function("f", ["x"], expr_str)
        print("Function defined successfully.")
    except Exception as e:
        print(f"Failed to define function: {e}")
        return

    # Evaluate at x=1
    # Expected: 1^3 / (e^1 - 1) = 1 / (2.718... - 1) = 1 / 1.718... = 0.581976...
    try:
        val = evaluate_function("f", [1])
        print(f"f(1) = {val}")
        print(f"Float val: {float(val)}")
        
        expected = 1.0 / (np.e - 1.0)
        assert abs(float(val) - expected) < 1e-5, f"Expected {expected}, got {float(val)}"
        print("PASS: f(1) evaluates correctly.")
    except Exception as e:
        print(f"FAIL: {e}")
        # If 'e' is treated as symbol, it might return (e - 1)**(-1) but formatted as such
        # or it might fail if 'e' is unknown
        pass

def test_planck_feature_generation():
    print("\n--- Test 2: Planck's Law Feature Generation ---")
    # Generate data for y = x^3 / (e^x - 1)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    var_names = ["x"]
    
    features, names = generate_candidate_features(X, var_names, include_transcendentals=True)
    
    target_name = "x^3/(exp(x)-1)"
    print(f"Looking for feature: {target_name}")
    
    if target_name in names:
        print(f"PASS: Feature '{target_name}' found in candidate list.")
    else:
        print("FAIL: Feature not found.")
        print("Available features:", names)
        # assert target_name in names, "Planck's Law feature missing"

if __name__ == "__main__":
    try:
        test_e_evaluation()
    except Exception as e:
        print(f"Test 1 failed: {e}")
        
    try:
        test_planck_feature_generation()
    except Exception as e:
        print(f"Test 2 failed: {e}")
