
import sys
import numpy as np
from unittest.mock import patch
from kalkulator_pkg.cli.repl_core import REPL
from kalkulator_pkg.symbolic_regression import GeneticSymbolicRegressor

def mock_fit(self, X, y, variable_names=None):
    print(f"\n[DEBUG] MOCK_FIT CALLED")
    print(f"[DEBUG] X shape: {X.shape}")
    print(f"[DEBUG] X values: {X.flatten()}")
    print(f"[DEBUG] y shape: {y.shape}")
    print(f"[DEBUG] y values: {y}")
    print(f"[DEBUG] variable_names: {variable_names}")
    
    # Return a dummy result to satisfy the caller
    class DummyPareto:
        def get_knee_point(self):
            return None
        def get_best(self): # Return a dummy solution
           class DummySol:
               mse = 0.0
               complexity = 1
               expression = "DEBUG_EXPR"
               sympy_expr = "DEBUG_EXPR"
           return DummySol()
    return DummyPareto()

if __name__ == "__main__":
    # Patch the fit method
    with patch('kalkulator_pkg.symbolic_regression.GeneticSymbolicRegressor.fit', side_effect=mock_fit, autospec=True):
        repl = REPL()
        
        # Test 1: The integer case
        print("\n--- TEST 1: Integers ---")
        repl.process_input("x=[1, 2, 3], y=[2, 4, 6], evolve f(x)")
        
        # Test 2: The float case (which gave 1/e result)
        print("\n--- TEST 2: Floats ---")
        repl.process_input("x=[1, 2, 3], y=[0.5, 1, 1.5], evolve f(x)")
