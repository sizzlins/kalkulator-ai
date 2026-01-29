
import numpy as np
import sys
import sympy as sp
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree

def reproduce_gamma_evolution():
    print("Reproduction: Testing 'gamma' (Factorial) Evolution via Genetic Engine")
    print("Target Function: y = x!")
    
    # Generate data for y = x! (1 to 6)
    X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y = np.array([1, 2, 6, 24, 120, 720])
    
    print(f"Data X: {X.flatten()}")
    print(f"Data y: {y}")
    
    # Configure Genetic Engine with Factorial operator
    # We use a high population/generations to ensure discovery if forensics fail,
    # but forensics should trigger since we provide no seeds.
    config = GeneticConfig(
        population_size=200,
        generations=20, # Should be enough with forensics
        verbose=True,   # Enable verbose to see forensic output
        operators=["add", "sub", "mul", "div", "pow", "factorial"], # Explicitly include factorial
        parsimony_coefficient=0.01
    )
    
    print("\n--- Initializing GeneticSymbolicRegressor ---")
    regressor = GeneticSymbolicRegressor(config)
    
    print("--- Fitting Model ---")
    # minimal setup
    try:
        regressor.fit(X, y, variable_names=['x'])
    except Exception as e:
        print(f"CRASH: Genetic Engine failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    best_expr = regressor.best_tree.to_string() if regressor.best_tree else "None"
    print(f"\nFound Function: {best_expr}")
    
    if regressor.best_tree:
        sympy_expr = regressor.best_tree.to_sympy()
        print(f"SymPy Expression: {sympy_expr}")
        
    # Check for gamma or factorial
    # Note: ExpressionTree might return 'factorial(x)' or 'gamma(x + 1)'
    res_str = str(best_expr).lower()
    sympy_str = str(sympy_expr).lower() if regressor.best_tree else ""
    
    if 'gamma' in res_str or 'factorial' in res_str or 'gamma' in sympy_str or 'factorial' in sympy_str:
        print("SUCCESS: Found function with gamma/factorial.")
    else:
        print(f"FAILURE: Did not find gamma/factorial. Found: {best_expr}")
        sys.exit(1)

if __name__ == "__main__":
    reproduce_gamma_evolution()
