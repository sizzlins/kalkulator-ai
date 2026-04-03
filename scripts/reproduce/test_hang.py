
import sys
import os
import numpy as np
import sympy as sp
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.parser import safe_sympy_parse
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
from kalkulator_pkg.config import ALLOWED_SYMPY_NAMES

def test_svd_seed_evaluation():
    print("Starting SVD Seed Evaluation Test...")
    
    # The problematic seed string from the log
    seed_str = "((0.1048 + x - 0.08406*x**2 - 0.2035*x**3 - 0.007566*x**4)/(-0.09163 - 1.236*x - 0.9391*x**2 - 0.08152*x**3 - 0.0004987*x**4)) * 1.21294 + 1.41759"
    
    variable_names = ["x"]
    
    print(f"Parsing seed: {seed_str}")
    
    try:
        # replicate initialization logic from strategies.py
        local_dict = {v: sp.Symbol(v) for v in variable_names}
        # partial replication of full_local_dict
        full_local_dict = {**ALLOWED_SYMPY_NAMES, **local_dict}
        
        expr = safe_sympy_parse(seed_str, local_dict=full_local_dict)
        print("Parsing successful.")
        
        tree = ExpressionTree.from_sympy(expr, variable_names)
        print(f"Tree created. Complexity: {tree.complexity()}")
        
        # Create dummy data
        X = np.linspace(0, 20, 254).reshape(-1, 1)
        print(f"Evaluating on {len(X)} points...")
        
        start_time = time.time()
        result = tree.evaluate_fast(X)
        duration = time.time() - start_time
        
        print(f"Evaluation complete in {duration:.4f}s")
        print(f"Result mean: {np.mean(result)}")
        print(f"Result min/max: {np.min(result)} / {np.max(result)}")
    except Exception as e:
        print(f"Error (Evaluation): {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing BFGS Optimization ---")
    try:
        from kalkulator_pkg.symbolic_regression.operators import optimize_constants_bfgs
        print("Starting BFGS...")
        start_bfgs = time.time()
        # Mock tree.fitness to avoid NoneType error if BFGS checks it
        tree.fitness = 100.0
        
        optimized_tree = optimize_constants_bfgs(tree, X, result, max_iter=10) # Optimize against its own result (perfect fit)
        duration_bfgs = time.time() - start_bfgs
        print(f"BFGS complete in {duration_bfgs:.4f}s")
    except ImportError:
        print("BFGS module not found or import error.")
    except Exception as e:
        print(f"BFGS Failed: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    test_svd_seed_evaluation()
