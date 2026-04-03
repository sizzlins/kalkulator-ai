
import time
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
import numba

def benchmark():
    print("Benchmarking Expression Evaluation...")
    
    # Setup Data
    N = 1_000_000
    x = np.linspace(0, 100, N)
    
    # 1. Simple Case: sin(x) + x**2
    expr_str_1 = "sin(x) + x**2"
    tree_1 = ExpressionTree.from_string(expr_str_1)
    
    # 2. Complex Case: deep tree
    # sin(sin(sin(x))) + cos(cos(cos(x))) + x**2 + x**3 + x**4 + exp(x/100)
    expr_str_2 = "sin(sin(sin(x))) + cos(cos(cos(x))) + x**2 + x**3 + x**4 + exp(x/100)"
    tree_2 = ExpressionTree.from_string(expr_str_2)
    
    # 3. Very Complex (Genetic Programming style)
    # (x + sin(x)) * (x - cos(x)) / (x**2 + 1) + exp(-x) * log(abs(x) + 1)
    expr_str_3 = "(x + sin(x)) * (x - cos(x)) / (x**2 + 1) + exp(-x) * log(abs(x) + 1)"
    tree_3 = ExpressionTree.from_string(expr_str_3)

    cases = [
        ("Simple", expr_str_1, tree_1),
        ("Medium", expr_str_2, tree_2),
        ("Complex", expr_str_3, tree_3),
    ]

    for name, expr_str, tree in cases:
        print(f"\n--- Case: {name} ---")
        print(f"Expr: {expr_str}")
        
    # 0. Warmup (Force JIT compilation of the evaluator kernel)
    print("Warming up Numba...")
    t0 = time.time()
    # Create a dummy tree to trigger _get_numba_evaluator and import
    ExpressionTree.from_string("x+1").evaluate(x[:10], use_numba=True)
    print(f"Warmup took: {time.time() - t0:.4f}s")

    for name, expr_str, tree in cases:
        print(f"\n--- Case: {name} ---")
        print(f"Expr: {expr_str}")
        
        # A. Baseline: ExpressionTree.evaluate (Python)
        t0 = time.time()
        res_py = tree.evaluate(x, use_numba=False)
        t_py = time.time() - t0
        print(f"Python RPN:        {t_py:.4f}s")
        
        # B. ExpressionTree.evaluate (Numba)
        # Should be fast if kernel is already compiled
        t0 = time.time()
        res_jit = tree.evaluate(x, use_numba=True)
        t_jit = time.time() - t0
        print(f"Numba RPN (Tree):  {t_jit:.4f}s (Speedup: {t_py/t_jit:.2f}x)")
        
        # C. Numba JIT (Explicit) - for comparison
        # (Skip explicit compilation benchmark to keep it simple, focus on Tree integration)
        
        # Verify correctness
        err = np.max(np.abs(res_py - res_jit))
        if err > 1e-6:
             print(f"WARNING: Divergence! Max error: {err}")

if __name__ == "__main__":
    benchmark()
