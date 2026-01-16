
import time
import numpy as np
import sympy as sp
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, NodeType

def benchmark():
    # 1. Setup Data
    X = np.random.rand(1000, 2) * 10 - 5
    variables = ["x0", "x1"]
    
    # 2. Create complex tree: sin(x0) * cos(x1) + (x0 - x1)^2 / 0.5
    # (Just a random complex structure)
    tree = ExpressionTree.random_tree(variables, max_depth=5, operators=['add', 'sub', 'mul', 'sin', 'cos', 'pow'], method='full')
    print(f"Tree: {tree}")
    
    # 3. Benchmark RPN (Current Implementation)
    start = time.time()
    for _ in range(100):
        # Clear cache to simulate new individual
        tree._rpn_stack = None 
        res_rpn = tree.evaluate(X)
    rpn_time = (time.time() - start) / 100
    print(f"RPN (Cold) Time: {rpn_time*1000:.4f} ms")
    
    # 4. Benchmark Lambdify (Manual)
    start = time.time()
    for _ in range(100):
        sym_expr = tree.to_sympy()  # This overhead is unavoidable for lambdify
        f = sp.lambdify(variables, sym_expr, modules='numpy')
        res_lamb = f(X[:,0], X[:,1])
    lamb_time = (time.time() - start) / 100
    print(f"Lambdify (Cold) Time: {lamb_time*1000:.4f} ms")
    
    # 5. Check Correctness (Use simpler tree to avoid overflow/clipping differences)
    simple_tree = ExpressionTree.random_tree(variables, max_depth=3, operators=['add', 'sub', 'mul', 'sin', 'cos'], method='full')
    simple_tree._rpn_stack = None
    res_rpn_safe = simple_tree.evaluate(X)
    
    sym_expr = simple_tree.to_sympy()
    f_ref = sp.lambdify(variables, sym_expr, modules='numpy')
    res_ref = f_ref(X[:,0], X[:,1])
    
    valid = np.isfinite(res_ref)
    if np.sum(valid) > 0:
        err = np.max(np.abs(res_rpn_safe[valid] - res_ref[valid]))
        print(f"Correctness Max Error: {err:.2e}")
        assert err < 1e-6, "RPN Evaluator incorrect on safe domain!"
    else:
        print("All NaNs (skipping correctness check)")
        
    print(f"Speedup: {lamb_time / rpn_time:.1f}x")

if __name__ == "__main__":
    benchmark()
