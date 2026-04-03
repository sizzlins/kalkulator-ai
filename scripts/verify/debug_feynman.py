
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from kalkulator_pkg.benchmarks.feynman_equations import FEYNMAN_EQUATIONS
    from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree

    eq = FEYNMAN_EQUATIONS[0]
    print(f"Eq: {eq.name}, Formula: {eq.formula}, Vars: {eq.variables}")

    print("Attempting ExpressionTree.from_string...")
    tree = ExpressionTree.from_string(eq.formula)
    print(f"Tree created. Variables: {tree.variables}")

    print("Setting variables...")
    tree.variables = eq.variables
    print(f"Tree variables set to: {tree.variables}")

    print("Compiling secure...")
    code = tree.compile_secure()
    print(f"Code compiled: {code}, type: {type(code)}")

    print("Preparing namespace...")
    namespace = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "abs": np.abs, "pi": np.pi, "e": np.e,
        "arcsin": np.arcsin, "arccos": np.arccos,
        "arctan": np.arctan, "sinh": np.sinh,
        "cosh": np.cosh, "tanh": np.tanh
    }

    print("Evaluating code in namespace...")
    func = eval(code, namespace)
    print(f"Func type: {type(func)}")
    
    print("Calling func(0)...")
    res = func(0)
    print(f"Result(0): {res}")

except Exception as e:
    import traceback
    print("=== EXCEPTION ===")
    traceback.print_exc()
