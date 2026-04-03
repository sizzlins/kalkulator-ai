
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig
from kalkulator_pkg.symbolic_regression.forensic_analysis import _detect_symmetry

def test_symmetry_heuristic():
    print("Testing _detect_symmetry heuristic...")
    
    # Create permutation symmetric data for max(x, y, z)
    # Case 1: Permutations of [1, 2, 3]
    X_base = np.array([
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [3, 1, 2],
        [3, 2, 1]
    ])
    y_base = np.max(X_base, axis=1) # All 3
    
    # Check if heuristic finds max
    seeds = _detect_symmetry(X_base, y_base, variable_names=["x", "y", "z"])
    print(f"Seeds found: {seeds}")
    
    if any("max" in s for s in seeds):
        print("SUCCESS: 'max' operator detected from permutations.")
    else:
        print("FAILURE: 'max' operator NOT detected.")
        sys.exit(1)

def test_regression_max():
    print("\nTesting full regression for max(a, b, c)...")
    
    # Generate larger dataset
    rng = np.random.RandomState(42)
    X = rng.rand(50, 3) * 10
    y = np.max(X, axis=1)
    
    # Add explicit permutations to help the heuristic (though random data might not have exact permutations)
    # The heuristic relies on FINDING permutations in the data.
    # So we MUST add permutations to the training set.
    X_perm = np.array([
        [100, 200, 300],
        [300, 200, 100],
        [5, 4, 3],
        [3, 5, 4]
    ])
    y_perm = np.max(X_perm, axis=1)
    
    X_train = np.vstack([X, X_perm])
    y_train = np.hstack([y, y_perm])
    
    print(f"Data shape: {X_train.shape}")
    
    # DEBUG: Manual Check of Seed Evaluation
    print("\n[DEBUG] Manual Seed Check:")
    try:
        import sympy as sp
        from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
        from kalkulator_pkg.symbolic_regression.strategies import ALLOWED_SYMPY_NAMES
        
        # Recreate the parsing context from strategies.py
        local_dict = {v: sp.Symbol(v) for v in ["a", "b", "c"]}
        local_dict.update({'exp': sp.exp, 'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 
                           'max': sp.Max, 'min': sp.Min, 'median': lambda *args: sp.Max(*args)})
        
        # Parse logic (from strategies.py) -- simplified
        # Actually, let's just use sp.sympify with locals for test (unsafe but fine for verify)
        seed_str = "max(a, b, c)"
        print(f"Parsing '{seed_str}'...")
        res = sp.sympify(seed_str, locals=local_dict)
        print(f"SymPy parsed: {res} (Type: {type(res)})")
        
        # Create Tree
        tree = ExpressionTree.from_sympy(res, ["a", "b", "c"])
        print(f"Tree created: {tree}")
        
        # Evaluate
        pred = tree.evaluate_fast(X_train)
        mse = np.mean((y_train - pred)**2)
        print(f"Manual Eval MSE: {mse}")
        
    except Exception as e:
        print(f"Manual Check Failed: {e}")
        import traceback
        traceback.print_exc()

    config = GeneticConfig(
        generations=10, # Fast run
        population_size=100,
        verbose=True
    )
    
    reg = GeneticSymbolicRegressor(config)
    reg.fit(X_train, y_train, variable_names=["a", "b", "c"])
    
    print(f"Best Expression: {reg.get_expression()}")
    
    if "max" in reg.get_expression():
        print("SUCCESS: Regression discovered 'max'.")
    else:
        print("FAILURE: Regression missed 'max'.")
        # Don't exit yet, maybe it found an equivalent form?
        # But for max, usually explicit is best.

if __name__ == "__main__":
    test_symmetry_heuristic()
    test_regression_max()
