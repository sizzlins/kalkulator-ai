
import sys
import os
import numpy as np

# Set up path
sys.path.insert(0, os.getcwd())

from kalkulator_pkg.cli.repl_commands import _handle_evolve
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

def test_overflow_fix():
    print("Testing for overflow warnings...")
    # Simulate the user's data that caused overflow
    # f(3,4) = 4, f(5,6) = 6 ... implies f(x,y) ~ y or max(x,y)
    
    # We'll use a larger range to provoke it if possible
    X = np.array([[3,4], [5,6], [7,8], [9,10], [99,99], [1000, 1000]])
    y = np.array([4, 6, 8, 10, 99, 1000])
    
    config = GeneticConfig(
        generations=5,
        population_size=50,
        verbose=True
    )
    regressor = GeneticSymbolicRegressor(config=config)
    
    try:
        # We can't capture warnings easily without warnings module, 
        # but if they appear in stdout/stderr we'll see them.
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            regressor.fit(X, y, ['x', 'y']) # fit signature confirmed to take X, y, var_names
            
            if len(w) > 0:
                print(f"[FAIL] Caught {len(w)} warnings:")
                for warning in w:
                    print(f"- {warning.category.__name__}: {warning.message}")
            else:
                print("[SUCCESS] No warnings captured.")
                
    except Exception as e:
        print(f"[CRASH] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_overflow_fix()
