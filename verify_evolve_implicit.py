
import sys
import numpy as np
from kalkulator_pkg.cli.repl_commands import _handle_evolve

def test_implicit_evolve():
    print("Testing Implicit Evolve...")
    
    # Mock REPL variables
    variables = {
        'x': 'np.array([1, 2, 3, 4, 5])',
        'y': 'np.array([2, 4, 6, 8, 10])'
    }
    
    cmd = "evolve f(x)"
    
    try:
        _handle_evolve(cmd, variables)
    except Exception as e:
        print(f"CRASH: {e}")

if __name__ == "__main__":
    test_implicit_evolve()
