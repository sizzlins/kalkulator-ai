
from kalkulator_pkg.cli.repl_commands import _handle_evolve
import numpy as np

def test_negative_disambiguation():
    print("Testing disambiguation with negative numbers...")
    
    # x includes negative number -> Abs(x) != x
    # x = -2, y = -4 (Target: 2x)
    # If model is 2*Abs(x), it would define y = 4 (Error)
    
    variables = {
        'x': np.array([-2, 1, 3]),
        'y': np.array([-4, 2, 6])
    }
    
    # We expect the AI to find 2*x and REJECT 2*Abs(x)
    
    try:
        _handle_evolve("evolve f(x)", variables)
    except Exception as e:
        print(f"Crash: {e}")

if __name__ == "__main__":
    test_negative_disambiguation()
