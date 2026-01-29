
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Mock evaluate_safely
sys.modules['kalkulator_pkg.worker'] = MagicMock()
sys.modules['kalkulator_pkg.worker'].evaluate_safely = MagicMock(return_value={"ok": True, "result": "1.0"})

# Mock function_manager
def mock_list_functions(ctx):
    return {
        "f": (["x", "y"], "x*y"), # 2 args
    }

sys.modules['kalkulator_pkg.function_manager'] = MagicMock()
sys.modules['kalkulator_pkg.function_manager'].list_functions = mock_list_functions

# Mock parser if needed (it is needed for imports)
sys.modules['kalkulator_pkg.parser'] = MagicMock()

from kalkulator_pkg.cli import repl_commands

def test_randomized_call():
    print("Testing randomized callr generation...")
    
    # Mock context
    ctx = MagicMock()
    
    # Override _CALL_SETS for testing (range of inputs)
    repl_commands._CALL_SETS["test"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Test randomize=True
    print("\n--- Testing callr f(x,y) ---")
    results = repl_commands._get_call_results(ctx, "f", "test", randomize=True)
    
    symmetry_broken = False
    for r in results:
        print(r)
        # Check if any call has f(a, b) where a != b
        # Format: f(a, b) = ...
        # Parse inputs
        try:
            call_part = r.split('=')[0].strip()
            args_str = call_part[2:-1] # strip f( and )
            args = [x.strip() for x in args_str.split(',')]
            if len(args) == 2 and args[0] != args[1]:
                symmetry_broken = True
        except:
            pass
            
    if symmetry_broken:
        print("\nSuccess: Symmetry broken! Found inputs where x != y.")
    else:
        print("\nFailure: Strict symmetry observed (x == y).")
        # Assert failure only if set is large enough that collision implies logic fail
        raise AssertionError("Randomization failed to break symmetry")

if __name__ == "__main__":
    test_randomized_call()
