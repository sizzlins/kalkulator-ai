
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Mock evaluate_safely BEFORE importing repl_commands
sys.modules['kalkulator_pkg.worker'] = MagicMock()
sys.modules['kalkulator_pkg.worker'].evaluate_safely = MagicMock(return_value={"ok": True, "result": "1.0"})

# Mock list_functions logic
def mock_list_functions(ctx):
    return {
        "f": (["x", "y"], "x*y"), # 2 args
        "g": (["x"], "x^2"),       # 1 arg
        "h": (["x", "y", "z"], "x+y+z"), # 3 args
    }

sys.modules['kalkulator_pkg.function_manager'] = MagicMock()
sys.modules['kalkulator_pkg.function_manager'].list_functions = mock_list_functions

from kalkulator_pkg.cli import repl_commands

def test_multi_arg_call():
    print("Testing multi-arg call generation...")
    
    # Mock context
    ctx = MagicMock()
    
    # Override _CALL_SETS for testing
    repl_commands._CALL_SETS["test"] = [1, 2]
    
    # Test 2-arg function 'f'
    print("\n--- Testing f(x,y) ---")
    results = repl_commands._get_call_results(ctx, "f", "test")
    for r in results:
        print(r)
        
    # Verify expected output format: f(1, 2), f(2, 1) (Shifted Rotation)
    assert any("f(1, 2)" in r for r in results), "Failed to generate f(1, 2)"
    assert any("f(2, 1)" in r for r in results), "Failed to generate f(2, 1)"
    
    # Test 1-arg function 'g'
    print("\n--- Testing g(x) ---")
    results = repl_commands._get_call_results(ctx, "g", "test")
    for r in results:
        print(r)
        
    # Verify expected output format: g(1), g(2)
    assert any("g(1)" in r for r in results), "Failed to generate g(1)"
    assert any("g(2)" in r for r in results), "Failed to generate g(2)"

    # Test 3-arg function 'h'
    print("\n--- Testing h(x,y,z) ---")
    results = repl_commands._get_call_results(ctx, "h", "test")
    for r in results:
        print(r)
    
    # Verify expected output format: h(1, 2, 1), h(2, 1, 2)
    assert any("h(1, 2, 1)" in r for r in results), "Failed to generate h(1, 2, 1)"
    assert any("h(2, 1, 2)" in r for r in results), "Failed to generate h(2, 1, 2)"

    print("\nSuccess!")

if __name__ == "__main__":
    test_multi_arg_call()
